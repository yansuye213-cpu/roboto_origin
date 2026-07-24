// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/model/centroidal_model_info.hpp"

#include <pinocchio/multibody/model.hpp>

#include <ocs2_centroidal_model/FactoryFunctions.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace whole_body_mpc {

namespace {

std::vector<int> make_joint_index_map(
    const ocs2::PinocchioInterface& pinocchio_interface,
    const std::vector<std::string>& joint_order) {
    const auto& model = pinocchio_interface.getModel();
    std::unordered_map<std::string, int> model_index_by_joint_name;
    for (pinocchio::JointIndex joint_id = 1; joint_id < model.joints.size(); joint_id++) {
        const auto& joint = model.joints[joint_id];
        if (joint.nq() == 1 && joint.nv() == 1) {
            const int model_joint_index = joint.idx_q() - 6;
            if (model_joint_index >= 0) {
                model_index_by_joint_name[model.names[joint_id]] = model_joint_index;
            }
        }
    }

    std::vector<int> model_joint_index_by_config_joint;
    model_joint_index_by_config_joint.reserve(joint_order.size());
    for (const std::string& joint_name : joint_order) {
        const auto it = model_index_by_joint_name.find(joint_name);
        if (it == model_index_by_joint_name.end()) {
            throw std::runtime_error(
                "OCS2 centroidal model joint not found in filtered URDF: " +
                joint_name);
        }
        model_joint_index_by_config_joint.push_back(it->second);
    }
    return model_joint_index_by_config_joint;
}

ocs2::vector_t make_nominal_joint_angles(
    const std::vector<double>& configured_nominal,
    const std::vector<int>& model_joint_index_by_config_joint,
    size_t actuated_dof_num) {
    ocs2::vector_t nominal = ocs2::vector_t::Zero(actuated_dof_num);
    if (configured_nominal.empty()) {
        return nominal;
    }
    if (configured_nominal.size() != model_joint_index_by_config_joint.size()) {
        throw std::runtime_error(
            "stand whole-body nominal joint angle size does not match joint_order");
    }
    for (size_t i = 0; i < configured_nominal.size(); i++) {
        const int model_index = model_joint_index_by_config_joint[i];
        if (model_index < 0 || model_index >= nominal.size()) {
            throw std::runtime_error("OCS2 centroidal model joint index is invalid");
        }
        nominal[model_index] = configured_nominal[i];
    }
    return nominal;
}

}  // namespace

Ocs2CentroidalModel::Ocs2CentroidalModel(
    ocs2::PinocchioInterface pinocchio_interface,
    ocs2::CentroidalModelInfo info,
    std::vector<int> model_joint_index_by_config_joint)
    : pinocchio_interface_(std::move(pinocchio_interface)),
      info_(std::move(info)),
      rbd_conversions_(pinocchio_interface_, info_),
      model_joint_index_by_config_joint_(
          std::move(model_joint_index_by_config_joint)) {}

Eigen::VectorXd Ocs2CentroidalModel::model_order_joint_vector(
    const Eigen::VectorXd& configured_order_vector) const {
    if (configured_order_vector.size() !=
        static_cast<int>(model_joint_index_by_config_joint_.size())) {
        throw std::runtime_error(
            "configured joint vector size does not match OCS2 centroidal joint_order");
    }
    Eigen::VectorXd model_order =
        Eigen::VectorXd::Zero(static_cast<int>(info_.actuatedDofNum));
    for (size_t i = 0; i < model_joint_index_by_config_joint_.size(); i++) {
        const int model_index = model_joint_index_by_config_joint_[i];
        if (model_index < 0 || model_index >= model_order.size()) {
            throw std::runtime_error("OCS2 centroidal model joint index is invalid");
        }
        model_order[model_index] = configured_order_vector[static_cast<int>(i)];
    }
    return model_order;
}

Eigen::VectorXd Ocs2CentroidalModel::configured_order_joint_vector(
    const Eigen::VectorXd& model_order_vector) const {
    if (model_order_vector.size() != static_cast<int>(info_.actuatedDofNum)) {
        throw std::runtime_error(
            "model joint vector size does not match OCS2 centroidal actuated DoF");
    }
    Eigen::VectorXd configured_order =
        Eigen::VectorXd::Zero(
            static_cast<int>(model_joint_index_by_config_joint_.size()));
    for (size_t i = 0; i < model_joint_index_by_config_joint_.size(); i++) {
        const int model_index = model_joint_index_by_config_joint_[i];
        if (model_index < 0 || model_index >= model_order_vector.size()) {
            throw std::runtime_error("OCS2 centroidal model joint index is invalid");
        }
        configured_order[static_cast<int>(i)] = model_order_vector[model_index];
    }
    return configured_order;
}

std::unique_ptr<Ocs2CentroidalModel> create_ocs2_centroidal_model(
    const Ocs2CentroidalModelConfig& config) {
    if (config.urdf_path.empty()) {
        throw std::runtime_error("OCS2 centroidal model requires a URDF path");
    }
    if (!std::filesystem::exists(config.urdf_path)) {
        throw std::runtime_error("OCS2 centroidal URDF path does not exist: " +
                                 config.urdf_path);
    }
    if (config.left_foot_frame.empty() || config.right_foot_frame.empty()) {
        throw std::runtime_error(
            "OCS2 centroidal model requires left and right foot frames");
    }
    if (config.joint_order.empty()) {
        throw std::runtime_error("OCS2 centroidal model requires joint_order");
    }

    ocs2::PinocchioInterface pinocchio_interface =
        ocs2::centroidal_model::createPinocchioInterface(config.urdf_path,
                                                         config.joint_order);
    const std::vector<int> model_joint_index_by_config_joint =
        make_joint_index_map(pinocchio_interface, config.joint_order);
    const ocs2::vector_t nominal_joint_angles =
        make_nominal_joint_angles(config.nominal_joint_angles,
                                  model_joint_index_by_config_joint,
                                  config.joint_order.size());
    std::vector<std::string> three_dof_contacts = {
        config.left_foot_frame, config.right_foot_frame};
    const std::vector<std::string> six_dof_contacts;
    ocs2::CentroidalModelInfo info =
        ocs2::centroidal_model::createCentroidalModelInfo(
            pinocchio_interface, ocs2::CentroidalModelType::FullCentroidalDynamics,
            nominal_joint_angles, three_dof_contacts, six_dof_contacts);
    if (info.numThreeDofContacts != 2 || info.numSixDofContacts != 0) {
        throw std::runtime_error(
            "OCS2 centroidal model expects exactly two 3-DoF foot contacts");
    }
    if (info.actuatedDofNum != config.joint_order.size()) {
        throw std::runtime_error(
            "OCS2 centroidal actuated DoF count does not match joint_order");
    }
    return std::make_unique<Ocs2CentroidalModel>(
        std::move(pinocchio_interface), std::move(info),
        model_joint_index_by_config_joint);
}

}  // namespace whole_body_mpc
