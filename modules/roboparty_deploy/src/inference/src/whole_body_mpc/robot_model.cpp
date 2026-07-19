// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/robot_model.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace whole_body_mpc {

struct RobotModel::Impl {
    pinocchio::Model model;
    pinocchio::Data data;
    pinocchio::FrameIndex base_frame_id = 0;
    pinocchio::FrameIndex left_foot_frame_id = 0;
    pinocchio::FrameIndex right_foot_frame_id = 0;
    std::vector<int> q_index_by_config_joint;
    std::vector<int> v_index_by_config_joint;
    double total_mass = 0.0;
};

RobotModel::RobotModel(Config config) : config_(std::move(config)) {
    validate_config();
    impl_ = std::make_unique<Impl>();

    if (config_.floating_base) {
        pinocchio::urdf::buildModel(config_.urdf_path, pinocchio::JointModelFreeFlyer(), impl_->model);
    } else {
        pinocchio::urdf::buildModel(config_.urdf_path, impl_->model);
    }
    impl_->data = pinocchio::Data(impl_->model);
    for (const auto& inertia : impl_->model.inertias) {
        impl_->total_mass += inertia.mass();
    }

    impl_->base_frame_id = impl_->model.getFrameId(config_.base_link);
    impl_->left_foot_frame_id = impl_->model.getFrameId(config_.left_foot_frame);
    impl_->right_foot_frame_id = impl_->model.getFrameId(config_.right_foot_frame);
    if (impl_->base_frame_id >= impl_->model.nframes) {
        throw std::runtime_error("base frame not found in URDF: " + config_.base_link);
    }
    if (impl_->left_foot_frame_id >= impl_->model.nframes) {
        throw std::runtime_error("left foot frame not found in URDF: " + config_.left_foot_frame);
    }
    if (impl_->right_foot_frame_id >= impl_->model.nframes) {
        throw std::runtime_error("right foot frame not found in URDF: " + config_.right_foot_frame);
    }

    std::unordered_map<std::string, std::pair<int, int>> joint_indices;
    for (pinocchio::JointIndex joint_id = 1; joint_id < impl_->model.joints.size(); joint_id++) {
        const std::string& joint_name = impl_->model.names[joint_id];
        const int nq = impl_->model.joints[joint_id].nq();
        const int nv = impl_->model.joints[joint_id].nv();
        if (nq == 1 && nv == 1) {
            model_joint_order_.push_back(joint_name);
            joint_indices[joint_name] = {impl_->model.joints[joint_id].idx_q(),
                                         impl_->model.joints[joint_id].idx_v()};
        }
    }

    impl_->q_index_by_config_joint.reserve(config_.joint_order.size());
    impl_->v_index_by_config_joint.reserve(config_.joint_order.size());
    for (const std::string& joint_name : config_.joint_order) {
        const auto it = joint_indices.find(joint_name);
        if (it == joint_indices.end()) {
            throw std::runtime_error("configured joint not found in URDF: " + joint_name);
        }
        impl_->q_index_by_config_joint.push_back(it->second.first);
        impl_->v_index_by_config_joint.push_back(it->second.second);
    }
}

RobotModel::~RobotModel() = default;

bool RobotModel::is_available() const {
    return true;
}

int RobotModel::nq() const {
    return impl_->model.nq;
}

int RobotModel::nv() const {
    return impl_->model.nv;
}

double RobotModel::total_mass() const {
    return impl_->total_mass;
}

Eigen::VectorXd RobotModel::neutral_configuration() const {
    return pinocchio::neutral(impl_->model);
}

Eigen::VectorXd RobotModel::zero_velocity() const {
    return Eigen::VectorXd::Zero(impl_->model.nv);
}

Eigen::VectorXd RobotModel::make_configuration(
    const std::vector<float>& joint_position, const Eigen::Vector3d& base_position,
    const Eigen::Quaterniond& base_orientation) const {
    if (joint_position.size() != config_.joint_order.size()) {
        throw std::runtime_error("joint_position size does not match stand_whole_body_joint_order");
    }
    Eigen::VectorXd q = neutral_configuration();
    if (config_.floating_base) {
        Eigen::Quaterniond normalized_base = base_orientation.normalized();
        q.segment<3>(0) = base_position;
        q.segment<4>(3) << normalized_base.x(), normalized_base.y(),
                           normalized_base.z(), normalized_base.w();
    }
    for (size_t i = 0; i < joint_position.size(); i++) {
        q[impl_->q_index_by_config_joint[i]] = static_cast<double>(joint_position[i]);
    }
    return q;
}

Eigen::VectorXd RobotModel::make_velocity(
    const std::vector<float>& joint_velocity, const Eigen::Vector3d& base_linear_velocity,
    const Eigen::Vector3d& base_angular_velocity) const {
    if (joint_velocity.size() != config_.joint_order.size()) {
        throw std::runtime_error("joint_velocity size does not match stand_whole_body_joint_order");
    }
    Eigen::VectorXd v = zero_velocity();
    if (config_.floating_base) {
        v.segment<3>(0) = base_linear_velocity;
        v.segment<3>(3) = base_angular_velocity;
    }
    for (size_t i = 0; i < joint_velocity.size(); i++) {
        v[impl_->v_index_by_config_joint[i]] = static_cast<double>(joint_velocity[i]);
    }
    return v;
}

RobotModel::Kinematics RobotModel::compute_kinematics(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v) const {
    if (q.size() != impl_->model.nq) {
        throw std::runtime_error("q size does not match Pinocchio model nq");
    }
    if (v.size() != impl_->model.nv) {
        throw std::runtime_error("v size does not match Pinocchio model nv");
    }

    pinocchio::forwardKinematics(impl_->model, impl_->data, q, v);
    pinocchio::updateFramePlacements(impl_->model, impl_->data);
    pinocchio::centerOfMass(impl_->model, impl_->data, q, v, false);
    pinocchio::computeJointJacobians(impl_->model, impl_->data, q);

    Kinematics output;
    output.com_position = impl_->data.com[0];
    output.com_velocity = impl_->data.vcom[0];

    const auto left_pose = impl_->data.oMf[impl_->left_foot_frame_id];
    output.left_foot_pose.linear() = left_pose.rotation();
    output.left_foot_pose.translation() = left_pose.translation();
    const auto right_pose = impl_->data.oMf[impl_->right_foot_frame_id];
    output.right_foot_pose.linear() = right_pose.rotation();
    output.right_foot_pose.translation() = right_pose.translation();

    output.left_foot_jacobian.setZero(6, impl_->model.nv);
    output.right_foot_jacobian.setZero(6, impl_->model.nv);
    pinocchio::getFrameJacobian(impl_->model, impl_->data, impl_->left_foot_frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, output.left_foot_jacobian);
    pinocchio::getFrameJacobian(impl_->model, impl_->data, impl_->right_foot_frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, output.right_foot_jacobian);
    return output;
}

Eigen::VectorXd RobotModel::nonlinear_effects(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v) const {
    if (q.size() != impl_->model.nq) {
        throw std::runtime_error("q size does not match Pinocchio model nq");
    }
    if (v.size() != impl_->model.nv) {
        throw std::runtime_error("v size does not match Pinocchio model nv");
    }
    return pinocchio::nonLinearEffects(impl_->model, impl_->data, q, v);
}

std::vector<double> RobotModel::configured_joint_torques(
    const Eigen::VectorXd& generalized_tau) const {
    if (generalized_tau.size() != impl_->model.nv) {
        throw std::runtime_error("generalized_tau size does not match Pinocchio model nv");
    }
    std::vector<double> output(config_.joint_order.size(), 0.0);
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = generalized_tau[impl_->v_index_by_config_joint[i]];
    }
    return output;
}

void RobotModel::validate_config() const {
    if (config_.urdf_path.empty()) {
        throw std::runtime_error("whole body robot model requires a URDF path");
    }
    if (!std::filesystem::exists(config_.urdf_path)) {
        throw std::runtime_error("URDF path does not exist: " + config_.urdf_path);
    }
    if (config_.base_link.empty()) {
        throw std::runtime_error("whole body robot model requires base_link");
    }
    if (config_.left_foot_frame.empty()) {
        throw std::runtime_error("whole body robot model requires left_foot_frame");
    }
    if (config_.right_foot_frame.empty()) {
        throw std::runtime_error("whole body robot model requires right_foot_frame");
    }
    if (config_.joint_order.empty()) {
        throw std::runtime_error("whole body robot model requires joint_order");
    }
}

}  // namespace whole_body_mpc
