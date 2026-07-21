// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/model/robot_model.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::unique_ptr<pinocchio::Model> model;
    std::unique_ptr<pinocchio::Data> data;
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

    auto model = std::make_unique<pinocchio::Model>();
    pinocchio::urdf::details::UrdfVisitor visitor(*model);
    if (config_.floating_base) {
        pinocchio::parsers::JointModel root_joint = pinocchio::JointModelFreeFlyer();
        const std::string root_joint_name = "root_joint";
        boost::optional<const pinocchio::parsers::JointModel&> root_joint_opt(root_joint);
        boost::optional<const std::string&> root_joint_name_opt(root_joint_name);
        pinocchio::urdf::details::parseRootTree(
            config_.urdf_path, visitor, root_joint_opt, root_joint_name_opt, false);
    } else {
        pinocchio::urdf::details::parseRootTree(
            config_.urdf_path, visitor, boost::none, boost::none, false);
    }
    impl_->model = std::move(model);
    impl_->data = std::make_unique<pinocchio::Data>(*impl_->model);

    const auto& pin_model = *impl_->model;
    for (const auto& inertia : pin_model.inertias) {
        impl_->total_mass += inertia.mass();
    }

    impl_->base_frame_id = pin_model.getFrameId(config_.base_link);
    impl_->left_foot_frame_id = pin_model.getFrameId(config_.left_foot_frame);
    impl_->right_foot_frame_id = pin_model.getFrameId(config_.right_foot_frame);
    if (impl_->base_frame_id >= pin_model.nframes) {
        throw std::runtime_error("base frame not found in URDF: " + config_.base_link);
    }
    if (impl_->left_foot_frame_id >= pin_model.nframes) {
        throw std::runtime_error("left foot frame not found in URDF: " + config_.left_foot_frame);
    }
    if (impl_->right_foot_frame_id >= pin_model.nframes) {
        throw std::runtime_error("right foot frame not found in URDF: " + config_.right_foot_frame);
    }

    std::unordered_map<std::string, std::pair<int, int>> joint_indices;
    for (pinocchio::JointIndex joint_id = 1; joint_id < pin_model.joints.size(); joint_id++) {
        const std::string& joint_name = pin_model.names[joint_id];
        const int nq = pin_model.joints[joint_id].nq();
        const int nv = pin_model.joints[joint_id].nv();
        if (nq == 1 && nv == 1) {
            model_joint_order_.push_back(joint_name);
            joint_indices[joint_name] = {pin_model.joints[joint_id].idx_q(),
                                         pin_model.joints[joint_id].idx_v()};
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
    return impl_->model->nq;
}

int RobotModel::nv() const {
    return impl_->model->nv;
}

double RobotModel::total_mass() const {
    return impl_->total_mass;
}

Eigen::VectorXd RobotModel::neutral_configuration() const {
    return pinocchio::neutral(*impl_->model);
}

Eigen::VectorXd RobotModel::zero_velocity() const {
    return Eigen::VectorXd::Zero(impl_->model->nv);
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
    const auto& pin_model = *impl_->model;
    auto& pin_data = *impl_->data;
    if (q.size() != pin_model.nq) {
        throw std::runtime_error("q size does not match Pinocchio model nq");
    }
    if (v.size() != pin_model.nv) {
        throw std::runtime_error("v size does not match Pinocchio model nv");
    }

    pinocchio::forwardKinematics(pin_model, pin_data, q, v);
    pinocchio::updateFramePlacements(pin_model, pin_data);
    pinocchio::centerOfMass(pin_model, pin_data, q, v, false);
    pinocchio::computeJointJacobians(pin_model, pin_data, q);
    pinocchio::computeJointJacobiansTimeVariation(pin_model, pin_data, q, v);
    pinocchio::crba(pin_model, pin_data, q);
    pin_data.M.triangularView<Eigen::StrictlyLower>() =
        pin_data.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::nonLinearEffects(pin_model, pin_data, q, v);

    Kinematics output;
    output.com_position = pin_data.com[0];
    output.com_velocity = pin_data.vcom[0];

    const auto left_pose = pin_data.oMf[impl_->left_foot_frame_id];
    output.left_foot_pose.linear() = left_pose.rotation();
    output.left_foot_pose.translation() = left_pose.translation();
    const auto right_pose = pin_data.oMf[impl_->right_foot_frame_id];
    output.right_foot_pose.linear() = right_pose.rotation();
    output.right_foot_pose.translation() = right_pose.translation();

    output.base_jacobian.setZero(6, pin_model.nv);
    output.left_foot_jacobian.setZero(6, pin_model.nv);
    output.right_foot_jacobian.setZero(6, pin_model.nv);
    pinocchio::getFrameJacobian(pin_model, pin_data, impl_->base_frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, output.base_jacobian);
    pinocchio::getFrameJacobian(pin_model, pin_data, impl_->left_foot_frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, output.left_foot_jacobian);
    pinocchio::getFrameJacobian(pin_model, pin_data, impl_->right_foot_frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, output.right_foot_jacobian);
    output.left_foot_velocity = (output.left_foot_jacobian * v).head<3>();
    output.right_foot_velocity = (output.right_foot_jacobian * v).head<3>();

    Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian_dot;
    jacobian_dot.setZero(6, pin_model.nv);
    pinocchio::getFrameJacobianTimeVariation(pin_model, pin_data, impl_->base_frame_id,
                                             pinocchio::LOCAL_WORLD_ALIGNED, jacobian_dot);
    output.base_jacobian_dot_v = jacobian_dot * v;
    pinocchio::getFrameJacobianTimeVariation(pin_model, pin_data, impl_->left_foot_frame_id,
                                             pinocchio::LOCAL_WORLD_ALIGNED, jacobian_dot);
    output.left_foot_jacobian_dot_v = jacobian_dot * v;
    pinocchio::getFrameJacobianTimeVariation(pin_model, pin_data, impl_->right_foot_frame_id,
                                             pinocchio::LOCAL_WORLD_ALIGNED, jacobian_dot);
    output.right_foot_jacobian_dot_v = jacobian_dot * v;
    output.mass_matrix = pin_data.M;
    output.nonlinear_effects = pin_data.nle;
    return output;
}

Eigen::VectorXd RobotModel::nonlinear_effects(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v) const {
    const auto& pin_model = *impl_->model;
    auto& pin_data = *impl_->data;
    if (q.size() != pin_model.nq) {
        throw std::runtime_error("q size does not match Pinocchio model nq");
    }
    if (v.size() != pin_model.nv) {
        throw std::runtime_error("v size does not match Pinocchio model nv");
    }
    return pinocchio::nonLinearEffects(pin_model, pin_data, q, v);
}

std::vector<double> RobotModel::configured_joint_torques(
    const Eigen::VectorXd& generalized_tau) const {
    if (generalized_tau.size() != impl_->model->nv) {
        throw std::runtime_error("generalized_tau size does not match Pinocchio model nv");
    }
    std::vector<double> output(config_.joint_order.size(), 0.0);
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = generalized_tau[impl_->v_index_by_config_joint[i]];
    }
    return output;
}

const std::vector<int>& RobotModel::configured_joint_velocity_indices() const {
    return impl_->v_index_by_config_joint;
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
