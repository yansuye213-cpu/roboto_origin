// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/whole_body_mpc_controller.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace whole_body_mpc {

WholeBodyMpcController::WholeBodyMpcController(const StandingStabilizer::Config& config)
    : config_(config) {
    validate_model_config();
    RobotModel::Config model_config;
    model_config.urdf_path = config_.whole_body_model_path;
    model_config.base_link = config_.whole_body_base_link;
    model_config.left_foot_frame = config_.whole_body_left_foot_link;
    model_config.right_foot_frame = config_.whole_body_right_foot_link;
    model_config.joint_order = config_.whole_body_joint_order;
    model_config.floating_base = true;
    robot_model_ = std::make_unique<RobotModel>(std::move(model_config));

    const Eigen::VectorXd q0 = robot_model_->neutral_configuration();
    const Eigen::VectorXd v0 = robot_model_->zero_velocity();
    neutral_kinematics_ = robot_model_->compute_kinematics(q0, v0);
}

void WholeBodyMpcController::reset() {}

StandingStabilizer::Command WholeBodyMpcController::apply(
    const StandingStabilizer::Measurement& measurement, float blend,
    const std::vector<float>& base_target, const std::vector<float>& kp,
    const std::vector<float>& kd, const std::vector<float>& current_joint_position,
    const std::vector<float>& current_joint_velocity) {
    (void)blend;
    const Eigen::Quaterniond base_orientation(
        measurement.qw, measurement.qx, measurement.qy, measurement.qz);
    const Eigen::VectorXd q = robot_model_->make_configuration(
        current_joint_position, Eigen::Vector3d::Zero(), base_orientation);
    const Eigen::VectorXd v = robot_model_->make_velocity(
        current_joint_velocity, Eigen::Vector3d::Zero(),
        Eigen::Vector3d(measurement.wx, measurement.wy, 0.0));
    latest_kinematics_ = robot_model_->compute_kinematics(q, v);

    StandingStabilizer::Command command;
    command.position = base_target;
    command.velocity.assign(config_.joint_num, 0.0f);
    command.kp = kp;
    command.kd = kd;
    command.tau.assign(config_.joint_num, 0.0f);
    throw std::runtime_error(
        "whole_body_mpc backend is configured but not implemented yet: next steps are URDF model loading, "
        "foot Jacobians, contact-force QP, and tau = J^T f torque mapping");
}

void WholeBodyMpcController::validate_model_config() const {
    if (config_.whole_body_model_path.empty()) {
        throw std::runtime_error("stand_whole_body_model_path is required for whole_body_mpc");
    }
    if (!std::filesystem::exists(config_.whole_body_model_path)) {
        throw std::runtime_error("stand_whole_body_model_path does not exist: " +
                                 config_.whole_body_model_path);
    }
    if (config_.whole_body_base_link.empty()) {
        throw std::runtime_error("stand_whole_body_base_link is required for whole_body_mpc");
    }
    if (config_.whole_body_left_foot_link.empty()) {
        throw std::runtime_error("stand_whole_body_left_foot_link is required for whole_body_mpc");
    }
    if (config_.whole_body_right_foot_link.empty()) {
        throw std::runtime_error("stand_whole_body_right_foot_link is required for whole_body_mpc");
    }
    if (config_.whole_body_joint_order.empty()) {
        throw std::runtime_error("stand_whole_body_joint_order is required for whole_body_mpc");
    }
}

}  // namespace whole_body_mpc
