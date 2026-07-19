// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "standing_stabilizer.hpp"

#include "whole_body_mpc/whole_body_mpc_controller.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

StandingStabilizer::StandingStabilizer(Config config) : config_(std::move(config)) {
    if (config_.joint_num <= 0) {
        throw std::runtime_error("StandingStabilizer joint_num must be positive");
    }
    if (config_.dt <= 0.0f) {
        throw std::runtime_error("StandingStabilizer dt must be positive");
    }
    if (config_.wbc_mpc_horizon <= 0 ||
        config_.wbc_mpc_orientation_weight < 0.0f ||
        config_.wbc_mpc_angular_rate_weight < 0.0f ||
        config_.wbc_mpc_com_weight < 0.0f ||
        config_.wbc_mpc_com_velocity_weight < 0.0f ||
        config_.wbc_mpc_control_weight <= 0.0f ||
        config_.wbc_mpc_max_angular_accel < 0.0f ||
        config_.wbc_mpc_max_com_accel < 0.0f) {
        throw std::runtime_error("StandingStabilizer WBC MPC parameters are invalid");
    }
    if (config_.wbc_qp_iterations <= 0) {
        throw std::runtime_error("StandingStabilizer wbc_qp_iterations must be positive");
    }
    if (config_.wbc_friction_coefficient < 0.0f ||
        config_.wbc_min_normal_force < 0.0f ||
        config_.wbc_max_normal_force < config_.wbc_min_normal_force ||
        config_.wbc_force_tracking_weight < 0.0f ||
        config_.wbc_moment_tracking_weight < 0.0f ||
        config_.wbc_regularization_weight < 0.0f ||
        config_.wbc_smooth_weight < 0.0f ||
        config_.wbc_max_body_moment < 0.0f ||
        config_.wbc_max_joint_torque < 0.0f) {
        throw std::runtime_error("StandingStabilizer WBC parameters are invalid");
    }
    if (!config_.joint_limits.empty() &&
        config_.joint_limits.size() != static_cast<size_t>(config_.joint_num * 2)) {
        throw std::runtime_error("StandingStabilizer joint_limits size mismatch");
    }
    if (!config_.wbc_torque_joint_scale.empty() &&
        config_.wbc_torque_joint_scale.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("StandingStabilizer wbc_torque_joint_scale size mismatch");
    }

    if (config_.whole_body_joint_order.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("StandingStabilizer whole_body_joint_order size mismatch");
    }

    whole_body_mpc_controller_ =
        std::make_unique<whole_body_mpc::WholeBodyMpcController>(config_);
}

StandingStabilizer::~StandingStabilizer() = default;

std::vector<std::string> StandingStabilizer::diagnostics() const {
    if (whole_body_mpc_controller_) {
        return whole_body_mpc_controller_->diagnostics();
    }
    return {};
}

void StandingStabilizer::reset() {
    if (whole_body_mpc_controller_) {
        whole_body_mpc_controller_->reset();
    }
}

StandingStabilizer::Measurement StandingStabilizer::measure(
    const std::vector<float>& quat, const std::vector<float>& angular_velocity) const {
    if (quat.size() < 4) {
        throw std::runtime_error("StandingStabilizer requires a quaternion with 4 values");
    }

    Eigen::Quaternionf q_b2w(quat[0], quat[1], quat[2], quat[3]);
    if (q_b2w.norm() <= 1.0e-6f) {
        q_b2w = Eigen::Quaternionf::Identity();
    } else {
        q_b2w.normalize();
    }

    const Eigen::Matrix3f R_b2w = q_b2w.toRotationMatrix();
    Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
    const Eigen::Vector3f gravity_b = q_b2w.inverse() * gravity_w;

    Measurement measurement;
    measurement.roll = std::atan2(R_b2w(2, 1), R_b2w(2, 2));
    measurement.pitch = std::asin(std::clamp(-R_b2w(2, 0), -1.0f, 1.0f));
    measurement.wx = angular_velocity.size() > 0 ? angular_velocity[0] : 0.0f;
    measurement.wy = angular_velocity.size() > 1 ? angular_velocity[1] : 0.0f;
    measurement.gravity_z = gravity_b.z();
    measurement.qw = q_b2w.w();
    measurement.qx = q_b2w.x();
    measurement.qy = q_b2w.y();
    measurement.qz = q_b2w.z();
    return measurement;
}

StandingStabilizer::Command StandingStabilizer::apply(
    const Measurement& measurement, float blend, const std::vector<float>& base_target,
    const std::vector<float>& kp, const std::vector<float>& kd,
    const std::vector<float>& current_joint_position,
    const std::vector<float>& current_joint_velocity) {
    if (whole_body_mpc_controller_) {
        return whole_body_mpc_controller_->apply(measurement, blend, base_target, kp, kd,
                                                 current_joint_position, current_joint_velocity);
    }
    throw std::runtime_error("StandingStabilizer has no active backend");
}
