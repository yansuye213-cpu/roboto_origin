// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "standing_stabilizer.hpp"

#include "whole_body_mpc/whole_body_mpc_controller.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

bool is_supported_wbc_mpc_backend(const std::string& backend) {
    return backend == "disabled" || backend == "ocs2";
}

}  // namespace

StandingStabilizer::StandingStabilizer(Config config) : config_(std::move(config)) {
    if (config_.joint_num <= 0) {
        throw std::runtime_error("StandingStabilizer joint_num must be positive");
    }
    if (config_.dt <= 0.0f) {
        throw std::runtime_error("StandingStabilizer dt must be positive");
    }
    if (config_.wbc_mpc_horizon <= 0 ||
        config_.wbc_mpc_dt <= 0.0f ||
        !is_supported_wbc_mpc_backend(config_.wbc_mpc_backend) ||
        config_.wbc_mpc_orientation_weight < 0.0f ||
        config_.wbc_mpc_angular_rate_weight < 0.0f ||
        config_.wbc_mpc_com_weight < 0.0f ||
        config_.wbc_mpc_com_velocity_weight < 0.0f ||
        config_.wbc_mpc_terminal_weight_scale < 0.0f ||
        config_.wbc_mpc_input_smooth_weight < 0.0f ||
        config_.wbc_mpc_force_weight <= 0.0f ||
        config_.wbc_mpc_qp_iterations < 0 ||
        config_.wbc_mpc_friction_barrier_mu < 0.0f ||
        config_.wbc_mpc_friction_barrier_delta <= 0.0f ||
        config_.wbc_mpc_friction_regularization <= 0.0f ||
        config_.wbc_mpc_max_angular_accel < 0.0f ||
        config_.wbc_mpc_max_com_accel < 0.0f ||
        config_.wbc_mpc_max_contact_force_delta < 0.0f ||
        !std::isfinite(config_.wbc_mpc_output_roll_sign) ||
        !std::isfinite(config_.wbc_mpc_output_pitch_sign) ||
        !std::isfinite(config_.wbc_mpc_output_roll_scale) ||
        !std::isfinite(config_.wbc_mpc_output_pitch_scale) ||
        std::abs(config_.wbc_mpc_output_roll_sign) < 1.0e-6f ||
        std::abs(config_.wbc_mpc_output_pitch_sign) < 1.0e-6f ||
        config_.wbc_mpc_output_roll_scale < 0.0f ||
        config_.wbc_mpc_output_pitch_scale < 0.0f ||
        config_.wbc_mpc_base_height_weight < 0.0f ||
        config_.wbc_mpc_yaw_weight < 0.0f ||
        config_.wbc_mpc_joint_angle_weight < 0.0f ||
        config_.wbc_mpc_joint_velocity_weight < 0.0f ||
        config_.wbc_mpc_swing_position_weight < 0.0f ||
        config_.wbc_mpc_joint_command_position_gain < 0.0f ||
        config_.wbc_mpc_joint_command_velocity_scale < 0.0f ||
        config_.wbc_mpc_joint_command_max_delta < 0.0f ||
        config_.wbc_mpc_joint_command_max_velocity < 0.0f ||
        config_.wbc_mpc_swing_time_scale < 0.0f ||
        config_.wbc_mpc_swing_lift_off_velocity < 0.0f ||
        config_.wbc_mpc_swing_touch_down_velocity < 0.0f ||
        config_.wbc_mpc_ad_model_folder.empty()) {
        throw std::runtime_error("StandingStabilizer WBC MPC parameters are invalid");
    }
    if (config_.wbc_state_velocity_filter_alpha < 0.0f ||
        config_.wbc_state_velocity_filter_alpha > 1.0f ||
        config_.wbc_state_max_base_linear_velocity < 0.0f) {
        throw std::runtime_error("StandingStabilizer WBC state estimation parameters are invalid");
    }
    if (config_.wbc_qp_iterations <= 0) {
        throw std::runtime_error("StandingStabilizer wbc_qp_iterations must be positive");
    }
    if (config_.wbc_active_set_iterations <= 0) {
        throw std::runtime_error("StandingStabilizer wbc_active_set_iterations must be positive");
    }
    if (config_.wbc_friction_coefficient < 0.0f ||
        config_.wbc_min_normal_force < 0.0f ||
        config_.wbc_max_normal_force < config_.wbc_min_normal_force ||
        config_.wbc_force_tracking_weight < 0.0f ||
        config_.wbc_moment_tracking_weight < 0.0f ||
        config_.wbc_regularization_weight < 0.0f ||
        config_.wbc_smooth_weight < 0.0f ||
        config_.wbc_max_body_moment < 0.0f ||
        config_.wbc_body_moment_rate_limit < 0.0f ||
        config_.wbc_body_moment_filter_weight < 0.0f ||
        config_.wbc_max_joint_torque < 0.0f ||
        config_.wbc_foot_half_length < 0.0f ||
        config_.wbc_foot_half_width < 0.0f) {
        throw std::runtime_error("StandingStabilizer WBC parameters are invalid");
    }
    if (config_.wbc_step_recovery_roll_trigger < 0.0f ||
        config_.wbc_step_recovery_pitch_trigger < 0.0f ||
        config_.wbc_step_recovery_rate_trigger < 0.0f ||
        config_.wbc_step_recovery_com_trigger < 0.0f ||
        config_.wbc_step_recovery_com_velocity_trigger < 0.0f ||
        config_.wbc_step_recovery_return_roll < 0.0f ||
        config_.wbc_step_recovery_return_pitch < 0.0f ||
        config_.wbc_step_recovery_return_rate < 0.0f ||
        config_.wbc_step_recovery_return_com < 0.0f ||
        config_.wbc_step_recovery_return_com_velocity < 0.0f ||
        config_.wbc_step_recovery_steps < 0 ||
        config_.wbc_step_recovery_swing_time <= 0.0f ||
        config_.wbc_step_recovery_double_support_time < 0.0f ||
        config_.wbc_step_recovery_settle_time < 0.0f ||
        config_.wbc_step_recovery_stable_time < 0.0f ||
        config_.wbc_step_recovery_cooldown < 0.0f ||
        config_.wbc_step_recovery_max_duration < 0.0f ||
        config_.wbc_step_recovery_min_step_x < 0.0f ||
        config_.wbc_step_recovery_min_step_y < 0.0f ||
        config_.wbc_step_recovery_max_step_x < 0.0f ||
        config_.wbc_step_recovery_max_step_y < 0.0f ||
        config_.wbc_step_recovery_min_step_x > config_.wbc_step_recovery_max_step_x ||
        config_.wbc_step_recovery_min_step_y > config_.wbc_step_recovery_max_step_y ||
        config_.wbc_step_recovery_capture_time < 0.0f ||
        config_.wbc_step_recovery_capture_gain < 0.0f ||
        config_.wbc_step_recovery_swing_height < 0.0f ||
        config_.wbc_swing_tracking_weight < 0.0f ||
        config_.wbc_swing_kp < 0.0f ||
        config_.wbc_swing_kd < 0.0f ||
        config_.wbc_swing_ik_gain < 0.0f ||
        config_.wbc_swing_ik_damping < 0.0f ||
        config_.wbc_swing_max_joint_delta < 0.0f ||
        config_.wbc_swing_max_joint_velocity < 0.0f) {
        throw std::runtime_error("StandingStabilizer WBC step recovery parameters are invalid");
    }
    if (!config_.joint_limits.empty() &&
        config_.joint_limits.size() != static_cast<size_t>(config_.joint_num * 2)) {
        throw std::runtime_error("StandingStabilizer joint_limits size mismatch");
    }
    if (!config_.wbc_torque_joint_scale.empty() &&
        config_.wbc_torque_joint_scale.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("StandingStabilizer wbc_torque_joint_scale size mismatch");
    }
    if (!config_.wbc_mpc_joint_command_joint_scale.empty() &&
        config_.wbc_mpc_joint_command_joint_scale.size() !=
            static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error(
            "StandingStabilizer wbc_mpc_joint_command_joint_scale size mismatch");
    }

    if (config_.whole_body_joint_order.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("StandingStabilizer whole_body_joint_order size mismatch");
    }
    if (!config_.whole_body_nominal_joint_angles.empty() &&
        config_.whole_body_nominal_joint_angles.size() !=
            static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error(
            "StandingStabilizer whole_body_nominal_joint_angles size mismatch");
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
    (void)measurement;
    (void)blend;
    (void)current_joint_position;
    (void)current_joint_velocity;
    const bool pure_stand_bypass =
        (!config_.wbc_mpc_enabled || config_.wbc_mpc_backend == "disabled") &&
        !config_.wbc_mpc_joint_command_enabled &&
        !config_.wbc_torque_enabled &&
        !config_.wbc_step_recovery_enabled;
    if (pure_stand_bypass) {
        Command command;
        command.position = base_target;
        command.velocity.assign(config_.joint_num, 0.0f);
        command.kp = kp;
        command.kd = kd;
        command.tau.assign(config_.joint_num, 0.0f);
        command.correction.wbc_mpc_backend = config_.wbc_mpc_backend;
        return command;
    }
    if (whole_body_mpc_controller_) {
        return whole_body_mpc_controller_->apply(measurement, blend, base_target, kp, kd,
                                                 current_joint_position, current_joint_velocity);
    }
    throw std::runtime_error("StandingStabilizer has no active backend");
}
