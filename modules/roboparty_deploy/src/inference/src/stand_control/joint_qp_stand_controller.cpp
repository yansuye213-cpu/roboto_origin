// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "stand_control/joint_qp_stand_controller.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stand_control {

JointQpStandController::JointQpStandController(const StandingStabilizer::Config& config)
    : config_(config) {
    reset();
}

void JointQpStandController::reset() {
    last_joint_delta_.assign(config_.joint_num, 0.0f);
}

StandingStabilizer::Command JointQpStandController::apply(
    const StandingStabilizer::Measurement& measurement, float blend,
    const std::vector<float>& base_target, const std::vector<float>& kp,
    const std::vector<float>& kd, const std::vector<float>& current_joint_position,
    const std::vector<float>& current_joint_velocity) {
    (void)current_joint_position;
    (void)current_joint_velocity;
    if (base_target.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("JointQpStandController base_target size mismatch");
    }
    if (last_joint_delta_.size() != static_cast<size_t>(config_.joint_num)) {
        reset();
    }

    StandingStabilizer::Command command;
    command.position = base_target;
    command.velocity.assign(config_.joint_num, 0.0f);
    command.kp = kp;
    command.kd = kd;
    command.tau.assign(config_.joint_num, 0.0f);

    StandingStabilizer::Correction correction;
    correction.roll_accel = solve_axis(measurement.roll, measurement.wx, config_.target_roll);
    correction.pitch_accel = solve_axis(measurement.pitch, measurement.wy, config_.target_pitch);
    correction.roll_correction = blend * std::clamp(correction.roll_accel * config_.roll_gain,
                                                    -config_.max_joint_correction,
                                                    config_.max_joint_correction);
    correction.pitch_correction = blend * std::clamp(correction.pitch_accel * config_.pitch_gain,
                                                     -config_.max_joint_correction,
                                                     config_.max_joint_correction);
    correction.qp_used = config_.qp_enabled;

    const std::vector<float> joint_delta = config_.qp_enabled
        ? solve_joint_qp(correction.roll_correction, correction.pitch_correction, base_target)
        : solve_scale_allocation(correction.roll_correction, correction.pitch_correction);
    apply_joint_delta(base_target, joint_delta, command.position);
    update_last_joint_delta(base_target, command.position, correction);
    command.correction = correction;
    return command;
}

float JointQpStandController::solve_axis(float angle, float rate, float target_angle) const {
    const float h = std::max(config_.dt, 1.0e-4f);
    Eigen::Matrix2f A;
    A << 1.0f, h,
         0.0f, 1.0f;
    Eigen::Vector2f B(0.5f * h * h, h);
    Eigen::Matrix2f Q = Eigen::Matrix2f::Zero();
    Q(0, 0) = config_.q_angle;
    Q(1, 1) = config_.q_rate;
    const float R = std::max(config_.r_accel, 1.0e-6f);
    Eigen::Matrix2f P = Q;
    Eigen::RowVector2f K = Eigen::RowVector2f::Zero();

    for (int i = 0; i < config_.horizon; i++) {
        const Eigen::Vector2f PB = P * B;
        const float denom = R + B.dot(PB);
        K = (B.transpose() * P * A) / denom;
        P = Q + A.transpose() * P * (A - B * K);
    }

    Eigen::Vector2f x(angle - target_angle, rate);
    return std::clamp(-K.dot(x), -config_.max_accel, config_.max_accel);
}

std::vector<float> JointQpStandController::solve_scale_allocation(
    float roll_correction, float pitch_correction) const {
    std::vector<float> joint_delta(config_.joint_num, 0.0f);
    for (int i = 0; i < config_.joint_num; i++) {
        joint_delta[i] += static_cast<float>(config_.roll_joint_scale[i]) * roll_correction;
        joint_delta[i] += static_cast<float>(config_.pitch_joint_scale[i]) * pitch_correction;
    }
    return joint_delta;
}

void JointQpStandController::apply_joint_delta(
    const std::vector<float>& base_target, const std::vector<float>& joint_delta,
    std::vector<float>& target) const {
    for (int i = 0; i < config_.joint_num; i++) {
        target[i] = base_target[i] + joint_delta[i];
        if (!config_.joint_limits.empty()) {
            target[i] = std::clamp(target[i],
                                   static_cast<float>(config_.joint_limits[i * 2]),
                                   static_cast<float>(config_.joint_limits[i * 2 + 1]));
        }
    }
}

void JointQpStandController::update_last_joint_delta(
    const std::vector<float>& base_target, const std::vector<float>& target,
    StandingStabilizer::Correction& correction) {
    Eigen::MatrixXf B(config_.joint_num, 2);
    Eigen::VectorXf actual_delta(config_.joint_num);
    correction.max_joint_delta = 0.0f;
    for (int i = 0; i < config_.joint_num; i++) {
        const float joint_delta = target[i] - base_target[i];
        last_joint_delta_[i] = joint_delta;
        actual_delta(i) = joint_delta;
        B(i, 0) = static_cast<float>(config_.roll_joint_scale[i]);
        B(i, 1) = static_cast<float>(config_.pitch_joint_scale[i]);
        correction.max_joint_delta = std::max(correction.max_joint_delta, std::abs(joint_delta));
    }
    const Eigen::Matrix2f gram =
        B.transpose() * B + 1.0e-6f * Eigen::Matrix2f::Identity();
    const Eigen::Vector2f allocated = gram.inverse() * B.transpose() * actual_delta;
    correction.roll_allocated = allocated(0);
    correction.pitch_allocated = allocated(1);
}

std::vector<float> JointQpStandController::solve_joint_qp(
    float roll_correction, float pitch_correction, const std::vector<float>& base_target) const {
    const int n = config_.joint_num;
    Eigen::MatrixXf B(n, 2);
    Eigen::VectorXf delta_ref(n);
    for (int i = 0; i < n; i++) {
        B(i, 0) = static_cast<float>(config_.roll_joint_scale[i]);
        B(i, 1) = static_cast<float>(config_.pitch_joint_scale[i]);
        delta_ref(i) = B(i, 0) * roll_correction + B(i, 1) * pitch_correction;
    }

    const Eigen::Vector2f desired(roll_correction, pitch_correction);
    const Eigen::Matrix2f gram =
        B.transpose() * B + 1.0e-6f * Eigen::Matrix2f::Identity();
    const Eigen::MatrixXf coordinate_map = gram.inverse() * B.transpose();
    Eigen::VectorXf last_delta(n);
    for (int i = 0; i < n; i++) {
        last_delta(i) = last_joint_delta_[i];
    }

    const float tracking_weight = config_.qp_tracking_weight;
    const float shape_weight = config_.qp_shape_weight;
    const float regularization_weight = config_.qp_regularization_weight;
    const float smooth_weight = config_.qp_smooth_weight;
    Eigen::MatrixXf H = (shape_weight + regularization_weight + smooth_weight + 1.0e-6f) *
                        Eigen::MatrixXf::Identity(n, n);
    Eigen::VectorXf g = -shape_weight * delta_ref - smooth_weight * last_delta;
    if (tracking_weight > 0.0f) {
        H += tracking_weight * coordinate_map.transpose() * coordinate_map;
        g += -tracking_weight * coordinate_map.transpose() * desired;
    }

    Eigen::VectorXf lower(n);
    Eigen::VectorXf upper(n);
    const float max_delta = config_.max_joint_correction;
    const float max_step = config_.qp_max_joint_velocity > 0.0f
        ? config_.qp_max_joint_velocity * config_.dt
        : 0.0f;
    for (int i = 0; i < n; i++) {
        float hard_lower = -max_delta;
        float hard_upper = max_delta;
        if (!config_.joint_limits.empty()) {
            hard_lower = std::max(hard_lower,
                                  static_cast<float>(config_.joint_limits[i * 2]) - base_target[i]);
            hard_upper = std::min(hard_upper,
                                  static_cast<float>(config_.joint_limits[i * 2 + 1]) - base_target[i]);
        }
        if (hard_lower > hard_upper) {
            const float joint_limit_delta = !config_.joint_limits.empty()
                ? std::clamp(0.0f,
                             static_cast<float>(config_.joint_limits[i * 2]) - base_target[i],
                             static_cast<float>(config_.joint_limits[i * 2 + 1]) - base_target[i])
                : 0.0f;
            hard_lower = joint_limit_delta;
            hard_upper = joint_limit_delta;
        }

        lower(i) = hard_lower;
        upper(i) = hard_upper;
        if (max_step > 0.0f) {
            const float rate_lower = std::max(hard_lower, last_joint_delta_[i] - max_step);
            const float rate_upper = std::min(hard_upper, last_joint_delta_[i] + max_step);
            if (rate_lower <= rate_upper) {
                lower(i) = rate_lower;
                upper(i) = rate_upper;
            }
        }
    }

    Eigen::VectorXf x = delta_ref;
    for (int i = 0; i < n; i++) {
        x(i) = std::clamp(x(i), lower(i), upper(i));
    }

    float lipschitz = 1.0e-6f;
    for (int row = 0; row < n; row++) {
        float row_sum = 0.0f;
        for (int col = 0; col < n; col++) {
            row_sum += std::abs(H(row, col));
        }
        lipschitz = std::max(lipschitz, row_sum);
    }
    const float step = 1.0f / lipschitz;
    for (int iter = 0; iter < config_.qp_iterations; iter++) {
        x -= step * (H * x + g);
        for (int i = 0; i < n; i++) {
            x(i) = std::clamp(x(i), lower(i), upper(i));
        }
    }

    std::vector<float> joint_delta(n, 0.0f);
    for (int i = 0; i < n; i++) {
        joint_delta[i] = x(i);
    }
    return joint_delta;
}

}  // namespace stand_control
