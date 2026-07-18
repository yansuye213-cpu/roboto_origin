// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "standing_stabilizer.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

StandingStabilizer::StandingStabilizer(Config config) : config_(std::move(config)) {
    if (config_.joint_num <= 0) {
        throw std::runtime_error("StandingStabilizer joint_num must be positive");
    }
    if (config_.horizon <= 0) {
        throw std::runtime_error("StandingStabilizer horizon must be positive");
    }
    if (config_.dt <= 0.0f) {
        throw std::runtime_error("StandingStabilizer dt must be positive");
    }
    if (config_.q_angle < 0.0f || config_.q_rate < 0.0f || config_.r_accel <= 0.0f) {
        throw std::runtime_error("StandingStabilizer weights are invalid");
    }
    if (config_.max_accel <= 0.0f || config_.max_joint_correction < 0.0f) {
        throw std::runtime_error("StandingStabilizer limits are invalid");
    }
    if (config_.roll_joint_scale.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("StandingStabilizer roll_joint_scale size mismatch");
    }
    if (config_.pitch_joint_scale.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("StandingStabilizer pitch_joint_scale size mismatch");
    }
    if (!config_.joint_limits.empty() &&
        config_.joint_limits.size() != static_cast<size_t>(config_.joint_num * 2)) {
        throw std::runtime_error("StandingStabilizer joint_limits size mismatch");
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
    return measurement;
}

StandingStabilizer::Correction StandingStabilizer::apply(
    const Measurement& measurement, float blend, std::vector<float>& target) const {
    if (target.size() != static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("StandingStabilizer target size mismatch");
    }

    Correction correction;
    correction.roll_accel = solve_axis(measurement.roll, measurement.wx, config_.target_roll);
    correction.pitch_accel = solve_axis(measurement.pitch, measurement.wy, config_.target_pitch);
    correction.roll_correction = blend * std::clamp(correction.roll_accel * config_.roll_gain,
                                                    -config_.max_joint_correction,
                                                    config_.max_joint_correction);
    correction.pitch_correction = blend * std::clamp(correction.pitch_accel * config_.pitch_gain,
                                                     -config_.max_joint_correction,
                                                     config_.max_joint_correction);

    for (int i = 0; i < config_.joint_num; i++) {
        target[i] += static_cast<float>(config_.roll_joint_scale[i]) * correction.roll_correction;
        target[i] += static_cast<float>(config_.pitch_joint_scale[i]) * correction.pitch_correction;
        if (!config_.joint_limits.empty()) {
            target[i] = std::clamp(target[i],
                                   static_cast<float>(config_.joint_limits[i * 2]),
                                   static_cast<float>(config_.joint_limits[i * 2 + 1]));
        }
    }
    return correction;
}

float StandingStabilizer::solve_axis(float angle, float rate, float target_angle) const {
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
