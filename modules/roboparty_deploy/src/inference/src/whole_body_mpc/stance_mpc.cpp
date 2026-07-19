// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/stance_mpc.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace whole_body_mpc {

namespace {

double clamp_abs(double value, double limit) {
    if (limit <= 0.0) {
        return 0.0;
    }
    return std::clamp(value, -limit, limit);
}

Eigen::Vector2d clamp_norm(const Eigen::Vector2d& value, double max_norm) {
    if (max_norm <= 0.0) {
        return Eigen::Vector2d::Zero();
    }
    const double norm = value.norm();
    if (norm <= max_norm || norm <= 1.0e-9) {
        return value;
    }
    return value * (max_norm / norm);
}

}  // namespace

StanceMpc::StanceMpc(Config config) : config_(config) {
    if (config_.horizon <= 0) {
        throw std::runtime_error("StanceMpc horizon must be positive");
    }
    if (config_.dt <= 0.0) {
        throw std::runtime_error("StanceMpc dt must be positive");
    }
    if (config_.orientation_weight < 0.0 ||
        config_.angular_rate_weight < 0.0 ||
        config_.com_weight < 0.0 ||
        config_.com_velocity_weight < 0.0 ||
        config_.control_weight <= 0.0 ||
        config_.max_angular_accel < 0.0 ||
        config_.max_com_accel < 0.0) {
        throw std::runtime_error("StanceMpc weights and limits are invalid");
    }
}

void StanceMpc::reset() {}

StanceMpc::Output StanceMpc::solve(const Input& input) const {
    constexpr int kStateDim = 8;
    constexpr int kControlDim = 4;
    using StateVector = Eigen::Matrix<double, kStateDim, 1>;
    using ControlVector = Eigen::Matrix<double, kControlDim, 1>;
    using StateMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;
    using ControlMatrix = Eigen::Matrix<double, kControlDim, kControlDim>;
    using DynamicsMatrix = Eigen::Matrix<double, kStateDim, kControlDim>;

    const double dt = config_.dt;
    StateMatrix A = StateMatrix::Identity();
    A(0, 2) = dt;
    A(1, 3) = dt;
    A(4, 6) = dt;
    A(5, 7) = dt;

    DynamicsMatrix B = DynamicsMatrix::Zero();
    B(0, 0) = 0.5 * dt * dt;
    B(1, 1) = 0.5 * dt * dt;
    B(2, 0) = dt;
    B(3, 1) = dt;
    B(4, 2) = 0.5 * dt * dt;
    B(5, 3) = 0.5 * dt * dt;
    B(6, 2) = dt;
    B(7, 3) = dt;

    StateMatrix Q = StateMatrix::Zero();
    Q(0, 0) = config_.orientation_weight;
    Q(1, 1) = config_.orientation_weight;
    Q(2, 2) = config_.angular_rate_weight;
    Q(3, 3) = config_.angular_rate_weight;
    Q(4, 4) = config_.com_weight;
    Q(5, 5) = config_.com_weight;
    Q(6, 6) = config_.com_velocity_weight;
    Q(7, 7) = config_.com_velocity_weight;

    ControlMatrix R = config_.control_weight * ControlMatrix::Identity();
    StateMatrix P = Q;
    Eigen::Matrix<double, kControlDim, kStateDim> K =
        Eigen::Matrix<double, kControlDim, kStateDim>::Zero();
    for (int i = 0; i < config_.horizon; i++) {
        const ControlMatrix hessian =
            R + B.transpose() * P * B +
            1.0e-9 * ControlMatrix::Identity();
        K = hessian.ldlt().solve(B.transpose() * P * A);
        P = Q + A.transpose() * P * (A - B * K);
    }

    Output output;
    output.state << input.roll - config_.target_roll,
                    input.pitch - config_.target_pitch,
                    input.wx,
                    input.wy,
                    input.com_offset_error.x(),
                    input.com_offset_error.y(),
                    input.com_velocity.x(),
                    input.com_velocity.y();
    output.control = -K * output.state;
    output.control(0) = clamp_abs(output.control(0), config_.max_angular_accel);
    output.control(1) = clamp_abs(output.control(1), config_.max_angular_accel);
    const Eigen::Vector2d com_accel =
        clamp_norm(output.control.segment<2>(2), config_.max_com_accel);
    output.control(2) = com_accel.x();
    output.control(3) = com_accel.y();

    output.desired_angular_acceleration << output.control(0), output.control(1), 0.0;
    output.desired_com_acceleration << output.control(2), output.control(3), 0.0;
    return output;
}

}  // namespace whole_body_mpc
