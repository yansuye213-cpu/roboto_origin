// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/estimation/base_state_estimator.hpp"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace whole_body_mpc {

namespace {

Eigen::Vector3d clamp_norm(const Eigen::Vector3d& value, double max_norm) {
    if (max_norm <= 0.0) {
        return Eigen::Vector3d::Zero();
    }
    const double norm = value.norm();
    if (norm <= max_norm || norm <= 1.0e-9) {
        return value;
    }
    return value * (max_norm / norm);
}

void add_foot_constraint(
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& foot_jacobian,
    const Eigen::VectorXd& velocity_without_base_linear,
    Eigen::MatrixXd& a,
    Eigen::VectorXd& b,
    int& row) {
    a.block(row, 0, 3, 3) = foot_jacobian.topRows<3>().leftCols<3>();
    b.segment<3>(row) = -foot_jacobian.topRows<3>() * velocity_without_base_linear;
    row += 3;
}

}  // namespace

BaseStateEstimator::BaseStateEstimator(Config config) : config_(config) {
    if (config_.velocity_filter_alpha < 0.0 || config_.velocity_filter_alpha > 1.0 ||
        config_.max_base_linear_velocity < 0.0) {
        throw std::runtime_error("BaseStateEstimator parameters are invalid");
    }
}

void BaseStateEstimator::reset() {
    filtered_base_linear_velocity_.setZero();
    has_filtered_velocity_ = false;
}

BaseStateEstimator::Output BaseStateEstimator::estimate(const Input& input) {
    Output output;
    if (!config_.enabled || input.kinematics == nullptr ||
        input.generalized_velocity_without_base_linear == nullptr) {
        return output;
    }

    const int foot_count =
        (input.left_contact ? 1 : 0) + (input.right_contact ? 1 : 0);
    if (foot_count <= 0) {
        return output;
    }

    const int rows = foot_count * 3;
    Eigen::MatrixXd a(rows, 3);
    Eigen::VectorXd b(rows);
    a.setZero();
    b.setZero();
    int row = 0;
    if (input.left_contact) {
        add_foot_constraint(input.kinematics->left_foot_jacobian,
                            *input.generalized_velocity_without_base_linear,
                            a, b, row);
    }
    if (input.right_contact) {
        add_foot_constraint(input.kinematics->right_foot_jacobian,
                            *input.generalized_velocity_without_base_linear,
                            a, b, row);
    }

    Eigen::Vector3d estimate =
        a.completeOrthogonalDecomposition().solve(b);
    if (!estimate.allFinite()) {
        estimate.setZero();
    }
    estimate = clamp_norm(estimate, config_.max_base_linear_velocity);

    if (has_filtered_velocity_) {
        const double alpha = config_.velocity_filter_alpha;
        filtered_base_linear_velocity_ =
            alpha * filtered_base_linear_velocity_ + (1.0 - alpha) * estimate;
    } else {
        filtered_base_linear_velocity_ = estimate;
        has_filtered_velocity_ = true;
    }

    output.base_linear_velocity = filtered_base_linear_velocity_;
    output.stance_velocity_residual =
        rows > 0 ? (a * filtered_base_linear_velocity_ - b).norm() /
                       static_cast<double>(rows)
                 : 0.0;
    output.constraint_rows = rows;
    output.used_contact_constraint = true;
    return output;
}

}  // namespace whole_body_mpc
