// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/contact_force_qp.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace whole_body_mpc {

namespace {

constexpr double kGravity = 9.80665;

Eigen::Matrix3d skew(const Eigen::Vector3d& value) {
    Eigen::Matrix3d output;
    output << 0.0, -value.z(), value.y(),
              value.z(), 0.0, -value.x(),
             -value.y(), value.x(), 0.0;
    return output;
}

}  // namespace

ContactForceQp::ContactForceQp(Config config) : config_(config) {
    if (config_.iterations <= 0) {
        throw std::runtime_error("ContactForceQp iterations must be positive");
    }
    if (config_.friction_coefficient < 0.0) {
        throw std::runtime_error("ContactForceQp friction_coefficient must be non-negative");
    }
    if (config_.min_normal_force < 0.0 || config_.max_normal_force < config_.min_normal_force) {
        throw std::runtime_error("ContactForceQp normal force limits are invalid");
    }
    if (config_.force_tracking_weight < 0.0 || config_.moment_tracking_weight < 0.0 ||
        config_.regularization_weight < 0.0 || config_.smooth_weight < 0.0) {
        throw std::runtime_error("ContactForceQp weights must be non-negative");
    }
    reset();
}

void ContactForceQp::reset() {
    last_solution_.setZero();
    has_last_solution_ = false;
}

ContactForceQp::Result ContactForceQp::solve(const Input& input) {
    if (input.mass <= 0.0) {
        throw std::runtime_error("ContactForceQp requires positive robot mass");
    }

    Eigen::Matrix<double, 6, 6> wrench_map = Eigen::Matrix<double, 6, 6>::Zero();
    wrench_map.block<3, 3>(0, 0).setIdentity();
    wrench_map.block<3, 3>(0, 3).setIdentity();
    wrench_map.block<3, 3>(3, 0) = skew(input.left_foot_position - input.com_position);
    wrench_map.block<3, 3>(3, 3) = skew(input.right_foot_position - input.com_position);

    Result result;
    result.target_wrench.head<3>() =
        input.mass * (input.desired_com_acceleration + Eigen::Vector3d(0.0, 0.0, kGravity));
    result.target_wrench.tail<3>() = input.desired_body_moment;

    Eigen::Matrix<double, 6, 1> nominal = Eigen::Matrix<double, 6, 1>::Zero();
    nominal.segment<3>(0) = 0.5 * result.target_wrench.head<3>();
    nominal.segment<3>(3) = 0.5 * result.target_wrench.head<3>();
    nominal = project_forces(nominal);

    Eigen::Matrix<double, 6, 6> weights = Eigen::Matrix<double, 6, 6>::Zero();
    weights.diagonal().head<3>().setConstant(config_.force_tracking_weight);
    weights.diagonal().tail<3>().setConstant(config_.moment_tracking_weight);

    const double regularization = std::max(config_.regularization_weight, 1.0e-9);
    Eigen::Matrix<double, 6, 6> hessian =
        wrench_map.transpose() * weights * wrench_map +
        (regularization + config_.smooth_weight) * Eigen::Matrix<double, 6, 6>::Identity();

    const Eigen::Matrix<double, 6, 1> smooth_target =
        has_last_solution_ ? last_solution_ : nominal;
    Eigen::Matrix<double, 6, 1> gradient_linear =
        -wrench_map.transpose() * weights * result.target_wrench -
        regularization * nominal -
        config_.smooth_weight * smooth_target;

    Eigen::Matrix<double, 6, 1> forces =
        hessian.ldlt().solve(-gradient_linear);
    forces = project_forces(forces);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigen_solver(hessian);
    double lipschitz = eigen_solver.eigenvalues().maxCoeff();
    if (!std::isfinite(lipschitz) || lipschitz <= 1.0e-9) {
        lipschitz = 1.0;
    }
    const double step = 1.0 / lipschitz;
    for (int iter = 0; iter < config_.iterations; iter++) {
        forces -= step * (hessian * forces + gradient_linear);
        forces = project_forces(forces);
    }

    result.left_force = forces.segment<3>(0);
    result.right_force = forces.segment<3>(3);
    result.achieved_wrench = wrench_map * forces;
    last_solution_ = forces;
    has_last_solution_ = true;
    return result;
}

Eigen::Matrix<double, 6, 1> ContactForceQp::project_forces(
    const Eigen::Matrix<double, 6, 1>& forces) const {
    Eigen::Matrix<double, 6, 1> output;
    output.segment<3>(0) = project_foot_force(forces.segment<3>(0));
    output.segment<3>(3) = project_foot_force(forces.segment<3>(3));
    return output;
}

Eigen::Vector3d ContactForceQp::project_foot_force(const Eigen::Vector3d& force) const {
    Eigen::Vector3d output = force;
    output.z() = std::clamp(output.z(), config_.min_normal_force, config_.max_normal_force);
    const double tangential_limit = config_.friction_coefficient * output.z();
    output.x() = std::clamp(output.x(), -tangential_limit, tangential_limit);
    output.y() = std::clamp(output.y(), -tangential_limit, tangential_limit);
    return output;
}

}  // namespace whole_body_mpc
