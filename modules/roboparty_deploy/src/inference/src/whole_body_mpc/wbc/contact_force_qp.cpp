// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/wbc/contact_force_qp.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

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
    last_solution_.resize(0);
    has_last_solution_ = false;
}

ContactForceQp::Result ContactForceQp::solve(const Input& input) {
    if (input.mass <= 0.0) {
        throw std::runtime_error("ContactForceQp requires positive robot mass");
    }

    Vector3dList contact_positions = input.contact_positions;
    int left_contact_count = input.left_contact_count;
    if (contact_positions.empty()) {
        contact_positions.push_back(input.left_foot_position);
        contact_positions.push_back(input.right_foot_position);
        left_contact_count = 1;
    }
    if (left_contact_count < 0 ||
        left_contact_count > static_cast<int>(contact_positions.size())) {
        throw std::runtime_error("ContactForceQp left_contact_count is invalid");
    }
    const int contact_count = static_cast<int>(contact_positions.size());
    const int force_dim = contact_count * 3;
    const int right_contact_count = contact_count - left_contact_count;

    Eigen::MatrixXd wrench_map(6, force_dim);
    wrench_map.setZero();
    for (int i = 0; i < contact_count; i++) {
        wrench_map.block<3, 3>(0, i * 3).setIdentity();
        wrench_map.block<3, 3>(3, i * 3) =
            skew(contact_positions[i] - input.com_position);
    }

    Result result;
    result.contact_forces.resize(contact_count, Eigen::Vector3d::Zero());

    Eigen::VectorXd nominal = Eigen::VectorXd::Zero(force_dim);
    if (input.desired_foot_forces_available) {
        for (int i = 0; i < left_contact_count; i++) {
            nominal.segment<3>(i * 3) =
                input.desired_left_force /
                static_cast<double>(std::max(left_contact_count, 1));
        }
        for (int i = left_contact_count; i < contact_count; i++) {
            nominal.segment<3>(i * 3) =
                input.desired_right_force /
                static_cast<double>(std::max(right_contact_count, 1));
        }
        nominal = project_forces(nominal, left_contact_count);
    } else {
        for (int i = 0; i < contact_count; i++) {
            nominal.segment<3>(i * 3) =
                input.mass * Eigen::Vector3d(0.0, 0.0, kGravity) /
                static_cast<double>(contact_count);
        }
        nominal = project_forces(nominal, left_contact_count);
    }
    result.target_wrench.head<3>() =
        input.mass * (input.desired_com_acceleration +
                      Eigen::Vector3d(0.0, 0.0, kGravity));
    result.target_wrench.tail<3>() = input.desired_body_moment;

    Eigen::Matrix<double, 6, 6> weights = Eigen::Matrix<double, 6, 6>::Zero();
    weights.diagonal().head<3>().setConstant(config_.force_tracking_weight);
    weights.diagonal().tail<3>().setConstant(config_.moment_tracking_weight);

    const double regularization = std::max(config_.regularization_weight, 1.0e-9);
    Eigen::MatrixXd hessian =
        wrench_map.transpose() * weights * wrench_map +
        (regularization + config_.smooth_weight) * Eigen::MatrixXd::Identity(force_dim, force_dim);

    const Eigen::VectorXd smooth_target =
        (has_last_solution_ && last_solution_.size() == force_dim) ? last_solution_ : nominal;
    Eigen::VectorXd gradient_linear =
        -wrench_map.transpose() * weights * result.target_wrench -
        regularization * nominal -
        config_.smooth_weight * smooth_target;

    Eigen::VectorXd forces =
        hessian.ldlt().solve(-gradient_linear);
    forces = project_forces(forces, left_contact_count);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(hessian);
    double lipschitz = eigen_solver.eigenvalues().maxCoeff();
    if (!std::isfinite(lipschitz) || lipschitz <= 1.0e-9) {
        lipschitz = 1.0;
    }
    const double step = 1.0 / lipschitz;
    for (int iter = 0; iter < config_.iterations; iter++) {
        forces -= step * (hessian * forces + gradient_linear);
        forces = project_forces(forces, left_contact_count);
    }

    for (int i = 0; i < contact_count; i++) {
        result.contact_forces[i] = forces.segment<3>(i * 3);
        if (i < left_contact_count) {
            result.left_force += result.contact_forces[i];
        } else {
            result.right_force += result.contact_forces[i];
        }
    }
    result.achieved_wrench = wrench_map * forces;
    last_solution_ = forces;
    has_last_solution_ = true;
    return result;
}

Eigen::VectorXd ContactForceQp::project_forces(
    const Eigen::VectorXd& forces, int left_contact_count) const {
    Eigen::VectorXd output = forces;
    const int contact_count = static_cast<int>(forces.size() / 3);
    const int right_contact_count = contact_count - left_contact_count;
    for (int i = 0; i < contact_count; i++) {
        const int same_foot_count =
            i < left_contact_count ? left_contact_count : right_contact_count;
        output.segment<3>(i * 3) =
            project_contact_force(forces.segment<3>(i * 3), same_foot_count);
    }
    return output;
}

Eigen::Vector3d ContactForceQp::project_contact_force(
    const Eigen::Vector3d& force, int contacts_on_same_foot) const {
    const double divisor = static_cast<double>(std::max(contacts_on_same_foot, 1));
    const double min_normal_force = config_.min_normal_force / divisor;
    const double max_normal_force = config_.max_normal_force / divisor;
    Eigen::Vector3d output = force;
    output.z() = std::clamp(output.z(), min_normal_force, max_normal_force);
    const double tangential_limit = config_.friction_coefficient * output.z();
    output.x() = std::clamp(output.x(), -tangential_limit, tangential_limit);
    output.y() = std::clamp(output.y(), -tangential_limit, tangential_limit);
    return output;
}

}  // namespace whole_body_mpc
