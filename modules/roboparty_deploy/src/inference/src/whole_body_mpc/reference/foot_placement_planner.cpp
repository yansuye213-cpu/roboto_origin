// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/reference/foot_placement_planner.hpp"

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

double sign_nonzero(double value) {
    return value >= 0.0 ? 1.0 : -1.0;
}

}  // namespace

FootPlacementPlanner::FootPlacementPlanner(Config config) : config_(config) {
    if (config_.capture_time < 0.0 || config_.capture_gain < 0.0 ||
        config_.min_step_x < 0.0 || config_.min_step_y < 0.0 ||
        config_.max_step_x < 0.0 || config_.max_step_y < 0.0 ||
        config_.min_step_x > config_.max_step_x ||
        config_.min_step_y > config_.max_step_y) {
        throw std::runtime_error("FootPlacementPlanner parameters are invalid");
    }
}

Eigen::Vector3d FootPlacementPlanner::compute_step_offset(const Input& input) const {
    const Eigen::Vector2d com_error =
        (input.com_position - input.support_center).head<2>();
    const Eigen::Vector2d com_velocity = input.com_velocity.head<2>();
    const Eigen::Vector2d capture_point =
        com_error + config_.capture_time * com_velocity;

    double step_x =
        config_.sagittal_sign *
        (config_.step_x_pitch_gain * input.pitch_error +
         config_.step_x_rate_gain * input.wy +
         config_.step_x_com_gain * com_error.x() +
         config_.step_x_com_velocity_gain * com_velocity.x() +
         config_.capture_gain * capture_point.x());
    double step_y =
        config_.lateral_sign *
        (config_.step_y_roll_gain * input.roll_error +
         config_.step_y_rate_gain * input.wx +
         config_.step_y_com_gain * com_error.y() +
         config_.step_y_com_velocity_gain * com_velocity.y() +
         config_.capture_gain * capture_point.y());

    if (std::abs(input.pitch_error) >= input.pitch_trigger &&
        std::abs(step_x) < config_.min_step_x) {
        step_x = config_.sagittal_sign *
                 sign_nonzero(input.pitch_error) * config_.min_step_x;
    }
    if (std::abs(input.roll_error) >= input.roll_trigger &&
        std::abs(step_y) < config_.min_step_y) {
        step_y = config_.lateral_sign *
                 sign_nonzero(input.roll_error) * config_.min_step_y;
    }

    return Eigen::Vector3d(clamp_abs(step_x, config_.max_step_x),
                           clamp_abs(step_y, config_.max_step_y), 0.0);
}

}  // namespace whole_body_mpc
