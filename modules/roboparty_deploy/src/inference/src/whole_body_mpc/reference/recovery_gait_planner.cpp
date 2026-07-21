// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/reference/recovery_gait_planner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace whole_body_mpc {

namespace {

double quintic(double s) {
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}

double quintic_dot(double s) {
    return 30.0 * s * s - 60.0 * s * s * s + 30.0 * s * s * s * s;
}

double quintic_ddot(double s) {
    return 60.0 * s - 180.0 * s * s + 120.0 * s * s * s;
}

FootPlacementPlanner::Config make_foot_placement_config(
    const RecoveryGaitPlanner::Config& config) {
    FootPlacementPlanner::Config output;
    output.step_x_pitch_gain = config.step_x_pitch_gain;
    output.step_x_rate_gain = config.step_x_rate_gain;
    output.step_x_com_gain = config.step_x_com_gain;
    output.step_x_com_velocity_gain = config.step_x_com_velocity_gain;
    output.step_y_roll_gain = config.step_y_roll_gain;
    output.step_y_rate_gain = config.step_y_rate_gain;
    output.step_y_com_gain = config.step_y_com_gain;
    output.step_y_com_velocity_gain = config.step_y_com_velocity_gain;
    output.capture_time = config.capture_time;
    output.capture_gain = config.capture_gain;
    output.min_step_x = config.min_step_x;
    output.min_step_y = config.min_step_y;
    output.max_step_x = config.max_step_x;
    output.max_step_y = config.max_step_y;
    output.sagittal_sign = config.sagittal_sign;
    output.lateral_sign = config.lateral_sign;
    return output;
}

}  // namespace

RecoveryGaitPlanner::RecoveryGaitPlanner(Config config)
    : config_(config),
      foot_placement_planner_(make_foot_placement_config(config_)) {
    if (config_.roll_trigger < 0.0 || config_.pitch_trigger < 0.0 ||
        config_.rate_trigger < 0.0 || config_.com_trigger < 0.0 ||
        config_.com_velocity_trigger < 0.0 || config_.return_roll < 0.0 ||
        config_.return_pitch < 0.0 || config_.return_rate < 0.0 ||
        config_.return_com < 0.0 || config_.return_com_velocity < 0.0 ||
        config_.step_count < 0 || config_.swing_time <= 0.0 ||
        config_.double_support_time < 0.0 || config_.settle_time < 0.0 ||
        config_.stable_time < 0.0 || config_.cooldown < 0.0 ||
        config_.max_duration < 0.0 || config_.min_step_x < 0.0 ||
        config_.min_step_y < 0.0 || config_.max_step_x < 0.0 ||
        config_.max_step_y < 0.0 || config_.capture_time < 0.0 ||
        config_.capture_gain < 0.0 || config_.swing_height < 0.0 ||
        config_.min_step_x > config_.max_step_x ||
        config_.min_step_y > config_.max_step_y) {
        throw std::runtime_error("RecoveryGaitPlanner parameters are invalid");
    }
}

void RecoveryGaitPlanner::reset() {
    phase_ = Phase::Idle;
    active_ = false;
    current_swing_left_ = config_.start_with_left;
    next_swing_left_ = config_.start_with_left;
    steps_completed_ = 0;
    planned_steps_ = 0;
    elapsed_ = 0.0;
    phase_elapsed_ = 0.0;
    stable_elapsed_ = 0.0;
    cooldown_elapsed_ = 1000000.0;
    swing_start_position_.setZero();
    swing_target_position_.setZero();
    latest_step_offset_.setZero();
}

RecoveryGaitPlanner::Reference RecoveryGaitPlanner::update(const Input& input) {
    const double dt = std::max(input.dt, 0.0);
    if (!active_) {
        cooldown_elapsed_ = std::min(cooldown_elapsed_ + dt, config_.cooldown);
        if (should_start(input)) {
            start_sequence(input);
        } else {
            return build_reference(input);
        }
    } else {
        elapsed_ += dt;
        phase_elapsed_ += dt;
    }

    if (phase_ == Phase::DoubleSupport &&
        phase_elapsed_ >= config_.double_support_time) {
        enter_swing(input);
    }

    if (phase_ == Phase::Swing && phase_elapsed_ >= config_.swing_time) {
        steps_completed_++;
        if (steps_completed_ >= planned_steps_) {
            enter_settle();
        } else {
            next_swing_left_ = !current_swing_left_;
            enter_double_support();
        }
    }

    if (phase_ == Phase::Settle) {
        if (is_stable(input)) {
            stable_elapsed_ += dt;
        } else {
            stable_elapsed_ = 0.0;
        }
        const bool stable_long_enough = stable_elapsed_ >= config_.stable_time;
        const bool settled_long_enough = phase_elapsed_ >= config_.settle_time;
        const bool timed_out =
            config_.max_duration > 0.0 && elapsed_ >= config_.max_duration;
        if ((stable_long_enough && settled_long_enough) || timed_out) {
            finish_sequence();
        }
    }

    return build_reference(input);
}

bool RecoveryGaitPlanner::should_start(const Input& input) const {
    if (!config_.enabled || active_ || cooldown_elapsed_ < config_.cooldown) {
        return false;
    }
    const double roll_error = input.roll - input.target_roll;
    const double pitch_error = input.pitch - input.target_pitch;
    const double angular_rate = std::max(std::abs(input.wx), std::abs(input.wy));
    const double com_error =
        (input.com_position - input.support_center).head<2>().norm();
    const double com_velocity = input.com_velocity.head<2>().norm();
    return (config_.roll_trigger > 0.0 && std::abs(roll_error) >= config_.roll_trigger) ||
           (config_.pitch_trigger > 0.0 && std::abs(pitch_error) >= config_.pitch_trigger) ||
           (config_.rate_trigger > 0.0 && angular_rate >= config_.rate_trigger) ||
           (config_.com_trigger > 0.0 && com_error >= config_.com_trigger) ||
           (config_.com_velocity_trigger > 0.0 &&
            com_velocity >= config_.com_velocity_trigger);
}

bool RecoveryGaitPlanner::is_stable(const Input& input) const {
    const double roll_error = input.roll - input.target_roll;
    const double pitch_error = input.pitch - input.target_pitch;
    const double angular_rate = std::max(std::abs(input.wx), std::abs(input.wy));
    const double com_error =
        (input.com_position - input.support_center).head<2>().norm();
    const double com_velocity = input.com_velocity.head<2>().norm();
    const bool attitude_stable =
        std::abs(roll_error) <= config_.return_roll &&
        std::abs(pitch_error) <= config_.return_pitch &&
        angular_rate <= config_.return_rate;
    const bool com_stable =
        (config_.return_com <= 0.0 || com_error <= config_.return_com) &&
        (config_.return_com_velocity <= 0.0 ||
         com_velocity <= config_.return_com_velocity);
    return attitude_stable && com_stable;
}

void RecoveryGaitPlanner::start_sequence(const Input& input) {
    active_ = true;
    phase_ = Phase::Idle;
    elapsed_ = 0.0;
    phase_elapsed_ = 0.0;
    stable_elapsed_ = 0.0;
    cooldown_elapsed_ = 0.0;
    steps_completed_ = 0;
    planned_steps_ = std::max(config_.step_count, 1);
    next_swing_left_ = choose_first_swing_left(input);
    enter_double_support();
}

void RecoveryGaitPlanner::enter_double_support() {
    phase_ = Phase::DoubleSupport;
    phase_elapsed_ = 0.0;
}

void RecoveryGaitPlanner::enter_swing(const Input& input) {
    phase_ = Phase::Swing;
    phase_elapsed_ = 0.0;
    current_swing_left_ = next_swing_left_;
    swing_start_position_ =
        current_swing_left_ ? input.left_foot_position : input.right_foot_position;
    latest_step_offset_ = compute_step_offset(input);
    swing_target_position_ = swing_start_position_ + latest_step_offset_;
    swing_target_position_.z() = swing_start_position_.z();
}

void RecoveryGaitPlanner::enter_settle() {
    phase_ = Phase::Settle;
    phase_elapsed_ = 0.0;
    stable_elapsed_ = 0.0;
}

void RecoveryGaitPlanner::finish_sequence() {
    phase_ = Phase::Idle;
    active_ = false;
    phase_elapsed_ = 0.0;
    elapsed_ = 0.0;
    stable_elapsed_ = 0.0;
    cooldown_elapsed_ = 0.0;
    latest_step_offset_.setZero();
}

bool RecoveryGaitPlanner::choose_first_swing_left(const Input& input) const {
    const double roll_error = input.roll - input.target_roll;
    if (std::abs(roll_error) >= config_.roll_trigger && config_.roll_trigger > 0.0) {
        const bool positive_roll = roll_error >= 0.0;
        return positive_roll == config_.first_swing_left_on_positive_roll;
    }
    return config_.start_with_left;
}

Eigen::Vector3d RecoveryGaitPlanner::compute_step_offset(const Input& input) const {
    if (!config_.step_placement_enabled) {
        return Eigen::Vector3d::Zero();
    }
    const double roll_error = input.roll - input.target_roll;
    const double pitch_error = input.pitch - input.target_pitch;
    FootPlacementPlanner::Input foot_input;
    foot_input.roll_error = roll_error;
    foot_input.pitch_error = pitch_error;
    foot_input.wx = input.wx;
    foot_input.wy = input.wy;
    foot_input.roll_trigger = config_.roll_trigger;
    foot_input.pitch_trigger = config_.pitch_trigger;
    foot_input.com_position = input.com_position;
    foot_input.com_velocity = input.com_velocity;
    foot_input.support_center = input.support_center;
    return foot_placement_planner_.compute_step_offset(foot_input);
}

RecoveryGaitPlanner::FootReference RecoveryGaitPlanner::swing_reference(bool left_swing) const {
    FootReference reference;
    if (phase_ != Phase::Swing || left_swing != current_swing_left_) {
        return reference;
    }

    const double duration = std::max(config_.swing_time, 1.0e-6);
    const double s = std::clamp(phase_elapsed_ / duration, 0.0, 1.0);
    const double h = quintic(s);
    const double hdot = quintic_dot(s) / duration;
    const double hddot = quintic_ddot(s) / (duration * duration);
    const double z_bump = 4.0 * s * (1.0 - s);
    const double z_bump_dot = 4.0 * (1.0 - 2.0 * s) / duration;
    const double z_bump_ddot = -8.0 / (duration * duration);
    const Eigen::Vector3d delta = swing_target_position_ - swing_start_position_;

    reference.active = true;
    reference.position = swing_start_position_ + h * delta;
    reference.position.z() += config_.swing_height * z_bump;
    reference.velocity = hdot * delta;
    reference.velocity.z() += config_.swing_height * z_bump_dot;
    reference.acceleration = hddot * delta;
    reference.acceleration.z() += config_.swing_height * z_bump_ddot;
    return reference;
}

RecoveryGaitPlanner::Reference RecoveryGaitPlanner::build_reference(const Input& input) const {
    Reference reference;
    reference.recovery_active = active_;
    reference.phase = phase_;
    reference.current_swing_left = current_swing_left_;
    reference.next_swing_left = next_swing_left_;
    reference.steps_completed = steps_completed_;
    reference.planned_steps = planned_steps_;
    reference.elapsed = elapsed_;
    reference.phase_elapsed = phase_elapsed_;
    reference.stable_elapsed = stable_elapsed_;
    reference.roll_error = input.roll - input.target_roll;
    reference.pitch_error = input.pitch - input.target_pitch;
    reference.angular_rate = std::max(std::abs(input.wx), std::abs(input.wy));
    reference.com_error =
        (input.com_position - input.support_center).head<2>().norm();
    reference.com_velocity = input.com_velocity.head<2>().norm();
    Eigen::Vector3d planned_step_offset = latest_step_offset_;
    if (active_ && phase_ == Phase::DoubleSupport && config_.step_placement_enabled) {
        planned_step_offset = compute_step_offset(input);
    }
    reference.step_x = planned_step_offset.x();
    reference.step_y = planned_step_offset.y();
    reference.left_foot_target = input.left_foot_position;
    reference.right_foot_target = input.right_foot_position;

    if (phase_ == Phase::Swing) {
        reference.left_swing = current_swing_left_;
        reference.right_swing = !current_swing_left_;
        reference.left_contact = !current_swing_left_;
        reference.right_contact = current_swing_left_;
        reference.left_foot = swing_reference(true);
        reference.right_foot = swing_reference(false);
        if (current_swing_left_) {
            reference.left_foot_target = swing_target_position_;
        } else {
            reference.right_foot_target = swing_target_position_;
        }
    } else if (active_ && phase_ == Phase::DoubleSupport) {
        if (next_swing_left_) {
            reference.left_foot_target = input.left_foot_position + planned_step_offset;
        } else {
            reference.right_foot_target = input.right_foot_position + planned_step_offset;
        }
    }

    return reference;
}

}  // namespace whole_body_mpc
