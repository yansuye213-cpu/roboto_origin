// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/reference/contact_schedule.hpp"

#include <algorithm>
#include <stdexcept>

namespace whole_body_mpc {

namespace {

using Phase = RecoveryGaitPlanner::Phase;

Eigen::Vector3d support_center_for(bool left_contact, bool right_contact,
                                   const Eigen::Vector3d& left_foot_position,
                                   const Eigen::Vector3d& right_foot_position) {
    if (left_contact && right_contact) {
        return 0.5 * (left_foot_position + right_foot_position);
    }
    if (left_contact) {
        return left_foot_position;
    }
    if (right_contact) {
        return right_foot_position;
    }
    return 0.5 * (left_foot_position + right_foot_position);
}

}  // namespace

ContactScheduleSample ContactSchedule::sample_at(double time) const {
    ContactScheduleSample fallback;
    if (samples.empty()) {
        return fallback;
    }
    const double query_time = std::max(time, 0.0);
    const auto it = std::lower_bound(
        samples.begin(), samples.end(), query_time,
        [](const ContactScheduleSample& sample, double value) {
            return sample.time < value;
        });
    if (it == samples.end()) {
        return samples.back();
    }
    return *it;
}

ContactSchedulePlanner::ContactSchedulePlanner(Config config)
    : config_(config) {
    if (config_.horizon <= 0 || config_.dt <= 0.0 ||
        config_.double_support_time < 0.0 || config_.swing_time <= 0.0 ||
        config_.settle_time < 0.0) {
        throw std::runtime_error("ContactSchedulePlanner parameters are invalid");
    }
}

ContactSchedule ContactSchedulePlanner::build(const Input& input) const {
    ContactSchedule schedule;
    schedule.enabled = config_.enabled;
    schedule.recovery_active = input.gait_reference.recovery_active;
    schedule.samples.reserve(static_cast<size_t>(config_.horizon));

    Eigen::Vector3d left_foot_position = input.left_foot_position;
    Eigen::Vector3d right_foot_position = input.right_foot_position;
    const Eigen::Vector3d step_offset(input.gait_reference.step_x,
                                      input.gait_reference.step_y, 0.0);

    Phase phase = input.gait_reference.phase;
    double phase_elapsed = input.gait_reference.phase_elapsed;
    int steps_completed = input.gait_reference.steps_completed;
    const int planned_steps = input.gait_reference.planned_steps;
    bool current_swing_left = input.gait_reference.current_swing_left;
    bool next_swing_left = input.gait_reference.next_swing_left;
    Eigen::Vector3d swing_target = current_swing_left
        ? input.gait_reference.left_foot_target
        : input.gait_reference.right_foot_target;

    for (int k = 0; k < config_.horizon; k++) {
        const double sample_time = static_cast<double>(k) * config_.dt;

        bool left_contact = true;
        bool right_contact = true;
        bool left_swing = false;
        bool right_swing = false;
        if (schedule.enabled && phase == Phase::Swing) {
            left_swing = current_swing_left;
            right_swing = !current_swing_left;
            left_contact = !left_swing;
            right_contact = !right_swing;
        }

        ContactScheduleSample sample;
        sample.time = sample_time;
        sample.left_contact = left_contact;
        sample.right_contact = right_contact;
        sample.left_swing = left_swing;
        sample.right_swing = right_swing;
        sample.left_foot_position = left_foot_position;
        sample.right_foot_position = right_foot_position;
        sample.support_center = support_center_for(
            left_contact, right_contact, left_foot_position, right_foot_position);
        schedule.samples.push_back(sample);

        if (!schedule.enabled || !schedule.recovery_active) {
            continue;
        }

        phase_elapsed += config_.dt;
        if (phase == Phase::DoubleSupport &&
            phase_elapsed >= config_.double_support_time) {
            phase = Phase::Swing;
            phase_elapsed = 0.0;
            current_swing_left = next_swing_left;
            swing_target =
                (current_swing_left ? left_foot_position : right_foot_position) +
                step_offset;
        } else if (phase == Phase::Swing &&
                   phase_elapsed >= config_.swing_time) {
            if (current_swing_left) {
                left_foot_position = swing_target;
            } else {
                right_foot_position = swing_target;
            }
            steps_completed++;
            if (steps_completed >= planned_steps) {
                phase = Phase::Settle;
                phase_elapsed = 0.0;
            } else {
                phase = Phase::DoubleSupport;
                phase_elapsed = 0.0;
                next_swing_left = !current_swing_left;
            }
        }
    }
    return schedule;
}

}  // namespace whole_body_mpc
