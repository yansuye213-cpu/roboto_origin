// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/reference/switched_model_reference_manager.hpp"

#include <algorithm>
#include <utility>

namespace whole_body_mpc {

SwitchedModelReferenceManager::SwitchedModelReferenceManager(
    ocs2::TargetTrajectories initial_target,
    ocs2::ModeSchedule initial_mode_schedule,
    SwingTrajectoryConfig swing_config)
    : ocs2::ReferenceManager(std::move(initial_target),
                             std::move(initial_mode_schedule)),
      swing_trajectory_(swing_config) {}

void SwitchedModelReferenceManager::set_reference_input(
    const ContactSchedule& contact_schedule,
    double time_origin,
    ContactFlags fallback_flags,
    const Eigen::Vector3d& left_foot_position,
    const Eigen::Vector3d& right_foot_position) {
    contact_schedule_ = contact_schedule;
    time_origin_ = time_origin;
    fallback_flags_ = fallback_flags;
    left_foot_position_ = left_foot_position;
    right_foot_position_ = right_foot_position;
}

void SwitchedModelReferenceManager::setModeSchedule(
    const ocs2::ModeSchedule& mode_schedule) {
    ocs2::ReferenceManager::setModeSchedule(mode_schedule);
}

void SwitchedModelReferenceManager::setModeSchedule(
    ocs2::ModeSchedule&& mode_schedule) {
    ocs2::ReferenceManager::setModeSchedule(std::move(mode_schedule));
}

ContactFlags SwitchedModelReferenceManager::contact_flags(double time) const {
    return contact_flags_at_time(getModeSchedule(), time);
}

void SwitchedModelReferenceManager::modifyReferences(
    ocs2::scalar_t init_time,
    ocs2::scalar_t final_time,
    const ocs2::vector_t& init_state,
    ocs2::TargetTrajectories& target_trajectories,
    ocs2::ModeSchedule& mode_schedule) {
    (void)init_state;
    if (contact_schedule_.enabled && !contact_schedule_.samples.empty()) {
        mode_schedule = to_ocs2_mode_schedule(contact_schedule_, time_origin_,
                                              fallback_flags_);
    }
    if (target_trajectories.size() == 1) {
        target_trajectories.timeTrajectory.push_back(final_time);
        target_trajectories.stateTrajectory.push_back(
            target_trajectories.stateTrajectory.front());
        target_trajectories.inputTrajectory.push_back(
            target_trajectories.inputTrajectory.front());
    }

    if (mode_schedule.modeSequence.empty()) {
        mode_schedule.modeSequence.push_back(
            mode_from_contact_flags(fallback_flags_));
    }
    if (mode_schedule.eventTimes.empty() ||
        mode_schedule.eventTimes.back() < final_time) {
        const double extension_time =
            std::max(static_cast<double>(final_time), init_time + 1.0e-3);
        mode_schedule.eventTimes.push_back(extension_time);
        mode_schedule.modeSequence.push_back(mode_schedule.modeSequence.back());
    }

    swing_trajectory_.update(mode_schedule, contact_schedule_, time_origin_,
                             left_foot_position_, right_foot_position_);
}

}  // namespace whole_body_mpc
