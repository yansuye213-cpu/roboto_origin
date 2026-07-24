// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/reference/swing_trajectory.hpp"

#include <algorithm>
#include <cmath>

namespace whole_body_mpc {

namespace {

constexpr size_t kLeftContactIndex = 0;
constexpr size_t kRightContactIndex = 1;
constexpr double kMinSegmentDuration = 1.0e-4;

}  // namespace

SwingTrajectoryPlanner::SwingTrajectoryPlanner(SwingTrajectoryConfig config)
    : config_(config) {
    fallback_position_[kLeftContactIndex].setZero();
    fallback_position_[kRightContactIndex].setZero();
}

bool SwingTrajectoryPlanner::CubicSegment::contains(double time) const {
    return time >= start_time && time <= end_time;
}

Eigen::Vector3d SwingTrajectoryPlanner::CubicSegment::position(double time) const {
    const double tau = std::clamp(time, start_time, end_time) - start_time;
    return a0 + tau * (a1 + tau * (a2 + tau * a3));
}

Eigen::Vector3d SwingTrajectoryPlanner::CubicSegment::velocity(double time) const {
    const double tau = std::clamp(time, start_time, end_time) - start_time;
    return a1 + tau * (2.0 * a2 + 3.0 * tau * a3);
}

Eigen::Vector3d SwingTrajectoryPlanner::CubicSegment::acceleration(double time) const {
    const double tau = std::clamp(time, start_time, end_time) - start_time;
    return 2.0 * a2 + 6.0 * tau * a3;
}

ContactFlags SwingTrajectoryPlanner::sample_flags(
    const ContactScheduleSample& sample) {
    return ContactFlags{sample.left_contact, sample.right_contact};
}

Eigen::Vector3d SwingTrajectoryPlanner::sample_foot_position(
    const ContactScheduleSample& sample,
    size_t contact_index) {
    return contact_index == kLeftContactIndex ? sample.left_foot_position
                                              : sample.right_foot_position;
}

bool SwingTrajectoryPlanner::sample_foot_swing(
    const ContactScheduleSample& sample,
    size_t contact_index) {
    return contact_index == kLeftContactIndex ? sample.left_swing
                                              : sample.right_swing;
}

double SwingTrajectoryPlanner::scaled_swing_height(
    double duration,
    const SwingTrajectoryConfig& config) {
    if (config.swing_height <= 0.0) {
        return 0.0;
    }
    if (config.swing_time_scale <= 0.0) {
        return config.swing_height;
    }
    return config.swing_height *
           std::clamp(duration / config.swing_time_scale, 0.0, 1.0);
}

SwingTrajectoryPlanner::CubicSegment SwingTrajectoryPlanner::make_segment(
    double start_time,
    double end_time,
    const Eigen::Vector3d& start_position,
    const Eigen::Vector3d& start_velocity,
    const Eigen::Vector3d& end_position,
    const Eigen::Vector3d& end_velocity) {
    const double duration = std::max(end_time - start_time, kMinSegmentDuration);
    CubicSegment segment;
    segment.start_time = start_time;
    segment.end_time = start_time + duration;
    segment.a0 = start_position;
    segment.a1 = start_velocity;
    segment.a2 =
        (3.0 * (end_position - start_position) / duration -
         2.0 * start_velocity - end_velocity) /
        duration;
    segment.a3 =
        (2.0 * (start_position - end_position) / duration +
         start_velocity + end_velocity) /
        (duration * duration);
    return segment;
}

void SwingTrajectoryPlanner::add_constant_segment(
    size_t contact_index,
    double start_time,
    double end_time,
    const Eigen::Vector3d& position) {
    if (contact_index >= segments_.size() || end_time <= start_time) {
        return;
    }
    segments_[contact_index].push_back(
        make_segment(start_time, end_time, position, Eigen::Vector3d::Zero(),
                     position, Eigen::Vector3d::Zero()));
}

void SwingTrajectoryPlanner::add_swing_segments(
    size_t contact_index,
    double start_time,
    double end_time,
    const Eigen::Vector3d& start_position,
    const Eigen::Vector3d& end_position) {
    if (contact_index >= segments_.size() || end_time <= start_time) {
        return;
    }

    const double duration = end_time - start_time;
    const double mid_time = 0.5 * (start_time + end_time);
    Eigen::Vector3d apex_position = 0.5 * (start_position + end_position);
    apex_position.z() =
        std::max(start_position.z(), end_position.z()) +
        scaled_swing_height(duration, config_);

    Eigen::Vector3d lift_off_velocity = Eigen::Vector3d::Zero();
    lift_off_velocity.z() = std::max(config_.lift_off_velocity, 0.0);
    Eigen::Vector3d touch_down_velocity = Eigen::Vector3d::Zero();
    touch_down_velocity.z() = -std::max(config_.touch_down_velocity, 0.0);

    segments_[contact_index].push_back(
        make_segment(start_time, mid_time, start_position,
                     lift_off_velocity, apex_position,
                     Eigen::Vector3d::Zero()));
    segments_[contact_index].push_back(
        make_segment(mid_time, end_time, apex_position,
                     Eigen::Vector3d::Zero(), end_position,
                     touch_down_velocity));
}

void SwingTrajectoryPlanner::update(
    const ocs2::ModeSchedule& mode_schedule,
    const ContactSchedule& contact_schedule,
    double time_origin,
    const Eigen::Vector3d& left_foot_position,
    const Eigen::Vector3d& right_foot_position) {
    fallback_position_[kLeftContactIndex] = left_foot_position;
    fallback_position_[kRightContactIndex] = right_foot_position;
    for (auto& segments : segments_) {
        segments.clear();
    }

    if (contact_schedule.samples.empty()) {
        const double final_time =
            mode_schedule.eventTimes.empty()
                ? time_origin + 1.0
                : std::max(time_origin + 1.0, mode_schedule.eventTimes.back());
        add_constant_segment(kLeftContactIndex, time_origin, final_time,
                             left_foot_position);
        add_constant_segment(kRightContactIndex, time_origin, final_time,
                             right_foot_position);
        return;
    }

    for (size_t contact_index = 0; contact_index < segments_.size();
         contact_index++) {
        size_t i = 0;
        while (i < contact_schedule.samples.size()) {
            const ContactScheduleSample& sample = contact_schedule.samples[i];
            const double segment_start = time_origin + std::max(sample.time, 0.0);
            size_t j = i + 1;
            const bool swing = sample_foot_swing(sample, contact_index);
            while (j < contact_schedule.samples.size() &&
                   sample_foot_swing(contact_schedule.samples[j],
                                     contact_index) == swing) {
                j++;
            }

            const size_t end_sample_index =
                std::min(j, contact_schedule.samples.size() - 1);
            const double segment_end =
                time_origin +
                std::max(contact_schedule.samples[end_sample_index].time,
                         sample.time);
            const Eigen::Vector3d start_position =
                sample_foot_position(sample, contact_index);
            const Eigen::Vector3d end_position =
                sample_foot_position(contact_schedule.samples[end_sample_index],
                                     contact_index);

            if (segment_end > segment_start) {
                if (swing) {
                    add_swing_segments(contact_index, segment_start, segment_end,
                                       start_position, end_position);
                } else {
                    add_constant_segment(contact_index, segment_start,
                                         segment_end, start_position);
                }
            }
            i = std::max(j, i + 1);
        }

        if (segments_[contact_index].empty()) {
            add_constant_segment(contact_index, time_origin, time_origin + 1.0,
                                 fallback_position_[contact_index]);
        }
    }
}

Eigen::Vector3d SwingTrajectoryPlanner::position(size_t contact_index,
                                                 double time) const {
    if (contact_index >= segments_.size()) {
        return Eigen::Vector3d::Zero();
    }
    const auto& segments = segments_[contact_index];
    for (const CubicSegment& segment : segments) {
        if (segment.contains(time)) {
            return segment.position(time);
        }
    }
    if (!segments.empty()) {
        return time < segments.front().start_time
                   ? segments.front().position(segments.front().start_time)
                   : segments.back().position(segments.back().end_time);
    }
    return fallback_position_[contact_index];
}

Eigen::Vector3d SwingTrajectoryPlanner::velocity(size_t contact_index,
                                                 double time) const {
    if (contact_index >= segments_.size()) {
        return Eigen::Vector3d::Zero();
    }
    const auto& segments = segments_[contact_index];
    for (const CubicSegment& segment : segments) {
        if (segment.contains(time)) {
            return segment.velocity(time);
        }
    }
    return Eigen::Vector3d::Zero();
}

Eigen::Vector3d SwingTrajectoryPlanner::acceleration(size_t contact_index,
                                                     double time) const {
    if (contact_index >= segments_.size()) {
        return Eigen::Vector3d::Zero();
    }
    const auto& segments = segments_[contact_index];
    for (const CubicSegment& segment : segments) {
        if (segment.contains(time)) {
            return segment.acceleration(time);
        }
    }
    return Eigen::Vector3d::Zero();
}

}  // namespace whole_body_mpc
