// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/reference/mode_schedule_adapter.hpp"

#include <algorithm>
#include <vector>

namespace whole_body_mpc {

size_t mode_from_contact_flags(ContactFlags flags) {
    if (flags.left && flags.right) {
        return kModeDoubleSupport;
    }
    if (flags.left) {
        return kModeLeftStance;
    }
    if (flags.right) {
        return kModeRightStance;
    }
    return kModeFlight;
}

ContactFlags contact_flags_from_mode(size_t mode) {
    switch (mode) {
        case kModeLeftStance:
            return ContactFlags{true, false};
        case kModeRightStance:
            return ContactFlags{false, true};
        case kModeFlight:
            return ContactFlags{false, false};
        case kModeDoubleSupport:
        default:
            return ContactFlags{true, true};
    }
}

int contact_count(ContactFlags flags) {
    return (flags.left ? 1 : 0) + (flags.right ? 1 : 0);
}

ocs2::ModeSchedule to_ocs2_mode_schedule(const ContactSchedule& schedule,
                                         double time_origin,
                                         ContactFlags fallback) {
    std::vector<ocs2::scalar_t> event_times;
    std::vector<size_t> mode_sequence;

    if (!schedule.enabled || schedule.samples.empty()) {
        mode_sequence.push_back(mode_from_contact_flags(fallback));
        return ocs2::ModeSchedule(std::move(event_times), std::move(mode_sequence));
    }

    const auto make_flags = [](const ContactScheduleSample& sample) {
        return ContactFlags{sample.left_contact, sample.right_contact};
    };

    size_t previous_mode = mode_from_contact_flags(make_flags(schedule.samples.front()));
    mode_sequence.push_back(previous_mode);
    double previous_event_time = time_origin;
    for (size_t i = 1; i < schedule.samples.size(); i++) {
        const size_t mode = mode_from_contact_flags(make_flags(schedule.samples[i]));
        const double event_time = time_origin + std::max(schedule.samples[i].time, 0.0);
        if (mode == previous_mode || event_time <= previous_event_time) {
            continue;
        }
        event_times.push_back(static_cast<ocs2::scalar_t>(event_time));
        mode_sequence.push_back(mode);
        previous_mode = mode;
        previous_event_time = event_time;
    }

    if (mode_sequence.empty()) {
        mode_sequence.push_back(mode_from_contact_flags(fallback));
    }
    return ocs2::ModeSchedule(std::move(event_times), std::move(mode_sequence));
}

ContactFlags contact_flags_at_time(const ocs2::ModeSchedule& mode_schedule,
                                   double time) {
    if (mode_schedule.modeSequence.empty()) {
        return ContactFlags{};
    }
    return contact_flags_from_mode(mode_schedule.modeAtTime(time));
}

}  // namespace whole_body_mpc
