// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/reference/contact_schedule.hpp"

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/ModeSchedule.h>

#include <cstddef>

namespace whole_body_mpc {

struct ContactFlags {
    bool left = true;
    bool right = true;
};

constexpr size_t kModeDoubleSupport = 0;
constexpr size_t kModeLeftStance = 1;
constexpr size_t kModeRightStance = 2;
constexpr size_t kModeFlight = 3;

size_t mode_from_contact_flags(ContactFlags flags);
ContactFlags contact_flags_from_mode(size_t mode);
int contact_count(ContactFlags flags);

ocs2::ModeSchedule to_ocs2_mode_schedule(const ContactSchedule& schedule,
                                         double time_origin,
                                         ContactFlags fallback);

ContactFlags contact_flags_at_time(const ocs2::ModeSchedule& mode_schedule,
                                   double time);

}  // namespace whole_body_mpc
