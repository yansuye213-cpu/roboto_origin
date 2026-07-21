// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/reference/recovery_gait_planner.hpp"

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <vector>

namespace whole_body_mpc {

struct ContactScheduleSample {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double time = 0.0;
    bool left_contact = true;
    bool right_contact = true;
    bool left_swing = false;
    bool right_swing = false;
    Eigen::Vector3d support_center = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
};

class ContactSchedule {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using SampleVector =
        std::vector<ContactScheduleSample,
                    Eigen::aligned_allocator<ContactScheduleSample>>;

    bool enabled = false;
    bool recovery_active = false;
    SampleVector samples;

    ContactScheduleSample sample_at(double time) const;
};

class ContactSchedulePlanner {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        bool enabled = true;
        int horizon = 20;
        double dt = 0.004;
        double double_support_time = 0.12;
        double swing_time = 0.45;
        double settle_time = 0.25;
    };

    struct Input {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RecoveryGaitPlanner::Reference gait_reference;
        Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
    };

    explicit ContactSchedulePlanner(Config config);

    ContactSchedule build(const Input& input) const;

   private:
    Config config_;
};

}  // namespace whole_body_mpc
