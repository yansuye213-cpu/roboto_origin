// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <Eigen/Core>

namespace whole_body_mpc {

class FootPlacementPlanner {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        double step_x_pitch_gain = 0.28;
        double step_x_rate_gain = 0.04;
        double step_x_com_gain = 0.30;
        double step_x_com_velocity_gain = 0.08;
        double step_y_roll_gain = 0.18;
        double step_y_rate_gain = 0.03;
        double step_y_com_gain = 0.30;
        double step_y_com_velocity_gain = 0.06;
        double capture_time = 0.25;
        double capture_gain = 0.35;
        double min_step_x = 0.03;
        double min_step_y = 0.02;
        double max_step_x = 0.12;
        double max_step_y = 0.07;
        double sagittal_sign = 1.0;
        double lateral_sign = 1.0;
    };

    struct Input {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double roll_error = 0.0;
        double pitch_error = 0.0;
        double wx = 0.0;
        double wy = 0.0;
        double roll_trigger = 0.0;
        double pitch_trigger = 0.0;
        Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d com_velocity = Eigen::Vector3d::Zero();
        Eigen::Vector3d support_center = Eigen::Vector3d::Zero();
    };

    explicit FootPlacementPlanner(Config config);

    Eigen::Vector3d compute_step_offset(const Input& input) const;

   private:
    Config config_;
};

}  // namespace whole_body_mpc
