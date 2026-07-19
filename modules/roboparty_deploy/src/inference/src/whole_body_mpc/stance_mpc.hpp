// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <Eigen/Core>

namespace whole_body_mpc {

class StanceMpc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        int horizon = 20;
        double dt = 0.004;
        double orientation_weight = 120.0;
        double angular_rate_weight = 1.4;
        double com_weight = 30.0;
        double com_velocity_weight = 4.0;
        double control_weight = 0.15;
        double max_angular_accel = 12.0;
        double max_com_accel = 2.0;
        double target_roll = 0.0;
        double target_pitch = 0.0;
    };

    struct Input {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double roll = 0.0;
        double pitch = 0.0;
        double wx = 0.0;
        double wy = 0.0;
        Eigen::Vector2d com_offset_error = Eigen::Vector2d::Zero();
        Eigen::Vector2d com_velocity = Eigen::Vector2d::Zero();
    };

    struct Output {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Eigen::Vector3d desired_com_acceleration = Eigen::Vector3d::Zero();
        Eigen::Vector3d desired_angular_acceleration = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 8, 1> state = Eigen::Matrix<double, 8, 1>::Zero();
        Eigen::Matrix<double, 4, 1> control = Eigen::Matrix<double, 4, 1>::Zero();
    };

    explicit StanceMpc(Config config);

    void reset();
    Output solve(const Input& input) const;

   private:
    Config config_;
};

}  // namespace whole_body_mpc
