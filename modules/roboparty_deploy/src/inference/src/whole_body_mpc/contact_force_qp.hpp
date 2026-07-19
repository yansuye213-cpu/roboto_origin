// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <Eigen/Core>

namespace whole_body_mpc {

class ContactForceQp {
   public:
    struct Config {
        int iterations = 24;
        double friction_coefficient = 0.45;
        double min_normal_force = 0.0;
        double max_normal_force = 250.0;
        double force_tracking_weight = 1.0;
        double moment_tracking_weight = 0.03;
        double regularization_weight = 1.0e-4;
        double smooth_weight = 0.02;
    };

    struct Input {
        double mass = 0.0;
        Eigen::Vector3d desired_com_acceleration = Eigen::Vector3d::Zero();
        Eigen::Vector3d desired_body_moment = Eigen::Vector3d::Zero();
        Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
    };

    struct Result {
        Eigen::Vector3d left_force = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_force = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, 1> target_wrench = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> achieved_wrench = Eigen::Matrix<double, 6, 1>::Zero();
    };

    explicit ContactForceQp(Config config);

    void reset();
    Result solve(const Input& input);

   private:
    Eigen::Matrix<double, 6, 1> project_forces(
        const Eigen::Matrix<double, 6, 1>& forces) const;
    Eigen::Vector3d project_foot_force(const Eigen::Vector3d& force) const;

    Config config_;
    Eigen::Matrix<double, 6, 1> last_solution_ = Eigen::Matrix<double, 6, 1>::Zero();
    bool has_last_solution_ = false;
};

}  // namespace whole_body_mpc
