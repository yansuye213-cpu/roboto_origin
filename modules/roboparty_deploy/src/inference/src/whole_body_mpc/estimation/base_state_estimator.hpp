// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/model/robot_model.hpp"

#include <Eigen/Core>

namespace whole_body_mpc {

class BaseStateEstimator {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        bool enabled = true;
        double velocity_filter_alpha = 0.85;
        double max_base_linear_velocity = 0.8;
    };

    struct Input {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        const RobotModel::Kinematics* kinematics = nullptr;
        const Eigen::VectorXd* generalized_velocity_without_base_linear = nullptr;
        bool left_contact = true;
        bool right_contact = true;
    };

    struct Output {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Eigen::Vector3d base_linear_velocity = Eigen::Vector3d::Zero();
        double stance_velocity_residual = 0.0;
        int constraint_rows = 0;
        bool used_contact_constraint = false;
    };

    explicit BaseStateEstimator(Config config);

    void reset();
    Output estimate(const Input& input);

   private:
    Config config_;
    Eigen::Vector3d filtered_base_linear_velocity_ = Eigen::Vector3d::Zero();
    bool has_filtered_velocity_ = false;
};

}  // namespace whole_body_mpc
