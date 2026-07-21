// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>

namespace whole_body_mpc {

using Vector3dList = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

class ContactForceQp {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        int iterations = 24;
        double friction_coefficient = 0.45;
        double min_normal_force = 0.0;
        double max_normal_force = 420.0;
        double force_tracking_weight = 1.0;
        double moment_tracking_weight = 0.03;
        double regularization_weight = 1.0e-4;
        double smooth_weight = 0.02;
    };

    struct Input {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double mass = 0.0;
        Eigen::Vector3d desired_com_acceleration = Eigen::Vector3d::Zero();
        Eigen::Vector3d desired_body_moment = Eigen::Vector3d::Zero();
        Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d desired_left_force = Eigen::Vector3d::Zero();
        Eigen::Vector3d desired_right_force = Eigen::Vector3d::Zero();
        Vector3dList contact_positions;
        int left_contact_count = 1;
        bool desired_foot_forces_available = false;
    };

    struct Result {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Eigen::Vector3d left_force = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_force = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, 1> target_wrench = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> achieved_wrench = Eigen::Matrix<double, 6, 1>::Zero();
        Vector3dList contact_forces;
    };

    explicit ContactForceQp(Config config);

    void reset();
    Result solve(const Input& input);

   private:
    Eigen::VectorXd project_forces(const Eigen::VectorXd& forces,
                                   int left_contact_count) const;
    Eigen::Vector3d project_contact_force(const Eigen::Vector3d& force,
                                          int contacts_on_same_foot) const;

    Config config_;
    Eigen::VectorXd last_solution_;
    bool has_last_solution_ = false;
};

}  // namespace whole_body_mpc
