// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

#include "whole_body_mpc/reference/contact_schedule.hpp"

namespace whole_body_mpc {

struct CentroidalMpcConfig {
    int horizon = 20;
    double dt = 0.004;
    double control_dt = 0.004;
    bool enabled = true;
    std::string backend = "ocs2";
    bool mrt_enabled = true;
    bool mrt_first_solve_blocking = false;
    double mrt_max_policy_age = 0.20;
    double orientation_weight = 120.0;
    double angular_rate_weight = 1.4;
    double com_weight = 30.0;
    double com_velocity_weight = 4.0;
    double terminal_weight_scale = 4.0;
    double input_smooth_weight = 0.02;
    double force_weight = 0.02;
    int qp_iterations = 40;
    bool terminal_cost_enabled = true;
    bool input_smoothing_enabled = true;
    bool contact_schedule_enabled = true;
    bool solver_constraints_enabled = true;
    bool zero_swing_force_constraint_enabled = true;
    bool normal_force_constraint_enabled = true;
    bool delta_force_constraint_enabled = true;
    bool friction_cone_constraint_enabled = true;
    bool stance_zero_velocity_constraint_enabled = true;
    bool swing_normal_velocity_constraint_enabled = true;
    bool swing_position_constraint_enabled = true;
    double friction_barrier_mu = 0.1;
    double friction_barrier_delta = 5.0;
    double friction_regularization = 25.0;
    double max_angular_accel = 12.0;
    double max_com_accel = 2.0;
    double max_contact_force_delta = 60.0;
    double friction_coefficient = 0.45;
    double min_normal_force = 0.0;
    double max_normal_force = 420.0;
    double base_height_weight = 5.0;
    double yaw_weight = 1.0;
    double joint_angle_weight = 0.5;
    double joint_velocity_weight = 0.02;
    double swing_position_weight = 10.0;
    double joint_velocity_limit = 2.0;
    double swing_height = 0.035;
    double swing_time_scale = 0.15;
    double swing_lift_off_velocity = 0.0;
    double swing_touch_down_velocity = 0.0;
    double target_roll = 0.0;
    double target_pitch = 0.0;
    std::string model_path;
    std::string left_foot_frame;
    std::string right_foot_frame;
    std::vector<std::string> joint_order;
    std::vector<double> nominal_joint_angles;
    std::vector<double> joint_position_limits;
    std::string ad_model_folder = "/tmp/roboparty_ocs2";
    bool ad_recompile = false;
    bool ad_verbose = false;
};

struct CentroidalMpcInput {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double roll = 0.0;
    double pitch = 0.0;
    double wx = 0.0;
    double wy = 0.0;
    Eigen::Vector2d com_offset_error = Eigen::Vector2d::Zero();
    Eigen::Vector2d com_velocity = Eigen::Vector2d::Zero();
    Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d support_center = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d neutral_com_offset = Eigen::Vector3d::Zero();
    Eigen::Vector3d base_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d base_orientation_zyx = Eigen::Vector3d::Zero();
    Eigen::Vector3d base_linear_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d base_angular_velocity = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_position = Eigen::VectorXd::Zero(0);
    Eigen::VectorXd joint_velocity = Eigen::VectorXd::Zero(0);
    double mass = 0.0;
    double roll_inertia = 1.0;
    double pitch_inertia = 1.0;
    bool left_contact = true;
    bool right_contact = true;
    ContactSchedule contact_schedule;
};

struct CentroidalMpcOutput {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d desired_com_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_angular_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_left_contact_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_right_contact_force = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 8, 1> state = Eigen::Matrix<double, 8, 1>::Zero();
    Eigen::Matrix<double, 4, 1> control = Eigen::Matrix<double, 4, 1>::Zero();
    Eigen::Matrix<double, 6, 1> contact_force_delta =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::VectorXd desired_joint_position = Eigen::VectorXd::Zero(0);
    Eigen::VectorXd desired_joint_velocity = Eigen::VectorXd::Zero(0);
    std::string backend = "disabled";
    bool solved = false;
    bool has_desired_contact_forces = false;
    bool has_desired_joint_command = false;
    int iterations = 0;
    double objective = 0.0;
};

}  // namespace whole_body_mpc
