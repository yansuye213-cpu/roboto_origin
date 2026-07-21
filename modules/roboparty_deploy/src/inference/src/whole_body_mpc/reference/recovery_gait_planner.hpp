// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/reference/foot_placement_planner.hpp"

#include <Eigen/Core>

namespace whole_body_mpc {

class RecoveryGaitPlanner {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum class Phase {
        Idle = 0,
        DoubleSupport = 1,
        Swing = 2,
        Settle = 3,
    };

    struct Config {
        bool enabled = false;
        bool step_placement_enabled = true;
        double roll_trigger = 0.24;
        double pitch_trigger = 0.22;
        double rate_trigger = 1.20;
        double com_trigger = 0.0;
        double com_velocity_trigger = 0.0;
        double return_roll = 0.10;
        double return_pitch = 0.12;
        double return_rate = 0.35;
        double return_com = 0.0;
        double return_com_velocity = 0.0;
        int step_count = 2;
        double swing_time = 0.45;
        double double_support_time = 0.12;
        double settle_time = 0.25;
        double stable_time = 0.30;
        double cooldown = 0.80;
        double max_duration = 2.50;
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
        double swing_height = 0.035;
        bool start_with_left = true;
        bool first_swing_left_on_positive_roll = true;
        double sagittal_sign = 1.0;
        double lateral_sign = 1.0;
    };

    struct Input {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double dt = 0.004;
        double roll = 0.0;
        double pitch = 0.0;
        double wx = 0.0;
        double wy = 0.0;
        double target_roll = 0.0;
        double target_pitch = 0.0;
        Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d com_velocity = Eigen::Vector3d::Zero();
        Eigen::Vector3d support_center = Eigen::Vector3d::Zero();
    };

    struct FootReference {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        bool active = false;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    };

    struct Reference {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        bool recovery_active = false;
        bool left_contact = true;
        bool right_contact = true;
        bool left_swing = false;
        bool right_swing = false;
        bool current_swing_left = true;
        bool next_swing_left = true;
        Phase phase = Phase::Idle;
        int steps_completed = 0;
        int planned_steps = 0;
        double elapsed = 0.0;
        double phase_elapsed = 0.0;
        double stable_elapsed = 0.0;
        double roll_error = 0.0;
        double pitch_error = 0.0;
        double angular_rate = 0.0;
        double com_error = 0.0;
        double com_velocity = 0.0;
        double step_x = 0.0;
        double step_y = 0.0;
        Eigen::Vector3d left_foot_target = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_foot_target = Eigen::Vector3d::Zero();
        FootReference left_foot;
        FootReference right_foot;
    };

    explicit RecoveryGaitPlanner(Config config);

    const Config& config() const { return config_; }
    void reset();
    Reference update(const Input& input);

   private:
    bool should_start(const Input& input) const;
    bool is_stable(const Input& input) const;
    void start_sequence(const Input& input);
    void enter_double_support();
    void enter_swing(const Input& input);
    void enter_settle();
    void finish_sequence();
    bool choose_first_swing_left(const Input& input) const;
    Eigen::Vector3d compute_step_offset(const Input& input) const;
    FootReference swing_reference(bool left_swing) const;
    Reference build_reference(const Input& input) const;

    Config config_;
    FootPlacementPlanner foot_placement_planner_;
    Phase phase_ = Phase::Idle;
    bool active_ = false;
    bool current_swing_left_ = true;
    bool next_swing_left_ = true;
    int steps_completed_ = 0;
    int planned_steps_ = 0;
    double elapsed_ = 0.0;
    double phase_elapsed_ = 0.0;
    double stable_elapsed_ = 0.0;
    double cooldown_elapsed_ = 1000000.0;
    Eigen::Vector3d swing_start_position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d swing_target_position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d latest_step_offset_ = Eigen::Vector3d::Zero();
};

}  // namespace whole_body_mpc
