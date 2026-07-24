// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace whole_body_mpc {
class WholeBodyMpcController;
}

class StandingStabilizer {
   public:
    struct Config {
        int joint_num = 0;
        float dt = 0.004f;
        std::string whole_body_model_path;
        std::string whole_body_base_link;
        std::string whole_body_left_foot_link;
        std::string whole_body_right_foot_link;
        std::vector<std::string> whole_body_joint_order;
        std::vector<double> whole_body_nominal_joint_angles;
        bool validate_whole_body_model = false;
        bool wbc_mpc_enabled = true;
        std::string wbc_mpc_backend = "ocs2";
        int wbc_mpc_horizon = 20;
        float wbc_mpc_dt = 0.012f;
        float wbc_target_roll = 0.0f;
        float wbc_target_pitch = 0.0f;
        float wbc_mpc_orientation_weight = 120.0f;
        float wbc_mpc_angular_rate_weight = 1.4f;
        float wbc_mpc_com_weight = 30.0f;
        float wbc_mpc_com_velocity_weight = 4.0f;
        float wbc_mpc_terminal_weight_scale = 4.0f;
        float wbc_mpc_input_smooth_weight = 0.02f;
        float wbc_mpc_force_weight = 0.02f;
        int wbc_mpc_qp_iterations = 40;
        bool wbc_mpc_terminal_cost_enabled = true;
        bool wbc_mpc_input_smoothing_enabled = true;
        bool wbc_mpc_contact_schedule_enabled = true;
        bool wbc_mpc_solver_constraints_enabled = true;
        bool wbc_mpc_zero_swing_force_constraint_enabled = true;
        bool wbc_mpc_normal_force_constraint_enabled = true;
        bool wbc_mpc_delta_force_constraint_enabled = true;
        bool wbc_mpc_friction_cone_constraint_enabled = true;
        float wbc_mpc_friction_barrier_mu = 0.1f;
        float wbc_mpc_friction_barrier_delta = 5.0f;
        float wbc_mpc_friction_regularization = 25.0f;
        float wbc_mpc_max_angular_accel = 12.0f;
        float wbc_mpc_max_com_accel = 2.0f;
        float wbc_mpc_max_contact_force_delta = 60.0f;
        float wbc_mpc_base_height_weight = 5.0f;
        float wbc_mpc_yaw_weight = 1.0f;
        float wbc_mpc_joint_angle_weight = 0.5f;
        float wbc_mpc_joint_velocity_weight = 0.02f;
        std::string wbc_mpc_ad_model_folder = "/tmp/roboparty_ocs2";
        bool wbc_mpc_ad_recompile = false;
        bool wbc_mpc_ad_verbose = false;
        bool wbc_state_estimation_enabled = true;
        float wbc_state_velocity_filter_alpha = 0.85f;
        float wbc_state_max_base_linear_velocity = 0.8f;
        bool wbc_contact_force_qp_enabled = true;
        bool wbc_whole_body_qp_enabled = true;
        bool wbc_floating_base_eom_enabled = true;
        bool wbc_stance_contact_constraint_enabled = true;
        bool wbc_friction_constraint_enabled = true;
        bool wbc_torque_limit_constraint_enabled = true;
        bool wbc_base_accel_task_enabled = true;
        bool wbc_contact_force_task_enabled = true;
        bool wbc_swing_task_enabled = true;
        bool wbc_qddot_regularization_enabled = true;
        bool wbc_tau_regularization_enabled = true;
        bool wbc_torque_enabled = false;
        int wbc_qp_iterations = 24;
        int wbc_active_set_iterations = 80;
        float wbc_friction_coefficient = 0.45f;
        float wbc_min_normal_force = 0.0f;
        float wbc_max_normal_force = 420.0f;
        float wbc_force_tracking_weight = 1.0f;
        float wbc_moment_tracking_weight = 0.03f;
        float wbc_regularization_weight = 1.0e-4f;
        float wbc_smooth_weight = 0.02f;
        float wbc_max_body_moment = 8.0f;
        float wbc_max_joint_torque = 0.6f;
        float wbc_foot_half_length = 0.065f;
        float wbc_foot_half_width = 0.035f;
        float wbc_foot_center_x = 0.060f;
        float wbc_foot_contact_z = -0.040f;
        bool wbc_virtual_foot_corners_enabled = true;
        bool wbc_step_recovery_enabled = false;
        bool wbc_step_placement_enabled = true;
        float wbc_step_recovery_roll_trigger = 0.24f;
        float wbc_step_recovery_pitch_trigger = 0.22f;
        float wbc_step_recovery_rate_trigger = 1.20f;
        float wbc_step_recovery_com_trigger = 0.0f;
        float wbc_step_recovery_com_velocity_trigger = 0.0f;
        float wbc_step_recovery_return_roll = 0.10f;
        float wbc_step_recovery_return_pitch = 0.12f;
        float wbc_step_recovery_return_rate = 0.35f;
        float wbc_step_recovery_return_com = 0.0f;
        float wbc_step_recovery_return_com_velocity = 0.0f;
        int wbc_step_recovery_steps = 2;
        float wbc_step_recovery_swing_time = 0.45f;
        float wbc_step_recovery_double_support_time = 0.12f;
        float wbc_step_recovery_settle_time = 0.25f;
        float wbc_step_recovery_stable_time = 0.30f;
        float wbc_step_recovery_cooldown = 0.80f;
        float wbc_step_recovery_max_duration = 2.50f;
        float wbc_step_recovery_step_x_pitch_gain = 0.28f;
        float wbc_step_recovery_step_x_rate_gain = 0.04f;
        float wbc_step_recovery_step_x_com_gain = 0.30f;
        float wbc_step_recovery_step_x_com_velocity_gain = 0.08f;
        float wbc_step_recovery_step_y_roll_gain = 0.18f;
        float wbc_step_recovery_step_y_rate_gain = 0.03f;
        float wbc_step_recovery_step_y_com_gain = 0.30f;
        float wbc_step_recovery_step_y_com_velocity_gain = 0.06f;
        float wbc_step_recovery_capture_time = 0.25f;
        float wbc_step_recovery_capture_gain = 0.35f;
        float wbc_step_recovery_min_step_x = 0.03f;
        float wbc_step_recovery_min_step_y = 0.02f;
        float wbc_step_recovery_max_step_x = 0.12f;
        float wbc_step_recovery_max_step_y = 0.07f;
        float wbc_step_recovery_swing_height = 0.035f;
        bool wbc_step_recovery_start_with_left = true;
        bool wbc_step_recovery_first_swing_left_on_positive_roll = true;
        float wbc_step_recovery_sagittal_sign = 1.0f;
        float wbc_step_recovery_lateral_sign = 1.0f;
        float wbc_swing_tracking_weight = 20.0f;
        bool wbc_swing_ik_enabled = true;
        float wbc_swing_kp = 60.0f;
        float wbc_swing_kd = 8.0f;
        float wbc_swing_ik_gain = 0.7f;
        float wbc_swing_ik_damping = 0.02f;
        float wbc_swing_max_joint_delta = 0.25f;
        float wbc_swing_max_joint_velocity = 2.0f;
        std::vector<double> wbc_torque_joint_scale;
        std::vector<double> joint_limits;
    };

    struct Measurement {
        float roll = 0.0f;
        float pitch = 0.0f;
        float wx = 0.0f;
        float wy = 0.0f;
        float gravity_z = -1.0f;
        float qw = 1.0f;
        float qx = 0.0f;
        float qy = 0.0f;
        float qz = 0.0f;
    };

    struct Correction {
        float mpc_roll_accel = 0.0f;
        float mpc_pitch_accel = 0.0f;
        float mpc_com_accel_x = 0.0f;
        float mpc_com_accel_y = 0.0f;
        float mpc_left_force_x = 0.0f;
        float mpc_left_force_y = 0.0f;
        float mpc_left_force_z = 0.0f;
        float mpc_right_force_x = 0.0f;
        float mpc_right_force_y = 0.0f;
        float mpc_right_force_z = 0.0f;
        std::string wbc_mpc_backend = "disabled";
        bool wbc_mpc_used = false;
        bool wbc_mpc_force_target_used = false;
        int wbc_mpc_iterations = 0;
        float wbc_mpc_objective = 0.0f;
        float wbc_left_normal_force = 0.0f;
        float wbc_right_normal_force = 0.0f;
        float wbc_roll_moment = 0.0f;
        float wbc_pitch_moment = 0.0f;
        float wbc_achieved_roll_moment = 0.0f;
        float wbc_achieved_pitch_moment = 0.0f;
        float wbc_max_raw_joint_torque = 0.0f;
        float wbc_max_joint_torque = 0.0f;
        float wbc_qp_violation = 0.0f;
        float wbc_dynamics_residual = 0.0f;
        float wbc_swing_error = 0.0f;
        float wbc_base_velocity_x = 0.0f;
        float wbc_base_velocity_y = 0.0f;
        float wbc_base_velocity_z = 0.0f;
        float wbc_com_velocity_x = 0.0f;
        float wbc_com_velocity_y = 0.0f;
        float wbc_state_estimator_residual = 0.0f;
        float wbc_step_x = 0.0f;
        float wbc_step_y = 0.0f;
        int wbc_contact_count = 0;
        int wbc_active_constraints = 0;
        int wbc_saturated_joint_count = 0;
        int wbc_max_torque_joint_index = -1;
        int wbc_step_phase = 0;
        int wbc_steps_completed = 0;
        int wbc_steps_planned = 0;
        int wbc_state_estimator_rows = 0;
        bool wbc_step_recovery_active = false;
        bool wbc_state_estimator_used = false;
        bool wbc_left_contact = true;
        bool wbc_right_contact = true;
        bool wbc_left_swing = false;
        bool wbc_right_swing = false;
        bool wbc_contact_force_qp_used = false;
        bool wbc_whole_body_qp_used = false;
        bool qp_used = false;
    };

    struct Command {
        std::vector<float> position;
        std::vector<float> velocity;
        std::vector<float> kp;
        std::vector<float> kd;
        std::vector<float> tau;
        Correction correction;
    };

    explicit StandingStabilizer(Config config);
    ~StandingStabilizer();

    const Config& config() const { return config_; }
    std::vector<std::string> diagnostics() const;

    void reset();
    Measurement measure(const std::vector<float>& quat, const std::vector<float>& angular_velocity) const;
    Command apply(const Measurement& measurement, float blend, const std::vector<float>& base_target,
                  const std::vector<float>& kp, const std::vector<float>& kd,
                  const std::vector<float>& current_joint_position,
                  const std::vector<float>& current_joint_velocity);

   private:
    Config config_;
    std::unique_ptr<whole_body_mpc::WholeBodyMpcController> whole_body_mpc_controller_;
};
