// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "inference_node.hpp"

void InferenceNode::load_config() {
    const std::string default_robot_dir = std::string(ROOT_DIR) + "robots/rpo";
    this->declare_parameter<std::string>("robot_name", "rpo");
    this->declare_parameter<std::string>("policy_name", "default");
    this->declare_parameter<std::string>("robot_config", default_robot_dir + "/robot.yaml");
    this->declare_parameter<std::string>("model_dir", default_robot_dir + "/models");
    this->declare_parameter<std::string>("motion_dir", default_robot_dir + "/motions");
    this->declare_parameter<std::vector<std::string>>("model_names", std::vector<std::string>{});
    this->declare_parameter<std::vector<std::string>>("motion_names", std::vector<std::string>{});
    this->declare_parameter<std::vector<std::string>>("obs_layouts", std::vector<std::string>{});
    this->declare_parameter<std::vector<std::string>>("extra_obs_layouts", std::vector<std::string>{});
    this->declare_parameter<std::vector<long int>>("frame_stacks", std::vector<long int>{});
    this->declare_parameter<std::vector<std::string>>("obs_stack_orders", std::vector<std::string>{});
    this->declare_parameter<float>("act_alpha", 0.9);
    this->declare_parameter<int>("intra_threads", -1);
    this->declare_parameter<std::string>("perception_obs_topic", "elevation_data");
    this->declare_parameter<int>("joint_num", 23);
    this->declare_parameter<int>("decimation", 10);
    this->declare_parameter<float>("dt", 0.001);
    this->declare_parameter<float>("obs_scales_lin_vel", 1.0);
    this->declare_parameter<float>("obs_scales_ang_vel", 1.0);
    this->declare_parameter<float>("obs_scales_dof_pos", 1.0);
    this->declare_parameter<float>("obs_scales_dof_vel", 1.0);
    this->declare_parameter<float>("obs_scales_gravity_b", 1.0);
    this->declare_parameter<float>("clip_observations", 100.0);
    this->declare_parameter<float>("action_scale", 0.3);
    this->declare_parameter<float>("clip_actions", 18.0);
    this->declare_parameter<std::vector<long int>>("usd2urdf", std::vector<long int>{});
    this->declare_parameter<std::vector<double>>("clip_cmd", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("joint_default_angle", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("stand_joint_angle", std::vector<double>{});
    this->declare_parameter<float>("stand_transition_time", 1.5);
    this->declare_parameter<std::string>("stand_whole_body_model_path", "");
    this->declare_parameter<std::string>("stand_whole_body_base_link", "");
    this->declare_parameter<std::string>("stand_whole_body_left_foot_link", "");
    this->declare_parameter<std::string>("stand_whole_body_right_foot_link", "");
    this->declare_parameter<std::vector<std::string>>("stand_whole_body_joint_order", std::vector<std::string>{});
    this->declare_parameter<bool>("stand_validate_whole_body_model", false);
    this->declare_parameter<bool>("stand_wbc_mpc_enabled", true);
    this->declare_parameter<std::string>("stand_wbc_mpc_backend", "ocs2");
    this->declare_parameter<bool>("stand_wbc_mpc_mrt_enabled", false);
    this->declare_parameter<bool>("stand_wbc_mpc_mrt_first_solve_blocking", true);
    this->declare_parameter<float>("stand_wbc_mpc_mrt_max_policy_age", 0.08);
    this->declare_parameter<int>("stand_wbc_mpc_horizon", 20);
    this->declare_parameter<float>("stand_wbc_mpc_dt", 0.012);
    this->declare_parameter<float>("stand_wbc_target_roll", 0.0);
    this->declare_parameter<float>("stand_wbc_target_pitch", 0.0);
    this->declare_parameter<float>("stand_wbc_mpc_orientation_weight", 120.0);
    this->declare_parameter<float>("stand_wbc_mpc_angular_rate_weight", 1.4);
    this->declare_parameter<float>("stand_wbc_mpc_com_weight", 30.0);
    this->declare_parameter<float>("stand_wbc_mpc_com_velocity_weight", 4.0);
    this->declare_parameter<float>("stand_wbc_mpc_terminal_weight_scale", 4.0);
    this->declare_parameter<float>("stand_wbc_mpc_input_smooth_weight", 0.02);
    this->declare_parameter<float>("stand_wbc_mpc_force_weight", 0.02);
    this->declare_parameter<int>("stand_wbc_mpc_qp_iterations", 40);
    this->declare_parameter<bool>("stand_wbc_mpc_terminal_cost_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_input_smoothing_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_contact_schedule_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_solver_constraints_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_zero_swing_force_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_normal_force_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_delta_force_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_friction_cone_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_stance_zero_velocity_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_swing_normal_velocity_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_mpc_swing_position_constraint_enabled", true);
    this->declare_parameter<float>("stand_wbc_mpc_friction_barrier_mu", 0.1);
    this->declare_parameter<float>("stand_wbc_mpc_friction_barrier_delta", 5.0);
    this->declare_parameter<float>("stand_wbc_mpc_friction_regularization", 25.0);
    this->declare_parameter<float>("stand_wbc_mpc_max_angular_accel", 12.0);
    this->declare_parameter<float>("stand_wbc_mpc_max_com_accel", 2.0);
    this->declare_parameter<float>("stand_wbc_mpc_max_contact_force_delta", 60.0);
    this->declare_parameter<float>("stand_wbc_mpc_base_height_weight", 5.0);
    this->declare_parameter<float>("stand_wbc_mpc_yaw_weight", 1.0);
    this->declare_parameter<float>("stand_wbc_mpc_joint_angle_weight", 0.5);
    this->declare_parameter<float>("stand_wbc_mpc_joint_velocity_weight", 0.02);
    this->declare_parameter<float>("stand_wbc_mpc_swing_position_weight", 10.0);
    this->declare_parameter<bool>("stand_wbc_mpc_joint_command_enabled", false);
    this->declare_parameter<float>("stand_wbc_mpc_joint_command_position_gain", 0.25);
    this->declare_parameter<float>("stand_wbc_mpc_joint_command_velocity_scale", 1.0);
    this->declare_parameter<float>("stand_wbc_mpc_swing_time_scale", 0.15);
    this->declare_parameter<float>("stand_wbc_mpc_swing_lift_off_velocity", 0.0);
    this->declare_parameter<float>("stand_wbc_mpc_swing_touch_down_velocity", 0.0);
    this->declare_parameter<std::string>("stand_wbc_mpc_ad_model_folder", "/tmp/roboparty_ocs2");
    this->declare_parameter<bool>("stand_wbc_mpc_ad_recompile", false);
    this->declare_parameter<bool>("stand_wbc_mpc_ad_verbose", false);
    this->declare_parameter<bool>("stand_wbc_state_estimation_enabled", true);
    this->declare_parameter<float>("stand_wbc_state_velocity_filter_alpha", 0.85);
    this->declare_parameter<float>("stand_wbc_state_max_base_linear_velocity", 0.8);
    this->declare_parameter<bool>("stand_wbc_contact_force_qp_enabled", true);
    this->declare_parameter<bool>("stand_wbc_whole_body_qp_enabled", true);
    this->declare_parameter<std::string>("stand_wbc_solver", "weighted");
    this->declare_parameter<bool>("stand_wbc_floating_base_eom_enabled", true);
    this->declare_parameter<bool>("stand_wbc_stance_contact_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_friction_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_torque_limit_constraint_enabled", true);
    this->declare_parameter<bool>("stand_wbc_base_accel_task_enabled", true);
    this->declare_parameter<bool>("stand_wbc_contact_force_task_enabled", true);
    this->declare_parameter<bool>("stand_wbc_swing_task_enabled", true);
    this->declare_parameter<bool>("stand_wbc_qddot_regularization_enabled", true);
    this->declare_parameter<bool>("stand_wbc_tau_regularization_enabled", true);
    this->declare_parameter<bool>("stand_wbc_enable_torque", false);
    this->declare_parameter<int>("stand_wbc_qp_iterations", 24);
    this->declare_parameter<int>("stand_wbc_active_set_iterations", 80);
    this->declare_parameter<float>("stand_wbc_friction_coefficient", 0.45);
    this->declare_parameter<float>("stand_wbc_min_normal_force", 0.0);
    this->declare_parameter<float>("stand_wbc_max_normal_force", 420.0);
    this->declare_parameter<float>("stand_wbc_force_tracking_weight", 1.0);
    this->declare_parameter<float>("stand_wbc_moment_tracking_weight", 0.03);
    this->declare_parameter<float>("stand_wbc_regularization_weight", 0.0001);
    this->declare_parameter<float>("stand_wbc_smooth_weight", 0.02);
    this->declare_parameter<float>("stand_wbc_max_body_moment", 8.0);
    this->declare_parameter<float>("stand_wbc_max_joint_torque", 0.6);
    this->declare_parameter<float>("stand_wbc_foot_half_length", 0.065);
    this->declare_parameter<float>("stand_wbc_foot_half_width", 0.035);
    this->declare_parameter<float>("stand_wbc_foot_center_x", 0.060);
    this->declare_parameter<float>("stand_wbc_foot_contact_z", -0.040);
    this->declare_parameter<bool>("stand_wbc_virtual_foot_corners_enabled", true);
    this->declare_parameter<bool>("stand_wbc_step_recovery_enabled", false);
    this->declare_parameter<bool>("stand_wbc_step_placement_enabled", true);
    this->declare_parameter<float>("stand_wbc_step_recovery_roll_trigger", 0.24);
    this->declare_parameter<float>("stand_wbc_step_recovery_pitch_trigger", 0.22);
    this->declare_parameter<float>("stand_wbc_step_recovery_rate_trigger", 1.20);
    this->declare_parameter<float>("stand_wbc_step_recovery_com_trigger", 0.0);
    this->declare_parameter<float>("stand_wbc_step_recovery_com_velocity_trigger", 0.0);
    this->declare_parameter<float>("stand_wbc_step_recovery_return_roll", 0.10);
    this->declare_parameter<float>("stand_wbc_step_recovery_return_pitch", 0.12);
    this->declare_parameter<float>("stand_wbc_step_recovery_return_rate", 0.35);
    this->declare_parameter<float>("stand_wbc_step_recovery_return_com", 0.0);
    this->declare_parameter<float>("stand_wbc_step_recovery_return_com_velocity", 0.0);
    this->declare_parameter<int>("stand_wbc_step_recovery_steps", 2);
    this->declare_parameter<float>("stand_wbc_step_recovery_swing_time", 0.45);
    this->declare_parameter<float>("stand_wbc_step_recovery_double_support_time", 0.12);
    this->declare_parameter<float>("stand_wbc_step_recovery_settle_time", 0.25);
    this->declare_parameter<float>("stand_wbc_step_recovery_stable_time", 0.30);
    this->declare_parameter<float>("stand_wbc_step_recovery_cooldown", 0.80);
    this->declare_parameter<float>("stand_wbc_step_recovery_max_duration", 2.50);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_x_pitch_gain", 0.28);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_x_rate_gain", 0.04);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_x_com_gain", 0.30);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_x_com_velocity_gain", 0.08);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_y_roll_gain", 0.18);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_y_rate_gain", 0.03);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_y_com_gain", 0.30);
    this->declare_parameter<float>("stand_wbc_step_recovery_step_y_com_velocity_gain", 0.06);
    this->declare_parameter<float>("stand_wbc_step_recovery_capture_time", 0.25);
    this->declare_parameter<float>("stand_wbc_step_recovery_capture_gain", 0.35);
    this->declare_parameter<float>("stand_wbc_step_recovery_min_step_x", 0.03);
    this->declare_parameter<float>("stand_wbc_step_recovery_min_step_y", 0.02);
    this->declare_parameter<float>("stand_wbc_step_recovery_max_step_x", 0.12);
    this->declare_parameter<float>("stand_wbc_step_recovery_max_step_y", 0.07);
    this->declare_parameter<float>("stand_wbc_step_recovery_swing_height", 0.035);
    this->declare_parameter<bool>("stand_wbc_step_recovery_start_with_left", true);
    this->declare_parameter<bool>("stand_wbc_step_recovery_first_swing_left_on_positive_roll", true);
    this->declare_parameter<float>("stand_wbc_step_recovery_sagittal_sign", 1.0);
    this->declare_parameter<float>("stand_wbc_step_recovery_lateral_sign", 1.0);
    this->declare_parameter<float>("stand_wbc_swing_tracking_weight", 20.0);
    this->declare_parameter<bool>("stand_wbc_swing_ik_enabled", true);
    this->declare_parameter<float>("stand_wbc_swing_kp", 60.0);
    this->declare_parameter<float>("stand_wbc_swing_kd", 8.0);
    this->declare_parameter<float>("stand_wbc_swing_ik_gain", 0.7);
    this->declare_parameter<float>("stand_wbc_swing_ik_damping", 0.02);
    this->declare_parameter<float>("stand_wbc_swing_max_joint_delta", 0.25);
    this->declare_parameter<float>("stand_wbc_swing_max_joint_velocity", 2.0);
    this->declare_parameter<std::vector<double>>("stand_wbc_torque_joint_scale", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("stand_kp", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("stand_kd", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("joint_limits", std::vector<double>{});
    this->declare_parameter<float>("gravity_z_upper", -0.5);
    std::vector<std::string> model_names;
    std::vector<std::string> motion_names;
    std::vector<std::string> obs_layouts;
    std::vector<std::string> extra_obs_layouts;
    std::vector<long int> frame_stacks;
    std::vector<std::string> obs_stack_orders;
    std::string robot_name;
    std::string policy_name;
    std::string model_dir;
    std::string motion_dir;
    this->get_parameter("robot_name", robot_name);
    this->get_parameter("policy_name", policy_name);
    this->get_parameter("robot_config", robot_config_path_);
    this->get_parameter("model_dir", model_dir);
    this->get_parameter("motion_dir", motion_dir);
    this->get_parameter("model_names", model_names);
    this->get_parameter("motion_names", motion_names);
    this->get_parameter("obs_layouts", obs_layouts);
    this->get_parameter("extra_obs_layouts", extra_obs_layouts);
    this->get_parameter("frame_stacks", frame_stacks);
    this->get_parameter("obs_stack_orders", obs_stack_orders);
    this->get_parameter("act_alpha", act_alpha_);
    this->get_parameter("intra_threads", intra_threads_);
    this->get_parameter("perception_obs_topic", perception_obs_topic_);
    this->get_parameter("joint_num", joint_num_);
    this->get_parameter("decimation", decimation_);
    this->get_parameter("dt", dt_);
    this->get_parameter("obs_scales_lin_vel", obs_scales_lin_vel_);
    this->get_parameter("obs_scales_ang_vel", obs_scales_ang_vel_);
    this->get_parameter("obs_scales_dof_pos", obs_scales_dof_pos_);
    this->get_parameter("obs_scales_dof_vel", obs_scales_dof_vel_);
    this->get_parameter("obs_scales_gravity_b", obs_scales_gravity_b_);
    this->get_parameter("clip_observations", clip_observations_);
    this->get_parameter("action_scale", action_scale_);
    this->get_parameter("clip_actions", clip_actions_);
    this->get_parameter("usd2urdf", usd2urdf_);
    this->get_parameter("clip_cmd", clip_cmd_);
    this->get_parameter("joint_default_angle", joint_default_angle_);
    this->get_parameter("stand_joint_angle", stand_joint_angle_);
    this->get_parameter("stand_transition_time", stand_transition_time_);
    this->get_parameter("stand_whole_body_model_path", stand_stabilizer_config_.whole_body_model_path);
    this->get_parameter("stand_whole_body_base_link", stand_stabilizer_config_.whole_body_base_link);
    this->get_parameter("stand_whole_body_left_foot_link", stand_stabilizer_config_.whole_body_left_foot_link);
    this->get_parameter("stand_whole_body_right_foot_link", stand_stabilizer_config_.whole_body_right_foot_link);
    this->get_parameter("stand_whole_body_joint_order", stand_stabilizer_config_.whole_body_joint_order);
    this->get_parameter("stand_validate_whole_body_model", stand_stabilizer_config_.validate_whole_body_model);
    this->get_parameter("stand_wbc_mpc_enabled", stand_stabilizer_config_.wbc_mpc_enabled);
    this->get_parameter("stand_wbc_mpc_backend", stand_stabilizer_config_.wbc_mpc_backend);
    this->get_parameter("stand_wbc_mpc_mrt_enabled", stand_stabilizer_config_.wbc_mpc_mrt_enabled);
    this->get_parameter("stand_wbc_mpc_mrt_first_solve_blocking", stand_stabilizer_config_.wbc_mpc_mrt_first_solve_blocking);
    this->get_parameter("stand_wbc_mpc_mrt_max_policy_age", stand_stabilizer_config_.wbc_mpc_mrt_max_policy_age);
    this->get_parameter("stand_wbc_mpc_horizon", stand_stabilizer_config_.wbc_mpc_horizon);
    this->get_parameter("stand_wbc_mpc_dt", stand_stabilizer_config_.wbc_mpc_dt);
    this->get_parameter("stand_wbc_target_roll", stand_stabilizer_config_.wbc_target_roll);
    this->get_parameter("stand_wbc_target_pitch", stand_stabilizer_config_.wbc_target_pitch);
    this->get_parameter("stand_wbc_mpc_orientation_weight", stand_stabilizer_config_.wbc_mpc_orientation_weight);
    this->get_parameter("stand_wbc_mpc_angular_rate_weight", stand_stabilizer_config_.wbc_mpc_angular_rate_weight);
    this->get_parameter("stand_wbc_mpc_com_weight", stand_stabilizer_config_.wbc_mpc_com_weight);
    this->get_parameter("stand_wbc_mpc_com_velocity_weight", stand_stabilizer_config_.wbc_mpc_com_velocity_weight);
    this->get_parameter("stand_wbc_mpc_terminal_weight_scale", stand_stabilizer_config_.wbc_mpc_terminal_weight_scale);
    this->get_parameter("stand_wbc_mpc_input_smooth_weight", stand_stabilizer_config_.wbc_mpc_input_smooth_weight);
    this->get_parameter("stand_wbc_mpc_force_weight", stand_stabilizer_config_.wbc_mpc_force_weight);
    this->get_parameter("stand_wbc_mpc_qp_iterations", stand_stabilizer_config_.wbc_mpc_qp_iterations);
    this->get_parameter("stand_wbc_mpc_terminal_cost_enabled", stand_stabilizer_config_.wbc_mpc_terminal_cost_enabled);
    this->get_parameter("stand_wbc_mpc_input_smoothing_enabled", stand_stabilizer_config_.wbc_mpc_input_smoothing_enabled);
    this->get_parameter("stand_wbc_mpc_contact_schedule_enabled", stand_stabilizer_config_.wbc_mpc_contact_schedule_enabled);
    this->get_parameter("stand_wbc_mpc_solver_constraints_enabled", stand_stabilizer_config_.wbc_mpc_solver_constraints_enabled);
    this->get_parameter("stand_wbc_mpc_zero_swing_force_constraint_enabled", stand_stabilizer_config_.wbc_mpc_zero_swing_force_constraint_enabled);
    this->get_parameter("stand_wbc_mpc_normal_force_constraint_enabled", stand_stabilizer_config_.wbc_mpc_normal_force_constraint_enabled);
    this->get_parameter("stand_wbc_mpc_delta_force_constraint_enabled", stand_stabilizer_config_.wbc_mpc_delta_force_constraint_enabled);
    this->get_parameter("stand_wbc_mpc_friction_cone_constraint_enabled", stand_stabilizer_config_.wbc_mpc_friction_cone_constraint_enabled);
    this->get_parameter("stand_wbc_mpc_stance_zero_velocity_constraint_enabled", stand_stabilizer_config_.wbc_mpc_stance_zero_velocity_constraint_enabled);
    this->get_parameter("stand_wbc_mpc_swing_normal_velocity_constraint_enabled", stand_stabilizer_config_.wbc_mpc_swing_normal_velocity_constraint_enabled);
    this->get_parameter("stand_wbc_mpc_swing_position_constraint_enabled", stand_stabilizer_config_.wbc_mpc_swing_position_constraint_enabled);
    this->get_parameter("stand_wbc_mpc_friction_barrier_mu", stand_stabilizer_config_.wbc_mpc_friction_barrier_mu);
    this->get_parameter("stand_wbc_mpc_friction_barrier_delta", stand_stabilizer_config_.wbc_mpc_friction_barrier_delta);
    this->get_parameter("stand_wbc_mpc_friction_regularization", stand_stabilizer_config_.wbc_mpc_friction_regularization);
    this->get_parameter("stand_wbc_mpc_max_angular_accel", stand_stabilizer_config_.wbc_mpc_max_angular_accel);
    this->get_parameter("stand_wbc_mpc_max_com_accel", stand_stabilizer_config_.wbc_mpc_max_com_accel);
    this->get_parameter("stand_wbc_mpc_max_contact_force_delta", stand_stabilizer_config_.wbc_mpc_max_contact_force_delta);
    this->get_parameter("stand_wbc_mpc_base_height_weight", stand_stabilizer_config_.wbc_mpc_base_height_weight);
    this->get_parameter("stand_wbc_mpc_yaw_weight", stand_stabilizer_config_.wbc_mpc_yaw_weight);
    this->get_parameter("stand_wbc_mpc_joint_angle_weight", stand_stabilizer_config_.wbc_mpc_joint_angle_weight);
    this->get_parameter("stand_wbc_mpc_joint_velocity_weight", stand_stabilizer_config_.wbc_mpc_joint_velocity_weight);
    this->get_parameter("stand_wbc_mpc_swing_position_weight", stand_stabilizer_config_.wbc_mpc_swing_position_weight);
    this->get_parameter("stand_wbc_mpc_joint_command_enabled", stand_stabilizer_config_.wbc_mpc_joint_command_enabled);
    this->get_parameter("stand_wbc_mpc_joint_command_position_gain", stand_stabilizer_config_.wbc_mpc_joint_command_position_gain);
    this->get_parameter("stand_wbc_mpc_joint_command_velocity_scale", stand_stabilizer_config_.wbc_mpc_joint_command_velocity_scale);
    this->get_parameter("stand_wbc_mpc_swing_time_scale", stand_stabilizer_config_.wbc_mpc_swing_time_scale);
    this->get_parameter("stand_wbc_mpc_swing_lift_off_velocity", stand_stabilizer_config_.wbc_mpc_swing_lift_off_velocity);
    this->get_parameter("stand_wbc_mpc_swing_touch_down_velocity", stand_stabilizer_config_.wbc_mpc_swing_touch_down_velocity);
    this->get_parameter("stand_wbc_mpc_ad_model_folder", stand_stabilizer_config_.wbc_mpc_ad_model_folder);
    this->get_parameter("stand_wbc_mpc_ad_recompile", stand_stabilizer_config_.wbc_mpc_ad_recompile);
    this->get_parameter("stand_wbc_mpc_ad_verbose", stand_stabilizer_config_.wbc_mpc_ad_verbose);
    this->get_parameter("stand_wbc_state_estimation_enabled", stand_stabilizer_config_.wbc_state_estimation_enabled);
    this->get_parameter("stand_wbc_state_velocity_filter_alpha", stand_stabilizer_config_.wbc_state_velocity_filter_alpha);
    this->get_parameter("stand_wbc_state_max_base_linear_velocity", stand_stabilizer_config_.wbc_state_max_base_linear_velocity);
    this->get_parameter("stand_wbc_contact_force_qp_enabled", stand_stabilizer_config_.wbc_contact_force_qp_enabled);
    this->get_parameter("stand_wbc_whole_body_qp_enabled", stand_stabilizer_config_.wbc_whole_body_qp_enabled);
    this->get_parameter("stand_wbc_solver", stand_stabilizer_config_.wbc_solver);
    this->get_parameter("stand_wbc_floating_base_eom_enabled", stand_stabilizer_config_.wbc_floating_base_eom_enabled);
    this->get_parameter("stand_wbc_stance_contact_constraint_enabled", stand_stabilizer_config_.wbc_stance_contact_constraint_enabled);
    this->get_parameter("stand_wbc_friction_constraint_enabled", stand_stabilizer_config_.wbc_friction_constraint_enabled);
    this->get_parameter("stand_wbc_torque_limit_constraint_enabled", stand_stabilizer_config_.wbc_torque_limit_constraint_enabled);
    this->get_parameter("stand_wbc_base_accel_task_enabled", stand_stabilizer_config_.wbc_base_accel_task_enabled);
    this->get_parameter("stand_wbc_contact_force_task_enabled", stand_stabilizer_config_.wbc_contact_force_task_enabled);
    this->get_parameter("stand_wbc_swing_task_enabled", stand_stabilizer_config_.wbc_swing_task_enabled);
    this->get_parameter("stand_wbc_qddot_regularization_enabled", stand_stabilizer_config_.wbc_qddot_regularization_enabled);
    this->get_parameter("stand_wbc_tau_regularization_enabled", stand_stabilizer_config_.wbc_tau_regularization_enabled);
    this->get_parameter("stand_wbc_enable_torque", stand_stabilizer_config_.wbc_torque_enabled);
    this->get_parameter("stand_wbc_qp_iterations", stand_stabilizer_config_.wbc_qp_iterations);
    this->get_parameter("stand_wbc_active_set_iterations", stand_stabilizer_config_.wbc_active_set_iterations);
    this->get_parameter("stand_wbc_friction_coefficient", stand_stabilizer_config_.wbc_friction_coefficient);
    this->get_parameter("stand_wbc_min_normal_force", stand_stabilizer_config_.wbc_min_normal_force);
    this->get_parameter("stand_wbc_max_normal_force", stand_stabilizer_config_.wbc_max_normal_force);
    this->get_parameter("stand_wbc_force_tracking_weight", stand_stabilizer_config_.wbc_force_tracking_weight);
    this->get_parameter("stand_wbc_moment_tracking_weight", stand_stabilizer_config_.wbc_moment_tracking_weight);
    this->get_parameter("stand_wbc_regularization_weight", stand_stabilizer_config_.wbc_regularization_weight);
    this->get_parameter("stand_wbc_smooth_weight", stand_stabilizer_config_.wbc_smooth_weight);
    this->get_parameter("stand_wbc_max_body_moment", stand_stabilizer_config_.wbc_max_body_moment);
    this->get_parameter("stand_wbc_max_joint_torque", stand_stabilizer_config_.wbc_max_joint_torque);
    this->get_parameter("stand_wbc_foot_half_length", stand_stabilizer_config_.wbc_foot_half_length);
    this->get_parameter("stand_wbc_foot_half_width", stand_stabilizer_config_.wbc_foot_half_width);
    this->get_parameter("stand_wbc_foot_center_x", stand_stabilizer_config_.wbc_foot_center_x);
    this->get_parameter("stand_wbc_foot_contact_z", stand_stabilizer_config_.wbc_foot_contact_z);
    this->get_parameter("stand_wbc_virtual_foot_corners_enabled", stand_stabilizer_config_.wbc_virtual_foot_corners_enabled);
    this->get_parameter("stand_wbc_step_recovery_enabled", stand_stabilizer_config_.wbc_step_recovery_enabled);
    this->get_parameter("stand_wbc_step_placement_enabled", stand_stabilizer_config_.wbc_step_placement_enabled);
    this->get_parameter("stand_wbc_step_recovery_roll_trigger", stand_stabilizer_config_.wbc_step_recovery_roll_trigger);
    this->get_parameter("stand_wbc_step_recovery_pitch_trigger", stand_stabilizer_config_.wbc_step_recovery_pitch_trigger);
    this->get_parameter("stand_wbc_step_recovery_rate_trigger", stand_stabilizer_config_.wbc_step_recovery_rate_trigger);
    this->get_parameter("stand_wbc_step_recovery_com_trigger", stand_stabilizer_config_.wbc_step_recovery_com_trigger);
    this->get_parameter("stand_wbc_step_recovery_com_velocity_trigger", stand_stabilizer_config_.wbc_step_recovery_com_velocity_trigger);
    this->get_parameter("stand_wbc_step_recovery_return_roll", stand_stabilizer_config_.wbc_step_recovery_return_roll);
    this->get_parameter("stand_wbc_step_recovery_return_pitch", stand_stabilizer_config_.wbc_step_recovery_return_pitch);
    this->get_parameter("stand_wbc_step_recovery_return_rate", stand_stabilizer_config_.wbc_step_recovery_return_rate);
    this->get_parameter("stand_wbc_step_recovery_return_com", stand_stabilizer_config_.wbc_step_recovery_return_com);
    this->get_parameter("stand_wbc_step_recovery_return_com_velocity", stand_stabilizer_config_.wbc_step_recovery_return_com_velocity);
    this->get_parameter("stand_wbc_step_recovery_steps", stand_stabilizer_config_.wbc_step_recovery_steps);
    this->get_parameter("stand_wbc_step_recovery_swing_time", stand_stabilizer_config_.wbc_step_recovery_swing_time);
    this->get_parameter("stand_wbc_step_recovery_double_support_time", stand_stabilizer_config_.wbc_step_recovery_double_support_time);
    this->get_parameter("stand_wbc_step_recovery_settle_time", stand_stabilizer_config_.wbc_step_recovery_settle_time);
    this->get_parameter("stand_wbc_step_recovery_stable_time", stand_stabilizer_config_.wbc_step_recovery_stable_time);
    this->get_parameter("stand_wbc_step_recovery_cooldown", stand_stabilizer_config_.wbc_step_recovery_cooldown);
    this->get_parameter("stand_wbc_step_recovery_max_duration", stand_stabilizer_config_.wbc_step_recovery_max_duration);
    this->get_parameter("stand_wbc_step_recovery_step_x_pitch_gain", stand_stabilizer_config_.wbc_step_recovery_step_x_pitch_gain);
    this->get_parameter("stand_wbc_step_recovery_step_x_rate_gain", stand_stabilizer_config_.wbc_step_recovery_step_x_rate_gain);
    this->get_parameter("stand_wbc_step_recovery_step_x_com_gain", stand_stabilizer_config_.wbc_step_recovery_step_x_com_gain);
    this->get_parameter("stand_wbc_step_recovery_step_x_com_velocity_gain", stand_stabilizer_config_.wbc_step_recovery_step_x_com_velocity_gain);
    this->get_parameter("stand_wbc_step_recovery_step_y_roll_gain", stand_stabilizer_config_.wbc_step_recovery_step_y_roll_gain);
    this->get_parameter("stand_wbc_step_recovery_step_y_rate_gain", stand_stabilizer_config_.wbc_step_recovery_step_y_rate_gain);
    this->get_parameter("stand_wbc_step_recovery_step_y_com_gain", stand_stabilizer_config_.wbc_step_recovery_step_y_com_gain);
    this->get_parameter("stand_wbc_step_recovery_step_y_com_velocity_gain", stand_stabilizer_config_.wbc_step_recovery_step_y_com_velocity_gain);
    this->get_parameter("stand_wbc_step_recovery_capture_time", stand_stabilizer_config_.wbc_step_recovery_capture_time);
    this->get_parameter("stand_wbc_step_recovery_capture_gain", stand_stabilizer_config_.wbc_step_recovery_capture_gain);
    this->get_parameter("stand_wbc_step_recovery_min_step_x", stand_stabilizer_config_.wbc_step_recovery_min_step_x);
    this->get_parameter("stand_wbc_step_recovery_min_step_y", stand_stabilizer_config_.wbc_step_recovery_min_step_y);
    this->get_parameter("stand_wbc_step_recovery_max_step_x", stand_stabilizer_config_.wbc_step_recovery_max_step_x);
    this->get_parameter("stand_wbc_step_recovery_max_step_y", stand_stabilizer_config_.wbc_step_recovery_max_step_y);
    this->get_parameter("stand_wbc_step_recovery_swing_height", stand_stabilizer_config_.wbc_step_recovery_swing_height);
    this->get_parameter("stand_wbc_step_recovery_start_with_left", stand_stabilizer_config_.wbc_step_recovery_start_with_left);
    this->get_parameter("stand_wbc_step_recovery_first_swing_left_on_positive_roll", stand_stabilizer_config_.wbc_step_recovery_first_swing_left_on_positive_roll);
    this->get_parameter("stand_wbc_step_recovery_sagittal_sign", stand_stabilizer_config_.wbc_step_recovery_sagittal_sign);
    this->get_parameter("stand_wbc_step_recovery_lateral_sign", stand_stabilizer_config_.wbc_step_recovery_lateral_sign);
    this->get_parameter("stand_wbc_swing_tracking_weight", stand_stabilizer_config_.wbc_swing_tracking_weight);
    this->get_parameter("stand_wbc_swing_ik_enabled", stand_stabilizer_config_.wbc_swing_ik_enabled);
    this->get_parameter("stand_wbc_swing_kp", stand_stabilizer_config_.wbc_swing_kp);
    this->get_parameter("stand_wbc_swing_kd", stand_stabilizer_config_.wbc_swing_kd);
    this->get_parameter("stand_wbc_swing_ik_gain", stand_stabilizer_config_.wbc_swing_ik_gain);
    this->get_parameter("stand_wbc_swing_ik_damping", stand_stabilizer_config_.wbc_swing_ik_damping);
    this->get_parameter("stand_wbc_swing_max_joint_delta", stand_stabilizer_config_.wbc_swing_max_joint_delta);
    this->get_parameter("stand_wbc_swing_max_joint_velocity", stand_stabilizer_config_.wbc_swing_max_joint_velocity);
    this->get_parameter("stand_wbc_torque_joint_scale", stand_stabilizer_config_.wbc_torque_joint_scale);
    if (!stand_stabilizer_config_.whole_body_model_path.empty()) {
        const std::filesystem::path model_path(stand_stabilizer_config_.whole_body_model_path);
        if (model_path.is_relative()) {
            stand_stabilizer_config_.whole_body_model_path =
                (std::filesystem::path(ROOT_DIR) / model_path).lexically_normal().string();
        }
    }
    std::vector<double> stand_kp_config;
    std::vector<double> stand_kd_config;
    this->get_parameter("stand_kp", stand_kp_config);
    this->get_parameter("stand_kd", stand_kd_config);
    this->get_parameter("joint_limits", joint_limits_);
    this->get_parameter("gravity_z_upper", gravity_z_upper_);

    if (joint_num_ <= 0) {
        throw std::runtime_error("joint_num must be positive");
    }
    if (clip_cmd_.size() != 6) {
        throw std::runtime_error("clip_cmd must contain [min_vx, max_vx, min_vy, max_vy, min_wz, max_wz]");
    }
    if (usd2urdf_.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("usd2urdf must have the same size as joint_num");
    }
    if (joint_default_angle_.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("joint_default_angle must have the same size as joint_num");
    }
    if (stand_joint_angle_.empty()) {
        stand_joint_angle_ = joint_default_angle_;
    }
    if (stand_joint_angle_.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("stand_joint_angle must be empty or have the same size as joint_num");
    }
    if (stand_transition_time_ <= 0.0f) {
        throw std::runtime_error("stand_transition_time must be positive");
    }
    stand_stabilizer_config_.joint_num = joint_num_;
    stand_stabilizer_config_.dt = dt_;
    const auto is_supported_wbc_mpc_backend = [](const std::string& backend) {
        return backend == "disabled" || backend == "ocs2";
    };
    if (stand_stabilizer_config_.wbc_mpc_horizon <= 0 ||
        stand_stabilizer_config_.wbc_mpc_dt <= 0.0f ||
        !is_supported_wbc_mpc_backend(stand_stabilizer_config_.wbc_mpc_backend) ||
        stand_stabilizer_config_.wbc_mpc_mrt_max_policy_age < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_orientation_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_angular_rate_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_com_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_com_velocity_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_terminal_weight_scale < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_input_smooth_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_force_weight <= 0.0f ||
        stand_stabilizer_config_.wbc_mpc_qp_iterations < 0 ||
        stand_stabilizer_config_.wbc_mpc_friction_barrier_mu < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_friction_barrier_delta <= 0.0f ||
        stand_stabilizer_config_.wbc_mpc_friction_regularization <= 0.0f ||
        stand_stabilizer_config_.wbc_mpc_max_angular_accel < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_max_com_accel < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_max_contact_force_delta < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_base_height_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_yaw_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_joint_angle_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_joint_velocity_weight < 0.0f ||
        stand_stabilizer_config_.wbc_mpc_ad_model_folder.empty()) {
        throw std::runtime_error("stand WBC MPC parameters are invalid");
    }
    if (stand_stabilizer_config_.wbc_state_velocity_filter_alpha < 0.0f ||
        stand_stabilizer_config_.wbc_state_velocity_filter_alpha > 1.0f ||
        stand_stabilizer_config_.wbc_state_max_base_linear_velocity < 0.0f) {
        throw std::runtime_error("stand WBC state estimation parameters are invalid");
    }
    if (stand_stabilizer_config_.wbc_qp_iterations <= 0) {
        throw std::runtime_error("stand_wbc_qp_iterations must be positive");
    }
    if (stand_stabilizer_config_.wbc_solver != "weighted" &&
        stand_stabilizer_config_.wbc_solver != "hierarchical") {
        throw std::runtime_error("stand_wbc_solver must be weighted or hierarchical");
    }
    if (stand_stabilizer_config_.wbc_active_set_iterations <= 0) {
        throw std::runtime_error("stand_wbc_active_set_iterations must be positive");
    }
    if (stand_stabilizer_config_.wbc_friction_coefficient < 0.0f ||
        stand_stabilizer_config_.wbc_min_normal_force < 0.0f ||
        stand_stabilizer_config_.wbc_max_normal_force < stand_stabilizer_config_.wbc_min_normal_force ||
        stand_stabilizer_config_.wbc_force_tracking_weight < 0.0f ||
        stand_stabilizer_config_.wbc_moment_tracking_weight < 0.0f ||
        stand_stabilizer_config_.wbc_regularization_weight < 0.0f ||
        stand_stabilizer_config_.wbc_smooth_weight < 0.0f ||
        stand_stabilizer_config_.wbc_max_body_moment < 0.0f ||
        stand_stabilizer_config_.wbc_max_joint_torque < 0.0f ||
        stand_stabilizer_config_.wbc_foot_half_length < 0.0f ||
        stand_stabilizer_config_.wbc_foot_half_width < 0.0f) {
        throw std::runtime_error("stand WBC parameters are invalid");
    }
    if (stand_stabilizer_config_.wbc_step_recovery_roll_trigger < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_pitch_trigger < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_rate_trigger < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_com_trigger < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_com_velocity_trigger < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_return_roll < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_return_pitch < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_return_rate < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_return_com < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_return_com_velocity < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_steps < 0 ||
        stand_stabilizer_config_.wbc_step_recovery_swing_time <= 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_double_support_time < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_settle_time < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_stable_time < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_cooldown < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_max_duration < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_min_step_x < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_min_step_y < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_max_step_x < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_max_step_y < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_min_step_x >
            stand_stabilizer_config_.wbc_step_recovery_max_step_x ||
        stand_stabilizer_config_.wbc_step_recovery_min_step_y >
            stand_stabilizer_config_.wbc_step_recovery_max_step_y ||
        stand_stabilizer_config_.wbc_step_recovery_capture_time < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_capture_gain < 0.0f ||
        stand_stabilizer_config_.wbc_step_recovery_swing_height < 0.0f ||
        stand_stabilizer_config_.wbc_swing_tracking_weight < 0.0f ||
        stand_stabilizer_config_.wbc_swing_kp < 0.0f ||
        stand_stabilizer_config_.wbc_swing_kd < 0.0f ||
        stand_stabilizer_config_.wbc_swing_ik_gain < 0.0f ||
        stand_stabilizer_config_.wbc_swing_ik_damping < 0.0f ||
        stand_stabilizer_config_.wbc_swing_max_joint_delta < 0.0f ||
        stand_stabilizer_config_.wbc_swing_max_joint_velocity < 0.0f) {
        throw std::runtime_error("stand WBC step recovery parameters are invalid");
    }
    if (stand_stabilizer_config_.wbc_torque_joint_scale.empty()) {
        stand_stabilizer_config_.wbc_torque_joint_scale.assign(joint_num_, 0.0);
        if (joint_num_ > 11) {
            stand_stabilizer_config_.wbc_torque_joint_scale[0] = 1.0;
            stand_stabilizer_config_.wbc_torque_joint_scale[1] = 1.0;
            stand_stabilizer_config_.wbc_torque_joint_scale[2] = 0.3;
            stand_stabilizer_config_.wbc_torque_joint_scale[3] = 1.0;
            stand_stabilizer_config_.wbc_torque_joint_scale[6] = 1.0;
            stand_stabilizer_config_.wbc_torque_joint_scale[7] = 1.0;
            stand_stabilizer_config_.wbc_torque_joint_scale[8] = 0.3;
            stand_stabilizer_config_.wbc_torque_joint_scale[9] = 1.0;
        }
    }
    if (stand_stabilizer_config_.wbc_torque_joint_scale.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("stand_wbc_torque_joint_scale must be empty or have the same size as joint_num");
    }
    stand_stabilizer_config_.whole_body_nominal_joint_angles = stand_joint_angle_;
    if (!stand_kp_config.empty() && stand_kp_config.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("stand_kp must be empty or have the same size as joint_num");
    }
    if (!stand_kd_config.empty() && stand_kd_config.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("stand_kd must be empty or have the same size as joint_num");
    }
    stand_kp_.assign(stand_kp_config.begin(), stand_kp_config.end());
    stand_kd_.assign(stand_kd_config.begin(), stand_kd_config.end());
    if (!joint_limits_.empty() && joint_limits_.size() != static_cast<size_t>(joint_num_ * 2)) {
        throw std::runtime_error("joint_limits must be empty or contain 2 values per joint");
    }
    stand_stabilizer_config_.joint_limits = joint_limits_;
    stand_stabilizer_ = std::make_unique<StandingStabilizer>(stand_stabilizer_config_);
    for (const std::string& diagnostic_line : stand_stabilizer_->diagnostics()) {
        RCLCPP_INFO(this->get_logger(), "%s", diagnostic_line.c_str());
    }
    for (size_t i = 0; i < usd2urdf_.size(); i++) {
        if (usd2urdf_[i] < 0 || usd2urdf_[i] >= joint_num_) {
            throw std::runtime_error("usd2urdf[" + std::to_string(i) + "] is out of joint range");
        }
        if (!joint_limits_.empty()) {
            const double lower = joint_limits_[i * 2];
            const double upper = joint_limits_[i * 2 + 1];
            if (stand_joint_angle_[i] < lower || stand_joint_angle_[i] > upper) {
                throw std::runtime_error("stand_joint_angle[" + std::to_string(i) + "] is out of joint range");
            }
        }
    }

    policies_.clear();
    motion_policy_indices_.clear();
    perception_obs_num_ = 0;
    const size_t policy_count = model_names.size();
    if (policy_count == 0) {
        throw std::runtime_error("model_names must contain at least one policy");
    }
    const auto require_policy_count = [policy_count](const auto& values, const std::string& name) {
        if (values.size() != policy_count) {
            throw std::runtime_error(name + " must have the same size as model_names");
        }
    };
    const auto require_empty_or_policy_count = [policy_count](const auto& values, const std::string& name) {
        if (!values.empty() && values.size() != policy_count) {
            throw std::runtime_error(name + " must be empty or have the same size as model_names");
        }
    };
    require_policy_count(obs_layouts, "obs_layouts");
    require_empty_or_policy_count(extra_obs_layouts, "extra_obs_layouts");
    require_policy_count(frame_stacks, "frame_stacks");
    require_policy_count(obs_stack_orders, "obs_stack_orders");
    require_empty_or_policy_count(motion_names, "motion_names");

    const auto resolve_asset_path = [](const std::string& base_dir, const std::string& asset_name) {
        const std::filesystem::path asset_path(asset_name);
        if (asset_path.is_absolute()) {
            return asset_path.string();
        }
        return (std::filesystem::path(base_dir) / asset_path).lexically_normal().string();
    };

    for (size_t i = 0; i < policy_count; i++) {
        const std::string& policy_model_name = model_names[i];
        const std::string policy_motion_name = motion_names.empty() ? "" : motion_names[i];
        const std::string policy_extra_obs_layout = extra_obs_layouts.empty() ? "" : extra_obs_layouts[i];
        const int policy_frame_stack = static_cast<int>(frame_stacks[i]);
        const std::string& policy_obs_stack_order_name = obs_stack_orders[i];
        if (policy_model_name.empty()) {
            throw std::runtime_error("model_names[" + std::to_string(i) + "] must not be empty");
        }
        if (policy_frame_stack <= 0) {
            throw std::runtime_error("frame_stacks[" + std::to_string(i) + "] must be positive");
        }

        PolicyRuntime policy;
        policy.name = policy_model_name;
        policy.model_path = resolve_asset_path(model_dir, policy_model_name);
        if (!policy_motion_name.empty()) {
            policy.motion_path = resolve_asset_path(motion_dir, policy_motion_name);
        }
        policy.obs_layout = parse_obs_layout(obs_layouts[i], "obs_layouts[" + std::to_string(i) + "]");
        policy.obs_layout_sizes.reserve(policy.obs_layout.size());
        for (const ObsSourceSpec& source : policy.obs_layout) {
            if ((source.name == "dof_pos" || source.name == "dof_vel" || source.name == "last_action") &&
                source.size < joint_num_) {
                throw std::runtime_error(source.name + " in obs_layouts[" + std::to_string(i) +
                                         "] must be at least joint_num");
            }
            policy.obs_layout_sizes.push_back(source.size);
            policy.obs_num += source.size;
            if (source.name == "perception") {
                perception_obs_num_ = source.size;
            }
        }
        if (!policy_extra_obs_layout.empty()) {
            policy.extra_obs_layout = parse_obs_layout(policy_extra_obs_layout, "extra_obs_layouts[" + std::to_string(i) + "]");
            for (const ObsSourceSpec& source : policy.extra_obs_layout) {
                policy.extra_obs_num += source.size;
                if (source.name == "perception") {
                    perception_obs_num_ = source.size;
                }
            }
        }
        policy.frame_stack = policy_frame_stack;
        policy.stack_order = parse_obs_stack_order(policy_obs_stack_order_name);
        if (!policy.motion_path.empty()) {
            motion_policy_indices_.push_back(static_cast<int>(policies_.size()));
        }
        policies_.push_back(std::move(policy));
    }
    RCLCPP_INFO(this->get_logger(), "robot_name: %s", robot_name.c_str());
    RCLCPP_INFO(this->get_logger(), "policy_name: %s", policy_name.c_str());
    RCLCPP_INFO(this->get_logger(), "robot_config: %s", robot_config_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "model_dir: %s", model_dir.c_str());
    RCLCPP_INFO(this->get_logger(), "motion_dir: %s", motion_dir.c_str());
    for(size_t i = 0; i < policies_.size(); i++) {
        RCLCPP_INFO(this->get_logger(), "policy %zu: %s", i, policies_[i].name.c_str());
        RCLCPP_INFO(this->get_logger(), "policy_model_path %zu: %s", i, policies_[i].model_path.c_str());
        if (!policies_[i].motion_path.empty()) {
            RCLCPP_INFO(this->get_logger(), "policy_motion_path %zu: %s", i, policies_[i].motion_path.c_str());
        }
    }
    RCLCPP_INFO(this->get_logger(), "act_alpha: %f", act_alpha_);
    RCLCPP_INFO(this->get_logger(), "intra_threads: %d", intra_threads_);
    RCLCPP_INFO(this->get_logger(), "supports_interrupt: %s", has_obs_source("interrupt") ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "has_motion_policy: %s", motion_policy_indices_.empty() ? "false" : "true");
    RCLCPP_INFO(this->get_logger(), "perception_obs_num: %d", perception_obs_num_);
    print_vector<std::string>("extra_obs_layouts", extra_obs_layouts);
    RCLCPP_INFO(this->get_logger(), "perception_obs_topic: %s", perception_obs_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "joint_num: %d", joint_num_);
    RCLCPP_INFO(this->get_logger(), "decimation: %d", decimation_);
    RCLCPP_INFO(this->get_logger(), "dt: %f", dt_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_lin_vel: %f", obs_scales_lin_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_ang_vel: %f", obs_scales_ang_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_dof_pos: %f", obs_scales_dof_pos_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_dof_vel: %f", obs_scales_dof_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_gravity_b: %f", obs_scales_gravity_b_);
    RCLCPP_INFO(this->get_logger(), "action_scale: %f", action_scale_);
    RCLCPP_INFO(this->get_logger(), "clip_actions: %f", clip_actions_);
    print_vector<long int>("usd2urdf", usd2urdf_);
    print_vector<double>("clip_cmd", clip_cmd_);
    print_vector<double>("joint_default_angle", joint_default_angle_);
    print_vector<double>("stand_joint_angle", stand_joint_angle_);
    RCLCPP_INFO(this->get_logger(), "stand_transition_time: %f", stand_transition_time_);
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_model_path: %s", stand_stabilizer_config_.whole_body_model_path.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_base_link: %s", stand_stabilizer_config_.whole_body_base_link.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_left_foot_link: %s", stand_stabilizer_config_.whole_body_left_foot_link.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_right_foot_link: %s", stand_stabilizer_config_.whole_body_right_foot_link.c_str());
    print_vector<std::string>("stand_whole_body_joint_order", stand_stabilizer_config_.whole_body_joint_order);
    RCLCPP_INFO(this->get_logger(), "stand_validate_whole_body_model: %s", stand_stabilizer_config_.validate_whole_body_model ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_enabled: %s", stand_stabilizer_config_.wbc_mpc_enabled ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_backend: %s", stand_stabilizer_config_.wbc_mpc_backend.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_horizon: %d", stand_stabilizer_config_.wbc_mpc_horizon);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_dt: %f", stand_stabilizer_config_.wbc_mpc_dt);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_target_roll: %f", stand_stabilizer_config_.wbc_target_roll);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_target_pitch: %f", stand_stabilizer_config_.wbc_target_pitch);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_orientation_weight: %f", stand_stabilizer_config_.wbc_mpc_orientation_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_angular_rate_weight: %f", stand_stabilizer_config_.wbc_mpc_angular_rate_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_com_weight: %f", stand_stabilizer_config_.wbc_mpc_com_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_com_velocity_weight: %f", stand_stabilizer_config_.wbc_mpc_com_velocity_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_terminal_weight_scale: %f", stand_stabilizer_config_.wbc_mpc_terminal_weight_scale);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_input_smooth_weight: %f", stand_stabilizer_config_.wbc_mpc_input_smooth_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_force_weight: %f", stand_stabilizer_config_.wbc_mpc_force_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_qp_iterations: %d", stand_stabilizer_config_.wbc_mpc_qp_iterations);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_mpc_cost_switches: terminal=%s input_smoothing=%s contact_schedule=%s",
                stand_stabilizer_config_.wbc_mpc_terminal_cost_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_input_smoothing_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_contact_schedule_enabled ? "true" : "false");
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_mpc_solver_constraints: enabled=%s zero_swing_force=%s normal_force=%s delta_force=%s friction_cone=%s stance_zero_velocity=%s swing_normal_velocity=%s swing_position=%s barrier=[%f, %f] friction_regularization=%f",
                stand_stabilizer_config_.wbc_mpc_solver_constraints_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_zero_swing_force_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_normal_force_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_delta_force_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_friction_cone_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_stance_zero_velocity_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_swing_normal_velocity_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_swing_position_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_friction_barrier_mu,
                stand_stabilizer_config_.wbc_mpc_friction_barrier_delta,
                stand_stabilizer_config_.wbc_mpc_friction_regularization);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_max_angular_accel: %f", stand_stabilizer_config_.wbc_mpc_max_angular_accel);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_max_com_accel: %f", stand_stabilizer_config_.wbc_mpc_max_com_accel);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_mpc_max_contact_force_delta: %f", stand_stabilizer_config_.wbc_mpc_max_contact_force_delta);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_mpc_full_centroidal_weights: base_height=%f yaw=%f joint_angle=%f joint_velocity=%f swing_position=%f swing_time_scale=%f joint_cmd=%s pos_gain=%f vel_scale=%f",
                stand_stabilizer_config_.wbc_mpc_base_height_weight,
                stand_stabilizer_config_.wbc_mpc_yaw_weight,
                stand_stabilizer_config_.wbc_mpc_joint_angle_weight,
                stand_stabilizer_config_.wbc_mpc_joint_velocity_weight,
                stand_stabilizer_config_.wbc_mpc_swing_position_weight,
                stand_stabilizer_config_.wbc_mpc_swing_time_scale,
                stand_stabilizer_config_.wbc_mpc_joint_command_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_joint_command_position_gain,
                stand_stabilizer_config_.wbc_mpc_joint_command_velocity_scale);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_mpc_ad: folder=%s recompile=%s verbose=%s",
                stand_stabilizer_config_.wbc_mpc_ad_model_folder.c_str(),
                stand_stabilizer_config_.wbc_mpc_ad_recompile ? "true" : "false",
                stand_stabilizer_config_.wbc_mpc_ad_verbose ? "true" : "false");
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_state_estimation: enabled=%s velocity_filter_alpha=%f max_base_linear_velocity=%f",
                stand_stabilizer_config_.wbc_state_estimation_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_state_velocity_filter_alpha,
                stand_stabilizer_config_.wbc_state_max_base_linear_velocity);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_algorithm_switches: contact_force_qp=%s whole_body_qp=%s swing_ik=%s virtual_foot_corners=%s step_placement=%s",
                stand_stabilizer_config_.wbc_contact_force_qp_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_whole_body_qp_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_swing_ik_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_virtual_foot_corners_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_step_placement_enabled ? "true" : "false");
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_constraint_switches: eom=%s stance_contact=%s friction=%s torque_limit=%s",
                stand_stabilizer_config_.wbc_floating_base_eom_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_stance_contact_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_friction_constraint_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_torque_limit_constraint_enabled ? "true" : "false");
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_task_switches: base_accel=%s contact_force=%s swing=%s qddot_reg=%s tau_reg=%s",
                stand_stabilizer_config_.wbc_base_accel_task_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_contact_force_task_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_swing_task_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_qddot_regularization_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_tau_regularization_enabled ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "stand_wbc_enable_torque: %s", stand_stabilizer_config_.wbc_torque_enabled ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "stand_wbc_qp_iterations: %d", stand_stabilizer_config_.wbc_qp_iterations);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_active_set_iterations: %d", stand_stabilizer_config_.wbc_active_set_iterations);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_friction_coefficient: %f", stand_stabilizer_config_.wbc_friction_coefficient);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_normal_force_limit: [%f, %f]",
                stand_stabilizer_config_.wbc_min_normal_force,
                stand_stabilizer_config_.wbc_max_normal_force);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_force_tracking_weight: %f", stand_stabilizer_config_.wbc_force_tracking_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_moment_tracking_weight: %f", stand_stabilizer_config_.wbc_moment_tracking_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_regularization_weight: %f", stand_stabilizer_config_.wbc_regularization_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_smooth_weight: %f", stand_stabilizer_config_.wbc_smooth_weight);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_max_body_moment: %f", stand_stabilizer_config_.wbc_max_body_moment);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_max_joint_torque: %f", stand_stabilizer_config_.wbc_max_joint_torque);
    RCLCPP_INFO(this->get_logger(), "stand_wbc_foot_geometry: half_length=%f half_width=%f center_x=%f contact_z=%f",
                stand_stabilizer_config_.wbc_foot_half_length,
                stand_stabilizer_config_.wbc_foot_half_width,
                stand_stabilizer_config_.wbc_foot_center_x,
                stand_stabilizer_config_.wbc_foot_contact_z);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_step_recovery: enabled=%s trigger_rp_rate_com=[%f, %f, %f, %f, %f] return_rp_rate_com=[%f, %f, %f, %f, %f]",
                stand_stabilizer_config_.wbc_step_recovery_enabled ? "true" : "false",
                stand_stabilizer_config_.wbc_step_recovery_roll_trigger,
                stand_stabilizer_config_.wbc_step_recovery_pitch_trigger,
                stand_stabilizer_config_.wbc_step_recovery_rate_trigger,
                stand_stabilizer_config_.wbc_step_recovery_com_trigger,
                stand_stabilizer_config_.wbc_step_recovery_com_velocity_trigger,
                stand_stabilizer_config_.wbc_step_recovery_return_roll,
                stand_stabilizer_config_.wbc_step_recovery_return_pitch,
                stand_stabilizer_config_.wbc_step_recovery_return_rate,
                stand_stabilizer_config_.wbc_step_recovery_return_com,
                stand_stabilizer_config_.wbc_step_recovery_return_com_velocity);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_step_timing: steps=%d double_support=%f swing=%f settle=%f stable=%f cooldown=%f max_duration=%f",
                stand_stabilizer_config_.wbc_step_recovery_steps,
                stand_stabilizer_config_.wbc_step_recovery_double_support_time,
                stand_stabilizer_config_.wbc_step_recovery_swing_time,
                stand_stabilizer_config_.wbc_step_recovery_settle_time,
                stand_stabilizer_config_.wbc_step_recovery_stable_time,
                stand_stabilizer_config_.wbc_step_recovery_cooldown,
                stand_stabilizer_config_.wbc_step_recovery_max_duration);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_step_gains: x[pitch=%f rate=%f com=%f com_vel=%f] y[roll=%f rate=%f com=%f com_vel=%f] capture=[time=%f gain=%f] min_xy=[%f, %f] max_xy=[%f, %f] signs=[%f, %f]",
                stand_stabilizer_config_.wbc_step_recovery_step_x_pitch_gain,
                stand_stabilizer_config_.wbc_step_recovery_step_x_rate_gain,
                stand_stabilizer_config_.wbc_step_recovery_step_x_com_gain,
                stand_stabilizer_config_.wbc_step_recovery_step_x_com_velocity_gain,
                stand_stabilizer_config_.wbc_step_recovery_step_y_roll_gain,
                stand_stabilizer_config_.wbc_step_recovery_step_y_rate_gain,
                stand_stabilizer_config_.wbc_step_recovery_step_y_com_gain,
                stand_stabilizer_config_.wbc_step_recovery_step_y_com_velocity_gain,
                stand_stabilizer_config_.wbc_step_recovery_capture_time,
                stand_stabilizer_config_.wbc_step_recovery_capture_gain,
                stand_stabilizer_config_.wbc_step_recovery_min_step_x,
                stand_stabilizer_config_.wbc_step_recovery_min_step_y,
                stand_stabilizer_config_.wbc_step_recovery_max_step_x,
                stand_stabilizer_config_.wbc_step_recovery_max_step_y,
                stand_stabilizer_config_.wbc_step_recovery_sagittal_sign,
                stand_stabilizer_config_.wbc_step_recovery_lateral_sign);
    RCLCPP_INFO(this->get_logger(),
                "stand_wbc_swing: height=%f tracking_weight=%f kp=%f kd=%f ik_gain=%f ik_damping=%f max_delta=%f max_vel=%f",
                stand_stabilizer_config_.wbc_step_recovery_swing_height,
                stand_stabilizer_config_.wbc_swing_tracking_weight,
                stand_stabilizer_config_.wbc_swing_kp,
                stand_stabilizer_config_.wbc_swing_kd,
                stand_stabilizer_config_.wbc_swing_ik_gain,
                stand_stabilizer_config_.wbc_swing_ik_damping,
                stand_stabilizer_config_.wbc_swing_max_joint_delta,
                stand_stabilizer_config_.wbc_swing_max_joint_velocity);
    print_vector<double>("stand_wbc_torque_joint_scale", stand_stabilizer_config_.wbc_torque_joint_scale);
    print_vector<float>("stand_kp", stand_kp_);
    print_vector<float>("stand_kd", stand_kd_);
    print_vector<double>("joint_limits", joint_limits_);
    RCLCPP_INFO(this->get_logger(), "gravity_z_upper: %f", gravity_z_upper_);
}

void InferenceNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    constexpr int kButtonA = 0;
    constexpr int kButtonB = 1;
    constexpr int kButtonY = 2;
    constexpr int kButtonX = 3;
    constexpr int kButtonLB = 4;
    constexpr int kButtonRB = 5;
    constexpr int kButtonLSB = 11;

    if (is_joy_control_){
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        cmd_vel_[0] = std::clamp(msg->axes[4] * clip_cmd_[1], clip_cmd_[0], clip_cmd_[1]);
        cmd_vel_[1] = std::clamp(msg->axes[3] * clip_cmd_[3], clip_cmd_[2], clip_cmd_[3]);
            if (msg->axes[2] < 0) {
            cmd_vel_[2] = std::clamp(-msg->axes[2] * clip_cmd_[5], clip_cmd_[4], clip_cmd_[5]);
            } else if (msg->axes[5] < 0) {
            cmd_vel_[2] = std::clamp(msg->axes[5] * clip_cmd_[5], clip_cmd_[4], clip_cmd_[5]);
            } else {
            cmd_vel_[2] = 0.0;
        }
    }
    if ((msg->buttons[kButtonX] == 1 && msg->buttons[kButtonX] != last_button0_)) {
        if(is_running_.load()){
            reset_runtime_state();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        if (robot_->is_init_.load()){
            robot_->deinit_motors();
            RCLCPP_INFO(this->get_logger(), "Motors deinitialized");
        } else {
            robot_->init_motors();
            RCLCPP_INFO(this->get_logger(), "Motors initialized");
        }
    }
    if (msg->buttons[kButtonA] == 1 && msg->buttons[kButtonA] != last_button1_) {
        if (is_running_.load()){
            reset_runtime_state();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        if (!robot_->is_init_.load()){
            RCLCPP_INFO(this->get_logger(), "Motors are not initialized!");
        } else {
            robot_->reset_joints(joint_default_angle_);
            RCLCPP_INFO(this->get_logger(), "Motors reset");
        }
    }
    if (msg->buttons[kButtonB] == 1 && msg->buttons[kButtonB] != last_button2_) {
        is_running_.store(!is_running_.load());
        RCLCPP_INFO(this->get_logger(), "Control %s", is_running_.load() ? "started" : "paused");
    }
    if (msg->buttons[kButtonY] == 1 && msg->buttons[kButtonY] != last_button3_) {
        is_joy_control_.store(!is_joy_control_);
        RCLCPP_INFO(this->get_logger(), "Controlled by %s", is_joy_control_.load() ? "joy" : "/cmd_vel");
    }
    if (supports_interrupt() || has_motion_policy()) {
        if (msg->buttons[kButtonLB] == 1 && msg->buttons[kButtonLB] != last_button4_) {
            const auto switch_while_paused = [this](auto&& switch_mode) {
                std::unique_lock<std::mutex> switch_lock(lb_switch_mutex_);
                const bool restore_running = is_running_.exchange(false);
                if (restore_running) {
                    RCLCPP_INFO(this->get_logger(), "Inference paused");
                }
                try {
                    switch_mode();
                } catch (...) {
                    if (restore_running) {
                        is_running_.store(true);
                        RCLCPP_INFO(this->get_logger(), "Inference started");
                    }
                    throw;
                }
                if (restore_running) {
                    is_running_.store(true);
                    RCLCPP_INFO(this->get_logger(), "Inference started");
                }
            };
            if (supports_interrupt()) {
                switch_while_paused([this]() {
                    std::unique_lock<std::mutex> lock(mode_mutex_);
                    is_interrupt_.store(!is_interrupt_.load());
                    RCLCPP_INFO(this->get_logger(), "Interrupt mode %s", is_interrupt_.load() ? "enabled" : "disabled");
                });
            } else if (has_motion_policy()) {
                switch_while_paused([this]() {
                    std::string policy_name;
                    std::unique_lock<std::mutex> lock(mode_mutex_);
                    is_motion_policy_.store(!is_motion_policy_.load());
                    active_policy_idx_ = is_motion_policy_.load() ? motion_policy_indices_[current_motion_policy_idx_] : 0;
                    reset_policy_runtime(active_policy());
                    policy_name = active_policy().name;
                    RCLCPP_INFO(this->get_logger(), "Policy enabled: %s", policy_name.c_str());
                });
            }
        }
        last_button4_ = msg->buttons[kButtonLB];
    }
    if (msg->buttons.size() > static_cast<size_t>(kButtonLSB)) {
        if (msg->buttons[kButtonLSB] == 1 && msg->buttons[kButtonLSB] != last_button_lsb_) {
            std::unique_lock<std::mutex> switch_lock(lb_switch_mutex_);
            if (!robot_->is_init_.load()) {
                RCLCPP_WARN(this->get_logger(), "Motors are not initialized, cannot enter stand mode");
            } else {
                std::unique_lock<std::mutex> lock(mode_mutex_);
                if (control_mode_ == ControlMode::Stand) {
                    control_mode_ = ControlMode::Policy;
                    stand_transition_active_ = false;
                    is_running_.store(false);
                    RCLCPP_INFO(this->get_logger(), "Stand mode disabled");
                } else {
                    control_mode_ = ControlMode::Stand;
                    is_interrupt_.store(false);
                    is_motion_policy_.store(false);
                    active_policy_idx_ = 0;
                    start_stand_transition_locked();
                    is_running_.store(true);
                    RCLCPP_INFO(this->get_logger(), "Stand mode enabled");
                }
            }
        }
        last_button_lsb_ = msg->buttons[kButtonLSB];
    }
    if (has_motion_policy()) {
        if (msg->buttons[kButtonRB] == 1 && msg->buttons[kButtonRB] != last_button5_) {
            std::unique_lock<std::mutex> lock(mode_mutex_);
            if (is_motion_policy_.load()) {
                RCLCPP_WARN(this->get_logger(), "Cannot switch motion policy while in motion policy mode");
            } else {
                current_motion_policy_idx_ = (current_motion_policy_idx_ + 1) % motion_policy_indices_.size();
                RCLCPP_INFO(this->get_logger(), "Selected policy: %s", policies_[motion_policy_indices_[current_motion_policy_idx_]].name.c_str());
            }
        }
        last_button5_ = msg->buttons[kButtonRB];
    }
    last_button0_ = msg->buttons[kButtonX];
    last_button1_ = msg->buttons[kButtonA];
    last_button2_ = msg->buttons[kButtonB];
    last_button3_ = msg->buttons[kButtonY];
}

void InferenceNode::subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg){
    if(!is_joy_control_){
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        cmd_vel_[0] = std::clamp(msg->linear.x, clip_cmd_[0], clip_cmd_[1]);
        cmd_vel_[1] = std::clamp(msg->linear.y, clip_cmd_[2], clip_cmd_[3]);
        cmd_vel_[2] = std::clamp(msg->angular.z, clip_cmd_[4], clip_cmd_[5]);
    }
}

void InferenceNode::subs_elevation_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg){
    if(perception_obs_num_ > 0){
        std::unique_lock<std::mutex> lock(perception_mutex_);
        if (msg->data.size() < perception_obs_buffer_.size()) {
            RCLCPP_WARN(this->get_logger(), "Perception obs message too small: got %zu, expected %zu", msg->data.size(), perception_obs_buffer_.size());
            std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
            return;
        }
        std::copy(msg->data.begin(), msg->data.begin() + perception_obs_buffer_.size(), perception_obs_buffer_.begin());
    }
}

void InferenceNode::subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg){
    if(supports_interrupt() && is_interrupt_.load()){
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        for(size_t i = 0; i < interrupt_action_.size(); i++){
            interrupt_action_[i] = msg->position[i];
        }
    }
}

void InferenceNode::reset_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is running, cannot reset joints.";
        return;
    }
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot reset joints.";
        return;
    }
    try {
        robot_->reset_joints(joint_default_angle_);
        response->success = true;
        response->message = "Joints reset successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::refresh_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot refresh motors.";
        return;
    }
    try {
        robot_->refresh_joints();
        response->success = true;
        response->message = "Motors refresh successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::read_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot read joints.";
        return;
    }
    try {
        response->success = true;
        response->message = "Joints read successfully";
        publish_joint_states();
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::read_imu_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_) {
        response->success = false;
        response->message = "IMU is not initialized, cannot read IMU.";
        return;
    }
    try {
        response->success = true;
        response->message = "IMU read successfully";
        publish_imu();
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::set_zeros_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot set zeros.";
        return;
    }
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is running, cannot set zeros.";
        return;
    }
    try {
        robot_->set_zeros();
        response->success = true;
        response->message = "Zeros set successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::clear_errors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_) {
        response->success = false;
        response->message = "Robot interface is not initialized, cannot clear errors.";
        return;
    }
    try {
        robot_->clear_errors();
        response->success = true;
        response->message = "Errors cleared successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::init_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are already initialized, cannot init motors.";
        return;
    }
    try {
        robot_->init_motors();
        response->success = true;
        response->message = "Motors initialized successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::deinit_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    try {
        robot_->deinit_motors();
        response->success = true;
        response->message = "Motor deinit commands sent successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::start_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is already running!";
        return;
    }
    is_running_.store(true);
    response->success = true;
    response->message = "Inference started";
}

void InferenceNode::stop_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                       std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!is_running_.load()) {
        response->success = false;
        response->message = "Inference is already stopped!";
        return;
    }
    is_running_.store(false);
    response->success = true;
    response->message = "Inference stopped";
}

void InferenceNode::publish_joint_states() {
    joint_pos_buffer_ = robot_->get_joint_q();
    joint_vel_buffer_ = robot_->get_joint_vel();
    joint_torques_buffer_ = robot_->get_joint_tau();
    joint_state_msg_.header.stamp = this->now();
    joint_state_msg_.effort.resize(joint_num_);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.position[i] = joint_pos_buffer_[i];
        joint_state_msg_.velocity[i] = joint_vel_buffer_[i];
        joint_state_msg_.effort[i] = joint_torques_buffer_[i];
    }
    joint_state_publisher_->publish(joint_state_msg_);
}

void InferenceNode::publish_action() {
    action_msg_.header.stamp = this->now();
    for (int i = 0; i < joint_num_; i++) {
        action_msg_.position[i] = act_[i];
    }
    action_publisher_->publish(action_msg_);
}

void InferenceNode::publish_imu() {
    quat_buffer_ = robot_->get_quat();
    ang_vel_buffer_ = robot_->get_ang_vel();
    auto msg = sensor_msgs::msg::Imu();
    msg.header.stamp = this->now();
    msg.orientation.w = quat_buffer_[0];
    msg.orientation.x = quat_buffer_[1];
    msg.orientation.y = quat_buffer_[2];
    msg.orientation.z = quat_buffer_[3];
    msg.angular_velocity.x = ang_vel_buffer_[0];
    msg.angular_velocity.y = ang_vel_buffer_[1];
    msg.angular_velocity.z = ang_vel_buffer_[2];
    imu_publisher_->publish(msg);
}
