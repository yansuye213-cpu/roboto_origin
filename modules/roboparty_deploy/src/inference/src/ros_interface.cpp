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
    this->declare_parameter<std::string>("stand_control_backend", "joint_qp");
    this->declare_parameter<int>("stand_mpc_horizon", 20);
    this->declare_parameter<float>("stand_mpc_q_angle", 120.0);
    this->declare_parameter<float>("stand_mpc_q_rate", 4.0);
    this->declare_parameter<float>("stand_mpc_r_accel", 0.08);
    this->declare_parameter<float>("stand_mpc_max_accel", 12.0);
    this->declare_parameter<float>("stand_mpc_roll_gain", 0.003);
    this->declare_parameter<float>("stand_mpc_pitch_gain", 0.003);
    this->declare_parameter<float>("stand_mpc_max_joint_correction", 0.08);
    this->declare_parameter<float>("stand_mpc_target_roll", 0.0);
    this->declare_parameter<float>("stand_mpc_target_pitch", 0.0);
    this->declare_parameter<bool>("stand_qp_enabled", false);
    this->declare_parameter<int>("stand_qp_iterations", 32);
    this->declare_parameter<float>("stand_qp_tracking_weight", 4.0);
    this->declare_parameter<float>("stand_qp_shape_weight", 0.25);
    this->declare_parameter<float>("stand_qp_regularization_weight", 0.02);
    this->declare_parameter<float>("stand_qp_smooth_weight", 0.8);
    this->declare_parameter<float>("stand_qp_max_joint_velocity", 4.0);
    this->declare_parameter<std::string>("stand_whole_body_model_path", "");
    this->declare_parameter<std::string>("stand_whole_body_base_link", "");
    this->declare_parameter<std::string>("stand_whole_body_left_foot_link", "");
    this->declare_parameter<std::string>("stand_whole_body_right_foot_link", "");
    this->declare_parameter<std::vector<std::string>>("stand_whole_body_joint_order", std::vector<std::string>{});
    this->declare_parameter<std::vector<double>>("stand_mpc_roll_joint_scale", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("stand_mpc_pitch_joint_scale", std::vector<double>{});
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
    this->get_parameter("stand_control_backend", stand_stabilizer_config_.control_backend);
    this->get_parameter("stand_mpc_horizon", stand_stabilizer_config_.horizon);
    this->get_parameter("stand_mpc_q_angle", stand_stabilizer_config_.q_angle);
    this->get_parameter("stand_mpc_q_rate", stand_stabilizer_config_.q_rate);
    this->get_parameter("stand_mpc_r_accel", stand_stabilizer_config_.r_accel);
    this->get_parameter("stand_mpc_max_accel", stand_stabilizer_config_.max_accel);
    this->get_parameter("stand_mpc_roll_gain", stand_stabilizer_config_.roll_gain);
    this->get_parameter("stand_mpc_pitch_gain", stand_stabilizer_config_.pitch_gain);
    this->get_parameter("stand_mpc_max_joint_correction", stand_stabilizer_config_.max_joint_correction);
    this->get_parameter("stand_mpc_target_roll", stand_stabilizer_config_.target_roll);
    this->get_parameter("stand_mpc_target_pitch", stand_stabilizer_config_.target_pitch);
    this->get_parameter("stand_qp_enabled", stand_stabilizer_config_.qp_enabled);
    this->get_parameter("stand_qp_iterations", stand_stabilizer_config_.qp_iterations);
    this->get_parameter("stand_qp_tracking_weight", stand_stabilizer_config_.qp_tracking_weight);
    this->get_parameter("stand_qp_shape_weight", stand_stabilizer_config_.qp_shape_weight);
    this->get_parameter("stand_qp_regularization_weight", stand_stabilizer_config_.qp_regularization_weight);
    this->get_parameter("stand_qp_smooth_weight", stand_stabilizer_config_.qp_smooth_weight);
    this->get_parameter("stand_qp_max_joint_velocity", stand_stabilizer_config_.qp_max_joint_velocity);
    this->get_parameter("stand_whole_body_model_path", stand_stabilizer_config_.whole_body_model_path);
    this->get_parameter("stand_whole_body_base_link", stand_stabilizer_config_.whole_body_base_link);
    this->get_parameter("stand_whole_body_left_foot_link", stand_stabilizer_config_.whole_body_left_foot_link);
    this->get_parameter("stand_whole_body_right_foot_link", stand_stabilizer_config_.whole_body_right_foot_link);
    this->get_parameter("stand_whole_body_joint_order", stand_stabilizer_config_.whole_body_joint_order);
    this->get_parameter("stand_mpc_roll_joint_scale", stand_stabilizer_config_.roll_joint_scale);
    this->get_parameter("stand_mpc_pitch_joint_scale", stand_stabilizer_config_.pitch_joint_scale);
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
    if (stand_stabilizer_config_.control_backend != "joint_qp" &&
        stand_stabilizer_config_.control_backend != "whole_body_mpc") {
        throw std::runtime_error("stand_control_backend must be joint_qp or whole_body_mpc");
    }
    stand_stabilizer_config_.joint_num = joint_num_;
    stand_stabilizer_config_.dt = dt_;
    if (stand_stabilizer_config_.horizon <= 0) {
        throw std::runtime_error("stand_mpc_horizon must be positive");
    }
    if (stand_stabilizer_config_.q_angle < 0.0f || stand_stabilizer_config_.q_rate < 0.0f ||
        stand_stabilizer_config_.r_accel <= 0.0f) {
        throw std::runtime_error("stand MPC weights must be non-negative, and stand_mpc_r_accel must be positive");
    }
    if (stand_stabilizer_config_.max_accel <= 0.0f || stand_stabilizer_config_.max_joint_correction < 0.0f) {
        throw std::runtime_error("stand MPC limits must be positive");
    }
    if (stand_stabilizer_config_.qp_iterations <= 0) {
        throw std::runtime_error("stand_qp_iterations must be positive");
    }
    if (stand_stabilizer_config_.qp_tracking_weight < 0.0f ||
        stand_stabilizer_config_.qp_shape_weight < 0.0f ||
        stand_stabilizer_config_.qp_regularization_weight < 0.0f ||
        stand_stabilizer_config_.qp_smooth_weight < 0.0f ||
        stand_stabilizer_config_.qp_max_joint_velocity < 0.0f) {
        throw std::runtime_error("stand QP parameters must be non-negative");
    }
    if (stand_stabilizer_config_.roll_joint_scale.empty()) {
        stand_stabilizer_config_.roll_joint_scale.assign(joint_num_, 0.0);
        if (joint_num_ > 11) {
            stand_stabilizer_config_.roll_joint_scale[1] = 0.35;
            stand_stabilizer_config_.roll_joint_scale[5] = 1.0;
            stand_stabilizer_config_.roll_joint_scale[7] = 0.35;
            stand_stabilizer_config_.roll_joint_scale[11] = 1.0;
        }
    }
    if (stand_stabilizer_config_.pitch_joint_scale.empty()) {
        stand_stabilizer_config_.pitch_joint_scale.assign(joint_num_, 0.0);
        if (joint_num_ > 10) {
            stand_stabilizer_config_.pitch_joint_scale[2] = 0.35;
            stand_stabilizer_config_.pitch_joint_scale[3] = -0.15;
            stand_stabilizer_config_.pitch_joint_scale[4] = 1.0;
            stand_stabilizer_config_.pitch_joint_scale[8] = 0.35;
            stand_stabilizer_config_.pitch_joint_scale[9] = -0.15;
            stand_stabilizer_config_.pitch_joint_scale[10] = 1.0;
        }
    }
    if (stand_stabilizer_config_.roll_joint_scale.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("stand_mpc_roll_joint_scale must be empty or have the same size as joint_num");
    }
    if (stand_stabilizer_config_.pitch_joint_scale.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("stand_mpc_pitch_joint_scale must be empty or have the same size as joint_num");
    }
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
    RCLCPP_INFO(this->get_logger(), "stand_control_backend: %s", stand_stabilizer_config_.control_backend.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_mpc_horizon: %d", stand_stabilizer_config_.horizon);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_q_angle: %f", stand_stabilizer_config_.q_angle);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_q_rate: %f", stand_stabilizer_config_.q_rate);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_r_accel: %f", stand_stabilizer_config_.r_accel);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_max_accel: %f", stand_stabilizer_config_.max_accel);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_roll_gain: %f", stand_stabilizer_config_.roll_gain);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_pitch_gain: %f", stand_stabilizer_config_.pitch_gain);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_max_joint_correction: %f", stand_stabilizer_config_.max_joint_correction);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_target_roll: %f", stand_stabilizer_config_.target_roll);
    RCLCPP_INFO(this->get_logger(), "stand_mpc_target_pitch: %f", stand_stabilizer_config_.target_pitch);
    RCLCPP_INFO(this->get_logger(), "stand_qp_enabled: %s", stand_stabilizer_config_.qp_enabled ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "stand_qp_iterations: %d", stand_stabilizer_config_.qp_iterations);
    RCLCPP_INFO(this->get_logger(), "stand_qp_tracking_weight: %f", stand_stabilizer_config_.qp_tracking_weight);
    RCLCPP_INFO(this->get_logger(), "stand_qp_shape_weight: %f", stand_stabilizer_config_.qp_shape_weight);
    RCLCPP_INFO(this->get_logger(), "stand_qp_regularization_weight: %f", stand_stabilizer_config_.qp_regularization_weight);
    RCLCPP_INFO(this->get_logger(), "stand_qp_smooth_weight: %f", stand_stabilizer_config_.qp_smooth_weight);
    RCLCPP_INFO(this->get_logger(), "stand_qp_max_joint_velocity: %f", stand_stabilizer_config_.qp_max_joint_velocity);
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_model_path: %s", stand_stabilizer_config_.whole_body_model_path.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_base_link: %s", stand_stabilizer_config_.whole_body_base_link.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_left_foot_link: %s", stand_stabilizer_config_.whole_body_left_foot_link.c_str());
    RCLCPP_INFO(this->get_logger(), "stand_whole_body_right_foot_link: %s", stand_stabilizer_config_.whole_body_right_foot_link.c_str());
    print_vector<std::string>("stand_whole_body_joint_order", stand_stabilizer_config_.whole_body_joint_order);
    print_vector<double>("stand_mpc_roll_joint_scale", stand_stabilizer_config_.roll_joint_scale);
    print_vector<double>("stand_mpc_pitch_joint_scale", stand_stabilizer_config_.pitch_joint_scale);
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
