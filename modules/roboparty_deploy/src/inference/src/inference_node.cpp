// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "inference_node.hpp"

ObsStackOrder InferenceNode::parse_obs_stack_order(const std::string& stack_order_name) {
    if (stack_order_name == "frame_major") {
        return ObsStackOrder::FrameMajor;
    }
    if (stack_order_name == "obs_major") {
        return ObsStackOrder::ObsMajor;
    }
    throw std::runtime_error("Unsupported obs stack order: " + stack_order_name);
}

void InferenceNode::update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                                       int obs_num, int frame_stack, ObsStackOrder stack_order,
                                       const std::vector<int>& field_sizes, bool is_first_frame) {
    if (stack_order == ObsStackOrder::FrameMajor) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin(), obs.end(), input_buffer.begin() + frame * obs_num);
            }
        } else {
            std::move(input_buffer.begin() + obs_num,
                      input_buffer.begin() + frame_stack * obs_num,
                      input_buffer.begin());
            std::copy(obs.begin(), obs.end(), input_buffer.begin() + (frame_stack - 1) * obs_num);
        }
        return;
    }

    int input_offset = 0;
    int obs_offset = 0;

    for (const int field_size : field_sizes) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                          input_buffer.begin() + input_offset + frame * field_size);
            }
        } else {
            std::move(input_buffer.begin() + input_offset + field_size,
                      input_buffer.begin() + input_offset + frame_stack * field_size,
                      input_buffer.begin() + input_offset);
            std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                      input_buffer.begin() + input_offset + (frame_stack - 1) * field_size);
        }
        input_offset += field_size * frame_stack;
        obs_offset += field_size;
    }
}

void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size){
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    ctx->num_inputs = ctx->session->GetInputCount();
    if (ctx->num_inputs != 1) {
        throw std::runtime_error("Only single-input ONNX models are supported: " + model_path);
    }
    ctx->input_names.resize(ctx->num_inputs);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
    }

    size_t model_input_size = 1;
    for (size_t i = 0; i < ctx->input_shape.size(); i++) {
        model_input_size *= static_cast<size_t>(ctx->input_shape[i]);
    }
    if (model_input_size != static_cast<size_t>(input_size)) {
        throw std::runtime_error(
            "ONNX input size mismatch for " + model_path + ": model expects " +
            std::to_string(model_input_size) + " values, but config provides " + std::to_string(input_size));
    }
    ctx->input_buffer.resize(input_size);

    ctx->num_outputs = ctx->session->GetOutputCount();
    if (ctx->num_outputs != 1) {
        throw std::runtime_error("Only single-output ONNX models are supported: " + model_path);
    }
    ctx->output_names.resize(ctx->num_outputs);

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->output_shape.empty()) {
            throw std::runtime_error("Unsupported scalar ONNX output for " + model_path);
        }
        if (ctx->output_shape[0] == -1) ctx->output_shape[0] = 1;
    }

    size_t model_output_size = 1;
    for (size_t i = 0; i < ctx->output_shape.size(); i++) {
        if (ctx->output_shape[i] <= 0) {
            throw std::runtime_error("Unsupported dynamic ONNX output shape for " + model_path);
        }
        model_output_size *= static_cast<size_t>(ctx->output_shape[i]);
    }
    if (usd2urdf_.size() > model_output_size) {
        throw std::runtime_error(
            "ONNX output size mismatch for " + model_path + ": model provides " +
            std::to_string(model_output_size) + " actions, but usd2urdf maps " +
            std::to_string(usd2urdf_.size()) + " actions");
    }
    ctx->output_buffer.resize(model_output_size);

    ctx->input_names_raw = std::vector<const char *>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char *>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(), ctx->input_shape.data(), ctx->input_shape.size()));
        
    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(), ctx->output_shape.data(), ctx->output_shape.size()));
}

void InferenceNode::reset_runtime_state() {
    is_running_.store(false);
    is_interrupt_.store(false);
    is_motion_policy_.store(false);
    active_policy_idx_ = 0;
    {
        std::unique_lock<std::mutex> lock(mode_mutex_);
        control_mode_ = ControlMode::Policy;
        stand_transition_elapsed_ = 0.0f;
        stand_transition_active_ = false;
    }
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(perception_mutex_);
        std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (int i = 0; i < joint_num_; i++) {
            act_[i] = static_cast<float>(joint_default_angle_[i]);
            last_act_[i] = static_cast<float>(joint_default_angle_[i]);
        }
    }
    if (supports_interrupt()) {
        if (joint_default_angle_.size() < interrupt_action_.size()) {
            throw std::runtime_error("joint_default_angle is smaller than interrupt_action");
        }
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        const size_t offset = joint_default_angle_.size() - interrupt_action_.size();
        for (size_t i = 0; i < interrupt_action_.size(); i++) {
            interrupt_action_[i] = static_cast<float>(joint_default_angle_[offset + i]);
        }
    }
    for (PolicyRuntime& policy : policies_) {
        reset_policy_runtime(policy);
    }
}

InferenceNode::PolicyRuntime& InferenceNode::active_policy() {
    return policies_[active_policy_idx_];
}

const InferenceNode::PolicyRuntime& InferenceNode::active_policy() const {
    return policies_[active_policy_idx_];
}

void InferenceNode::initialize_runtime_state() {
    active_policy_idx_ = 0;

    joint_state_msg_.name.resize(joint_num_);
    joint_state_msg_.position.assign(joint_num_, 0.0f);
    joint_state_msg_.velocity.assign(joint_num_, 0.0f);
    joint_state_msg_.effort.assign(joint_num_, 0.0f);
    action_msg_.name.resize(joint_num_);
    action_msg_.position.assign(joint_num_, 0.0f);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.name[i] = "joint_" + std::to_string(i + 1);
        action_msg_.name[i] = "action_" + std::to_string(i + 1);
    }

    cmd_vel_.assign(3, 0.0f);
    act_.assign(joint_num_, 0.0f);
    last_act_.assign(joint_num_, 0.0f);
    stand_start_action_.assign(joint_num_, 0.0f);
    joint_pos_buffer_.assign(joint_num_, 0.0f);
    joint_vel_buffer_.assign(joint_num_, 0.0f);
    joint_torques_buffer_.assign(joint_num_, 0.0f);
    quat_buffer_.assign(4, 0.0f);
    ang_vel_buffer_.assign(3, 0.0f);
    if (has_obs_source("perception")) {
        perception_obs_buffer_.assign(perception_obs_num_, 0.0f);
    } else {
        perception_obs_buffer_.clear();
    }
    if (has_obs_source("interrupt")) {
        interrupt_action_.assign(10, 0.0f);
    } else {
        interrupt_action_.clear();
    }
}

bool InferenceNode::has_motion_policy() const {
    return !motion_policy_indices_.empty();
}

bool InferenceNode::supports_interrupt() const {
    return !interrupt_action_.empty();
}

void InferenceNode::reset_policy_runtime(PolicyRuntime& policy) {
    std::fill(policy.obs.begin(), policy.obs.end(), 0.0f);
    for (auto& segment : policy.obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    for (auto& segment : policy.extra_obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    if (policy.ctx) {
        std::fill(policy.ctx->input_buffer.begin(), policy.ctx->input_buffer.end(), 0.0f);
        std::fill(policy.ctx->output_buffer.begin(), policy.ctx->output_buffer.end(), 0.0f);
    }
    policy.motion_frame = 0;
    policy.is_first_frame = true;
}

void InferenceNode::start_stand_transition_locked() {
    stand_transition_elapsed_ = 0.0f;
    stand_transition_active_ = true;
    if (stand_stabilizer_) {
        stand_stabilizer_->reset();
    }
    stand_start_action_.assign(joint_num_, 0.0f);
    std::unique_lock<std::mutex> lock(act_mutex_);
    for (int i = 0; i < joint_num_; i++) {
        stand_start_action_[i] = last_act_[i];
    }
}

void InferenceNode::apply_stand_action() {
    quat_buffer_ = robot_->get_quat();
    ang_vel_buffer_ = robot_->get_ang_vel();
    joint_pos_buffer_ = robot_->get_joint_q();
    joint_vel_buffer_ = robot_->get_joint_vel();
    const StandingStabilizer::Measurement measurement =
        stand_stabilizer_->measure(quat_buffer_, ang_vel_buffer_);
    if (measurement.gravity_z > gravity_z_upper_) {
        RCLCPP_FATAL(this->get_logger(), "Robot fell down in stand mode! Shutting down...");
        rclcpp::shutdown();
        throw std::runtime_error("Robot fell down in stand mode");
    }

    std::vector<float> target(joint_num_, 0.0f);
    float mpc_blend = 1.0f;
    {
        std::unique_lock<std::mutex> lock(mode_mutex_);
        if (stand_start_action_.size() != static_cast<size_t>(joint_num_)) {
            stand_start_action_.assign(joint_num_, 0.0f);
            for (int i = 0; i < joint_num_; i++) {
                stand_start_action_[i] = static_cast<float>(joint_default_angle_[i]);
            }
        }
        float blend = 1.0f;
        if (stand_transition_active_) {
            stand_transition_elapsed_ += dt_;
            const float duration = std::max(stand_transition_time_, dt_);
            blend = std::clamp(stand_transition_elapsed_ / duration, 0.0f, 1.0f);
            blend = blend * blend * (3.0f - 2.0f * blend);
            if (stand_transition_elapsed_ >= duration) {
                stand_transition_active_ = false;
            }
        }
        mpc_blend = blend;
        for (int i = 0; i < joint_num_; i++) {
            target[i] = stand_start_action_[i] +
                        blend * (static_cast<float>(stand_joint_angle_[i]) - stand_start_action_[i]);
        }
    }

    const StandingStabilizer::Command command =
        stand_stabilizer_->apply(measurement, mpc_blend, target, stand_kp_, stand_kd_,
                                 joint_pos_buffer_, joint_vel_buffer_);
    const StandingStabilizer::Correction& correction = command.correction;
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                         "stand ctrl[whole_body_mpc]: roll=%.4f pitch=%.4f wx=%.4f wy=%.4f mpc_acc=[%.3f, %.3f, %.3f, %.3f] qp=%d wbc_fz=[%.1f, %.1f] wbc_moment_des=[%.2f, %.2f] wbc_moment_act=[%.2f, %.2f] max_tau=%.3f raw_tau=%.3f sat=%d tau_j=%d",
                         measurement.roll, measurement.pitch, measurement.wx, measurement.wy,
                         correction.mpc_roll_accel, correction.mpc_pitch_accel,
                         correction.mpc_com_accel_x, correction.mpc_com_accel_y,
                         correction.qp_used ? 1 : 0,
                         correction.wbc_left_normal_force, correction.wbc_right_normal_force,
                         correction.wbc_roll_moment, correction.wbc_pitch_moment,
                         correction.wbc_achieved_roll_moment,
                         correction.wbc_achieved_pitch_moment,
                         correction.wbc_max_joint_torque,
                         correction.wbc_max_raw_joint_torque,
                         correction.wbc_saturated_joint_count,
                         correction.wbc_max_torque_joint_index);

    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        act_ = command.position;
        last_act_ = command.position;
    }
    robot_->apply_action(command.position, command.velocity, command.kp, command.kd, command.tau);
    publish_joint_states();
    publish_imu();
    publish_action();
}

void InferenceNode::apply_action() {
    if(!is_running_.load() || !robot_->is_init_.load()){
        return;
    }
    {
        std::unique_lock<std::mutex> mode_lock(mode_mutex_);
        if (control_mode_ == ControlMode::Stand) {
            mode_lock.unlock();
            apply_stand_action();
            return;
        }
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (size_t i = 0; i < act_.size(); i++) {
            last_act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
        }
    }
    robot_->apply_action(last_act_);
}

void InferenceNode::control() {
    pthread_setname_np(pthread_self(), "control");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_WARN(this->get_logger(), "Failed to set realtime priority for control thread; continuing without SCHED_FIFO");
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000000));
    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        try {
            apply_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in control thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;
        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_WARN(this->get_logger(), "Failed to set realtime priority for inference thread; continuing without SCHED_FIFO");
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));

    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        if(!is_running_.load()){
            std::this_thread::sleep_for(period);
            continue;
        }

        try {
            std::unique_lock<std::mutex> mode_lock(mode_mutex_);
            if (control_mode_ == ControlMode::Stand) {
                mode_lock.unlock();
                std::this_thread::sleep_for(period);
                continue;
            }
            auto& policy = active_policy();
            update_obs_segments(policy.obs_segments, policy.obs_layout);
            publish_imu();
            publish_joint_states();
            flatten_obs_segments(policy.obs_segments, policy.obs.begin());

            std::transform(policy.obs.begin(), policy.obs.end(), policy.obs.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });

            update_stacked_obs(policy.ctx->input_buffer, policy.obs, policy.obs_num, policy.frame_stack,
                               policy.stack_order, policy.obs_layout_sizes, policy.is_first_frame);
            if(policy.extra_obs_num > 0){
                update_obs_segments(policy.extra_obs_segments, policy.extra_obs_layout);
                flatten_obs_segments(policy.extra_obs_segments, policy.ctx->input_buffer.begin() + policy.frame_stack * policy.obs_num);
            }
            if (policy.motion_loader) {
                step_motion_frame();
            }
            policy.is_first_frame = false;

            policy.ctx->session->Run(Ort::RunOptions{nullptr},
                policy.ctx->input_names_raw.data(), policy.ctx->input_tensor.get(), policy.ctx->num_inputs,
                policy.ctx->output_names_raw.data(), policy.ctx->output_tensor.get(), policy.ctx->num_outputs);

            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                for (float& action : policy.ctx->output_buffer) {
                    action = std::clamp(action, -clip_actions_, clip_actions_);
                }
                for (size_t i = 0; i < usd2urdf_.size(); i++) {
                    const size_t joint_idx = static_cast<size_t>(usd2urdf_[i]);
                    act_[joint_idx] = policy.ctx->output_buffer[i];
                    act_[joint_idx] = act_[joint_idx] * action_scale_ + joint_default_angle_[joint_idx];
                }
                if(supports_interrupt() && is_interrupt_.load()){
                    std::unique_lock<std::mutex> lock(interrupt_mutex_);
                    for (size_t i = 0; i < interrupt_action_.size(); i++) {
                        act_[act_.size() - interrupt_action_.size() + i] = interrupt_action_[i];
                    }
                }
                publish_action();
            }
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in inference thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        auto loop_end = std::chrono::steady_clock::now();
        // 使用微秒进行计算
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;

        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        } else {
            // 警告信息也使用更精确的单位
            RCLCPP_WARN(this->get_logger(), "Inference loop overran! Took %lld us, but period is %lld us.", static_cast<long long>(elapsed_time.count()), static_cast<long long>(period.count()));
        }
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "mlockall failed.");
    }
    pthread_setname_np(pthread_self(), "main");
    struct sched_param sp{}; sp.sched_priority = 50;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "Failed to set realtime priority for main thread; continuing without SCHED_FIFO");
    }
    std::shared_ptr<InferenceNode> node;
    try {
        node = std::make_shared<InferenceNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "Press 'X' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'A' to reset motors");
        RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause control");
        RCLCPP_INFO(node->get_logger(), "Press 'Y' to switch between Gamepad Control / cmd_vel Control");
        if (node->supports_interrupt() || node->has_motion_policy()) {
            RCLCPP_INFO(node->get_logger(), "Press 'LB' to switch policy mode (available in beyondmimic / interrupt modes)");
        }
        RCLCPP_INFO(node->get_logger(), "Press 'LSB' to enter/exit stand mode");
        if (node->has_motion_policy()){
            RCLCPP_INFO(node->get_logger(), "Press 'RB' to switch motion sequence (available in beyondmimic mode)");
        }
        RCLCPP_INFO(node->get_logger(), "Right Stick: Control forward, backward, left and right movement");
        RCLCPP_INFO(node->get_logger(), "LT/RT: Control turning (left / right rotation)");
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    node.reset();
    return 0;
}
