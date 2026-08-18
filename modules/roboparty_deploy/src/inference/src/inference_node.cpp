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

bool InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx,
                                std::string model_path,
                                int input_size,
                                std::string& disabled_reason) {
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }
    disabled_reason.clear();

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
        disabled_reason =
            "ONNX input size mismatch: model expects " +
            std::to_string(model_input_size) + " values, config provides " +
            std::to_string(input_size);
        ctx.reset();
        return false;
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
    if (model_output_size != usd2urdf_.size()) {
        disabled_reason =
            "ONNX output size mismatch: model provides " +
            std::to_string(model_output_size) + " actions, usd2urdf maps " +
            std::to_string(usd2urdf_.size()) + " actions";
        ctx.reset();
        return false;
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
    return true;
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
        policy_transition_elapsed_ = 0.0f;
        policy_transition_active_ = false;
        policy_transition_target_ready_ = false;
    }
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        clear_velocity_command_locked();
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
            policy_filtered_action_[i] = static_cast<float>(joint_default_angle_[i]);
            policy_transition_offset_[i] = 0.0f;
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
    policy_start_action_.assign(joint_num_, 0.0f);
    policy_transition_offset_.assign(joint_num_, 0.0f);
    policy_filtered_action_.assign(joint_num_, 0.0f);
    joint_pos_buffer_.assign(joint_num_, 0.0f);
    joint_vel_buffer_.assign(joint_num_, 0.0f);
    joint_torques_buffer_.assign(joint_num_, 0.0f);
    joint_limit_violation_active_.assign(joint_num_, false);
    quat_buffer_.assign(4, 0.0f);
    ang_vel_buffer_.assign(3, 0.0f);
    if (has_obs_source("perception")) {
        perception_obs_buffer_.assign(perception_obs_num_, 0.0f);
    } else {
        perception_obs_buffer_.clear();
    }
    if (has_obs_source("interrupt")) {
        interrupt_action_.assign(interrupt_action_size_, 0.0f);
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

void InferenceNode::start_run_log() {
    if (!run_logger_ || run_logger_->active()) {
        return;
    }
    std::string error_message;
    if (!run_logger_->start(active_policy().name, error_message)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to start policy CSV log: %s",
                     error_message.c_str());
        return;
    }
    RCLCPP_INFO(this->get_logger(), "Policy CSV log started: %s",
                std::filesystem::absolute(run_logger_->current_path()).c_str());
}

void InferenceNode::finish_run_log(const std::string& reason,
                                   const std::string& detail,
                                   bool clean_exit) {
    if (!run_logger_ || !run_logger_->active()) {
        return;
    }
    run_logger_->finish(reason, detail, clean_exit);
    RCLCPP_INFO(this->get_logger(), "Policy CSV log finished: %s", reason.c_str());
}

void InferenceNode::handle_runtime_fault(const std::string& source,
                                         const std::string& detail) noexcept {
    bool expected = false;
    if (!runtime_fault_handling_.compare_exchange_strong(expected, true)) {
        return;
    }

    // Stop producing commands before touching motor state. The node and its
    // worker threads remain alive so the operator can initialize and retry.
    is_running_.store(false);
    const bool motors_were_initialized =
        robot_ && robot_->is_init_.exchange(false);
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        clear_velocity_command_locked();
    }

    if (run_logger_ && run_logger_->active()) {
        try {
            run_logger_->record_event("runtime_fault", source + ": " + detail);
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to enqueue runtime fault event");
        }
    }

    std::string final_detail = detail;
    try {
        std::lock_guard<std::mutex> lifecycle_lock(motor_lifecycle_mutex_);
        // A start request that was already in progress may have raced with the
        // first store. Reassert the stopped state while lifecycle operations
        // are serialized, then send the hardware disable commands.
        is_running_.store(false);
        const bool motors_initialized_after_lock =
            robot_ && robot_->is_init_.exchange(false);
        if (robot_ &&
            (motors_were_initialized || motors_initialized_after_lock)) {
            robot_->deinit_motors();
        }
    } catch (const std::exception& e) {
        final_detail += "; motor deinit failed: ";
        final_detail += e.what();
        RCLCPP_ERROR(this->get_logger(),
                     "Motor deinitialization after runtime fault failed: %s",
                     e.what());
    } catch (...) {
        final_detail += "; motor deinit failed with an unknown error";
        RCLCPP_ERROR(this->get_logger(),
                     "Motor deinitialization after runtime fault failed with "
                     "an unknown error");
    }

    try {
        finish_run_log(source, final_detail, false);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(),
                     "Failed to finalize policy CSV after runtime fault: %s",
                     e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(),
                     "Failed to finalize policy CSV after runtime fault");
    }

    RCLCPP_ERROR(this->get_logger(),
                 "Runtime fault handled without stopping the node: %s: %s. "
                 "Motors are disabled; initialize and prepare the robot before retrying.",
                 source.c_str(), final_detail.c_str());
    runtime_fault_handling_.store(false);
}

void InferenceNode::record_policy_sample(size_t clamped_joint_count) {
    if (!run_logger_ || !run_logger_->active()) {
        return;
    }

    PolicyLogContext context;
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        context.cmd_vx = cmd_vel_[0];
        context.cmd_vy = cmd_vel_[1];
        context.cmd_yaw = cmd_vel_[2];
    }
    if (quat_buffer_.size() == 4) {
        Eigen::Quaternionf q(quat_buffer_[0], quat_buffer_[1],
                             quat_buffer_[2], quat_buffer_[3]);
        if (q.norm() > 1.0e-6f) {
            q.normalize();
            const Eigen::Matrix3f rotation = q.toRotationMatrix();
            context.imu_roll = std::atan2(rotation(2, 1), rotation(2, 2));
            context.imu_pitch = std::asin(
                std::clamp(-rotation(2, 0), -1.0f, 1.0f));
            const Eigen::Vector3f gravity_body =
                q.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            context.gravity_z = gravity_body.z();
        }
    }
    if (ang_vel_buffer_.size() == 3) {
        context.imu_wx = ang_vel_buffer_[0];
        context.imu_wy = ang_vel_buffer_[1];
        context.imu_wz = ang_vel_buffer_[2];
    }
    if (policy_transition_active_) {
        const float duration = std::max(policy_transition_time_, dt_);
        context.transition_phase = std::clamp(
            policy_transition_elapsed_ / duration, 0.0f, 1.0f);
    }
    context.clamped_joint_count = clamped_joint_count;
    run_logger_->record_sample(context);
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

void InferenceNode::sync_action_reference(const std::vector<float>& joint_q) {
    if (joint_q.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("Joint feedback size does not match joint_num");
    }
    std::unique_lock<std::mutex> lock(act_mutex_);
    act_ = joint_q;
    last_act_ = joint_q;
}

void InferenceNode::start_policy_transition_locked(const std::vector<float>& current_joint_q) {
    if (current_joint_q.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("Joint feedback size does not match joint_num");
    }
    control_mode_ = ControlMode::Policy;
    stand_transition_active_ = false;
    policy_transition_elapsed_ = 0.0f;
    policy_transition_active_ = true;
    policy_transition_target_ready_ = false;
    policy_start_action_ = current_joint_q;
    std::fill(policy_transition_offset_.begin(), policy_transition_offset_.end(), 0.0f);
    policy_filtered_action_ = current_joint_q;
    reset_policy_runtime(active_policy());
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        act_ = current_joint_q;
        last_act_ = current_joint_q;
    }
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        clear_velocity_command_locked();
    }
    std::fill(joint_limit_violation_active_.begin(),
              joint_limit_violation_active_.end(), false);
    start_run_log();
    is_running_.store(true);
}

void InferenceNode::enter_safe_stand_locked(const std::vector<float>& current_joint_q) {
    sync_action_reference(current_joint_q);
    control_mode_ = ControlMode::Stand;
    policy_transition_active_ = false;
    policy_transition_target_ready_ = false;
    is_interrupt_.store(false);
    is_motion_policy_.store(false);
    active_policy_idx_ = 0;
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        clear_velocity_command_locked();
    }
    start_stand_transition_locked();
    is_running_.store(true);
}

void InferenceNode::apply_stand_action() {
    const auto stand_start = std::chrono::steady_clock::now();
    quat_buffer_ = robot_->get_quat();
    ang_vel_buffer_ = robot_->get_ang_vel();
    joint_pos_buffer_ = robot_->get_joint_q();
    joint_vel_buffer_ = robot_->get_joint_vel();
    const StandingStabilizer::Measurement measurement =
        stand_stabilizer_->measure(quat_buffer_, ang_vel_buffer_);
    if (measurement.gravity_z > gravity_z_upper_) {
        RCLCPP_ERROR(this->get_logger(),
                     "Robot fell down in stand mode; disabling motors and stopping control");
        throw std::runtime_error(
            "Robot fell down in stand mode: gravity_z=" +
            std::to_string(measurement.gravity_z) +
            ", threshold=" + std::to_string(gravity_z_upper_));
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
    const auto stand_calc_end = std::chrono::steady_clock::now();
    const auto stand_calc_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            stand_calc_end - stand_start).count();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                         "stand ctrl[whole_body_mpc]: roll=%.4f pitch=%.4f wx=%.4f wy=%.4f base_v=[%.3f, %.3f, %.3f] com_v=[%.3f, %.3f] com_err=[%.3f, %.3f] est=%d/%d est_res=%.3g mpc=%s/%d iter=%d mpc_us=%d wbc_us=%d calc_us=%lld obj=%.3g mpc_acc=[%.3f, %.3f, %.3f, %.3f] mpc_f=%d jcmd=%d jcmd_delta=%.5f jcmd_vel=%.3f jcmd_j=%d L=[%.1f,%.1f,%.1f] R=[%.1f,%.1f,%.1f] step=%d phase=%d contacts=[%d,%d]/%d swing=[%d,%d] steps=%d/%d step_xy=[%.3f, %.3f] swing_err=%.3f force_qp=%d wbc_qp=%d active=%d viol=%.3g dyn_res=%.3g wbc_fz=[%.1f, %.1f] wbc_moment_des=[%.2f, %.2f] wbc_moment_act=[%.2f, %.2f] max_tau=%.3f raw_tau=%.3f sat=%d tau_j=%d",
                         measurement.roll, measurement.pitch, measurement.wx, measurement.wy,
                         correction.wbc_base_velocity_x,
                         correction.wbc_base_velocity_y,
                         correction.wbc_base_velocity_z,
                         correction.wbc_com_velocity_x,
                         correction.wbc_com_velocity_y,
                         correction.wbc_com_error_x,
                         correction.wbc_com_error_y,
                         correction.wbc_state_estimator_used ? 1 : 0,
                         correction.wbc_state_estimator_rows,
                         correction.wbc_state_estimator_residual,
                         correction.wbc_mpc_backend.c_str(),
                         correction.wbc_mpc_used ? 1 : 0,
                         correction.wbc_mpc_iterations,
                         correction.wbc_mpc_solve_us,
                         correction.wbc_whole_body_solve_us,
                         static_cast<long long>(stand_calc_us),
                         correction.wbc_mpc_objective,
                         correction.mpc_roll_accel, correction.mpc_pitch_accel,
                         correction.mpc_com_accel_x, correction.mpc_com_accel_y,
                         correction.wbc_mpc_force_target_used ? 1 : 0,
                         correction.mpc_joint_command_used ? 1 : 0,
                         correction.mpc_joint_command_max_delta,
                         correction.mpc_joint_command_max_velocity,
                         correction.mpc_joint_command_max_joint_index,
                         correction.mpc_left_force_x,
                         correction.mpc_left_force_y,
                         correction.mpc_left_force_z,
                         correction.mpc_right_force_x,
                         correction.mpc_right_force_y,
                         correction.mpc_right_force_z,
                         correction.wbc_step_recovery_active ? 1 : 0,
                         correction.wbc_step_phase,
                         correction.wbc_left_contact ? 1 : 0,
                         correction.wbc_right_contact ? 1 : 0,
                         correction.wbc_contact_count,
                         correction.wbc_left_swing ? 1 : 0,
                         correction.wbc_right_swing ? 1 : 0,
                         correction.wbc_steps_completed,
                         correction.wbc_steps_planned,
                         correction.wbc_step_x,
                         correction.wbc_step_y,
                         correction.wbc_swing_error,
                         correction.wbc_contact_force_qp_used ? 1 : 0,
                         correction.wbc_whole_body_qp_used ? 1 : 0,
                         correction.wbc_active_constraints,
                         correction.wbc_qp_violation,
                         correction.wbc_dynamics_residual,
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
    if (!is_running_.load() || !robot_->is_init_.load()) {
        return;
    }
    std::vector<float> command;
    std::unique_lock<std::mutex> mode_lock(mode_mutex_);
    if (control_mode_ == ControlMode::Stand) {
        mode_lock.unlock();
        apply_stand_action();
        return;
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        if (policy_transition_active_ && !policy_transition_target_ready_) {
            command = policy_start_action_;
        } else {
            float remaining_offset = 0.0f;
            if (policy_transition_active_) {
                policy_transition_elapsed_ += dt_;
                const float duration = std::max(policy_transition_time_, dt_);
                const float phase = std::clamp(policy_transition_elapsed_ / duration, 0.0f, 1.0f);
                const float phase2 = phase * phase;
                const float phase3 = phase2 * phase;
                const float smootherstep = phase3 * (phase * (phase * 6.0f - 15.0f) + 10.0f);
                remaining_offset = 1.0f - smootherstep;
                if (policy_transition_elapsed_ >= duration) {
                    policy_transition_active_ = false;
                }
            }
            // Decay only the handoff mismatch; do not attenuate subsequent policy motion.
            for (size_t i = 0; i < act_.size(); i++) {
                policy_filtered_action_[i] =
                    act_alpha_ * act_[i] + (1.0f - act_alpha_) * policy_filtered_action_[i];
                float target = policy_filtered_action_[i] +
                               remaining_offset * policy_transition_offset_[i];
                if (remaining_offset > 0.0f && !joint_limits_.empty()) {
                    const float lower = static_cast<float>(joint_limits_[i * 2]) +
                                        policy_joint_limit_margin_;
                    const float upper = static_cast<float>(joint_limits_[i * 2 + 1]) -
                                        policy_joint_limit_margin_;
                    target = std::clamp(target, lower, upper);
                }
                last_act_[i] = target;
            }
            command = last_act_;
        }
    }
    mode_lock.unlock();
    robot_->apply_action(command);
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
            if (run_logger_ && run_logger_->active() && robot_->is_init_.load()) {
                robot_->sample_telemetry(telemetry_snapshot_);
                run_logger_->record_control(telemetry_snapshot_);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in control thread: %s", e.what());
            handle_runtime_fault("control_exception", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(),
                         "Unknown exception in control thread");
            handle_runtime_fault("control_exception", "unknown exception");
        }
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;
        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        } else {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Control loop overran! Took %lld us, but period is %lld us.",
                static_cast<long long>(elapsed_time.count()),
                static_cast<long long>(period.count()));
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
            if (control_mode_ != ControlMode::Policy) {
                mode_lock.unlock();
                std::this_thread::sleep_for(period);
                continue;
            }
            auto& policy = active_policy();
            if (!policy.inference_enabled || !policy.ctx) {
                RCLCPP_ERROR_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Active policy %s is disabled: %s",
                    policy.name.c_str(), policy.disabled_reason.c_str());
                finish_run_log("policy_disabled", policy.disabled_reason, false);
                is_running_.store(false);
                mode_lock.unlock();
                std::this_thread::sleep_for(period);
                continue;
            }
            update_obs_segments(policy.obs_segments, policy.obs_layout);
            publish_imu();
            publish_joint_states();
            flatten_obs_segments(policy.obs_segments, policy.obs.begin());

            std::transform(policy.obs.begin(), policy.obs.end(), policy.obs.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });

            update_stacked_obs(policy.ctx->input_buffer, policy.obs, policy.obs_num, policy.frame_stack,
                               policy.stack_order, policy.obs_layout_sizes, policy.is_first_frame);
            if (policy.extra_obs_num > 0) {
                update_obs_segments(policy.extra_obs_segments, policy.extra_obs_layout);
                flatten_obs_segments(
                    policy.extra_obs_segments,
                    policy.ctx->input_buffer.begin() + policy.frame_stack * policy.obs_num);
            }
            if (policy.motion_loader) {
                step_motion_frame();
            }
            policy.is_first_frame = false;

            policy.ctx->session->Run(
                Ort::RunOptions{nullptr},
                policy.ctx->input_names_raw.data(), policy.ctx->input_tensor.get(), policy.ctx->num_inputs,
                policy.ctx->output_names_raw.data(), policy.ctx->output_tensor.get(), policy.ctx->num_outputs);

            size_t clamped_joint_count = 0;
            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                for (size_t i = 0; i < usd2urdf_.size(); i++) {
                    const size_t joint_idx = static_cast<size_t>(usd2urdf_[i]);
                    const float action =
                        std::clamp(policy.ctx->output_buffer[i], -clip_actions_, clip_actions_);
                    float target = static_cast<float>(
                        policy_joint_signs_[i] * action * action_scale_ +
                        joint_default_angle_[joint_idx]);
                    if (!joint_limits_.empty()) {
                        const float lower = static_cast<float>(joint_limits_[joint_idx * 2]) +
                                            policy_joint_limit_margin_;
                        const float upper = static_cast<float>(joint_limits_[joint_idx * 2 + 1]) -
                                            policy_joint_limit_margin_;
                        const float clamped_target = std::clamp(target, lower, upper);
                        clamped_joint_count += clamped_target != target ? 1 : 0;
                        target = clamped_target;
                    }
                    act_[joint_idx] = target;
                }
                if (clamped_joint_count > 0) {
                    RCLCPP_WARN_THROTTLE(
                        this->get_logger(), *this->get_clock(), 1000,
                        "Policy targets clamped by mechanical limits for %zu joint(s)",
                        clamped_joint_count);
                }
                if (supports_interrupt() && is_interrupt_.load()) {
                    std::unique_lock<std::mutex> lock(interrupt_mutex_);
                    for (size_t i = 0; i < interrupt_action_.size(); i++) {
                        act_[act_.size() - interrupt_action_.size() + i] = interrupt_action_[i];
                    }
                }
                if (policy_transition_active_ && !policy_transition_target_ready_) {
                    for (size_t i = 0; i < act_.size(); i++) {
                        policy_filtered_action_[i] = act_[i];
                        policy_transition_offset_[i] = policy_start_action_[i] - act_[i];
                    }
                    policy_transition_target_ready_ = true;
                }
                publish_action();
            }
            record_policy_sample(clamped_joint_count);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in inference thread: %s", e.what());
            handle_runtime_fault("inference_exception", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(),
                         "Unknown exception in inference thread");
            handle_runtime_fault("inference_exception", "unknown exception");
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
        RCLCPP_INFO(node->get_logger(), "Press 'RSB' to move to the policy default pose");
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
