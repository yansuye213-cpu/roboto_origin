// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/whole_body_mpc_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace whole_body_mpc {

namespace {

constexpr double kGravity = 9.80665;

std::string format_vector3(const Eigen::Vector3d& value) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(5)
       << "[" << value.x() << ", " << value.y() << ", " << value.z() << "]";
    return ss.str();
}

std::string format_joint_names(const std::vector<std::string>& names) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < names.size(); i++) {
        ss << names[i];
        if (i + 1 < names.size()) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

std::string format_pose(const Eigen::Isometry3d& pose) {
    const Eigen::Vector3d rpy = pose.linear().eulerAngles(0, 1, 2);
    std::ostringstream ss;
    ss << "xyz=" << format_vector3(pose.translation())
       << " rpy=" << format_vector3(rpy);
    return ss.str();
}

Eigen::Vector3d base_zyx_from_quaternion(const Eigen::Quaterniond& q_b2w) {
    Eigen::Quaterniond q = q_b2w;
    if (q.norm() <= 1.0e-9) {
        q = Eigen::Quaterniond::Identity();
    } else {
        q.normalize();
    }
    const Eigen::Matrix3d R = q.toRotationMatrix();
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return Eigen::Vector3d(yaw, pitch, roll);
}

Eigen::VectorXd vector_from_float_list(const std::vector<float>& values) {
    Eigen::VectorXd output(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        output[static_cast<int>(i)] = static_cast<double>(values[i]);
    }
    return output;
}

double clamp_abs(double value, double limit) {
    if (limit <= 0.0) {
        return 0.0;
    }
    return std::clamp(value, -limit, limit);
}

bool contains_token(const std::string& value, const std::string& token) {
    return value.find(token) != std::string::npos;
}

bool is_left_leg_joint_name(const std::string& name) {
    return contains_token(name, "left") &&
           (contains_token(name, "leg") || contains_token(name, "knee") ||
            contains_token(name, "ankle") || contains_token(name, "hip"));
}

bool is_right_leg_joint_name(const std::string& name) {
    return contains_token(name, "right") &&
           (contains_token(name, "leg") || contains_token(name, "knee") ||
            contains_token(name, "ankle") || contains_token(name, "hip"));
}

Eigen::Matrix3d skew(const Eigen::Vector3d& value) {
    Eigen::Matrix3d output;
    output << 0.0, -value.z(), value.y(),
              value.z(), 0.0, -value.x(),
             -value.y(), value.x(), 0.0;
    return output;
}

Eigen::Matrix<double, 6, 1> contact_wrench(
    const Eigen::Vector3d& com_position, const Vector3dList& positions,
    const Vector3dList& forces) {
    Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
    const size_t count = std::min(positions.size(), forces.size());
    for (size_t i = 0; i < count; i++) {
        wrench.head<3>() += forces[i];
        wrench.tail<3>() += (positions[i] - com_position).cross(forces[i]);
    }
    return wrench;
}

}  // namespace

WholeBodyMpcController::WholeBodyMpcController(const StandingStabilizer::Config& config)
    : config_(config) {
    validate_model_config();
    RobotModel::Config model_config;
    model_config.urdf_path = config_.whole_body_model_path;
    model_config.base_link = config_.whole_body_base_link;
    model_config.left_foot_frame = config_.whole_body_left_foot_link;
    model_config.right_foot_frame = config_.whole_body_right_foot_link;
    model_config.joint_order = config_.whole_body_joint_order;
    model_config.floating_base = true;
    robot_model_ = std::make_unique<RobotModel>(std::move(model_config));
    initialize_leg_joint_indices();

    BaseStateEstimator::Config state_estimator_config;
    state_estimator_config.enabled = config_.wbc_state_estimation_enabled;
    state_estimator_config.velocity_filter_alpha =
        config_.wbc_state_velocity_filter_alpha;
    state_estimator_config.max_base_linear_velocity =
        config_.wbc_state_max_base_linear_velocity;
    base_state_estimator_ =
        std::make_unique<BaseStateEstimator>(state_estimator_config);

    const Eigen::VectorXd q0 = robot_model_->neutral_configuration();
    const Eigen::VectorXd v0 = robot_model_->zero_velocity();
    neutral_kinematics_ = robot_model_->compute_kinematics(q0, v0);
    robot_mass_ = robot_model_->total_mass();
    if (robot_mass_ <= 0.0) {
        throw std::runtime_error("whole_body_mpc robot mass must be positive");
    }
    const Eigen::Vector3d neutral_foot_midpoint =
        0.5 * (foot_contact_center(neutral_kinematics_.left_foot_pose) +
               foot_contact_center(neutral_kinematics_.right_foot_pose));
    neutral_com_offset_xy_ =
        (neutral_kinematics_.com_position - neutral_foot_midpoint).head<2>();

    CentroidalMpc::Config mpc_config;
    mpc_config.enabled = config_.wbc_mpc_enabled;
    mpc_config.backend = config_.wbc_mpc_backend;
    mpc_config.mrt_enabled = config_.wbc_mpc_mrt_enabled;
    mpc_config.mrt_first_solve_blocking =
        config_.wbc_mpc_mrt_first_solve_blocking;
    mpc_config.mrt_max_policy_age = config_.wbc_mpc_mrt_max_policy_age;
    mpc_config.horizon = config_.wbc_mpc_horizon;
    mpc_config.dt = config_.wbc_mpc_dt;
    mpc_config.control_dt = config_.dt;
    mpc_config.orientation_weight = config_.wbc_mpc_orientation_weight;
    mpc_config.angular_rate_weight = config_.wbc_mpc_angular_rate_weight;
    mpc_config.com_weight = config_.wbc_mpc_com_weight;
    mpc_config.com_velocity_weight = config_.wbc_mpc_com_velocity_weight;
    mpc_config.terminal_weight_scale = config_.wbc_mpc_terminal_weight_scale;
    mpc_config.input_smooth_weight = config_.wbc_mpc_input_smooth_weight;
    mpc_config.force_weight = config_.wbc_mpc_force_weight;
    mpc_config.qp_iterations = config_.wbc_mpc_qp_iterations;
    mpc_config.terminal_cost_enabled = config_.wbc_mpc_terminal_cost_enabled;
    mpc_config.input_smoothing_enabled = config_.wbc_mpc_input_smoothing_enabled;
    mpc_config.contact_schedule_enabled = config_.wbc_mpc_contact_schedule_enabled;
    mpc_config.solver_constraints_enabled =
        config_.wbc_mpc_solver_constraints_enabled;
    mpc_config.zero_swing_force_constraint_enabled =
        config_.wbc_mpc_zero_swing_force_constraint_enabled;
    mpc_config.normal_force_constraint_enabled =
        config_.wbc_mpc_normal_force_constraint_enabled;
    mpc_config.delta_force_constraint_enabled =
        config_.wbc_mpc_delta_force_constraint_enabled;
    mpc_config.friction_cone_constraint_enabled =
        config_.wbc_mpc_friction_cone_constraint_enabled;
    mpc_config.stance_zero_velocity_constraint_enabled =
        config_.wbc_mpc_stance_zero_velocity_constraint_enabled;
    mpc_config.swing_normal_velocity_constraint_enabled =
        config_.wbc_mpc_swing_normal_velocity_constraint_enabled;
    mpc_config.swing_position_constraint_enabled =
        config_.wbc_mpc_swing_position_constraint_enabled;
    mpc_config.friction_barrier_mu = config_.wbc_mpc_friction_barrier_mu;
    mpc_config.friction_barrier_delta = config_.wbc_mpc_friction_barrier_delta;
    mpc_config.friction_regularization = config_.wbc_mpc_friction_regularization;
    mpc_config.max_angular_accel = config_.wbc_mpc_max_angular_accel;
    mpc_config.max_com_accel = config_.wbc_mpc_max_com_accel;
    mpc_config.max_contact_force_delta = config_.wbc_mpc_max_contact_force_delta;
    mpc_config.friction_coefficient = config_.wbc_friction_coefficient;
    mpc_config.min_normal_force = config_.wbc_min_normal_force;
    mpc_config.max_normal_force = config_.wbc_max_normal_force;
    mpc_config.base_height_weight = config_.wbc_mpc_base_height_weight;
    mpc_config.yaw_weight = config_.wbc_mpc_yaw_weight;
    mpc_config.joint_angle_weight = config_.wbc_mpc_joint_angle_weight;
    mpc_config.joint_velocity_weight = config_.wbc_mpc_joint_velocity_weight;
    mpc_config.swing_position_weight = config_.wbc_mpc_swing_position_weight;
    mpc_config.joint_velocity_limit = config_.wbc_swing_max_joint_velocity;
    mpc_config.swing_height = config_.wbc_step_recovery_swing_height;
    mpc_config.swing_time_scale = config_.wbc_mpc_swing_time_scale;
    mpc_config.swing_lift_off_velocity =
        config_.wbc_mpc_swing_lift_off_velocity;
    mpc_config.swing_touch_down_velocity =
        config_.wbc_mpc_swing_touch_down_velocity;
    mpc_config.target_roll = config_.wbc_target_roll;
    mpc_config.target_pitch = config_.wbc_target_pitch;
    mpc_config.model_path = config_.whole_body_model_path;
    mpc_config.left_foot_frame = config_.whole_body_left_foot_link;
    mpc_config.right_foot_frame = config_.whole_body_right_foot_link;
    mpc_config.joint_order = config_.whole_body_joint_order;
    mpc_config.nominal_joint_angles = config_.whole_body_nominal_joint_angles;
    mpc_config.joint_position_limits = config_.joint_limits;
    mpc_config.ad_model_folder = config_.wbc_mpc_ad_model_folder;
    mpc_config.ad_recompile = config_.wbc_mpc_ad_recompile;
    mpc_config.ad_verbose = config_.wbc_mpc_ad_verbose;
    centroidal_mpc_ = std::make_unique<CentroidalMpc>(mpc_config);

    ContactForceQp::Config qp_config;
    qp_config.iterations = config_.wbc_qp_iterations;
    qp_config.friction_coefficient = config_.wbc_friction_coefficient;
    qp_config.min_normal_force = config_.wbc_min_normal_force;
    qp_config.max_normal_force = config_.wbc_max_normal_force;
    qp_config.force_tracking_weight = config_.wbc_force_tracking_weight;
    qp_config.moment_tracking_weight = config_.wbc_moment_tracking_weight;
    qp_config.regularization_weight = config_.wbc_regularization_weight;
    qp_config.smooth_weight = config_.wbc_smooth_weight;
    contact_force_qp_ = std::make_unique<ContactForceQp>(qp_config);

    WholeBodyWbc::Config whole_body_wbc_config;
    whole_body_wbc_config.joint_num = config_.joint_num;
    whole_body_wbc_config.solver = config_.wbc_solver;
    whole_body_wbc_config.enabled = config_.wbc_whole_body_qp_enabled;
    whole_body_wbc_config.floating_base_eom_enabled =
        config_.wbc_floating_base_eom_enabled;
    whole_body_wbc_config.stance_contact_constraint_enabled =
        config_.wbc_stance_contact_constraint_enabled;
    whole_body_wbc_config.friction_constraint_enabled =
        config_.wbc_friction_constraint_enabled;
    whole_body_wbc_config.torque_limit_constraint_enabled =
        config_.wbc_torque_limit_constraint_enabled;
    whole_body_wbc_config.base_accel_task_enabled =
        config_.wbc_base_accel_task_enabled;
    whole_body_wbc_config.contact_force_task_enabled =
        config_.wbc_contact_force_task_enabled;
    whole_body_wbc_config.swing_task_enabled = config_.wbc_swing_task_enabled;
    whole_body_wbc_config.qddot_regularization_enabled =
        config_.wbc_qddot_regularization_enabled;
    whole_body_wbc_config.tau_regularization_enabled =
        config_.wbc_tau_regularization_enabled;
    whole_body_wbc_config.torque_enabled = config_.wbc_torque_enabled;
    whole_body_wbc_config.active_set_iterations =
        config_.wbc_active_set_iterations;
    whole_body_wbc_config.friction_coefficient =
        config_.wbc_friction_coefficient;
    whole_body_wbc_config.min_normal_force = config_.wbc_min_normal_force;
    whole_body_wbc_config.max_normal_force = config_.wbc_max_normal_force;
    whole_body_wbc_config.force_tracking_weight =
        config_.wbc_force_tracking_weight;
    whole_body_wbc_config.moment_tracking_weight =
        config_.wbc_moment_tracking_weight;
    whole_body_wbc_config.regularization_weight =
        config_.wbc_regularization_weight;
    whole_body_wbc_config.smooth_weight = config_.wbc_smooth_weight;
    whole_body_wbc_config.swing_tracking_weight =
        config_.wbc_swing_tracking_weight;
    whole_body_wbc_config.swing_kp = config_.wbc_swing_kp;
    whole_body_wbc_config.swing_kd = config_.wbc_swing_kd;
    whole_body_wbc_config.max_joint_torque = config_.wbc_max_joint_torque;
    whole_body_wbc_config.torque_joint_scale =
        config_.wbc_torque_joint_scale;
    whole_body_wbc_ = create_whole_body_wbc(std::move(whole_body_wbc_config));

    RecoveryGaitPlanner::Config gait_config;
    gait_config.enabled = config_.wbc_step_recovery_enabled;
    gait_config.step_placement_enabled = config_.wbc_step_placement_enabled;
    gait_config.roll_trigger = config_.wbc_step_recovery_roll_trigger;
    gait_config.pitch_trigger = config_.wbc_step_recovery_pitch_trigger;
    gait_config.rate_trigger = config_.wbc_step_recovery_rate_trigger;
    gait_config.com_trigger = config_.wbc_step_recovery_com_trigger;
    gait_config.com_velocity_trigger = config_.wbc_step_recovery_com_velocity_trigger;
    gait_config.return_roll = config_.wbc_step_recovery_return_roll;
    gait_config.return_pitch = config_.wbc_step_recovery_return_pitch;
    gait_config.return_rate = config_.wbc_step_recovery_return_rate;
    gait_config.return_com = config_.wbc_step_recovery_return_com;
    gait_config.return_com_velocity = config_.wbc_step_recovery_return_com_velocity;
    gait_config.step_count = config_.wbc_step_recovery_steps;
    gait_config.swing_time = config_.wbc_step_recovery_swing_time;
    gait_config.double_support_time = config_.wbc_step_recovery_double_support_time;
    gait_config.settle_time = config_.wbc_step_recovery_settle_time;
    gait_config.stable_time = config_.wbc_step_recovery_stable_time;
    gait_config.cooldown = config_.wbc_step_recovery_cooldown;
    gait_config.max_duration = config_.wbc_step_recovery_max_duration;
    gait_config.step_x_pitch_gain = config_.wbc_step_recovery_step_x_pitch_gain;
    gait_config.step_x_rate_gain = config_.wbc_step_recovery_step_x_rate_gain;
    gait_config.step_x_com_gain = config_.wbc_step_recovery_step_x_com_gain;
    gait_config.step_x_com_velocity_gain =
        config_.wbc_step_recovery_step_x_com_velocity_gain;
    gait_config.step_y_roll_gain = config_.wbc_step_recovery_step_y_roll_gain;
    gait_config.step_y_rate_gain = config_.wbc_step_recovery_step_y_rate_gain;
    gait_config.step_y_com_gain = config_.wbc_step_recovery_step_y_com_gain;
    gait_config.step_y_com_velocity_gain =
        config_.wbc_step_recovery_step_y_com_velocity_gain;
    gait_config.capture_time = config_.wbc_step_recovery_capture_time;
    gait_config.capture_gain = config_.wbc_step_recovery_capture_gain;
    gait_config.min_step_x = config_.wbc_step_recovery_min_step_x;
    gait_config.min_step_y = config_.wbc_step_recovery_min_step_y;
    gait_config.max_step_x = config_.wbc_step_recovery_max_step_x;
    gait_config.max_step_y = config_.wbc_step_recovery_max_step_y;
    gait_config.swing_height = config_.wbc_step_recovery_swing_height;
    gait_config.start_with_left = config_.wbc_step_recovery_start_with_left;
    gait_config.first_swing_left_on_positive_roll =
        config_.wbc_step_recovery_first_swing_left_on_positive_roll;
    gait_config.sagittal_sign = config_.wbc_step_recovery_sagittal_sign;
    gait_config.lateral_sign = config_.wbc_step_recovery_lateral_sign;
    recovery_gait_planner_ = std::make_unique<RecoveryGaitPlanner>(gait_config);

    ContactSchedulePlanner::Config schedule_config;
    schedule_config.enabled = config_.wbc_mpc_contact_schedule_enabled;
    schedule_config.horizon = config_.wbc_mpc_horizon;
    schedule_config.dt = config_.wbc_mpc_dt;
    schedule_config.double_support_time = config_.wbc_step_recovery_double_support_time;
    schedule_config.swing_time = config_.wbc_step_recovery_swing_time;
    schedule_config.settle_time = config_.wbc_step_recovery_settle_time;
    contact_schedule_planner_ =
        std::make_unique<ContactSchedulePlanner>(schedule_config);
}

void WholeBodyMpcController::reset() {
    if (base_state_estimator_) {
        base_state_estimator_->reset();
    }
    if (centroidal_mpc_) {
        centroidal_mpc_->reset();
    }
    if (contact_force_qp_) {
        contact_force_qp_->reset();
    }
    if (whole_body_wbc_) {
        whole_body_wbc_->reset();
    }
    if (recovery_gait_planner_) {
        recovery_gait_planner_->reset();
    }
    latest_state_estimate_ = BaseStateEstimator::Output{};
    state_estimator_left_contact_ = true;
    state_estimator_right_contact_ = true;
    has_smoothed_mpc_output_ = false;
    has_smoothed_body_moment_ = false;
    smoothed_mpc_angular_acceleration_.setZero();
    smoothed_mpc_com_acceleration_.setZero();
    smoothed_mpc_left_force_.setZero();
    smoothed_mpc_right_force_.setZero();
    smoothed_body_moment_.setZero();
}

std::vector<std::string> WholeBodyMpcController::diagnostics() const {
    std::vector<std::string> lines;
    lines.emplace_back(
        "whole_body_mpc model: urdf=" + config_.whole_body_model_path +
        " nq=" + std::to_string(robot_model_->nq()) +
        " nv=" + std::to_string(robot_model_->nv()) +
        " mass=" + std::to_string(robot_mass_) +
        " configured_joints=" + std::to_string(robot_model_->configured_joint_order().size()) +
        " model_1d_joints=" + std::to_string(robot_model_->model_joint_order().size()));
    lines.emplace_back(
        "whole_body_mpc frames: base=" + config_.whole_body_base_link +
        " left_foot=" + config_.whole_body_left_foot_link +
        " right_foot=" + config_.whole_body_right_foot_link);
    lines.emplace_back(
        "whole_body_mpc neutral_com: xyz=" + format_vector3(neutral_kinematics_.com_position));
    lines.emplace_back(
        "whole_body_mpc neutral_left_foot: " + format_pose(neutral_kinematics_.left_foot_pose));
    lines.emplace_back(
        "whole_body_mpc neutral_right_foot: " + format_pose(neutral_kinematics_.right_foot_pose));
    lines.emplace_back(
        "whole_body_mpc foot_jacobians: left=" +
        std::to_string(neutral_kinematics_.left_foot_jacobian.rows()) + "x" +
        std::to_string(neutral_kinematics_.left_foot_jacobian.cols()) +
        " right=" + std::to_string(neutral_kinematics_.right_foot_jacobian.rows()) + "x" +
        std::to_string(neutral_kinematics_.right_foot_jacobian.cols()));
    lines.emplace_back(
        "whole_body_mpc neutral_com_xy_offset_from_feet: " +
        format_vector3(Eigen::Vector3d(neutral_com_offset_xy_.x(), neutral_com_offset_xy_.y(), 0.0)));
    lines.emplace_back(
        "whole_body_mpc centroidal_mpc: horizon=" +
        std::to_string(config_.wbc_mpc_horizon) +
        " mpc_dt=" + std::to_string(config_.wbc_mpc_dt) +
        " control_dt=" + std::to_string(config_.dt) +
        " enabled=" + std::string(config_.wbc_mpc_enabled ? "true" : "false") +
        " backend=" + centroidal_mpc_->backend_name() +
        " mrt=" +
        std::string(config_.wbc_mpc_mrt_enabled ? "enabled" : "disabled") +
        " first_blocking=" +
        std::string(config_.wbc_mpc_mrt_first_solve_blocking ? "true" : "false") +
        " max_policy_age=" +
        std::to_string(config_.wbc_mpc_mrt_max_policy_age) +
        " terminal_cost=" +
        std::string(config_.wbc_mpc_terminal_cost_enabled ? "true" : "false") +
        " input_smoothing=" +
        std::string(config_.wbc_mpc_input_smoothing_enabled ? "true" : "false") +
        " contact_schedule=" +
        std::string(config_.wbc_mpc_contact_schedule_enabled ? "true" : "false") +
        " constraints=" +
        std::string(config_.wbc_mpc_solver_constraints_enabled ? "true" : "false") +
        " qp_iterations=" + std::to_string(config_.wbc_mpc_qp_iterations));
    lines.emplace_back(
        "whole_body_mpc centroidal_mpc_weights: horizon=" +
        std::to_string(config_.wbc_mpc_horizon) +
        " target_rp=[" + std::to_string(config_.wbc_target_roll) + ", " +
        std::to_string(config_.wbc_target_pitch) + "]" +
        " max_accel=[" + std::to_string(config_.wbc_mpc_max_angular_accel) + ", " +
        std::to_string(config_.wbc_mpc_max_com_accel) + "]" +
        " force_weight=" + std::to_string(config_.wbc_mpc_force_weight) +
        " joint_angle_weight=" +
        std::to_string(config_.wbc_mpc_joint_angle_weight) +
        " joint_velocity_weight=" +
        std::to_string(config_.wbc_mpc_joint_velocity_weight) +
        " max_force_delta=" +
        std::to_string(config_.wbc_mpc_max_contact_force_delta) +
        " output_signs=[" + std::to_string(config_.wbc_mpc_output_roll_sign) +
        ", " + std::to_string(config_.wbc_mpc_output_pitch_sign) + "]" +
        " output_scales=[" + std::to_string(config_.wbc_mpc_output_roll_scale) +
        ", " + std::to_string(config_.wbc_mpc_output_pitch_scale) + "]");
    lines.emplace_back(
        "whole_body_mpc body_moment_limit: max=" +
        std::to_string(config_.wbc_max_body_moment) +
        " rate=" + std::to_string(config_.wbc_body_moment_rate_limit) +
        " filter_weight=" +
        std::to_string(config_.wbc_body_moment_filter_weight));
    lines.emplace_back(
        "whole_body_mpc ocs2_full_centroidal: ad_folder=" +
        config_.wbc_mpc_ad_model_folder +
        " ad_recompile=" +
        std::string(config_.wbc_mpc_ad_recompile ? "true" : "false") +
        " ad_verbose=" +
        std::string(config_.wbc_mpc_ad_verbose ? "true" : "false") +
        " foot_contacts=[3dof:" + config_.whole_body_left_foot_link + ", " +
        config_.whole_body_right_foot_link + "]");
    lines.emplace_back(
        "whole_body_mpc centroidal_mpc_constraints: zero_swing_force=" +
        std::string(config_.wbc_mpc_zero_swing_force_constraint_enabled ? "true" : "false") +
        " normal_force=" +
        std::string(config_.wbc_mpc_normal_force_constraint_enabled ? "true" : "false") +
        " delta_force=" +
        std::string(config_.wbc_mpc_delta_force_constraint_enabled ? "true" : "false") +
        " friction_cone=" +
        std::string(config_.wbc_mpc_friction_cone_constraint_enabled ? "true" : "false") +
        " stance_zero_velocity=" +
        std::string(config_.wbc_mpc_stance_zero_velocity_constraint_enabled ? "true" : "false") +
        " swing_normal_velocity=" +
        std::string(config_.wbc_mpc_swing_normal_velocity_constraint_enabled ? "true" : "false") +
        " swing_position=" +
        std::string(config_.wbc_mpc_swing_position_constraint_enabled ? "true" : "false") +
        " barrier=[" + std::to_string(config_.wbc_mpc_friction_barrier_mu) + ", " +
        std::to_string(config_.wbc_mpc_friction_barrier_delta) + "]" +
        " friction_regularization=" +
        std::to_string(config_.wbc_mpc_friction_regularization));
    lines.emplace_back(
        "whole_body_mpc state_estimation: " +
        std::string(config_.wbc_state_estimation_enabled ? "enabled" : "disabled") +
        " velocity_filter_alpha=" +
        std::to_string(config_.wbc_state_velocity_filter_alpha) +
        " max_base_linear_velocity=" +
        std::to_string(config_.wbc_state_max_base_linear_velocity));
    lines.emplace_back(
        "whole_body_mpc contacts: virtual_foot_corners=" +
        std::string((config_.wbc_virtual_foot_corners_enabled &&
                     config_.wbc_foot_half_length > 0.0f &&
                     config_.wbc_foot_half_width > 0.0f) ? "enabled" : "disabled") +
        " half_length=" + std::to_string(config_.wbc_foot_half_length) +
        " half_width=" + std::to_string(config_.wbc_foot_half_width) +
        " center_x=" + std::to_string(config_.wbc_foot_center_x) +
        " contact_z=" + std::to_string(config_.wbc_foot_contact_z));
    lines.emplace_back(
        "whole_body_mpc qp_switches: contact_force_qp=" +
        std::string(config_.wbc_contact_force_qp_enabled ? "enabled" : "disabled") +
        " whole_body_qp=" +
        std::string(config_.wbc_whole_body_qp_enabled ? "enabled" : "disabled") +
        " solver=" + (whole_body_wbc_ ? whole_body_wbc_->name() : std::string("none")) +
        " floating_base_eom=" +
        std::string(config_.wbc_floating_base_eom_enabled ? "enabled" : "disabled") +
        " stance_contact=" +
        std::string(config_.wbc_stance_contact_constraint_enabled ? "enabled" : "disabled") +
        " friction=" +
        std::string(config_.wbc_friction_constraint_enabled ? "enabled" : "disabled") +
        " torque_limit=" +
        std::string(config_.wbc_torque_limit_constraint_enabled ? "enabled" : "disabled"));
    lines.emplace_back(
        "whole_body_mpc task_switches: base_accel=" +
        std::string(config_.wbc_base_accel_task_enabled ? "enabled" : "disabled") +
        " contact_force=" +
        std::string(config_.wbc_contact_force_task_enabled ? "enabled" : "disabled") +
        " swing=" +
        std::string(config_.wbc_swing_task_enabled ? "enabled" : "disabled") +
        " qddot_reg=" +
        std::string(config_.wbc_qddot_regularization_enabled ? "enabled" : "disabled") +
        " tau_reg=" +
        std::string(config_.wbc_tau_regularization_enabled ? "enabled" : "disabled") +
        " swing_ik=" +
        std::string(config_.wbc_swing_ik_enabled ? "enabled" : "disabled") +
        " mpc_joint_command=" +
        std::string(config_.wbc_mpc_joint_command_enabled ? "enabled" : "disabled"));
    lines.emplace_back(
        "whole_body_mpc torque_output: " +
        std::string(config_.wbc_torque_enabled ? "enabled" : "disabled") +
        " max_joint_torque=" + std::to_string(config_.wbc_max_joint_torque) +
        " active_set_iterations=" + std::to_string(config_.wbc_active_set_iterations));
    lines.emplace_back(
        "whole_body_mpc step_recovery: " +
        std::string(config_.wbc_step_recovery_enabled ? "enabled" : "disabled") +
        " step_placement=" +
        std::string(config_.wbc_step_placement_enabled ? "enabled" : "disabled") +
        " triggers=[roll " + std::to_string(config_.wbc_step_recovery_roll_trigger) +
        ", pitch " + std::to_string(config_.wbc_step_recovery_pitch_trigger) +
        ", rate " + std::to_string(config_.wbc_step_recovery_rate_trigger) +
        ", com " + std::to_string(config_.wbc_step_recovery_com_trigger) +
        ", com_vel " + std::to_string(config_.wbc_step_recovery_com_velocity_trigger) +
        "] steps=" + std::to_string(config_.wbc_step_recovery_steps) +
        " swing_time=" + std::to_string(config_.wbc_step_recovery_swing_time) +
        " swing_height=" + std::to_string(config_.wbc_step_recovery_swing_height));
    lines.emplace_back(
        "whole_body_mpc step_gains: x[pitch=" +
        std::to_string(config_.wbc_step_recovery_step_x_pitch_gain) +
        ", rate=" + std::to_string(config_.wbc_step_recovery_step_x_rate_gain) +
        ", com=" + std::to_string(config_.wbc_step_recovery_step_x_com_gain) +
        ", com_vel=" +
        std::to_string(config_.wbc_step_recovery_step_x_com_velocity_gain) +
        "] y[roll=" + std::to_string(config_.wbc_step_recovery_step_y_roll_gain) +
        ", rate=" + std::to_string(config_.wbc_step_recovery_step_y_rate_gain) +
        ", com=" + std::to_string(config_.wbc_step_recovery_step_y_com_gain) +
        ", com_vel=" +
        std::to_string(config_.wbc_step_recovery_step_y_com_velocity_gain) +
        "] capture=[time=" +
        std::to_string(config_.wbc_step_recovery_capture_time) +
        ", gain=" + std::to_string(config_.wbc_step_recovery_capture_gain) + "]");
    lines.emplace_back(
        "whole_body_mpc swing_tracking: weight=" +
        std::to_string(config_.wbc_swing_tracking_weight) +
        " kp=" + std::to_string(config_.wbc_swing_kp) +
        " kd=" + std::to_string(config_.wbc_swing_kd) +
        " ik_gain=" + std::to_string(config_.wbc_swing_ik_gain) +
        " left_leg_joints=" + std::to_string(left_leg_joint_indices_.size()) +
        " right_leg_joints=" + std::to_string(right_leg_joint_indices_.size()));
    lines.emplace_back(
        "whole_body_mpc configured_joint_order: " +
        format_joint_names(robot_model_->configured_joint_order()));
    if (robot_model_->model_joint_order() != robot_model_->configured_joint_order()) {
        lines.emplace_back(
            "whole_body_mpc model_joint_order: " +
            format_joint_names(robot_model_->model_joint_order()));
        lines.emplace_back(
            "whole_body_mpc note: configured_joint_order differs from Pinocchio model order; "
            "runtime joint_q/joint_vel must follow configured_joint_order.");
    }
    return lines;
}

StandingStabilizer::Command WholeBodyMpcController::apply(
    const StandingStabilizer::Measurement& measurement, float blend,
    const std::vector<float>& base_target, const std::vector<float>& kp,
    const std::vector<float>& kd, const std::vector<float>& current_joint_position,
    const std::vector<float>& current_joint_velocity) {
    (void)blend;
    const Eigen::Quaterniond base_orientation(
        measurement.qw, measurement.qx, measurement.qy, measurement.qz);
    const Eigen::VectorXd q = robot_model_->make_configuration(
        current_joint_position, Eigen::Vector3d::Zero(), base_orientation);
    const Eigen::VectorXd v_without_base_linear = robot_model_->make_velocity(
        current_joint_velocity, Eigen::Vector3d::Zero(),
        Eigen::Vector3d(measurement.wx, measurement.wy, 0.0));
    const RobotModel::Kinematics seed_kinematics =
        robot_model_->compute_kinematics(q, v_without_base_linear);
    BaseStateEstimator::Input state_input;
    state_input.kinematics = &seed_kinematics;
    state_input.generalized_velocity_without_base_linear = &v_without_base_linear;
    state_input.left_contact = state_estimator_left_contact_;
    state_input.right_contact = state_estimator_right_contact_;
    latest_state_estimate_ = base_state_estimator_->estimate(state_input);

    const Eigen::VectorXd v = robot_model_->make_velocity(
        current_joint_velocity, latest_state_estimate_.base_linear_velocity,
        Eigen::Vector3d(measurement.wx, measurement.wy, 0.0));
    latest_kinematics_ = robot_model_->compute_kinematics(q, v);
    const RecoveryGaitPlanner::Reference gait_reference =
        build_gait_reference(measurement, latest_kinematics_);
    const ContactSchedule contact_schedule =
        build_contact_schedule(latest_kinematics_, gait_reference);
    state_estimator_left_contact_ = gait_reference.left_contact;
    state_estimator_right_contact_ = gait_reference.right_contact;
    const ContactPointSet contacts =
        build_contact_point_set(latest_kinematics_, gait_reference);

    StandingStabilizer::Command command;
    command.position = base_target;
    command.velocity.assign(config_.joint_num, 0.0f);
    command.kp = kp;
    command.kd = kd;
    apply_swing_ik_targets(latest_kinematics_, gait_reference, current_joint_position,
                           command.position, command.velocity);

    StandingStabilizer::Correction correction;
    const CentroidalMpc::Input mpc_input =
        build_centroidal_mpc_input(measurement, contacts, gait_reference,
                                   contact_schedule, current_joint_position,
                                   current_joint_velocity);
    correction.wbc_com_error_x = static_cast<float>(mpc_input.com_offset_error.x());
    correction.wbc_com_error_y = static_cast<float>(mpc_input.com_offset_error.y());
    const auto mpc_solve_start = std::chrono::steady_clock::now();
    CentroidalMpc::Output mpc_output = centroidal_mpc_->solve(mpc_input);
    const auto mpc_solve_end = std::chrono::steady_clock::now();
    if (mpc_output.has_desired_contact_forces) {
        mpc_output.desired_angular_acceleration.x() *=
            static_cast<double>(config_.wbc_mpc_output_roll_scale);
        mpc_output.desired_angular_acceleration.y() *=
            static_cast<double>(config_.wbc_mpc_output_pitch_scale);
    } else {
        mpc_output.desired_angular_acceleration.x() *=
            static_cast<double>(config_.wbc_mpc_output_roll_sign) *
            static_cast<double>(config_.wbc_mpc_output_roll_scale);
        mpc_output.desired_angular_acceleration.y() *=
            static_cast<double>(config_.wbc_mpc_output_pitch_sign) *
            static_cast<double>(config_.wbc_mpc_output_pitch_scale);
    }
    mpc_output.control[0] = mpc_output.desired_angular_acceleration.x();
    mpc_output.control[1] = mpc_output.desired_angular_acceleration.y();
    smooth_mpc_output(mpc_output);
    correction.wbc_mpc_backend = mpc_output.backend;
    correction.wbc_mpc_used = mpc_output.solved;
    correction.wbc_mpc_iterations = mpc_output.iterations;
    correction.wbc_mpc_solve_us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            mpc_solve_end - mpc_solve_start).count());
    correction.wbc_mpc_objective = static_cast<float>(mpc_output.objective);
    correction.mpc_roll_accel =
        static_cast<float>(mpc_output.desired_angular_acceleration.x());
    correction.mpc_pitch_accel =
        static_cast<float>(mpc_output.desired_angular_acceleration.y());
    correction.mpc_com_accel_x =
        static_cast<float>(mpc_output.desired_com_acceleration.x());
    correction.mpc_com_accel_y =
        static_cast<float>(mpc_output.desired_com_acceleration.y());
    correction.mpc_left_force_x =
        static_cast<float>(mpc_output.desired_left_contact_force.x());
    correction.mpc_left_force_y =
        static_cast<float>(mpc_output.desired_left_contact_force.y());
    correction.mpc_left_force_z =
        static_cast<float>(mpc_output.desired_left_contact_force.z());
    correction.mpc_right_force_x =
        static_cast<float>(mpc_output.desired_right_contact_force.x());
    correction.mpc_right_force_y =
        static_cast<float>(mpc_output.desired_right_contact_force.y());
    correction.mpc_right_force_z =
        static_cast<float>(mpc_output.desired_right_contact_force.z());
    correction.wbc_mpc_force_target_used = mpc_output.has_desired_contact_forces;
    apply_mpc_joint_command(mpc_output, current_joint_position,
                            command.position, command.velocity, correction);

    ContactForceQp::Input qp_input = build_contact_qp_input(mpc_output, contacts);
    const ContactForceQp::Result contact_result =
        config_.wbc_contact_force_qp_enabled
            ? contact_force_qp_->solve(qp_input)
            : make_nominal_contact_result(qp_input);
    correction.wbc_contact_force_qp_used = config_.wbc_contact_force_qp_enabled;
    correction.wbc_left_normal_force = static_cast<float>(contact_result.left_force.z());
    correction.wbc_right_normal_force = static_cast<float>(contact_result.right_force.z());
    correction.wbc_roll_moment = static_cast<float>(qp_input.desired_body_moment.x());
    correction.wbc_pitch_moment = static_cast<float>(qp_input.desired_body_moment.y());
    correction.wbc_achieved_roll_moment =
        static_cast<float>(contact_result.achieved_wrench.tail<3>().x());
    correction.wbc_achieved_pitch_moment =
        static_cast<float>(contact_result.achieved_wrench.tail<3>().y());

    WholeBodyWbc::Input wbc_input;
    wbc_input.kinematics = &latest_kinematics_;
    wbc_input.contacts = &contacts;
    wbc_input.gait_reference = &gait_reference;
    wbc_input.mpc_output = &mpc_output;
    wbc_input.contact_result = &contact_result;
    wbc_input.configured_joint_velocity_indices =
        &robot_model_->configured_joint_velocity_indices();
    wbc_input.blend = blend;
    const auto wbc_solve_start = std::chrono::steady_clock::now();
    const WholeBodyWbc::Result wbc_result =
        whole_body_wbc_ ? whole_body_wbc_->solve(wbc_input)
                        : WholeBodyWbc::Result{};
    const auto wbc_solve_end = std::chrono::steady_clock::now();
    correction.wbc_whole_body_solve_us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            wbc_solve_end - wbc_solve_start).count());
    correction.wbc_whole_body_qp_used = config_.wbc_whole_body_qp_enabled;
    correction.qp_used = correction.wbc_whole_body_qp_used;
    correction.wbc_left_normal_force = static_cast<float>(wbc_result.left_force.z());
    correction.wbc_right_normal_force = static_cast<float>(wbc_result.right_force.z());
    correction.wbc_achieved_roll_moment =
        static_cast<float>(wbc_result.achieved_wrench.tail<3>().x());
    correction.wbc_achieved_pitch_moment =
        static_cast<float>(wbc_result.achieved_wrench.tail<3>().y());
    correction.wbc_max_raw_joint_torque =
        static_cast<float>(wbc_result.max_raw_joint_torque);
    correction.wbc_max_joint_torque =
        static_cast<float>(wbc_result.max_command_joint_torque);
    correction.wbc_saturated_joint_count = wbc_result.saturated_joint_count;
    correction.wbc_max_torque_joint_index = wbc_result.max_torque_joint_index;
    correction.wbc_qp_violation =
        static_cast<float>(wbc_result.max_qp_violation);
    correction.wbc_dynamics_residual =
        static_cast<float>(wbc_result.dynamics_residual);
    correction.wbc_swing_error =
        static_cast<float>(wbc_result.swing_error);
    correction.wbc_base_velocity_x =
        static_cast<float>(latest_state_estimate_.base_linear_velocity.x());
    correction.wbc_base_velocity_y =
        static_cast<float>(latest_state_estimate_.base_linear_velocity.y());
    correction.wbc_base_velocity_z =
        static_cast<float>(latest_state_estimate_.base_linear_velocity.z());
    correction.wbc_com_velocity_x =
        static_cast<float>(latest_kinematics_.com_velocity.x());
    correction.wbc_com_velocity_y =
        static_cast<float>(latest_kinematics_.com_velocity.y());
    correction.wbc_state_estimator_residual =
        static_cast<float>(latest_state_estimate_.stance_velocity_residual);
    correction.wbc_state_estimator_rows = latest_state_estimate_.constraint_rows;
    correction.wbc_state_estimator_used =
        latest_state_estimate_.used_contact_constraint;
    correction.wbc_step_x = static_cast<float>(gait_reference.step_x);
    correction.wbc_step_y = static_cast<float>(gait_reference.step_y);
    correction.wbc_contact_count = wbc_result.contact_count;
    correction.wbc_active_constraints = wbc_result.active_constraint_count;
    correction.wbc_step_recovery_active = gait_reference.recovery_active;
    correction.wbc_step_phase = static_cast<int>(gait_reference.phase);
    correction.wbc_steps_completed = gait_reference.steps_completed;
    correction.wbc_steps_planned = gait_reference.planned_steps;
    correction.wbc_left_contact = gait_reference.left_contact;
    correction.wbc_right_contact = gait_reference.right_contact;
    correction.wbc_left_swing = gait_reference.left_swing;
    correction.wbc_right_swing = gait_reference.right_swing;
    command.tau = wbc_result.tau;
    command.correction = correction;
    return command;
}

void WholeBodyMpcController::smooth_mpc_output(
    CentroidalMpc::Output& mpc_output) {
    if (!config_.wbc_mpc_input_smoothing_enabled ||
        config_.wbc_mpc_input_smooth_weight <= 0.0f ||
        !mpc_output.has_desired_contact_forces) {
        if (!mpc_output.has_desired_contact_forces) {
            has_smoothed_mpc_output_ = false;
        }
        return;
    }

    const double alpha =
        1.0 / (1.0 + static_cast<double>(config_.wbc_mpc_input_smooth_weight));
    if (!has_smoothed_mpc_output_) {
        smoothed_mpc_angular_acceleration_ =
            mpc_output.desired_angular_acceleration;
        smoothed_mpc_com_acceleration_ = mpc_output.desired_com_acceleration;
        smoothed_mpc_left_force_ = mpc_output.desired_left_contact_force;
        smoothed_mpc_right_force_ = mpc_output.desired_right_contact_force;
        has_smoothed_mpc_output_ = true;
    } else {
        smoothed_mpc_angular_acceleration_ =
            (1.0 - alpha) * smoothed_mpc_angular_acceleration_ +
            alpha * mpc_output.desired_angular_acceleration;
        smoothed_mpc_com_acceleration_ =
            (1.0 - alpha) * smoothed_mpc_com_acceleration_ +
            alpha * mpc_output.desired_com_acceleration;
        smoothed_mpc_left_force_ =
            (1.0 - alpha) * smoothed_mpc_left_force_ +
            alpha * mpc_output.desired_left_contact_force;
        smoothed_mpc_right_force_ =
            (1.0 - alpha) * smoothed_mpc_right_force_ +
            alpha * mpc_output.desired_right_contact_force;
    }

    mpc_output.desired_angular_acceleration =
        smoothed_mpc_angular_acceleration_;
    mpc_output.desired_com_acceleration = smoothed_mpc_com_acceleration_;
    mpc_output.desired_left_contact_force = smoothed_mpc_left_force_;
    mpc_output.desired_right_contact_force = smoothed_mpc_right_force_;
    mpc_output.control[0] = mpc_output.desired_angular_acceleration.x();
    mpc_output.control[1] = mpc_output.desired_angular_acceleration.y();
    mpc_output.control[2] = mpc_output.desired_com_acceleration.x();
    mpc_output.control[3] = mpc_output.desired_com_acceleration.y();
}

Eigen::Vector3d WholeBodyMpcController::foot_contact_center(
    const Eigen::Isometry3d& foot_pose) const {
    return foot_pose.translation() +
           foot_pose.linear() *
               Eigen::Vector3d(static_cast<double>(config_.wbc_foot_center_x), 0.0,
                               static_cast<double>(config_.wbc_foot_contact_z));
}

CentroidalMpc::Input WholeBodyMpcController::build_centroidal_mpc_input(
    const StandingStabilizer::Measurement& measurement,
    const ContactPointSet& contacts,
    const RecoveryGaitPlanner::Reference& gait_reference,
    const ContactSchedule& contact_schedule,
    const std::vector<float>& current_joint_position,
    const std::vector<float>& current_joint_velocity) const {
    Eigen::Vector3d support_center = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& position : contacts.positions) {
        support_center += position;
    }
    support_center /= static_cast<double>(std::max<size_t>(contacts.positions.size(), 1));
    const Eigen::Vector2d current_com_offset_xy =
        (latest_kinematics_.com_position - support_center).head<2>();

    CentroidalMpc::Input input;
    input.roll = measurement.roll;
    input.pitch = measurement.pitch;
    input.wx = measurement.wx;
    input.wy = measurement.wy;
    input.com_offset_error = current_com_offset_xy - neutral_com_offset_xy_;
    input.com_velocity = latest_kinematics_.com_velocity.head<2>();
    input.com_position = latest_kinematics_.com_position;
    input.support_center = support_center;
    input.left_foot_position = foot_contact_center(latest_kinematics_.left_foot_pose);
    input.right_foot_position = foot_contact_center(latest_kinematics_.right_foot_pose);
    input.neutral_com_offset << neutral_com_offset_xy_.x(), neutral_com_offset_xy_.y(), 0.0;
    const Eigen::Quaterniond base_orientation(
        measurement.qw, measurement.qx, measurement.qy, measurement.qz);
    Eigen::Quaterniond normalized_base_orientation = base_orientation;
    if (normalized_base_orientation.norm() <= 1.0e-9) {
        normalized_base_orientation = Eigen::Quaterniond::Identity();
    } else {
        normalized_base_orientation.normalize();
    }
    input.base_position = Eigen::Vector3d::Zero();
    input.base_orientation_zyx =
        base_zyx_from_quaternion(normalized_base_orientation);
    input.base_linear_velocity = latest_state_estimate_.base_linear_velocity;
    input.base_angular_velocity =
        normalized_base_orientation *
        Eigen::Vector3d(measurement.wx, measurement.wy, 0.0);
    input.joint_position = vector_from_float_list(current_joint_position);
    input.joint_velocity = vector_from_float_list(current_joint_velocity);
    input.mass = robot_mass_;
    if (latest_kinematics_.mass_matrix.rows() >= 6 &&
        latest_kinematics_.mass_matrix.cols() >= 6) {
        const Eigen::Matrix3d base_inertia =
            latest_kinematics_.mass_matrix.block<3, 3>(3, 3);
        input.roll_inertia = std::max(std::abs(base_inertia(0, 0)), 1.0e-3);
        input.pitch_inertia = std::max(std::abs(base_inertia(1, 1)), 1.0e-3);
    }
    input.left_contact = gait_reference.left_contact;
    input.right_contact = gait_reference.right_contact;
    input.contact_schedule = contact_schedule;
    return input;
}

ContactSchedule WholeBodyMpcController::build_contact_schedule(
    const RobotModel::Kinematics& kinematics,
    const RecoveryGaitPlanner::Reference& gait_reference) const {
    ContactSchedulePlanner::Input input;
    input.gait_reference = gait_reference;
    input.left_foot_position = foot_contact_center(kinematics.left_foot_pose);
    input.right_foot_position = foot_contact_center(kinematics.right_foot_pose);
    return contact_schedule_planner_->build(input);
}

RecoveryGaitPlanner::Reference WholeBodyMpcController::build_gait_reference(
    const StandingStabilizer::Measurement& measurement,
    const RobotModel::Kinematics& kinematics) {
    RecoveryGaitPlanner::Input input;
    input.dt = config_.dt;
    input.roll = measurement.roll;
    input.pitch = measurement.pitch;
    input.wx = measurement.wx;
    input.wy = measurement.wy;
    input.target_roll = config_.wbc_target_roll;
    input.target_pitch = config_.wbc_target_pitch;
    input.left_foot_position = foot_contact_center(kinematics.left_foot_pose);
    input.right_foot_position = foot_contact_center(kinematics.right_foot_pose);
    input.com_position = kinematics.com_position;
    input.com_velocity = kinematics.com_velocity;
    input.support_center =
        0.5 * (input.left_foot_position + input.right_foot_position);
    return recovery_gait_planner_->update(input);
}

WholeBodyMpcController::ContactPointSet WholeBodyMpcController::build_contact_point_set(
    const RobotModel::Kinematics& kinematics,
    const RecoveryGaitPlanner::Reference& gait_reference) const {
    ContactPointSet contacts;
    const int nv = robot_model_->nv();
    const bool use_corners =
        config_.wbc_virtual_foot_corners_enabled &&
        config_.wbc_foot_half_length > 0.0f && config_.wbc_foot_half_width > 0.0f;
    const int contacts_per_foot = use_corners ? 4 : 1;
    contacts.positions.reserve(contacts_per_foot * 2);
    contacts.jacobian.setZero(contacts_per_foot * 2 * 3, nv);
    contacts.jacobian_dot_v.setZero(contacts_per_foot * 2 * 3);

    const auto add_foot_contacts =
        [&](const Eigen::Isometry3d& foot_pose,
            const Eigen::Matrix<double, 6, Eigen::Dynamic>& foot_jacobian,
            const Eigen::Matrix<double, 6, 1>& foot_jdot_v) {
            Vector3dList local_points;
            if (use_corners) {
                const double cx = config_.wbc_foot_center_x;
                const double hx = config_.wbc_foot_half_length;
                const double hy = config_.wbc_foot_half_width;
                const double z = config_.wbc_foot_contact_z;
                local_points.emplace_back(cx + hx, hy, z);
                local_points.emplace_back(cx + hx, -hy, z);
                local_points.emplace_back(cx - hx, hy, z);
                local_points.emplace_back(cx - hx, -hy, z);
            } else {
                local_points.emplace_back(0.0, 0.0, 0.0);
            }

            for (const Eigen::Vector3d& local_point : local_points) {
                const int point_index = static_cast<int>(contacts.positions.size());
                const Eigen::Vector3d world_offset = foot_pose.linear() * local_point;
                contacts.positions.push_back(foot_pose.translation() + world_offset);
                contacts.jacobian.block(point_index * 3, 0, 3, nv) =
                    foot_jacobian.topRows<3>() -
                    skew(world_offset) * foot_jacobian.bottomRows<3>();
                contacts.jacobian_dot_v.segment<3>(point_index * 3) =
                    foot_jdot_v.head<3>() - skew(world_offset) * foot_jdot_v.tail<3>();
            }
        };

    if (gait_reference.left_contact) {
        add_foot_contacts(kinematics.left_foot_pose, kinematics.left_foot_jacobian,
                          kinematics.left_foot_jacobian_dot_v);
        contacts.left_contact_count = static_cast<int>(contacts.positions.size());
    } else {
        contacts.left_contact_count = 0;
    }
    if (gait_reference.right_contact) {
        add_foot_contacts(kinematics.right_foot_pose, kinematics.right_foot_jacobian,
                          kinematics.right_foot_jacobian_dot_v);
    }
    if (contacts.positions.empty()) {
        throw std::runtime_error("whole_body_mpc gait reference has no stance contacts");
    }
    contacts.jacobian.conservativeResize(static_cast<int>(contacts.positions.size()) * 3, nv);
    contacts.jacobian_dot_v.conservativeResize(static_cast<int>(contacts.positions.size()) * 3);
    return contacts;
}

ContactForceQp::Input WholeBodyMpcController::build_contact_qp_input(
    const CentroidalMpc::Output& mpc_output,
    const ContactPointSet& contacts) {
    ContactForceQp::Input input;
    input.mass = robot_mass_;
    input.com_position = latest_kinematics_.com_position;
    input.left_foot_position = latest_kinematics_.left_foot_pose.translation();
    input.right_foot_position = latest_kinematics_.right_foot_pose.translation();
    input.contact_positions = contacts.positions;
    input.left_contact_count = contacts.left_contact_count;
    input.desired_com_acceleration = mpc_output.desired_com_acceleration;
    input.desired_left_force = mpc_output.desired_left_contact_force;
    input.desired_right_force = mpc_output.desired_right_contact_force;
    input.desired_foot_forces_available =
        mpc_output.has_desired_contact_forces;

    Eigen::Vector3d desired_body_moment = Eigen::Vector3d::Zero();
    if (latest_kinematics_.mass_matrix.rows() >= 6 &&
        latest_kinematics_.mass_matrix.cols() >= 6) {
        desired_body_moment =
            latest_kinematics_.mass_matrix.block<3, 3>(3, 3) *
            mpc_output.desired_angular_acceleration;
    }
    Eigen::Vector3d limited_body_moment(
        clamp_abs(desired_body_moment.x(), config_.wbc_max_body_moment),
        clamp_abs(desired_body_moment.y(), config_.wbc_max_body_moment),
        0.0);
    input.desired_body_moment = smooth_body_moment_target(limited_body_moment);
    return input;
}

Eigen::Vector3d WholeBodyMpcController::smooth_body_moment_target(
    const Eigen::Vector3d& target) {
    const bool filter_enabled =
        config_.wbc_body_moment_filter_weight > 0.0f;
    const bool rate_limit_enabled =
        config_.wbc_body_moment_rate_limit > 0.0f &&
        config_.dt > 0.0f;
    if (!filter_enabled && !rate_limit_enabled) {
        has_smoothed_body_moment_ = false;
        smoothed_body_moment_ = target;
        return target;
    }

    if (!has_smoothed_body_moment_) {
        smoothed_body_moment_.setZero();
        has_smoothed_body_moment_ = true;
    }

    Eigen::Vector3d filtered_target = target;
    if (filter_enabled) {
        const double alpha =
            1.0 / (1.0 + static_cast<double>(
                             config_.wbc_body_moment_filter_weight));
        filtered_target =
            (1.0 - alpha) * smoothed_body_moment_ + alpha * target;
    }

    Eigen::Vector3d output = filtered_target;
    if (rate_limit_enabled) {
        const double max_delta =
            static_cast<double>(config_.wbc_body_moment_rate_limit) *
            static_cast<double>(config_.dt);
        const Eigen::Vector3d delta = filtered_target - smoothed_body_moment_;
        output = smoothed_body_moment_ +
                 Eigen::Vector3d(clamp_abs(delta.x(), max_delta),
                                 clamp_abs(delta.y(), max_delta),
                                 0.0);
    }

    output.x() = clamp_abs(output.x(), config_.wbc_max_body_moment);
    output.y() = clamp_abs(output.y(), config_.wbc_max_body_moment);
    output.z() = 0.0;
    smoothed_body_moment_ = output;
    return output;
}

ContactForceQp::Result WholeBodyMpcController::make_nominal_contact_result(
    const ContactForceQp::Input& input) const {
    ContactForceQp::Result result;
    const int contact_count = static_cast<int>(input.contact_positions.size());
    if (contact_count <= 0 || input.mass <= 0.0) {
        return result;
    }

    result.contact_forces.resize(contact_count, Eigen::Vector3d::Zero());
    const int right_contact_count = contact_count - input.left_contact_count;
    result.target_wrench.head<3>() =
        input.mass * (input.desired_com_acceleration +
                      Eigen::Vector3d(0.0, 0.0, kGravity));
    result.target_wrench.tail<3>() = input.desired_body_moment;
    if (input.desired_foot_forces_available) {
        for (int i = 0; i < contact_count; i++) {
            const bool left = i < input.left_contact_count;
            const int same_foot_count =
                left ? input.left_contact_count : right_contact_count;
            const double divisor = static_cast<double>(std::max(same_foot_count, 1));
            result.contact_forces[i] =
                (left ? input.desired_left_force : input.desired_right_force) / divisor;
            if (left) {
                result.left_force += result.contact_forces[i];
            } else {
                result.right_force += result.contact_forces[i];
            }
        }
        result.achieved_wrench =
            contact_wrench(input.com_position, input.contact_positions, result.contact_forces);
        return result;
    }

    const Eigen::Vector3d nominal_force =
        result.target_wrench.head<3>() / static_cast<double>(contact_count);
    for (int i = 0; i < contact_count; i++) {
        const int same_foot_count =
            i < input.left_contact_count ? input.left_contact_count : right_contact_count;
        const double divisor = static_cast<double>(std::max(same_foot_count, 1));
        const double min_normal_force =
            static_cast<double>(config_.wbc_min_normal_force) / divisor;
        const double max_normal_force =
            static_cast<double>(config_.wbc_max_normal_force) / divisor;
        Eigen::Vector3d force = nominal_force;
        force.z() = std::clamp(force.z(), min_normal_force, max_normal_force);
        const double tangential_limit =
            static_cast<double>(config_.wbc_friction_coefficient) * force.z();
        force.x() = std::clamp(force.x(), -tangential_limit, tangential_limit);
        force.y() = std::clamp(force.y(), -tangential_limit, tangential_limit);

        result.contact_forces[i] = force;
        if (i < input.left_contact_count) {
            result.left_force += force;
        } else {
            result.right_force += force;
        }
    }
    result.achieved_wrench =
        contact_wrench(input.com_position, input.contact_positions, result.contact_forces);
    return result;
}

void WholeBodyMpcController::apply_mpc_joint_command(
    const CentroidalMpc::Output& mpc_output,
    const std::vector<float>& current_joint_position,
    std::vector<float>& command_position,
    std::vector<float>& command_velocity,
    StandingStabilizer::Correction& correction) const {
    if (!config_.wbc_mpc_joint_command_enabled ||
        !mpc_output.has_desired_joint_command ||
        current_joint_position.size() != command_position.size() ||
        current_joint_position.size() != command_velocity.size() ||
        mpc_output.desired_joint_velocity.size() !=
            static_cast<int>(current_joint_position.size())) {
        return;
    }

    std::vector<int> controlled_joint_indices = left_leg_joint_indices_;
    controlled_joint_indices.insert(controlled_joint_indices.end(),
                                    right_leg_joint_indices_.begin(),
                                    right_leg_joint_indices_.end());
    const double position_gain =
        std::clamp(static_cast<double>(config_.wbc_mpc_joint_command_position_gain),
                   0.0, 1.0);
    const double velocity_scale =
        std::max(static_cast<double>(config_.wbc_mpc_joint_command_velocity_scale),
                 0.0);
    const double velocity_limit = std::min(
        static_cast<double>(config_.wbc_swing_max_joint_velocity),
        static_cast<double>(config_.wbc_mpc_joint_command_max_velocity));
    const double max_delta =
        static_cast<double>(config_.wbc_mpc_joint_command_max_delta);

    for (int joint_index : controlled_joint_indices) {
        if (joint_index < 0 ||
            joint_index >= static_cast<int>(command_position.size())) {
            continue;
        }
        const double joint_scale =
            config_.wbc_mpc_joint_command_joint_scale.empty()
                ? 1.0
                : std::clamp(
                      config_.wbc_mpc_joint_command_joint_scale
                          [static_cast<size_t>(joint_index)],
                      0.0, 1.0);
        if (joint_scale <= 0.0) {
            continue;
        }
        const double desired_velocity =
            joint_scale * mpc_output.desired_joint_velocity[joint_index];
        const double limited_desired_velocity =
            clamp_abs(desired_velocity, velocity_limit);
        double delta =
            position_gain * static_cast<double>(config_.dt) *
            limited_desired_velocity;
        if (max_delta > 0.0) {
            delta = clamp_abs(delta, max_delta);
        }
        double target =
            static_cast<double>(current_joint_position[joint_index]) + delta;
        if (!config_.joint_limits.empty()) {
            const int lower_idx = joint_index * 2;
            target = std::clamp(target, config_.joint_limits[lower_idx],
                                config_.joint_limits[lower_idx + 1]);
        }
        command_position[joint_index] = static_cast<float>(target);
        const double command_vel =
            clamp_abs(velocity_scale * limited_desired_velocity,
                      velocity_limit);
        command_velocity[joint_index] = static_cast<float>(
            command_vel);
        const double applied_delta =
            target - static_cast<double>(current_joint_position[joint_index]);
        if (std::abs(applied_delta) > correction.mpc_joint_command_max_delta) {
            correction.mpc_joint_command_max_delta =
                static_cast<float>(std::abs(applied_delta));
            correction.mpc_joint_command_max_joint_index = joint_index;
        }
        correction.mpc_joint_command_max_velocity =
            std::max(correction.mpc_joint_command_max_velocity,
                     static_cast<float>(std::abs(command_vel)));
        correction.mpc_joint_command_used = true;
    }
}

void WholeBodyMpcController::apply_swing_ik_targets(
    const RobotModel::Kinematics& kinematics,
    const RecoveryGaitPlanner::Reference& gait_reference,
    const std::vector<float>& current_joint_position,
    std::vector<float>& command_position,
    std::vector<float>& command_velocity) const {
    if (!config_.wbc_swing_ik_enabled || !gait_reference.recovery_active ||
        current_joint_position.size() != command_position.size()) {
        return;
    }
    if (gait_reference.left_foot.active) {
        apply_foot_ik_target(
            kinematics.left_foot_jacobian,
            gait_reference.left_foot.position - kinematics.left_foot_pose.translation(),
            gait_reference.left_foot.velocity,
            left_leg_joint_indices_, current_joint_position,
            command_position, command_velocity);
    }
    if (gait_reference.right_foot.active) {
        apply_foot_ik_target(
            kinematics.right_foot_jacobian,
            gait_reference.right_foot.position - kinematics.right_foot_pose.translation(),
            gait_reference.right_foot.velocity,
            right_leg_joint_indices_, current_joint_position,
            command_position, command_velocity);
    }
}

void WholeBodyMpcController::apply_foot_ik_target(
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& foot_jacobian,
    const Eigen::Vector3d& position_error,
    const Eigen::Vector3d& desired_velocity,
    const std::vector<int>& command_joint_indices,
    const std::vector<float>& current_joint_position,
    std::vector<float>& command_position,
    std::vector<float>& command_velocity) const {
    if (command_joint_indices.empty()) {
        return;
    }
    const auto& joint_v_indices = robot_model_->configured_joint_velocity_indices();
    Eigen::MatrixXd leg_jacobian(3, static_cast<int>(command_joint_indices.size()));
    for (int col = 0; col < static_cast<int>(command_joint_indices.size()); col++) {
        const int command_index = command_joint_indices[col];
        leg_jacobian.col(col) = foot_jacobian.topRows<3>().col(joint_v_indices[command_index]);
    }

    const double damping = std::max(static_cast<double>(config_.wbc_swing_ik_damping), 1.0e-6);
    const Eigen::Matrix3d damped =
        leg_jacobian * leg_jacobian.transpose() +
        damping * damping * Eigen::Matrix3d::Identity();
    const Eigen::VectorXd joint_delta =
        leg_jacobian.transpose() *
        damped.ldlt().solve(static_cast<double>(config_.wbc_swing_ik_gain) * position_error);
    const Eigen::VectorXd joint_velocity =
        leg_jacobian.transpose() * damped.ldlt().solve(desired_velocity);

    for (int col = 0; col < static_cast<int>(command_joint_indices.size()); col++) {
        const int command_index = command_joint_indices[col];
        if (command_index < 0 || command_index >= static_cast<int>(command_position.size())) {
            continue;
        }
        const double delta = clamp_abs(
            joint_delta[col], static_cast<double>(config_.wbc_swing_max_joint_delta));
        double target = static_cast<double>(current_joint_position[command_index]) + delta;
        if (!config_.joint_limits.empty()) {
            const int lower_idx = command_index * 2;
            target = std::clamp(target, config_.joint_limits[lower_idx],
                                config_.joint_limits[lower_idx + 1]);
        }
        command_position[command_index] = static_cast<float>(target);
        command_velocity[command_index] = static_cast<float>(
            clamp_abs(joint_velocity[col],
                      static_cast<double>(config_.wbc_swing_max_joint_velocity)));
    }
}

void WholeBodyMpcController::initialize_leg_joint_indices() {
    left_leg_joint_indices_.clear();
    right_leg_joint_indices_.clear();
    const std::vector<std::string>& joint_order = robot_model_->configured_joint_order();
    for (int i = 0; i < static_cast<int>(joint_order.size()); i++) {
        if (is_left_leg_joint_name(joint_order[i])) {
            left_leg_joint_indices_.push_back(i);
        } else if (is_right_leg_joint_name(joint_order[i])) {
            right_leg_joint_indices_.push_back(i);
        }
    }
    if (left_leg_joint_indices_.empty() || right_leg_joint_indices_.empty()) {
        throw std::runtime_error("whole_body_mpc could not identify left/right leg joints");
    }
}

void WholeBodyMpcController::validate_model_config() const {
    if (config_.whole_body_model_path.empty()) {
        throw std::runtime_error("stand_whole_body_model_path is required for whole_body_mpc");
    }
    if (!std::filesystem::exists(config_.whole_body_model_path)) {
        throw std::runtime_error("stand_whole_body_model_path does not exist: " +
                                 config_.whole_body_model_path);
    }
    if (config_.whole_body_base_link.empty()) {
        throw std::runtime_error("stand_whole_body_base_link is required for whole_body_mpc");
    }
    if (config_.whole_body_left_foot_link.empty()) {
        throw std::runtime_error("stand_whole_body_left_foot_link is required for whole_body_mpc");
    }
    if (config_.whole_body_right_foot_link.empty()) {
        throw std::runtime_error("stand_whole_body_right_foot_link is required for whole_body_mpc");
    }
    if (config_.whole_body_joint_order.empty()) {
        throw std::runtime_error("stand_whole_body_joint_order is required for whole_body_mpc");
    }
}

}  // namespace whole_body_mpc
