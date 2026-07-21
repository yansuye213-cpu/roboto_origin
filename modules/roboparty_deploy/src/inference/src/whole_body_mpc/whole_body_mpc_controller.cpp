// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/whole_body_mpc_controller.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/QR>
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

struct LinearQpResult {
    Eigen::VectorXd x;
    int active_constraint_count = 0;
    double max_violation = 0.0;
};

bool contains_active_constraint(const std::vector<int>& active, int index) {
    return std::find(active.begin(), active.end(), index) != active.end();
}

LinearQpResult solve_active_set_qp(
    const Eigen::MatrixXd& hessian,
    const Eigen::VectorXd& linear_rhs,
    const Eigen::MatrixXd& equality,
    const Eigen::VectorXd& equality_rhs,
    const Eigen::MatrixXd& inequality,
    const Eigen::VectorXd& inequality_rhs,
    int max_iterations) {
    constexpr double kPrimalTolerance = 1.0e-5;
    constexpr double kDualTolerance = 1.0e-6;

    const int decision_dim = static_cast<int>(hessian.rows());
    const int equality_rows = static_cast<int>(equality.rows());
    std::vector<int> active;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(decision_dim);
    Eigen::VectorXd active_lambda;
    double max_violation = 0.0;

    const auto solve_with_active_set = [&]() {
        const int active_rows = static_cast<int>(active.size());
        const int constraint_rows = equality_rows + active_rows;
        Eigen::MatrixXd kkt(decision_dim + constraint_rows,
                            decision_dim + constraint_rows);
        Eigen::VectorXd rhs(decision_dim + constraint_rows);
        kkt.setZero();
        rhs.setZero();
        kkt.block(0, 0, decision_dim, decision_dim) = hessian;
        rhs.head(decision_dim) = linear_rhs;
        if (equality_rows > 0) {
            kkt.block(0, decision_dim, decision_dim, equality_rows) =
                equality.transpose();
            kkt.block(decision_dim, 0, equality_rows, decision_dim) = equality;
            rhs.segment(decision_dim, equality_rows) = equality_rhs;
        }
        for (int i = 0; i < active_rows; i++) {
            const int constraint_index = active[i];
            kkt.block(0, decision_dim + equality_rows + i, decision_dim, 1) =
                inequality.row(constraint_index).transpose();
            kkt.block(decision_dim + equality_rows + i, 0, 1, decision_dim) =
                inequality.row(constraint_index);
            rhs[decision_dim + equality_rows + i] =
                inequality_rhs[constraint_index];
        }

        const Eigen::VectorXd solution =
            kkt.completeOrthogonalDecomposition().solve(rhs);
        x = solution.head(decision_dim);
        if (active_rows > 0) {
            active_lambda = solution.segment(decision_dim + equality_rows, active_rows);
        } else {
            active_lambda.resize(0);
        }
    };

    const int iteration_limit = std::max(max_iterations, 1);
    for (int iter = 0; iter < iteration_limit; iter++) {
        solve_with_active_set();
        max_violation = 0.0;
        int most_violated = -1;
        if (inequality.rows() > 0) {
            const Eigen::VectorXd violation = inequality * x - inequality_rhs;
            for (int i = 0; i < violation.size(); i++) {
                if (contains_active_constraint(active, i)) {
                    continue;
                }
                if (violation[i] > max_violation) {
                    max_violation = violation[i];
                    most_violated = i;
                }
            }
        }
        if (most_violated >= 0 && max_violation > kPrimalTolerance) {
            active.push_back(most_violated);
            continue;
        }

        int most_negative_lambda = -1;
        double min_lambda = 0.0;
        for (int i = 0; i < active_lambda.size(); i++) {
            if (active_lambda[i] < min_lambda) {
                min_lambda = active_lambda[i];
                most_negative_lambda = i;
            }
        }
        if (most_negative_lambda >= 0 && min_lambda < -kDualTolerance) {
            active.erase(active.begin() + most_negative_lambda);
            continue;
        }

        LinearQpResult result;
        result.x = x;
        result.active_constraint_count = static_cast<int>(active.size());
        result.max_violation = std::max(max_violation, 0.0);
        return result;
    }

    solve_with_active_set();
    if (inequality.rows() > 0) {
        const Eigen::VectorXd violation = inequality * x - inequality_rhs;
        max_violation = std::max(0.0, violation.maxCoeff());
    }
    LinearQpResult result;
    result.x = x;
    result.active_constraint_count = static_cast<int>(active.size());
    result.max_violation = max_violation;
    return result;
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
    mpc_config.friction_barrier_mu = config_.wbc_mpc_friction_barrier_mu;
    mpc_config.friction_barrier_delta = config_.wbc_mpc_friction_barrier_delta;
    mpc_config.friction_regularization = config_.wbc_mpc_friction_regularization;
    mpc_config.max_angular_accel = config_.wbc_mpc_max_angular_accel;
    mpc_config.max_com_accel = config_.wbc_mpc_max_com_accel;
    mpc_config.max_contact_force_delta = config_.wbc_mpc_max_contact_force_delta;
    mpc_config.friction_coefficient = config_.wbc_friction_coefficient;
    mpc_config.min_normal_force = config_.wbc_min_normal_force;
    mpc_config.max_normal_force = config_.wbc_max_normal_force;
    mpc_config.target_roll = config_.wbc_target_roll;
    mpc_config.target_pitch = config_.wbc_target_pitch;
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
    if (recovery_gait_planner_) {
        recovery_gait_planner_->reset();
    }
    latest_state_estimate_ = BaseStateEstimator::Output{};
    state_estimator_left_contact_ = true;
    state_estimator_right_contact_ = true;
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
        " max_force_delta=" +
        std::to_string(config_.wbc_mpc_max_contact_force_delta));
    lines.emplace_back(
        "whole_body_mpc centroidal_mpc_constraints: zero_swing_force=" +
        std::string(config_.wbc_mpc_zero_swing_force_constraint_enabled ? "true" : "false") +
        " normal_force=" +
        std::string(config_.wbc_mpc_normal_force_constraint_enabled ? "true" : "false") +
        " delta_force=" +
        std::string(config_.wbc_mpc_delta_force_constraint_enabled ? "true" : "false") +
        " friction_cone=" +
        std::string(config_.wbc_mpc_friction_cone_constraint_enabled ? "true" : "false") +
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
        std::string(config_.wbc_swing_ik_enabled ? "enabled" : "disabled"));
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
                                   contact_schedule);
    const CentroidalMpc::Output mpc_output = centroidal_mpc_->solve(mpc_input);
    correction.wbc_mpc_backend = mpc_output.backend;
    correction.wbc_mpc_used = mpc_output.solved;
    correction.wbc_mpc_iterations = mpc_output.iterations;
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

    const WbcQpResult wbc_result =
        solve_whole_body_wbc_qp(latest_kinematics_, contacts, gait_reference,
                                mpc_output, contact_result, blend);
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
    const ContactSchedule& contact_schedule) const {
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
    const ContactPointSet& contacts) const {
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
    input.desired_body_moment << clamp_abs(desired_body_moment.x(), config_.wbc_max_body_moment),
                                 clamp_abs(desired_body_moment.y(), config_.wbc_max_body_moment),
                                 0.0;
    return input;
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

WholeBodyMpcController::WbcQpResult WholeBodyMpcController::solve_whole_body_wbc_qp(
    const RobotModel::Kinematics& kinematics,
    const ContactPointSet& contacts,
    const RecoveryGaitPlanner::Reference& gait_reference,
    const CentroidalMpc::Output& mpc_output,
    const ContactForceQp::Result& contact_result,
    float blend) const {
    WbcQpResult result;
    result.tau.assign(config_.joint_num, 0.0f);
    const int nv = robot_model_->nv();
    const int contact_count = static_cast<int>(contacts.positions.size());
    const int contact_dim = contact_count * 3;
    const int tau_dim = config_.joint_num;
    const int qddot_offset = 0;
    const int force_offset = nv;
    const int tau_offset = nv + contact_dim;
    const int decision_dim = nv + contact_dim + tau_dim;
    const auto& joint_v_indices = robot_model_->configured_joint_velocity_indices();
    result.contact_count = contact_count;

    if (contact_count <= 0 || contacts.jacobian.rows() != contact_dim ||
        contacts.jacobian.cols() != nv || contacts.jacobian_dot_v.size() != contact_dim) {
        throw std::runtime_error("whole_body_mpc contact point set is invalid");
    }

    if (!config_.wbc_whole_body_qp_enabled) {
        result.left_force = contact_result.left_force;
        result.right_force = contact_result.right_force;
        result.achieved_wrench = contact_result.achieved_wrench;
        return result;
    }

    Eigen::VectorXd contact_force_target = Eigen::VectorXd::Zero(contact_dim);
    if (static_cast<int>(contact_result.contact_forces.size()) == contact_count) {
        for (int i = 0; i < contact_count; i++) {
            contact_force_target.segment<3>(i * 3) = contact_result.contact_forces[i];
        }
    }

    const int dynamics_rows = config_.wbc_floating_base_eom_enabled ? nv : 0;
    const int stance_contact_rows =
        config_.wbc_stance_contact_constraint_enabled ? contact_dim : 0;
    Eigen::MatrixXd equality(dynamics_rows + stance_contact_rows, decision_dim);
    Eigen::VectorXd equality_rhs(dynamics_rows + stance_contact_rows);
    equality.setZero();
    equality_rhs.setZero();
    int equality_row = 0;
    if (config_.wbc_floating_base_eom_enabled) {
        equality.block(equality_row, qddot_offset, nv, nv) = kinematics.mass_matrix;
        equality.block(equality_row, force_offset, nv, contact_dim) =
            -contacts.jacobian.transpose();
        for (int i = 0; i < tau_dim; i++) {
            equality(equality_row + joint_v_indices[i], tau_offset + i) = -1.0;
        }
        equality_rhs.segment(equality_row, nv) = -kinematics.nonlinear_effects;
        equality_row += nv;
    }
    if (config_.wbc_stance_contact_constraint_enabled) {
        equality.block(equality_row, qddot_offset, contact_dim, nv) = contacts.jacobian;
        equality_rhs.segment(equality_row, contact_dim) = -contacts.jacobian_dot_v;
        equality_row += contact_dim;
    }

    Eigen::Matrix<double, 6, 1> desired_base_accel =
        Eigen::Matrix<double, 6, 1>::Zero();
    desired_base_accel.head<3>() = mpc_output.desired_com_acceleration;
    desired_base_accel.tail<3>() = mpc_output.desired_angular_acceleration;

    const double base_weight =
        std::sqrt(std::max(static_cast<double>(config_.wbc_moment_tracking_weight), 1.0e-9));
    const double force_weight =
        std::sqrt(std::max(static_cast<double>(config_.wbc_force_tracking_weight), 0.0));
    const double qddot_weight =
        std::sqrt(std::max(static_cast<double>(config_.wbc_regularization_weight), 1.0e-9));
    const double tau_weight =
        std::sqrt(std::max(static_cast<double>(config_.wbc_smooth_weight), 1.0e-9));
    const double swing_weight =
        std::sqrt(std::max(static_cast<double>(config_.wbc_swing_tracking_weight), 0.0));
    const int swing_dim =
        config_.wbc_swing_task_enabled
            ? (gait_reference.left_swing ? 3 : 0) +
                  (gait_reference.right_swing ? 3 : 0)
            : 0;

    const int base_task_rows = config_.wbc_base_accel_task_enabled ? 6 : 0;
    const int force_task_rows = config_.wbc_contact_force_task_enabled ? contact_dim : 0;
    const int qddot_task_rows = config_.wbc_qddot_regularization_enabled ? nv : 0;
    const int tau_task_rows = config_.wbc_tau_regularization_enabled ? tau_dim : 0;
    const int task_rows =
        base_task_rows + force_task_rows + swing_dim + qddot_task_rows + tau_task_rows;
    Eigen::MatrixXd task(task_rows, decision_dim);
    Eigen::VectorXd task_rhs(task_rows);
    task.setZero();
    task_rhs.setZero();
    int row = 0;
    if (config_.wbc_base_accel_task_enabled) {
        task.block(row, qddot_offset, 6, nv) = base_weight * kinematics.base_jacobian;
        task_rhs.segment(row, 6) =
            base_weight * (desired_base_accel - kinematics.base_jacobian_dot_v);
        row += 6;
    }
    if (config_.wbc_contact_force_task_enabled) {
        task.block(row, force_offset, contact_dim, contact_dim) =
            force_weight * Eigen::MatrixXd::Identity(contact_dim, contact_dim);
        task_rhs.segment(row, contact_dim) = force_weight * contact_force_target;
        row += contact_dim;
    }
    const auto add_swing_task =
        [&](const RecoveryGaitPlanner::FootReference& foot_reference,
            const Eigen::Vector3d& current_position,
            const Eigen::Vector3d& current_velocity,
            const Eigen::Matrix<double, 6, Eigen::Dynamic>& foot_jacobian,
            const Eigen::Matrix<double, 6, 1>& foot_jacobian_dot_v) {
            if (!config_.wbc_swing_task_enabled || !foot_reference.active) {
                return;
            }
            const Eigen::Vector3d position_error =
                foot_reference.position - current_position;
            const Eigen::Vector3d velocity_error =
                foot_reference.velocity - current_velocity;
            const Eigen::Vector3d desired_acceleration =
                foot_reference.acceleration +
                static_cast<double>(config_.wbc_swing_kp) * position_error +
                static_cast<double>(config_.wbc_swing_kd) * velocity_error;
            result.swing_error = std::max(result.swing_error, position_error.norm());
            task.block(row, qddot_offset, 3, nv) =
                swing_weight * foot_jacobian.topRows<3>();
            task_rhs.segment(row, 3) =
                swing_weight * (desired_acceleration - foot_jacobian_dot_v.head<3>());
            row += 3;
        };
    add_swing_task(gait_reference.left_foot,
                   kinematics.left_foot_pose.translation(),
                   kinematics.left_foot_velocity,
                   kinematics.left_foot_jacobian,
                   kinematics.left_foot_jacobian_dot_v);
    add_swing_task(gait_reference.right_foot,
                   kinematics.right_foot_pose.translation(),
                   kinematics.right_foot_velocity,
                   kinematics.right_foot_jacobian,
                   kinematics.right_foot_jacobian_dot_v);
    if (config_.wbc_qddot_regularization_enabled) {
        task.block(row, qddot_offset, nv, nv) =
            qddot_weight * Eigen::MatrixXd::Identity(nv, nv);
        row += nv;
    }
    if (config_.wbc_tau_regularization_enabled) {
        task.block(row, tau_offset, tau_dim, tau_dim) =
            tau_weight * Eigen::MatrixXd::Identity(tau_dim, tau_dim);
        row += tau_dim;
    }

    Eigen::MatrixXd hessian =
        task.transpose() * task + 1.0e-8 * Eigen::MatrixXd::Identity(decision_dim, decision_dim);
    Eigen::VectorXd gradient_rhs = task.transpose() * task_rhs;

    const int friction_rows = config_.wbc_friction_constraint_enabled ? contact_count * 6 : 0;
    const int torque_limit_rows = config_.wbc_torque_limit_constraint_enabled ? tau_dim * 2 : 0;
    const int inequality_rows = friction_rows + torque_limit_rows;
    Eigen::MatrixXd inequality(inequality_rows, decision_dim);
    Eigen::VectorXd inequality_rhs(inequality_rows);
    inequality.setZero();
    inequality_rhs.setZero();
    int ineq_row = 0;
    const int right_contact_count = contact_count - contacts.left_contact_count;
    if (config_.wbc_friction_constraint_enabled) {
        for (int i = 0; i < contact_count; i++) {
            const int same_foot_count =
                i < contacts.left_contact_count ? contacts.left_contact_count : right_contact_count;
            const double normal_divisor = static_cast<double>(std::max(same_foot_count, 1));
            const double min_normal_force =
                static_cast<double>(config_.wbc_min_normal_force) / normal_divisor;
            const double max_normal_force =
                static_cast<double>(config_.wbc_max_normal_force) / normal_divisor;
            const int fx = force_offset + i * 3;
            const int fy = fx + 1;
            const int fz = fx + 2;
            const double mu = static_cast<double>(config_.wbc_friction_coefficient);

            inequality(ineq_row, fx) = 1.0;
            inequality(ineq_row, fz) = -mu;
            inequality_rhs[ineq_row++] = 0.0;
            inequality(ineq_row, fx) = -1.0;
            inequality(ineq_row, fz) = -mu;
            inequality_rhs[ineq_row++] = 0.0;
            inequality(ineq_row, fy) = 1.0;
            inequality(ineq_row, fz) = -mu;
            inequality_rhs[ineq_row++] = 0.0;
            inequality(ineq_row, fy) = -1.0;
            inequality(ineq_row, fz) = -mu;
            inequality_rhs[ineq_row++] = 0.0;
            inequality(ineq_row, fz) = -1.0;
            inequality_rhs[ineq_row++] = -min_normal_force;
            inequality(ineq_row, fz) = 1.0;
            inequality_rhs[ineq_row++] = max_normal_force;
        }
    }

    const double torque_blend = std::clamp(static_cast<double>(blend), 0.0, 1.0);
    if (config_.wbc_torque_limit_constraint_enabled) {
        for (int i = 0; i < tau_dim; i++) {
            const double scale = config_.wbc_torque_joint_scale.empty()
                ? 0.0
                : std::abs(config_.wbc_torque_joint_scale[i]);
            const double torque_bound =
                torque_blend * scale * static_cast<double>(config_.wbc_max_joint_torque);
            inequality(ineq_row, tau_offset + i) = 1.0;
            inequality_rhs[ineq_row++] = torque_bound;
            inequality(ineq_row, tau_offset + i) = -1.0;
            inequality_rhs[ineq_row++] = torque_bound;
        }
    }

    const LinearQpResult qp_solution = solve_active_set_qp(
        hessian, gradient_rhs, equality, equality_rhs, inequality, inequality_rhs,
        config_.wbc_active_set_iterations);
    const Eigen::VectorXd solution = qp_solution.x;
    result.max_qp_violation = qp_solution.max_violation;
    result.active_constraint_count = qp_solution.active_constraint_count;
    result.dynamics_residual = (equality * solution - equality_rhs).lpNorm<Eigen::Infinity>();

    Eigen::VectorXd solved_force = solution.segment(force_offset, contact_dim);
    Vector3dList solved_contact_forces;
    solved_contact_forces.resize(contact_count, Eigen::Vector3d::Zero());
    for (int i = 0; i < contact_count; i++) {
        solved_contact_forces[i] = solved_force.segment<3>(i * 3);
        if (i < contacts.left_contact_count) {
            result.left_force += solved_contact_forces[i];
        } else {
            result.right_force += solved_contact_forces[i];
        }
    }
    result.achieved_wrench = contact_wrench(
        kinematics.com_position, contacts.positions, solved_contact_forces);

    const Eigen::VectorXd solved_tau = solution.segment(tau_offset, tau_dim);
    for (int i = 0; i < config_.joint_num; i++) {
        const double scale = config_.wbc_torque_joint_scale.empty()
            ? 0.0
            : std::abs(config_.wbc_torque_joint_scale[i]);
        const double torque_bound =
            torque_blend * scale * static_cast<double>(config_.wbc_max_joint_torque);
        double command_tau = solved_tau[i];
        if (!std::isfinite(command_tau)) {
            command_tau = 0.0;
        }
        const double raw_abs_tau = std::abs(command_tau);
        if (raw_abs_tau > result.max_raw_joint_torque) {
            result.max_raw_joint_torque = raw_abs_tau;
            result.max_torque_joint_index = i;
        }
        command_tau = clamp_abs(command_tau, torque_bound);
        if (torque_bound > 1.0e-6 && raw_abs_tau > torque_bound - 1.0e-5) {
            result.saturated_joint_count++;
        }
        if (config_.wbc_torque_enabled) {
            result.tau[i] = static_cast<float>(command_tau);
            result.max_command_joint_torque =
                std::max(result.max_command_joint_torque, std::abs(command_tau));
        }
    }
    return result;
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
