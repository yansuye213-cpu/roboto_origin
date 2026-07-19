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

namespace whole_body_mpc {

namespace {

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

Eigen::Matrix<double, 6, 1> contact_wrench(
    const Eigen::Vector3d& com_position,
    const Eigen::Vector3d& left_foot_position,
    const Eigen::Vector3d& right_foot_position,
    const Eigen::Vector3d& left_force,
    const Eigen::Vector3d& right_force) {
    Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
    wrench.head<3>() = left_force + right_force;
    wrench.tail<3>() =
        (left_foot_position - com_position).cross(left_force) +
        (right_foot_position - com_position).cross(right_force);
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

    const Eigen::VectorXd q0 = robot_model_->neutral_configuration();
    const Eigen::VectorXd v0 = robot_model_->zero_velocity();
    neutral_kinematics_ = robot_model_->compute_kinematics(q0, v0);
    robot_mass_ = robot_model_->total_mass();
    if (robot_mass_ <= 0.0) {
        throw std::runtime_error("whole_body_mpc robot mass must be positive");
    }
    const Eigen::Vector3d neutral_foot_midpoint =
        0.5 * (neutral_kinematics_.left_foot_pose.translation() +
               neutral_kinematics_.right_foot_pose.translation());
    neutral_com_offset_xy_ =
        (neutral_kinematics_.com_position - neutral_foot_midpoint).head<2>();

    StanceMpc::Config mpc_config;
    mpc_config.horizon = config_.wbc_mpc_horizon;
    mpc_config.dt = config_.dt;
    mpc_config.orientation_weight = config_.wbc_mpc_orientation_weight;
    mpc_config.angular_rate_weight = config_.wbc_mpc_angular_rate_weight;
    mpc_config.com_weight = config_.wbc_mpc_com_weight;
    mpc_config.com_velocity_weight = config_.wbc_mpc_com_velocity_weight;
    mpc_config.control_weight = config_.wbc_mpc_control_weight;
    mpc_config.max_angular_accel = config_.wbc_mpc_max_angular_accel;
    mpc_config.max_com_accel = config_.wbc_mpc_max_com_accel;
    mpc_config.target_roll = config_.wbc_target_roll;
    mpc_config.target_pitch = config_.wbc_target_pitch;
    stance_mpc_ = std::make_unique<StanceMpc>(mpc_config);

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
}

void WholeBodyMpcController::reset() {
    if (stance_mpc_) {
        stance_mpc_->reset();
    }
    if (contact_force_qp_) {
        contact_force_qp_->reset();
    }
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
        "whole_body_mpc upper_stance_mpc: horizon=" +
        std::to_string(config_.wbc_mpc_horizon) +
        " target_rp=[" + std::to_string(config_.wbc_target_roll) + ", " +
        std::to_string(config_.wbc_target_pitch) + "]" +
        " max_accel=[" + std::to_string(config_.wbc_mpc_max_angular_accel) + ", " +
        std::to_string(config_.wbc_mpc_max_com_accel) + "]");
    lines.emplace_back(
        "whole_body_mpc torque_output: " +
        std::string(config_.wbc_torque_enabled ? "enabled" : "disabled") +
        " max_joint_torque=" + std::to_string(config_.wbc_max_joint_torque));
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
    const Eigen::VectorXd v = robot_model_->make_velocity(
        current_joint_velocity, Eigen::Vector3d::Zero(),
        Eigen::Vector3d(measurement.wx, measurement.wy, 0.0));
    latest_kinematics_ = robot_model_->compute_kinematics(q, v);

    StandingStabilizer::Command command;
    command.position = base_target;
    command.velocity.assign(config_.joint_num, 0.0f);
    command.kp = kp;
    command.kd = kd;

    StandingStabilizer::Correction correction;
    const StanceMpc::Input mpc_input = build_stance_mpc_input(measurement);
    const StanceMpc::Output mpc_output = stance_mpc_->solve(mpc_input);
    correction.mpc_roll_accel =
        static_cast<float>(mpc_output.desired_angular_acceleration.x());
    correction.mpc_pitch_accel =
        static_cast<float>(mpc_output.desired_angular_acceleration.y());
    correction.mpc_com_accel_x =
        static_cast<float>(mpc_output.desired_com_acceleration.x());
    correction.mpc_com_accel_y =
        static_cast<float>(mpc_output.desired_com_acceleration.y());

    ContactForceQp::Input qp_input = build_contact_qp_input(mpc_output);
    const ContactForceQp::Result contact_result = contact_force_qp_->solve(qp_input);
    correction.qp_used = true;
    correction.wbc_left_normal_force = static_cast<float>(contact_result.left_force.z());
    correction.wbc_right_normal_force = static_cast<float>(contact_result.right_force.z());
    correction.wbc_roll_moment = static_cast<float>(qp_input.desired_body_moment.x());
    correction.wbc_pitch_moment = static_cast<float>(qp_input.desired_body_moment.y());
    correction.wbc_achieved_roll_moment =
        static_cast<float>(contact_result.achieved_wrench.tail<3>().x());
    correction.wbc_achieved_pitch_moment =
        static_cast<float>(contact_result.achieved_wrench.tail<3>().y());

    const WbcQpResult wbc_result =
        solve_stance_wbc_qp(latest_kinematics_, mpc_output, contact_result, blend);
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
    command.tau = wbc_result.tau;
    command.correction = correction;
    return command;
}

StanceMpc::Input WholeBodyMpcController::build_stance_mpc_input(
    const StandingStabilizer::Measurement& measurement) const {
    const Eigen::Vector3d foot_midpoint =
        0.5 * (latest_kinematics_.left_foot_pose.translation() +
               latest_kinematics_.right_foot_pose.translation());
    const Eigen::Vector2d current_com_offset_xy =
        (latest_kinematics_.com_position - foot_midpoint).head<2>();

    StanceMpc::Input input;
    input.roll = measurement.roll;
    input.pitch = measurement.pitch;
    input.wx = measurement.wx;
    input.wy = measurement.wy;
    input.com_offset_error = current_com_offset_xy - neutral_com_offset_xy_;
    input.com_velocity = latest_kinematics_.com_velocity.head<2>();
    return input;
}

ContactForceQp::Input WholeBodyMpcController::build_contact_qp_input(
    const StanceMpc::Output& mpc_output) const {
    ContactForceQp::Input input;
    input.mass = robot_mass_;
    input.com_position = latest_kinematics_.com_position;
    input.left_foot_position = latest_kinematics_.left_foot_pose.translation();
    input.right_foot_position = latest_kinematics_.right_foot_pose.translation();
    input.desired_com_acceleration = mpc_output.desired_com_acceleration;

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

WholeBodyMpcController::WbcQpResult WholeBodyMpcController::solve_stance_wbc_qp(
    const RobotModel::Kinematics& kinematics,
    const StanceMpc::Output& mpc_output,
    const ContactForceQp::Result& contact_result,
    float blend) const {
    WbcQpResult result;
    result.tau.assign(config_.joint_num, 0.0f);
    const int nv = robot_model_->nv();
    const int contact_dim = 6;
    const int tau_dim = config_.joint_num;
    const int qddot_offset = 0;
    const int force_offset = nv;
    const int tau_offset = nv + contact_dim;
    const int decision_dim = nv + contact_dim + tau_dim;
    const auto& joint_v_indices = robot_model_->configured_joint_velocity_indices();

    Eigen::MatrixXd contact_jacobian(contact_dim, nv);
    contact_jacobian.block(0, 0, 3, nv) = kinematics.left_foot_jacobian.topRows<3>();
    contact_jacobian.block(3, 0, 3, nv) = kinematics.right_foot_jacobian.topRows<3>();
    Eigen::VectorXd contact_jdot_v(contact_dim);
    contact_jdot_v.segment<3>(0) = kinematics.left_foot_jacobian_dot_v.head<3>();
    contact_jdot_v.segment<3>(3) = kinematics.right_foot_jacobian_dot_v.head<3>();

    Eigen::MatrixXd equality(nv + contact_dim, decision_dim);
    Eigen::VectorXd equality_rhs(nv + contact_dim);
    equality.setZero();
    equality_rhs.setZero();
    equality.block(0, qddot_offset, nv, nv) = kinematics.mass_matrix;
    equality.block(0, force_offset, nv, contact_dim) = -contact_jacobian.transpose();
    for (int i = 0; i < tau_dim; i++) {
        equality(joint_v_indices[i], tau_offset + i) = -1.0;
    }
    equality_rhs.head(nv) = -kinematics.nonlinear_effects;
    equality.block(nv, qddot_offset, contact_dim, nv) = contact_jacobian;
    equality_rhs.tail(contact_dim) = -contact_jdot_v;

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

    const int task_rows = 6 + contact_dim + nv + tau_dim;
    Eigen::MatrixXd task(task_rows, decision_dim);
    Eigen::VectorXd task_rhs(task_rows);
    task.setZero();
    task_rhs.setZero();
    int row = 0;
    task.block(row, qddot_offset, 6, nv) = base_weight * kinematics.base_jacobian;
    task_rhs.segment(row, 6) =
        base_weight * (desired_base_accel - kinematics.base_jacobian_dot_v);
    row += 6;
    task.block(row, force_offset, contact_dim, contact_dim) =
        force_weight * Eigen::MatrixXd::Identity(contact_dim, contact_dim);
    task_rhs.segment(row, 3) = force_weight * contact_result.left_force;
    task_rhs.segment(row + 3, 3) = force_weight * contact_result.right_force;
    row += contact_dim;
    task.block(row, qddot_offset, nv, nv) =
        qddot_weight * Eigen::MatrixXd::Identity(nv, nv);
    row += nv;
    task.block(row, tau_offset, tau_dim, tau_dim) =
        tau_weight * Eigen::MatrixXd::Identity(tau_dim, tau_dim);

    Eigen::MatrixXd hessian =
        task.transpose() * task + 1.0e-8 * Eigen::MatrixXd::Identity(decision_dim, decision_dim);
    Eigen::VectorXd gradient_rhs = task.transpose() * task_rhs;
    Eigen::MatrixXd kkt(decision_dim + equality.rows(), decision_dim + equality.rows());
    Eigen::VectorXd kkt_rhs(decision_dim + equality.rows());
    kkt.setZero();
    kkt.block(0, 0, decision_dim, decision_dim) = hessian;
    kkt.block(0, decision_dim, decision_dim, equality.rows()) = equality.transpose();
    kkt.block(decision_dim, 0, equality.rows(), decision_dim) = equality;
    kkt_rhs.head(decision_dim) = gradient_rhs;
    kkt_rhs.tail(equality.rows()) = equality_rhs;

    const Eigen::VectorXd solution = kkt.completeOrthogonalDecomposition().solve(kkt_rhs);
    Eigen::VectorXd solved_force = solution.segment(force_offset, contact_dim);
    result.left_force = solved_force.segment<3>(0);
    result.right_force = solved_force.segment<3>(3);
    result.achieved_wrench = contact_wrench(
        kinematics.com_position,
        kinematics.left_foot_pose.translation(),
        kinematics.right_foot_pose.translation(),
        result.left_force,
        result.right_force);

    const Eigen::VectorXd solved_tau = solution.segment(tau_offset, tau_dim);
    const double torque_blend = std::clamp(static_cast<double>(blend), 0.0, 1.0);
    for (int i = 0; i < config_.joint_num; i++) {
        const double scale = config_.wbc_torque_joint_scale.empty()
            ? 0.0
            : config_.wbc_torque_joint_scale[i];
        double scaled_tau = torque_blend * scale * solved_tau[i];
        if (!std::isfinite(scaled_tau)) {
            scaled_tau = 0.0;
        }
        const double raw_abs_tau = std::abs(scaled_tau);
        if (raw_abs_tau > result.max_raw_joint_torque) {
            result.max_raw_joint_torque = raw_abs_tau;
            result.max_torque_joint_index = i;
        }
        const double command_tau = clamp_abs(scaled_tau, config_.wbc_max_joint_torque);
        if (raw_abs_tau > static_cast<double>(config_.wbc_max_joint_torque) + 1.0e-6) {
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
