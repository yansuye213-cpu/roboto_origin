// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/whole_body_mpc_controller.hpp"

#include <algorithm>
#include <cmath>
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

Eigen::Vector2d clamp_norm(const Eigen::Vector2d& value, double max_norm) {
    if (max_norm <= 0.0) {
        return Eigen::Vector2d::Zero();
    }
    const double norm = value.norm();
    if (norm <= max_norm || norm <= 1.0e-9) {
        return value;
    }
    return value * (max_norm / norm);
}

double clamp_abs(double value, double limit) {
    if (limit <= 0.0) {
        return 0.0;
    }
    return std::clamp(value, -limit, limit);
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
    ContactForceQp::Input qp_input = build_contact_qp_input(measurement);
    const ContactForceQp::Result contact_result = contact_force_qp_->solve(qp_input);
    correction.qp_used = true;
    correction.wbc_left_normal_force = static_cast<float>(contact_result.left_force.z());
    correction.wbc_right_normal_force = static_cast<float>(contact_result.right_force.z());
    correction.wbc_roll_moment = static_cast<float>(qp_input.desired_body_moment.x());
    correction.wbc_pitch_moment = static_cast<float>(qp_input.desired_body_moment.y());

    command.tau = compute_joint_torque_command(
        latest_kinematics_, contact_result, q, v, blend, correction);
    command.correction = correction;
    return command;
}

ContactForceQp::Input WholeBodyMpcController::build_contact_qp_input(
    const StandingStabilizer::Measurement& measurement) const {
    ContactForceQp::Input input;
    input.mass = robot_mass_;
    input.com_position = latest_kinematics_.com_position;
    input.left_foot_position = latest_kinematics_.left_foot_pose.translation();
    input.right_foot_position = latest_kinematics_.right_foot_pose.translation();

    const Eigen::Vector3d foot_midpoint =
        0.5 * (input.left_foot_position + input.right_foot_position);
    const Eigen::Vector2d current_com_offset_xy =
        (input.com_position - foot_midpoint).head<2>();
    const Eigen::Vector2d com_offset_error =
        current_com_offset_xy - neutral_com_offset_xy_;
    Eigen::Vector2d desired_xy_accel =
        -static_cast<double>(config_.wbc_com_kp) * com_offset_error -
        static_cast<double>(config_.wbc_com_kd) * latest_kinematics_.com_velocity.head<2>();
    desired_xy_accel = clamp_norm(desired_xy_accel, config_.wbc_max_com_accel);
    input.desired_com_acceleration << desired_xy_accel.x(), desired_xy_accel.y(), 0.0;

    const double roll_moment =
        -static_cast<double>(config_.wbc_roll_moment_kp) *
            (static_cast<double>(measurement.roll) - static_cast<double>(config_.target_roll)) -
        static_cast<double>(config_.wbc_roll_moment_kd) * static_cast<double>(measurement.wx);
    const double pitch_moment =
        -static_cast<double>(config_.wbc_pitch_moment_kp) *
            (static_cast<double>(measurement.pitch) - static_cast<double>(config_.target_pitch)) -
        static_cast<double>(config_.wbc_pitch_moment_kd) * static_cast<double>(measurement.wy);
    input.desired_body_moment << clamp_abs(roll_moment, config_.wbc_max_body_moment),
                                 clamp_abs(pitch_moment, config_.wbc_max_body_moment),
                                 0.0;
    return input;
}

std::vector<float> WholeBodyMpcController::compute_joint_torque_command(
    const RobotModel::Kinematics& kinematics,
    const ContactForceQp::Result& contact_result,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& v,
    float blend,
    StandingStabilizer::Correction& correction) const {
    std::vector<float> tau(config_.joint_num, 0.0f);
    if (!config_.wbc_torque_enabled || config_.wbc_max_joint_torque <= 0.0f) {
        return tau;
    }

    Eigen::Matrix<double, 6, 1> left_wrench = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> right_wrench = Eigen::Matrix<double, 6, 1>::Zero();
    left_wrench.head<3>() = contact_result.left_force;
    right_wrench.head<3>() = contact_result.right_force;

    const Eigen::VectorXd nonlinear = robot_model_->nonlinear_effects(q, v);
    const Eigen::VectorXd contact_generalized =
        kinematics.left_foot_jacobian.transpose() * left_wrench +
        kinematics.right_foot_jacobian.transpose() * right_wrench;
    const Eigen::VectorXd generalized_tau = nonlinear - contact_generalized;
    const std::vector<double> joint_tau = robot_model_->configured_joint_torques(generalized_tau);

    const double torque_blend = std::clamp(static_cast<double>(blend), 0.0, 1.0);
    for (int i = 0; i < config_.joint_num; i++) {
        const double scale = config_.wbc_torque_joint_scale.empty()
            ? 0.0
            : config_.wbc_torque_joint_scale[i];
        double command_tau = torque_blend * scale * joint_tau[i];
        if (!std::isfinite(command_tau)) {
            command_tau = 0.0;
        }
        command_tau = clamp_abs(command_tau, config_.wbc_max_joint_torque);
        tau[i] = static_cast<float>(command_tau);
        correction.wbc_max_joint_torque =
            std::max(correction.wbc_max_joint_torque, static_cast<float>(std::abs(command_tau)));
    }
    return tau;
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
