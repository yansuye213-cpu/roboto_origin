// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/wbc/whole_body_wbc.hpp"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace whole_body_mpc {

namespace {

double clamp_abs(double value, double limit) {
    if (limit <= 0.0) {
        return 0.0;
    }
    return std::clamp(value, -limit, limit);
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
            active_lambda = solution.segment(decision_dim + equality_rows,
                                             active_rows);
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

void append_rows(Eigen::MatrixXd& matrix, Eigen::VectorXd& vector,
                 const Eigen::MatrixXd& rows,
                 const Eigen::VectorXd& values) {
    if (rows.rows() <= 0) {
        return;
    }
    const int old_rows = static_cast<int>(matrix.rows());
    matrix.conservativeResize(old_rows + rows.rows(), Eigen::NoChange);
    vector.conservativeResize(old_rows + rows.rows());
    matrix.block(old_rows, 0, rows.rows(), rows.cols()) = rows;
    vector.segment(old_rows, rows.rows()) = values;
}

Eigen::MatrixXd stack_matrices(const std::vector<Eigen::MatrixXd>& matrices,
                               int cols) {
    int rows = 0;
    for (const auto& matrix : matrices) {
        rows += static_cast<int>(matrix.rows());
    }
    Eigen::MatrixXd output(rows, cols);
    output.setZero();
    int row = 0;
    for (const auto& matrix : matrices) {
        if (matrix.rows() <= 0) {
            continue;
        }
        output.block(row, 0, matrix.rows(), cols) = matrix;
        row += static_cast<int>(matrix.rows());
    }
    return output;
}

Eigen::VectorXd stack_vectors(const std::vector<Eigen::VectorXd>& vectors) {
    int rows = 0;
    for (const auto& vector : vectors) {
        rows += static_cast<int>(vector.size());
    }
    Eigen::VectorXd output(rows);
    output.setZero();
    int row = 0;
    for (const auto& vector : vectors) {
        if (vector.size() <= 0) {
            continue;
        }
        output.segment(row, vector.size()) = vector;
        row += static_cast<int>(vector.size());
    }
    return output;
}

struct WholeBodyWbcProblem {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int nv = 0;
    int contact_count = 0;
    int contact_dim = 0;
    int tau_dim = 0;
    int qddot_offset = 0;
    int force_offset = 0;
    int tau_offset = 0;
    int decision_dim = 0;
    double torque_blend = 1.0;
    double swing_error = 0.0;
    Eigen::MatrixXd equality;
    Eigen::VectorXd equality_rhs;
    Eigen::MatrixXd inequality;
    Eigen::VectorXd inequality_rhs;
    Eigen::MatrixXd base_task;
    Eigen::VectorXd base_task_rhs;
    Eigen::MatrixXd force_task;
    Eigen::VectorXd force_task_rhs;
    Eigen::MatrixXd swing_task;
    Eigen::VectorXd swing_task_rhs;
    Eigen::MatrixXd qddot_task;
    Eigen::VectorXd qddot_task_rhs;
    Eigen::MatrixXd tau_task;
    Eigen::VectorXd tau_task_rhs;
};

Eigen::VectorXd make_contact_force_target(
    const ContactForceQp::Result& contact_result,
    int contact_count,
    int contact_dim) {
    Eigen::VectorXd contact_force_target = Eigen::VectorXd::Zero(contact_dim);
    if (static_cast<int>(contact_result.contact_forces.size()) == contact_count) {
        for (int i = 0; i < contact_count; i++) {
            contact_force_target.segment<3>(i * 3) =
                contact_result.contact_forces[i];
        }
    }
    return contact_force_target;
}

WholeBodyWbcProblem build_problem(const WholeBodyWbc::Config& config,
                                  const WholeBodyWbc::Input& input) {
    const auto& kinematics = *input.kinematics;
    const auto& contacts = *input.contacts;
    const auto& gait_reference = *input.gait_reference;
    const auto& mpc_output = *input.mpc_output;
    const auto& contact_result = *input.contact_result;
    const auto& joint_v_indices = *input.configured_joint_velocity_indices;

    WholeBodyWbcProblem problem;
    problem.nv = static_cast<int>(kinematics.mass_matrix.rows());
    problem.contact_count = static_cast<int>(contacts.positions.size());
    problem.contact_dim = problem.contact_count * 3;
    problem.tau_dim = config.joint_num;
    problem.qddot_offset = 0;
    problem.force_offset = problem.nv;
    problem.tau_offset = problem.nv + problem.contact_dim;
    problem.decision_dim = problem.nv + problem.contact_dim + problem.tau_dim;
    problem.torque_blend = std::clamp(input.blend, 0.0, 1.0);

    const int dynamics_rows = config.floating_base_eom_enabled ? problem.nv : 0;
    const int stance_contact_rows =
        config.stance_contact_constraint_enabled ? problem.contact_dim : 0;
    problem.equality.setZero(dynamics_rows + stance_contact_rows,
                             problem.decision_dim);
    problem.equality_rhs.setZero(dynamics_rows + stance_contact_rows);
    int equality_row = 0;
    if (config.floating_base_eom_enabled) {
        problem.equality.block(equality_row, problem.qddot_offset,
                               problem.nv, problem.nv) =
            kinematics.mass_matrix;
        problem.equality.block(equality_row, problem.force_offset,
                               problem.nv, problem.contact_dim) =
            -contacts.jacobian.transpose();
        for (int i = 0; i < problem.tau_dim; i++) {
            problem.equality(equality_row + joint_v_indices[i],
                             problem.tau_offset + i) = -1.0;
        }
        problem.equality_rhs.segment(equality_row, problem.nv) =
            -kinematics.nonlinear_effects;
        equality_row += problem.nv;
    }
    if (config.stance_contact_constraint_enabled) {
        problem.equality.block(equality_row, problem.qddot_offset,
                               problem.contact_dim, problem.nv) =
            contacts.jacobian;
        problem.equality_rhs.segment(equality_row, problem.contact_dim) =
            -contacts.jacobian_dot_v;
        equality_row += problem.contact_dim;
    }

    const int friction_rows =
        config.friction_constraint_enabled ? problem.contact_count * 6 : 0;
    const int torque_limit_rows =
        config.torque_limit_constraint_enabled ? problem.tau_dim * 2 : 0;
    problem.inequality.setZero(friction_rows + torque_limit_rows,
                               problem.decision_dim);
    problem.inequality_rhs.setZero(friction_rows + torque_limit_rows);
    int ineq_row = 0;
    const int right_contact_count =
        problem.contact_count - contacts.left_contact_count;
    if (config.friction_constraint_enabled) {
        for (int i = 0; i < problem.contact_count; i++) {
            const int same_foot_count =
                i < contacts.left_contact_count ? contacts.left_contact_count
                                                : right_contact_count;
            const double normal_divisor =
                static_cast<double>(std::max(same_foot_count, 1));
            const double min_normal_force =
                config.min_normal_force / normal_divisor;
            const double max_normal_force =
                config.max_normal_force / normal_divisor;
            const int fx = problem.force_offset + i * 3;
            const int fy = fx + 1;
            const int fz = fx + 2;
            const double mu = config.friction_coefficient;

            problem.inequality(ineq_row, fx) = 1.0;
            problem.inequality(ineq_row, fz) = -mu;
            problem.inequality_rhs[ineq_row++] = 0.0;
            problem.inequality(ineq_row, fx) = -1.0;
            problem.inequality(ineq_row, fz) = -mu;
            problem.inequality_rhs[ineq_row++] = 0.0;
            problem.inequality(ineq_row, fy) = 1.0;
            problem.inequality(ineq_row, fz) = -mu;
            problem.inequality_rhs[ineq_row++] = 0.0;
            problem.inequality(ineq_row, fy) = -1.0;
            problem.inequality(ineq_row, fz) = -mu;
            problem.inequality_rhs[ineq_row++] = 0.0;
            problem.inequality(ineq_row, fz) = -1.0;
            problem.inequality_rhs[ineq_row++] = -min_normal_force;
            problem.inequality(ineq_row, fz) = 1.0;
            problem.inequality_rhs[ineq_row++] = max_normal_force;
        }
    }
    if (config.torque_limit_constraint_enabled) {
        for (int i = 0; i < problem.tau_dim; i++) {
            const double scale = config.torque_joint_scale.empty()
                ? 0.0
                : std::abs(config.torque_joint_scale[i]);
            const double torque_bound =
                problem.torque_blend * scale * config.max_joint_torque;
            problem.inequality(ineq_row, problem.tau_offset + i) = 1.0;
            problem.inequality_rhs[ineq_row++] = torque_bound;
            problem.inequality(ineq_row, problem.tau_offset + i) = -1.0;
            problem.inequality_rhs[ineq_row++] = torque_bound;
        }
    }

    const double base_weight =
        std::sqrt(std::max(config.moment_tracking_weight, 1.0e-9));
    const double force_weight =
        std::sqrt(std::max(config.force_tracking_weight, 0.0));
    const double qddot_weight =
        std::sqrt(std::max(config.regularization_weight, 1.0e-9));
    const double tau_weight =
        std::sqrt(std::max(config.smooth_weight, 1.0e-9));
    const double swing_weight =
        std::sqrt(std::max(config.swing_tracking_weight, 0.0));

    if (config.base_accel_task_enabled) {
        Eigen::Matrix<double, 6, 1> desired_base_accel =
            Eigen::Matrix<double, 6, 1>::Zero();
        desired_base_accel.head<3>() = mpc_output.desired_com_acceleration;
        desired_base_accel.tail<3>() = mpc_output.desired_angular_acceleration;
        problem.base_task.setZero(6, problem.decision_dim);
        problem.base_task.block(0, problem.qddot_offset, 6, problem.nv) =
            base_weight * kinematics.base_jacobian;
        problem.base_task_rhs =
            base_weight * (desired_base_accel - kinematics.base_jacobian_dot_v);
    } else {
        problem.base_task.setZero(0, problem.decision_dim);
        problem.base_task_rhs.resize(0);
    }

    if (config.contact_force_task_enabled) {
        problem.force_task.setZero(problem.contact_dim, problem.decision_dim);
        problem.force_task.block(0, problem.force_offset,
                                 problem.contact_dim, problem.contact_dim) =
            force_weight *
            Eigen::MatrixXd::Identity(problem.contact_dim, problem.contact_dim);
        problem.force_task_rhs =
            force_weight * make_contact_force_target(
                               contact_result, problem.contact_count,
                               problem.contact_dim);
    } else {
        problem.force_task.setZero(0, problem.decision_dim);
        problem.force_task_rhs.resize(0);
    }

    const int swing_dim =
        config.swing_task_enabled
            ? (gait_reference.left_swing ? 3 : 0) +
                  (gait_reference.right_swing ? 3 : 0)
            : 0;
    problem.swing_task.setZero(swing_dim, problem.decision_dim);
    problem.swing_task_rhs.setZero(swing_dim);
    int swing_row = 0;
    const auto add_swing_task =
        [&](const RecoveryGaitPlanner::FootReference& foot_reference,
            const Eigen::Vector3d& current_position,
            const Eigen::Vector3d& current_velocity,
            const Eigen::Matrix<double, 6, Eigen::Dynamic>& foot_jacobian,
            const Eigen::Matrix<double, 6, 1>& foot_jacobian_dot_v) {
            if (!config.swing_task_enabled || !foot_reference.active) {
                return;
            }
            const Eigen::Vector3d position_error =
                foot_reference.position - current_position;
            const Eigen::Vector3d velocity_error =
                foot_reference.velocity - current_velocity;
            const Eigen::Vector3d desired_acceleration =
                foot_reference.acceleration +
                config.swing_kp * position_error +
                config.swing_kd * velocity_error;
            problem.swing_error =
                std::max(problem.swing_error, position_error.norm());
            problem.swing_task.block(swing_row, problem.qddot_offset,
                                     3, problem.nv) =
                swing_weight * foot_jacobian.topRows<3>();
            problem.swing_task_rhs.segment(swing_row, 3) =
                swing_weight *
                (desired_acceleration - foot_jacobian_dot_v.head<3>());
            swing_row += 3;
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

    if (config.qddot_regularization_enabled) {
        problem.qddot_task.setZero(problem.nv, problem.decision_dim);
        problem.qddot_task.block(0, problem.qddot_offset,
                                 problem.nv, problem.nv) =
            qddot_weight * Eigen::MatrixXd::Identity(problem.nv, problem.nv);
        problem.qddot_task_rhs = Eigen::VectorXd::Zero(problem.nv);
    } else {
        problem.qddot_task.setZero(0, problem.decision_dim);
        problem.qddot_task_rhs.resize(0);
    }
    if (config.tau_regularization_enabled) {
        problem.tau_task.setZero(problem.tau_dim, problem.decision_dim);
        problem.tau_task.block(0, problem.tau_offset,
                               problem.tau_dim, problem.tau_dim) =
            tau_weight *
            Eigen::MatrixXd::Identity(problem.tau_dim, problem.tau_dim);
        problem.tau_task_rhs = Eigen::VectorXd::Zero(problem.tau_dim);
    } else {
        problem.tau_task.setZero(0, problem.decision_dim);
        problem.tau_task_rhs.resize(0);
    }

    return problem;
}

LinearQpResult solve_task_qp(const WholeBodyWbcProblem& problem,
                             const Eigen::MatrixXd& task,
                             const Eigen::VectorXd& task_rhs,
                             const Eigen::MatrixXd& equality,
                             const Eigen::VectorXd& equality_rhs,
                             int active_set_iterations) {
    Eigen::MatrixXd hessian =
        1.0e-8 *
        Eigen::MatrixXd::Identity(problem.decision_dim, problem.decision_dim);
    Eigen::VectorXd linear_rhs = Eigen::VectorXd::Zero(problem.decision_dim);
    if (task.rows() > 0) {
        hessian += task.transpose() * task;
        linear_rhs = task.transpose() * task_rhs;
    }
    return solve_active_set_qp(hessian, linear_rhs, equality, equality_rhs,
                               problem.inequality, problem.inequality_rhs,
                               active_set_iterations);
}

WholeBodyWbc::Result make_disabled_result(const WholeBodyWbc::Config& config,
                                          const WholeBodyWbc::Input& input) {
    WholeBodyWbc::Result result;
    result.tau.assign(config.joint_num, 0.0f);
    if (input.contacts) {
        result.contact_count =
            static_cast<int>(input.contacts->positions.size());
    }
    if (input.contact_result) {
        result.left_force = input.contact_result->left_force;
        result.right_force = input.contact_result->right_force;
        result.achieved_wrench = input.contact_result->achieved_wrench;
    }
    return result;
}

WholeBodyWbc::Result extract_result(const WholeBodyWbc::Config& config,
                                    const WholeBodyWbc::Input& input,
                                    const WholeBodyWbcProblem& problem,
                                    const LinearQpResult& qp_solution) {
    const auto& kinematics = *input.kinematics;
    const auto& contacts = *input.contacts;
    WholeBodyWbc::Result result;
    result.tau.assign(config.joint_num, 0.0f);
    result.contact_count = problem.contact_count;
    result.swing_error = problem.swing_error;
    result.max_qp_violation = qp_solution.max_violation;
    result.active_constraint_count = qp_solution.active_constraint_count;
    result.dynamics_residual =
        problem.equality.rows() > 0
            ? (problem.equality * qp_solution.x - problem.equality_rhs)
                  .lpNorm<Eigen::Infinity>()
            : 0.0;

    const Eigen::VectorXd solved_force =
        qp_solution.x.segment(problem.force_offset, problem.contact_dim);
    Vector3dList solved_contact_forces;
    solved_contact_forces.resize(problem.contact_count, Eigen::Vector3d::Zero());
    for (int i = 0; i < problem.contact_count; i++) {
        solved_contact_forces[i] = solved_force.segment<3>(i * 3);
        if (i < contacts.left_contact_count) {
            result.left_force += solved_contact_forces[i];
        } else {
            result.right_force += solved_contact_forces[i];
        }
    }
    result.achieved_wrench = contact_wrench(
        kinematics.com_position, contacts.positions, solved_contact_forces);

    const Eigen::VectorXd solved_tau =
        qp_solution.x.segment(problem.tau_offset, problem.tau_dim);
    for (int i = 0; i < config.joint_num; i++) {
        const double scale = config.torque_joint_scale.empty()
            ? 0.0
            : std::abs(config.torque_joint_scale[i]);
        const double torque_bound =
            problem.torque_blend * scale * config.max_joint_torque;
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
        if (torque_bound > 1.0e-6 &&
            raw_abs_tau > torque_bound - 1.0e-5) {
            result.saturated_joint_count++;
        }
        if (config.torque_enabled) {
            result.tau[i] = static_cast<float>(command_tau);
            result.max_command_joint_torque =
                std::max(result.max_command_joint_torque,
                         std::abs(command_tau));
        }
    }
    return result;
}

class WeightedWholeBodyWbc final : public WholeBodyWbc {
   public:
    explicit WeightedWholeBodyWbc(Config config)
        : WholeBodyWbc(std::move(config)) {}

    const std::string& name() const override { return name_; }

    Result solve(const Input& input) const override {
        validate_input(input);
        if (!config().enabled) {
            return make_disabled_result(config(), input);
        }
        const WholeBodyWbcProblem problem = build_problem(config(), input);
        const Eigen::MatrixXd task = stack_matrices(
            {problem.base_task, problem.force_task, problem.swing_task,
             problem.qddot_task, problem.tau_task},
            problem.decision_dim);
        const Eigen::VectorXd task_rhs = stack_vectors(
            {problem.base_task_rhs, problem.force_task_rhs,
             problem.swing_task_rhs, problem.qddot_task_rhs,
             problem.tau_task_rhs});
        const LinearQpResult qp_solution =
            solve_task_qp(problem, task, task_rhs, problem.equality,
                          problem.equality_rhs,
                          config().active_set_iterations);
        return extract_result(config(), input, problem, qp_solution);
    }

   private:
    std::string name_ = "weighted";
};

class HierarchicalWholeBodyWbc final : public WholeBodyWbc {
   public:
    explicit HierarchicalWholeBodyWbc(Config config)
        : WholeBodyWbc(std::move(config)) {}

    const std::string& name() const override { return name_; }

    Result solve(const Input& input) const override {
        validate_input(input);
        if (!config().enabled) {
            return make_disabled_result(config(), input);
        }
        const WholeBodyWbcProblem problem = build_problem(config(), input);
        if (problem.base_task.rows() <= 0) {
            const Eigen::MatrixXd task = stack_matrices(
                {problem.force_task, problem.swing_task, problem.qddot_task,
                 problem.tau_task},
                problem.decision_dim);
            const Eigen::VectorXd task_rhs = stack_vectors(
                {problem.force_task_rhs, problem.swing_task_rhs,
                 problem.qddot_task_rhs, problem.tau_task_rhs});
            const LinearQpResult qp_solution =
                solve_task_qp(problem, task, task_rhs, problem.equality,
                              problem.equality_rhs,
                              config().active_set_iterations);
            return extract_result(config(), input, problem, qp_solution);
        }

        const Eigen::MatrixXd priority0_task = stack_matrices(
            {problem.base_task, problem.qddot_task, problem.tau_task},
            problem.decision_dim);
        const Eigen::VectorXd priority0_rhs = stack_vectors(
            {problem.base_task_rhs, problem.qddot_task_rhs,
             problem.tau_task_rhs});
        const LinearQpResult priority0_solution =
            solve_task_qp(problem, priority0_task, priority0_rhs,
                          problem.equality, problem.equality_rhs,
                          config().active_set_iterations);

        Eigen::MatrixXd equality = problem.equality;
        Eigen::VectorXd equality_rhs = problem.equality_rhs;
        append_rows(equality, equality_rhs, problem.base_task,
                    problem.base_task * priority0_solution.x);

        const Eigen::MatrixXd priority1_task = stack_matrices(
            {problem.force_task, problem.swing_task, problem.qddot_task,
             problem.tau_task},
            problem.decision_dim);
        const Eigen::VectorXd priority1_rhs = stack_vectors(
            {problem.force_task_rhs, problem.swing_task_rhs,
             problem.qddot_task_rhs, problem.tau_task_rhs});
        LinearQpResult priority1_solution =
            solve_task_qp(problem, priority1_task, priority1_rhs, equality,
                          equality_rhs, config().active_set_iterations);
        priority1_solution.max_violation =
            std::max(priority1_solution.max_violation,
                     priority0_solution.max_violation);
        return extract_result(config(), input, problem, priority1_solution);
    }

   private:
    std::string name_ = "hierarchical";
};

}  // namespace

WholeBodyWbc::WholeBodyWbc(Config config) : config_(std::move(config)) {
    if (config_.joint_num <= 0) {
        throw std::runtime_error("WholeBodyWbc joint_num must be positive");
    }
    if (config_.active_set_iterations <= 0) {
        throw std::runtime_error(
            "WholeBodyWbc active_set_iterations must be positive");
    }
    if (config_.friction_coefficient < 0.0 ||
        config_.min_normal_force < 0.0 ||
        config_.max_normal_force < config_.min_normal_force ||
        config_.force_tracking_weight < 0.0 ||
        config_.moment_tracking_weight < 0.0 ||
        config_.regularization_weight < 0.0 ||
        config_.smooth_weight < 0.0 ||
        config_.swing_tracking_weight < 0.0 ||
        config_.swing_kp < 0.0 ||
        config_.swing_kd < 0.0 ||
        config_.max_joint_torque < 0.0) {
        throw std::runtime_error("WholeBodyWbc parameters are invalid");
    }
    if (!config_.torque_joint_scale.empty() &&
        config_.torque_joint_scale.size() !=
            static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error(
            "WholeBodyWbc torque_joint_scale size mismatch");
    }
}

void WholeBodyWbc::validate_input(const Input& input) const {
    if (!input.kinematics || !input.contacts || !input.gait_reference ||
        !input.mpc_output || !input.contact_result ||
        !input.configured_joint_velocity_indices) {
        throw std::runtime_error("WholeBodyWbc input has null fields");
    }
    const auto& kinematics = *input.kinematics;
    const auto& contacts = *input.contacts;
    const int nv = static_cast<int>(kinematics.mass_matrix.rows());
    const int contact_count = static_cast<int>(contacts.positions.size());
    const int contact_dim = contact_count * 3;
    if (contact_count <= 0 ||
        kinematics.mass_matrix.rows() != kinematics.mass_matrix.cols() ||
        kinematics.mass_matrix.cols() <= 0 ||
        kinematics.nonlinear_effects.size() != nv ||
        kinematics.base_jacobian.rows() != 6 ||
        kinematics.base_jacobian.cols() != nv ||
        kinematics.base_jacobian_dot_v.size() != 6 ||
        contacts.jacobian.rows() != contact_dim ||
        contacts.jacobian.cols() != nv ||
        contacts.jacobian_dot_v.size() != contact_dim ||
        contacts.left_contact_count < 0 ||
        contacts.left_contact_count > contact_count ||
        input.configured_joint_velocity_indices->size() !=
            static_cast<size_t>(config_.joint_num)) {
        throw std::runtime_error("WholeBodyWbc input dimensions are invalid");
    }
    for (int index : *input.configured_joint_velocity_indices) {
        if (index < 0 || index >= nv) {
            throw std::runtime_error(
                "WholeBodyWbc joint velocity index is invalid");
        }
    }
}

std::unique_ptr<WholeBodyWbc> create_whole_body_wbc(WholeBodyWbc::Config config) {
    if (config.solver == "weighted") {
        return std::make_unique<WeightedWholeBodyWbc>(std::move(config));
    }
    if (config.solver == "hierarchical") {
        return std::make_unique<HierarchicalWholeBodyWbc>(std::move(config));
    }
    throw std::runtime_error("unsupported stand_wbc_solver: " + config.solver);
}

}  // namespace whole_body_mpc
