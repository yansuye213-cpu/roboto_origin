// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/model/robot_model.hpp"
#include "whole_body_mpc/mpc/centroidal_mpc_types.hpp"
#include "whole_body_mpc/reference/recovery_gait_planner.hpp"
#include "whole_body_mpc/wbc/contact_force_qp.hpp"

#include <memory>
#include <string>
#include <vector>

namespace whole_body_mpc {

class WholeBodyWbc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        int joint_num = 0;
        std::string solver = "weighted";
        bool enabled = true;
        bool floating_base_eom_enabled = true;
        bool stance_contact_constraint_enabled = true;
        bool friction_constraint_enabled = true;
        bool torque_limit_constraint_enabled = true;
        bool base_accel_task_enabled = true;
        bool contact_force_task_enabled = true;
        bool swing_task_enabled = true;
        bool qddot_regularization_enabled = true;
        bool tau_regularization_enabled = true;
        bool torque_enabled = false;
        int active_set_iterations = 80;
        double friction_coefficient = 0.45;
        double min_normal_force = 0.0;
        double max_normal_force = 420.0;
        double force_tracking_weight = 1.0;
        double moment_tracking_weight = 0.03;
        double regularization_weight = 1.0e-4;
        double smooth_weight = 0.02;
        double swing_tracking_weight = 20.0;
        double swing_kp = 60.0;
        double swing_kd = 8.0;
        double max_joint_torque = 0.6;
        std::vector<double> torque_joint_scale;
    };

    struct ContactPointSet {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Vector3dList positions;
        int left_contact_count = 0;
        Eigen::MatrixXd jacobian;
        Eigen::VectorXd jacobian_dot_v;
    };

    struct Input {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        const RobotModel::Kinematics* kinematics = nullptr;
        const ContactPointSet* contacts = nullptr;
        const RecoveryGaitPlanner::Reference* gait_reference = nullptr;
        const CentroidalMpcOutput* mpc_output = nullptr;
        const ContactForceQp::Result* contact_result = nullptr;
        const std::vector<int>* configured_joint_velocity_indices = nullptr;
        double blend = 1.0;
    };

    struct Result {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        std::vector<float> tau;
        Eigen::Vector3d left_force = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_force = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, 1> achieved_wrench =
            Eigen::Matrix<double, 6, 1>::Zero();
        double max_raw_joint_torque = 0.0;
        double max_command_joint_torque = 0.0;
        double max_qp_violation = 0.0;
        double dynamics_residual = 0.0;
        double swing_error = 0.0;
        int contact_count = 0;
        int active_constraint_count = 0;
        int saturated_joint_count = 0;
        int max_torque_joint_index = -1;
    };

    explicit WholeBodyWbc(Config config);
    virtual ~WholeBodyWbc() = default;

    virtual const std::string& name() const = 0;
    virtual void reset() {}
    virtual Result solve(const Input& input) const = 0;

   protected:
    const Config& config() const { return config_; }
    void validate_input(const Input& input) const;

   private:
    Config config_;
};

std::unique_ptr<WholeBodyWbc> create_whole_body_wbc(WholeBodyWbc::Config config);

}  // namespace whole_body_mpc
