// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "standing_stabilizer.hpp"
#include "whole_body_mpc/contact_force_qp.hpp"
#include "whole_body_mpc/robot_model.hpp"
#include "whole_body_mpc/stance_mpc.hpp"

#include <memory>
#include <string>
#include <vector>

namespace whole_body_mpc {

class WholeBodyMpcController {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit WholeBodyMpcController(const StandingStabilizer::Config& config);

    void reset();
    std::vector<std::string> diagnostics() const;

    StandingStabilizer::Command apply(const StandingStabilizer::Measurement& measurement, float blend,
                                      const std::vector<float>& base_target,
                                      const std::vector<float>& kp,
                                      const std::vector<float>& kd,
                                      const std::vector<float>& current_joint_position,
                                      const std::vector<float>& current_joint_velocity);

    const std::string& model_path() const { return config_.whole_body_model_path; }

   private:
    struct WbcQpResult {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        std::vector<float> tau;
        Eigen::Vector3d left_force = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_force = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, 1> achieved_wrench = Eigen::Matrix<double, 6, 1>::Zero();
        double max_raw_joint_torque = 0.0;
        double max_command_joint_torque = 0.0;
        int saturated_joint_count = 0;
        int max_torque_joint_index = -1;
    };

    void validate_model_config() const;
    StanceMpc::Input build_stance_mpc_input(
        const StandingStabilizer::Measurement& measurement) const;
    ContactForceQp::Input build_contact_qp_input(
        const StanceMpc::Output& mpc_output) const;
    WbcQpResult solve_stance_wbc_qp(
        const RobotModel::Kinematics& kinematics,
        const StanceMpc::Output& mpc_output,
        const ContactForceQp::Result& contact_result,
        float blend) const;

    StandingStabilizer::Config config_;
    std::unique_ptr<RobotModel> robot_model_;
    std::unique_ptr<StanceMpc> stance_mpc_;
    std::unique_ptr<ContactForceQp> contact_force_qp_;
    RobotModel::Kinematics neutral_kinematics_;
    RobotModel::Kinematics latest_kinematics_;
    Eigen::Vector2d neutral_com_offset_xy_ = Eigen::Vector2d::Zero();
    double robot_mass_ = 0.0;
};

}  // namespace whole_body_mpc
