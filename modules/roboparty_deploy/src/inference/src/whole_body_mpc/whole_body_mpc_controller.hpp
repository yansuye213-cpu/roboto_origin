// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "standing_stabilizer.hpp"
#include "whole_body_mpc/contact_force_qp.hpp"
#include "whole_body_mpc/robot_model.hpp"

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
    void validate_model_config() const;
    ContactForceQp::Input build_contact_qp_input(
        const StandingStabilizer::Measurement& measurement) const;
    std::vector<float> compute_joint_torque_command(
        const RobotModel::Kinematics& kinematics,
        const ContactForceQp::Result& contact_result,
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& v,
        float blend,
        StandingStabilizer::Correction& correction) const;

    StandingStabilizer::Config config_;
    std::unique_ptr<RobotModel> robot_model_;
    std::unique_ptr<ContactForceQp> contact_force_qp_;
    RobotModel::Kinematics neutral_kinematics_;
    RobotModel::Kinematics latest_kinematics_;
    Eigen::Vector2d neutral_com_offset_xy_ = Eigen::Vector2d::Zero();
    double robot_mass_ = 0.0;
};

}  // namespace whole_body_mpc
