// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "standing_stabilizer.hpp"
#include "whole_body_mpc/estimation/base_state_estimator.hpp"
#include "whole_body_mpc/model/robot_model.hpp"
#include "whole_body_mpc/mpc/centroidal_mpc.hpp"
#include "whole_body_mpc/reference/contact_schedule.hpp"
#include "whole_body_mpc/reference/recovery_gait_planner.hpp"
#include "whole_body_mpc/wbc/contact_force_qp.hpp"
#include "whole_body_mpc/wbc/whole_body_wbc.hpp"

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
    using ContactPointSet = WholeBodyWbc::ContactPointSet;

    void validate_model_config() const;
    Eigen::Vector3d foot_contact_center(const Eigen::Isometry3d& foot_pose) const;
    CentroidalMpc::Input build_centroidal_mpc_input(
        const StandingStabilizer::Measurement& measurement,
        const ContactPointSet& contacts,
        const RecoveryGaitPlanner::Reference& gait_reference,
        const ContactSchedule& contact_schedule,
        const std::vector<float>& current_joint_position,
        const std::vector<float>& current_joint_velocity) const;
    ContactForceQp::Input build_contact_qp_input(
        const CentroidalMpc::Output& mpc_output,
        const ContactPointSet& contacts) const;
    ContactForceQp::Result make_nominal_contact_result(
        const ContactForceQp::Input& input) const;
    ContactPointSet build_contact_point_set(
        const RobotModel::Kinematics& kinematics,
        const RecoveryGaitPlanner::Reference& gait_reference) const;
    RecoveryGaitPlanner::Reference build_gait_reference(
        const StandingStabilizer::Measurement& measurement,
        const RobotModel::Kinematics& kinematics);
    ContactSchedule build_contact_schedule(
        const RobotModel::Kinematics& kinematics,
        const RecoveryGaitPlanner::Reference& gait_reference) const;
    void apply_swing_ik_targets(
        const RobotModel::Kinematics& kinematics,
        const RecoveryGaitPlanner::Reference& gait_reference,
        const std::vector<float>& current_joint_position,
        std::vector<float>& command_position,
        std::vector<float>& command_velocity) const;
    void apply_mpc_joint_command(
        const CentroidalMpc::Output& mpc_output,
        const std::vector<float>& current_joint_position,
        std::vector<float>& command_position,
        std::vector<float>& command_velocity) const;
    void apply_foot_ik_target(
        const Eigen::Matrix<double, 6, Eigen::Dynamic>& foot_jacobian,
        const Eigen::Vector3d& position_error,
        const Eigen::Vector3d& desired_velocity,
        const std::vector<int>& command_joint_indices,
        const std::vector<float>& current_joint_position,
        std::vector<float>& command_position,
        std::vector<float>& command_velocity) const;
    void initialize_leg_joint_indices();

    StandingStabilizer::Config config_;
    std::unique_ptr<RobotModel> robot_model_;
    std::unique_ptr<BaseStateEstimator> base_state_estimator_;
    std::unique_ptr<CentroidalMpc> centroidal_mpc_;
    std::unique_ptr<ContactForceQp> contact_force_qp_;
    std::unique_ptr<WholeBodyWbc> whole_body_wbc_;
    std::unique_ptr<RecoveryGaitPlanner> recovery_gait_planner_;
    std::unique_ptr<ContactSchedulePlanner> contact_schedule_planner_;
    RobotModel::Kinematics neutral_kinematics_;
    RobotModel::Kinematics latest_kinematics_;
    BaseStateEstimator::Output latest_state_estimate_;
    Eigen::Vector2d neutral_com_offset_xy_ = Eigen::Vector2d::Zero();
    double robot_mass_ = 0.0;
    bool state_estimator_left_contact_ = true;
    bool state_estimator_right_contact_ = true;
    std::vector<int> left_leg_joint_indices_;
    std::vector<int> right_leg_joint_indices_;
};

}  // namespace whole_body_mpc
