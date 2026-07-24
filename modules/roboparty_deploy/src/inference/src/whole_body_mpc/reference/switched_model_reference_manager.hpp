// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/reference/contact_schedule.hpp"
#include "whole_body_mpc/reference/mode_schedule_adapter.hpp"
#include "whole_body_mpc/reference/swing_trajectory.hpp"

#include <Eigen/Core>

#include <ocs2_oc/synchronized_module/ReferenceManager.h>

namespace whole_body_mpc {

class SwitchedModelReferenceManager final : public ocs2::ReferenceManager {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SwitchedModelReferenceManager(ocs2::TargetTrajectories initial_target,
                                  ocs2::ModeSchedule initial_mode_schedule,
                                  SwingTrajectoryConfig swing_config);

    void set_reference_input(const ContactSchedule& contact_schedule,
                             double time_origin,
                             ContactFlags fallback_flags,
                             const Eigen::Vector3d& left_foot_position,
                             const Eigen::Vector3d& right_foot_position);

    void setModeSchedule(const ocs2::ModeSchedule& mode_schedule) override;
    void setModeSchedule(ocs2::ModeSchedule&& mode_schedule) override;

    ContactFlags contact_flags(double time) const;
    const SwingTrajectoryPlanner& swing_trajectory() const {
        return swing_trajectory_;
    }

   private:
    void modifyReferences(ocs2::scalar_t init_time,
                          ocs2::scalar_t final_time,
                          const ocs2::vector_t& init_state,
                          ocs2::TargetTrajectories& target_trajectories,
                          ocs2::ModeSchedule& mode_schedule) override;

    ContactSchedule contact_schedule_;
    SwingTrajectoryPlanner swing_trajectory_;
    ContactFlags fallback_flags_{true, true};
    Eigen::Vector3d left_foot_position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_position_ = Eigen::Vector3d::Zero();
    double time_origin_ = 0.0;
};

}  // namespace whole_body_mpc
