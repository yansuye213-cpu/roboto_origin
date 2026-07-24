// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/reference/contact_schedule.hpp"
#include "whole_body_mpc/reference/mode_schedule_adapter.hpp"

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <array>
#include <vector>

#include <ocs2_core/reference/ModeSchedule.h>

namespace whole_body_mpc {

struct SwingTrajectoryConfig {
    double lift_off_velocity = 0.0;
    double touch_down_velocity = 0.0;
    double swing_height = 0.035;
    double swing_time_scale = 0.15;
};

class SwingTrajectoryPlanner {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit SwingTrajectoryPlanner(SwingTrajectoryConfig config = {});

    void update(const ocs2::ModeSchedule& mode_schedule,
                const ContactSchedule& contact_schedule,
                double time_origin,
                const Eigen::Vector3d& left_foot_position,
                const Eigen::Vector3d& right_foot_position);

    Eigen::Vector3d position(size_t contact_index, double time) const;
    Eigen::Vector3d velocity(size_t contact_index, double time) const;
    Eigen::Vector3d acceleration(size_t contact_index, double time) const;

   private:
    struct CubicSegment {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double start_time = 0.0;
        double end_time = 0.0;
        Eigen::Vector3d a0 = Eigen::Vector3d::Zero();
        Eigen::Vector3d a1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d a2 = Eigen::Vector3d::Zero();
        Eigen::Vector3d a3 = Eigen::Vector3d::Zero();

        bool contains(double time) const;
        Eigen::Vector3d position(double time) const;
        Eigen::Vector3d velocity(double time) const;
        Eigen::Vector3d acceleration(double time) const;
    };

    using SegmentVector =
        std::vector<CubicSegment, Eigen::aligned_allocator<CubicSegment>>;

    void add_constant_segment(size_t contact_index,
                              double start_time,
                              double end_time,
                              const Eigen::Vector3d& position);
    void add_swing_segments(size_t contact_index,
                            double start_time,
                            double end_time,
                            const Eigen::Vector3d& start_position,
                            const Eigen::Vector3d& end_position);
    static CubicSegment make_segment(double start_time,
                                     double end_time,
                                     const Eigen::Vector3d& start_position,
                                     const Eigen::Vector3d& start_velocity,
                                     const Eigen::Vector3d& end_position,
                                     const Eigen::Vector3d& end_velocity);
    static ContactFlags sample_flags(const ContactScheduleSample& sample);
    static Eigen::Vector3d sample_foot_position(const ContactScheduleSample& sample,
                                                size_t contact_index);
    static bool sample_foot_swing(const ContactScheduleSample& sample,
                                  size_t contact_index);
    static double scaled_swing_height(double duration,
                                      const SwingTrajectoryConfig& config);

    SwingTrajectoryConfig config_;
    std::array<SegmentVector, 2> segments_;
    std::array<Eigen::Vector3d, 2> fallback_position_;
};

}  // namespace whole_body_mpc
