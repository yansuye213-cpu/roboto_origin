// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "standing_stabilizer.hpp"

#include <vector>

namespace stand_control {

class JointQpStandController {
   public:
    explicit JointQpStandController(const StandingStabilizer::Config& config);

    void reset();
    StandingStabilizer::Command apply(const StandingStabilizer::Measurement& measurement, float blend,
                                      const std::vector<float>& base_target,
                                      const std::vector<float>& kp,
                                      const std::vector<float>& kd,
                                      const std::vector<float>& current_joint_position,
                                      const std::vector<float>& current_joint_velocity);

   private:
    float solve_axis(float angle, float rate, float target_angle) const;
    std::vector<float> solve_joint_qp(float roll_correction, float pitch_correction,
                                      const std::vector<float>& base_target) const;
    std::vector<float> solve_scale_allocation(float roll_correction, float pitch_correction) const;
    void apply_joint_delta(const std::vector<float>& base_target, const std::vector<float>& joint_delta,
                           std::vector<float>& target) const;
    void update_last_joint_delta(const std::vector<float>& base_target, const std::vector<float>& target,
                                 StandingStabilizer::Correction& correction);

    StandingStabilizer::Config config_;
    std::vector<float> last_joint_delta_;
};

}  // namespace stand_control
