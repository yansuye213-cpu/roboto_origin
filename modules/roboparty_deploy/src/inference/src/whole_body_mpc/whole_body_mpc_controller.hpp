// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "standing_stabilizer.hpp"

#include <string>
#include <vector>

namespace whole_body_mpc {

class WholeBodyMpcController {
   public:
    explicit WholeBodyMpcController(const StandingStabilizer::Config& config);

    void reset();
    StandingStabilizer::Command apply(const StandingStabilizer::Measurement& measurement, float blend,
                                      const std::vector<float>& base_target,
                                      const std::vector<float>& kp,
                                      const std::vector<float>& kd);

    const std::string& model_path() const { return config_.whole_body_model_path; }

   private:
    void validate_model_config() const;

    StandingStabilizer::Config config_;
};

}  // namespace whole_body_mpc
