// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/whole_body_mpc_controller.hpp"

#include <filesystem>
#include <stdexcept>

namespace whole_body_mpc {

WholeBodyMpcController::WholeBodyMpcController(const StandingStabilizer::Config& config)
    : config_(config) {
    validate_model_config();
}

void WholeBodyMpcController::reset() {}

StandingStabilizer::Command WholeBodyMpcController::apply(
    const StandingStabilizer::Measurement& measurement, float blend,
    const std::vector<float>& base_target, const std::vector<float>& kp,
    const std::vector<float>& kd) {
    (void)measurement;
    (void)blend;
    (void)base_target;
    (void)kp;
    (void)kd;
    throw std::runtime_error(
        "whole_body_mpc backend is configured but not implemented yet: next steps are URDF model loading, "
        "foot Jacobians, contact-force QP, and tau = J^T f torque mapping");
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
}

}  // namespace whole_body_mpc
