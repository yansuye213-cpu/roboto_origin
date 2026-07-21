// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/mpc/centroidal_mpc.hpp"

#ifdef ROBOPARTY_WITH_OCS2
#include "whole_body_mpc/mpc/ocs2_centroidal_mpc.hpp"
#endif

#include <stdexcept>
#include <utility>

namespace whole_body_mpc {

namespace {

bool is_supported_backend_name(const std::string& backend) {
    return backend == "disabled" || backend == "ocs2";
}

}  // namespace

CentroidalMpc::CentroidalMpc(Config config) : config_(std::move(config)) {
    if (config_.horizon <= 0) {
        throw std::runtime_error("CentroidalMpc horizon must be positive");
    }
    if (config_.dt <= 0.0 || config_.control_dt <= 0.0) {
        throw std::runtime_error("CentroidalMpc dt must be positive");
    }
    if (config_.orientation_weight < 0.0 ||
        config_.angular_rate_weight < 0.0 ||
        config_.com_weight < 0.0 ||
        config_.com_velocity_weight < 0.0 ||
        config_.terminal_weight_scale < 0.0 ||
        config_.input_smooth_weight < 0.0 ||
        config_.force_weight <= 0.0 ||
        config_.qp_iterations < 0 ||
        config_.friction_barrier_mu < 0.0 ||
        config_.friction_barrier_delta <= 0.0 ||
        config_.friction_regularization <= 0.0 ||
        config_.max_angular_accel < 0.0 ||
        config_.max_com_accel < 0.0 ||
        config_.max_contact_force_delta < 0.0 ||
        config_.friction_coefficient < 0.0 ||
        config_.min_normal_force < 0.0 ||
        config_.max_normal_force < config_.min_normal_force) {
        throw std::runtime_error("CentroidalMpc weights and limits are invalid");
    }
    if (!is_supported_backend_name(config_.backend)) {
        throw std::runtime_error("Unsupported CentroidalMpc backend: " + config_.backend);
    }
    backend_name_ = config_.enabled ? config_.backend : "disabled";
    backend_ = make_backend(config_);
}

void CentroidalMpc::reset() {
    if (backend_) {
        backend_->reset();
    }
}

CentroidalMpc::Output CentroidalMpc::solve(const Input& input) {
    Output output;
    output.state << input.roll - config_.target_roll,
                    input.pitch - config_.target_pitch,
                    input.wx,
                    input.wy,
                    input.com_offset_error.x(),
                    input.com_offset_error.y(),
                    input.com_velocity.x(),
                    input.com_velocity.y();
    output.backend = backend_name_;
    if (!config_.enabled || !backend_) {
        return output;
    }
    output = backend_->solve(input);
    return output;
}

std::unique_ptr<CentroidalMpcBackend> CentroidalMpc::make_backend(
    const Config& config) {
    if (!config.enabled || config.backend == "disabled") {
        return nullptr;
    }
    if (config.backend == "ocs2") {
#ifdef ROBOPARTY_WITH_OCS2
        return std::make_unique<Ocs2CentroidalMpc>(config);
#else
        throw std::runtime_error(
            "stand_wbc_mpc_backend=ocs2 requested, but OCS2 is not linked into "
            "roboparty_inference. Source /home/yansuye/ocs2_ws/install/setup.bash "
            "before building this package.");
#endif
    }
    throw std::runtime_error("Unsupported CentroidalMpc backend: " + config.backend);
}

}  // namespace whole_body_mpc
