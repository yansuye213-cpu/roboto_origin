// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/mpc/centroidal_mpc_backend.hpp"

#include <memory>

#include <ocs2_core/Types.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_sqp/SqpMpc.h>

namespace whole_body_mpc {

struct Ocs2CentroidalMpcData;

class Ocs2CentroidalMpc final : public CentroidalMpcBackend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit Ocs2CentroidalMpc(CentroidalMpcConfig config);

    const std::string& name() const override { return name_; }
    void reset() override;
    CentroidalMpcOutput solve(const CentroidalMpcInput& input) override;

   private:
    static constexpr int kStateDim = 8;
    static constexpr int kInputDim = 6;

    Eigen::Matrix<double, kStateDim, 1> make_state(
        const CentroidalMpcInput& input) const;
    ocs2::matrix_t state_weight(bool terminal) const;
    ocs2::matrix_t input_weight() const;
    ocs2::TargetTrajectories make_target_trajectories(
        const CentroidalMpcInput& input, double time) const;
    Eigen::Matrix<double, kInputDim, 1> clamp_force_delta(
        const Eigen::Matrix<double, kInputDim, 1>& control,
        const CentroidalMpcInput& input) const;
    Eigen::Vector3d nominal_foot_force(bool in_contact, int contact_feet,
                                       const CentroidalMpcInput& input) const;
    Eigen::Vector3d project_foot_force(const Eigen::Vector3d& force,
                                       bool in_contact) const;
    void fill_output(const Eigen::Matrix<double, kInputDim, 1>& control,
                     const Eigen::Matrix<double, kStateDim, 1>& state,
                     const CentroidalMpcInput& input,
                     int iterations,
                     double objective,
                     CentroidalMpcOutput& output) const;
    void update_dynamics_data(const CentroidalMpcInput& input);
    void configure_solver();

    CentroidalMpcConfig config_;
    std::string name_ = "ocs2";
    std::shared_ptr<Ocs2CentroidalMpcData> dynamics_data_;
    ocs2::OptimalControlProblem problem_;
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
    ocs2::DefaultInitializer initializer_;
    std::unique_ptr<ocs2::SqpMpc> mpc_;
    Eigen::Matrix<double, kInputDim, 1> last_control_ =
        Eigen::Matrix<double, kInputDim, 1>::Zero();
    double time_ = 0.0;
};

}  // namespace whole_body_mpc
