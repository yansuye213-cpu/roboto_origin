// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/mpc/centroidal_mpc_backend.hpp"

#include <memory>

#include <ocs2_core/Types.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_sqp/SqpMpc.h>

namespace whole_body_mpc {

class Ocs2CentroidalModel;

class Ocs2CentroidalMpc final : public CentroidalMpcBackend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit Ocs2CentroidalMpc(CentroidalMpcConfig config);
    ~Ocs2CentroidalMpc() override;

    const std::string& name() const override { return name_; }
    void reset() override;
    CentroidalMpcOutput solve(const CentroidalMpcInput& input) override;

   private:
    ocs2::vector_t make_state(const CentroidalMpcInput& input);
    ocs2::matrix_t state_weight(bool terminal) const;
    ocs2::matrix_t input_weight() const;
    ocs2::TargetTrajectories make_target_trajectories(
        const CentroidalMpcInput& input, double time,
        const ocs2::vector_t& current_state) const;
    ocs2::vector_t nominal_input(bool left_contact, bool right_contact) const;
    ocs2::vector_t project_input(const ocs2::vector_t& control,
                                 bool left_contact,
                                 bool right_contact) const;
    Eigen::Vector3d project_foot_force(const Eigen::Vector3d& force,
                                       bool in_contact) const;
    void fill_output(const ocs2::vector_t& control,
                     const ocs2::vector_t& full_state,
                     const CentroidalMpcInput& input,
                     int iterations,
                     double objective,
                     CentroidalMpcOutput& output) const;
    void validate_input(const CentroidalMpcInput& input) const;
    void configure_solver();

    CentroidalMpcConfig config_;
    std::string name_ = "ocs2";
    std::unique_ptr<Ocs2CentroidalModel> model_;
    ocs2::OptimalControlProblem problem_;
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
    std::unique_ptr<ocs2::Initializer> initializer_;
    std::unique_ptr<ocs2::SqpMpc> mpc_;
    ocs2::vector_t last_input_;
    ocs2::vector_t nominal_joint_position_;
    bool has_last_input_ = false;
    double time_ = 0.0;
};

}  // namespace whole_body_mpc
