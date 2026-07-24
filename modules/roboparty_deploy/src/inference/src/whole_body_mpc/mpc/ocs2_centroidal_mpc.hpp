// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/mpc/centroidal_mpc_backend.hpp"

#include <memory>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

#include <ocs2_core/Types.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_sqp/SqpMpc.h>

namespace whole_body_mpc {

class Ocs2CentroidalModel;
class SwitchedModelReferenceManager;

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
    struct SolveRequest;
    struct PolicySnapshot;
    SolveRequest make_solve_request(const CentroidalMpcInput& input,
                                    const ocs2::vector_t& state,
                                    double time) const;
    bool run_solver_iteration(const SolveRequest& request,
                              PolicySnapshot& policy);
    bool compute_control_from_policy(const ocs2::PrimalSolution& solution,
                                     double time,
                                     const ocs2::vector_t& state,
                                     ocs2::vector_t& control) const;
    bool try_get_policy_snapshot(double time, PolicySnapshot& policy) const;
    void publish_policy_snapshot(PolicySnapshot policy);
    void start_mrt_worker();
    void stop_mrt_worker();
    void enqueue_mrt_request(SolveRequest request);
    void mrt_worker_loop();
    void validate_input(const CentroidalMpcInput& input) const;
    void configure_solver();

    CentroidalMpcConfig config_;
    std::string name_ = "ocs2";
    std::unique_ptr<Ocs2CentroidalModel> model_;
    ocs2::OptimalControlProblem problem_;
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
    std::shared_ptr<SwitchedModelReferenceManager> switched_reference_manager_;
    std::unique_ptr<ocs2::Initializer> initializer_;
    std::unique_ptr<ocs2::SqpMpc> mpc_;
    ocs2::vector_t last_input_;
    ocs2::vector_t nominal_joint_position_;
    ocs2::vector_t joint_position_lower_;
    ocs2::vector_t joint_position_upper_;
    bool has_last_input_ = false;
    double time_ = 0.0;
    mutable std::mutex solver_mutex_;
    std::mutex mrt_request_mutex_;
    std::condition_variable mrt_request_cv_;
    std::thread mrt_worker_;
    bool mrt_worker_running_ = false;
    bool mrt_stop_requested_ = false;
    bool mrt_request_pending_ = false;
    uint64_t mrt_next_sequence_ = 1;
    std::unique_ptr<SolveRequest> mrt_pending_request_;
    mutable std::mutex mrt_policy_mutex_;
    std::unique_ptr<PolicySnapshot> mrt_policy_;
};

}  // namespace whole_body_mpc
