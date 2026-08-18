// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/mpc/ocs2_centroidal_mpc.hpp"

#include "whole_body_mpc/model/centroidal_model_info.hpp"
#include "whole_body_mpc/reference/mode_schedule_adapter.hpp"
#include "whole_body_mpc/reference/switched_model_reference_manager.hpp"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamicsAD.h>
#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/QuadraticPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace whole_body_mpc {

namespace {

constexpr double kGravity = 9.80665;
constexpr size_t kLeftContactIndex = 0;
constexpr size_t kRightContactIndex = 1;

double clamp_abs(double value, double limit) {
    if (limit <= 0.0) {
        return 0.0;
    }
    return std::clamp(value, -limit, limit);
}

Eigen::Vector2d clamp_norm(const Eigen::Vector2d& value, double max_norm) {
    if (max_norm <= 0.0) {
        return Eigen::Vector2d::Zero();
    }
    const double norm = value.norm();
    if (norm <= max_norm || norm <= 1.0e-9) {
        return value;
    }
    return value * (max_norm / norm);
}

bool foot_in_contact(ContactFlags flags, size_t contact_index) {
    return contact_index == kLeftContactIndex ? flags.left : flags.right;
}

int force_offset(size_t contact_index) {
    return static_cast<int>(3 * contact_index);
}

ocs2::vector_t make_weight_compensating_input(
    const ocs2::CentroidalModelInfo& info,
    ContactFlags flags) {
    ocs2::vector_t input = ocs2::vector_t::Zero(info.inputDim);
    const int stance_feet = contact_count(flags);
    if (stance_feet <= 0) {
        return input;
    }
    const Eigen::Vector3d force(0.0, 0.0,
                                info.robotMass * kGravity /
                                    static_cast<double>(stance_feet));
    if (flags.left) {
        ocs2::centroidal_model::getContactForces(
            input, kLeftContactIndex, info) = force;
    }
    if (flags.right) {
        ocs2::centroidal_model::getContactForces(
            input, kRightContactIndex, info) = force;
    }
    return input;
}

ContactFlags flags_from_reference_manager(
    const std::shared_ptr<ocs2::ReferenceManager>& reference_manager,
    double time) {
    if (!reference_manager) {
        return ContactFlags{};
    }
    return contact_flags_at_time(reference_manager->getModeSchedule(), time);
}

class PinocchioCentroidalSystemDynamics final
    : public ocs2::SystemDynamicsBase {
   public:
    PinocchioCentroidalSystemDynamics(
        const ocs2::PinocchioInterface& pinocchio_interface,
        const ocs2::CentroidalModelInfo& info,
        const CentroidalMpcConfig& config)
        : dynamics_(pinocchio_interface, info, "roboparty_centroidal_dynamics",
                    config.ad_model_folder, config.ad_recompile,
                    config.ad_verbose) {}

    PinocchioCentroidalSystemDynamics(
        const PinocchioCentroidalSystemDynamics& other)
        : ocs2::SystemDynamicsBase(other), dynamics_(other.dynamics_) {}

    PinocchioCentroidalSystemDynamics* clone() const override {
        return new PinocchioCentroidalSystemDynamics(*this);
    }

    ocs2::vector_t computeFlowMap(ocs2::scalar_t time,
                                  const ocs2::vector_t& state,
                                  const ocs2::vector_t& input,
                                  const ocs2::PreComputation&) override {
        return dynamics_.getValue(time, state, input);
    }

    ocs2::VectorFunctionLinearApproximation linearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation&) override {
        return dynamics_.getLinearApproximation(time, state, input);
    }

   private:
    ocs2::PinocchioCentroidalDynamicsAD dynamics_;
};

class WeightCompensatingInitializer final : public ocs2::Initializer {
   public:
    WeightCompensatingInitializer(
        ocs2::CentroidalModelInfo info,
        std::shared_ptr<ocs2::ReferenceManager> reference_manager)
        : info_(std::move(info)),
          reference_manager_(std::move(reference_manager)) {}

    WeightCompensatingInitializer(const WeightCompensatingInitializer& other)
        : ocs2::Initializer(other),
          info_(other.info_),
          reference_manager_(other.reference_manager_) {}

    WeightCompensatingInitializer* clone() const override {
        return new WeightCompensatingInitializer(*this);
    }

    void compute(ocs2::scalar_t time,
                 const ocs2::vector_t& state,
                 ocs2::scalar_t,
                 ocs2::vector_t& input,
                 ocs2::vector_t& next_state) override {
        input = make_weight_compensating_input(
            info_, flags_from_reference_manager(reference_manager_, time));
        next_state = state;
    }

   private:
    ocs2::CentroidalModelInfo info_;
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
};

std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
make_foot_kinematics(const Ocs2CentroidalModel& model,
                     const ocs2::CentroidalModelInfo& info,
                     const CentroidalMpcConfig& config,
                     const std::string& foot_frame,
                     const std::string& model_name) {
    const ocs2::CentroidalModelInfoCppAd info_cppad = info.toCppAd();
    const ocs2::CentroidalModelPinocchioMappingCppAd pinocchio_mapping(
        info_cppad);
    auto update_callback =
        [info_cppad](const ocs2::ad_vector_t& state,
                     ocs2::PinocchioInterfaceCppAd& pinocchio_interface_ad) {
            const ocs2::ad_vector_t q =
                ocs2::centroidal_model::getGeneralizedCoordinates(
                    state, info_cppad);
            ocs2::updateCentroidalDynamics(pinocchio_interface_ad, info_cppad,
                                           q);
        };
    return std::make_unique<ocs2::PinocchioEndEffectorKinematicsCppAd>(
        model.pinocchio_interface(), pinocchio_mapping,
        std::vector<std::string>{foot_frame}, info.stateDim, info.inputDim,
        update_callback, model_name, config.ad_model_folder,
        config.ad_recompile, config.ad_verbose);
}

class FootZeroVelocityConstraint final : public ocs2::StateInputConstraint {
   public:
    FootZeroVelocityConstraint(
        std::shared_ptr<SwitchedModelReferenceManager> reference_manager,
        const ocs2::EndEffectorKinematics<ocs2::scalar_t>& kinematics,
        size_t contact_index)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          reference_manager_(std::move(reference_manager)),
          kinematics_(kinematics.clone()),
          contact_index_(contact_index) {}

    FootZeroVelocityConstraint(const FootZeroVelocityConstraint& other)
        : ocs2::StateInputConstraint(other),
          reference_manager_(other.reference_manager_),
          kinematics_(other.kinematics_->clone()),
          contact_index_(other.contact_index_) {}

    FootZeroVelocityConstraint* clone() const override {
        return new FootZeroVelocityConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return foot_in_contact(reference_manager_->contact_flags(time),
                               contact_index_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override { return 3; }

    ocs2::vector_t getValue(ocs2::scalar_t,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        return kinematics_->getVelocity(state, input).front();
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation&) const override {
        return kinematics_->getVelocityLinearApproximation(state, input)
            .front();
    }

   private:
    std::shared_ptr<SwitchedModelReferenceManager> reference_manager_;
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>> kinematics_;
    size_t contact_index_ = 0;
};

class SwingFootNormalVelocityConstraint final
    : public ocs2::StateInputConstraint {
   public:
    SwingFootNormalVelocityConstraint(
        std::shared_ptr<SwitchedModelReferenceManager> reference_manager,
        const ocs2::EndEffectorKinematics<ocs2::scalar_t>& kinematics,
        size_t contact_index)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          reference_manager_(std::move(reference_manager)),
          kinematics_(kinematics.clone()),
          contact_index_(contact_index) {}

    SwingFootNormalVelocityConstraint(
        const SwingFootNormalVelocityConstraint& other)
        : ocs2::StateInputConstraint(other),
          reference_manager_(other.reference_manager_),
          kinematics_(other.kinematics_->clone()),
          contact_index_(other.contact_index_) {}

    SwingFootNormalVelocityConstraint* clone() const override {
        return new SwingFootNormalVelocityConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return !foot_in_contact(reference_manager_->contact_flags(time),
                                contact_index_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override { return 1; }

    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        const Eigen::Vector3d velocity =
            kinematics_->getVelocity(state, input).front();
        const Eigen::Vector3d reference_velocity =
            reference_manager_->swing_trajectory().velocity(contact_index_,
                                                            time);
        return (ocs2::vector_t(1) << velocity.z() - reference_velocity.z())
            .finished();
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        (void)pre_computation;
        const ocs2::VectorFunctionLinearApproximation velocity =
            kinematics_->getVelocityLinearApproximation(state, input).front();
        ocs2::VectorFunctionLinearApproximation output =
            ocs2::VectorFunctionLinearApproximation::Zero(1, state.size(),
                                                          input.size());
        output.f[0] =
            velocity.f.z() -
            reference_manager_->swing_trajectory().velocity(contact_index_,
                                                            time)
                .z();
        output.dfdx = velocity.dfdx.row(2);
        output.dfdu = velocity.dfdu.row(2);
        return output;
    }

   private:
    std::shared_ptr<SwitchedModelReferenceManager> reference_manager_;
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>> kinematics_;
    size_t contact_index_ = 0;
};

class SwingFootPositionConstraint final : public ocs2::StateInputConstraint {
   public:
    SwingFootPositionConstraint(
        std::shared_ptr<SwitchedModelReferenceManager> reference_manager,
        const ocs2::EndEffectorKinematics<ocs2::scalar_t>& kinematics,
        size_t contact_index)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          reference_manager_(std::move(reference_manager)),
          kinematics_(kinematics.clone()),
          contact_index_(contact_index) {}

    SwingFootPositionConstraint(const SwingFootPositionConstraint& other)
        : ocs2::StateInputConstraint(other),
          reference_manager_(other.reference_manager_),
          kinematics_(other.kinematics_->clone()),
          contact_index_(other.contact_index_) {}

    SwingFootPositionConstraint* clone() const override {
        return new SwingFootPositionConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return !foot_in_contact(reference_manager_->contact_flags(time),
                                contact_index_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override { return 3; }

    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)input;
        return kinematics_->getPosition(state).front() -
               reference_manager_->swing_trajectory().position(contact_index_,
                                                               time);
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        (void)input;
        (void)pre_computation;
        ocs2::VectorFunctionLinearApproximation output =
            kinematics_->getPositionLinearApproximation(state).front();
        output.f -=
            reference_manager_->swing_trajectory().position(contact_index_,
                                                            time);
        return output;
    }

   private:
    std::shared_ptr<SwitchedModelReferenceManager> reference_manager_;
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>> kinematics_;
    size_t contact_index_ = 0;
};

class SwingFootZeroForceConstraint final : public ocs2::StateInputConstraint {
   public:
    SwingFootZeroForceConstraint(
        std::shared_ptr<ocs2::ReferenceManager> reference_manager,
        ocs2::CentroidalModelInfo info,
        size_t contact_index)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          reference_manager_(std::move(reference_manager)),
          info_(std::move(info)),
          contact_index_(contact_index) {}

    SwingFootZeroForceConstraint(const SwingFootZeroForceConstraint& other)
        : ocs2::StateInputConstraint(other),
          reference_manager_(other.reference_manager_),
          info_(other.info_),
          contact_index_(other.contact_index_) {}

    SwingFootZeroForceConstraint* clone() const override {
        return new SwingFootZeroForceConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return !foot_in_contact(
            flags_from_reference_manager(reference_manager_, time),
            contact_index_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override { return 3; }

    ocs2::vector_t getValue(ocs2::scalar_t,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        ocs2::vector_t value(3);
        value = ocs2::centroidal_model::getContactForces(
            input, contact_index_, info_);
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(3, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(3, input.size());
        approximation.dfdu.block<3, 3>(0, force_offset(contact_index_))
            .setIdentity();
        return approximation;
    }

   private:
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
    ocs2::CentroidalModelInfo info_;
    size_t contact_index_ = 0;
};

class ContactFootNormalForceBoundsConstraint final
    : public ocs2::StateInputConstraint {
   public:
    ContactFootNormalForceBoundsConstraint(
        std::shared_ptr<ocs2::ReferenceManager> reference_manager,
        ocs2::CentroidalModelInfo info,
        CentroidalMpcConfig config,
        size_t contact_index)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          reference_manager_(std::move(reference_manager)),
          info_(std::move(info)),
          config_(std::move(config)),
          contact_index_(contact_index) {}

    ContactFootNormalForceBoundsConstraint(
        const ContactFootNormalForceBoundsConstraint& other)
        : ocs2::StateInputConstraint(other),
          reference_manager_(other.reference_manager_),
          info_(other.info_),
          config_(other.config_),
          contact_index_(other.contact_index_) {}

    ContactFootNormalForceBoundsConstraint* clone() const override {
        return new ContactFootNormalForceBoundsConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return foot_in_contact(
            flags_from_reference_manager(reference_manager_, time),
            contact_index_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override { return 2; }

    ocs2::vector_t getValue(ocs2::scalar_t,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        const double fz = ocs2::centroidal_model::getContactForces(
                              input, contact_index_, info_)
                              .z();
        ocs2::vector_t value(2);
        value << fz - config_.min_normal_force,
                 config_.max_normal_force - fz;
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(2, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(2, input.size());
        const int z_index = force_offset(contact_index_) + 2;
        approximation.dfdu(0, z_index) = 1.0;
        approximation.dfdu(1, z_index) = -1.0;
        return approximation;
    }

   private:
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
    ocs2::CentroidalModelInfo info_;
    CentroidalMpcConfig config_;
    size_t contact_index_ = 0;
};

class ContactForceDeviationBoundsConstraint final
    : public ocs2::StateInputConstraint {
   public:
    ContactForceDeviationBoundsConstraint(
        std::shared_ptr<ocs2::ReferenceManager> reference_manager,
        ocs2::CentroidalModelInfo info,
        CentroidalMpcConfig config)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          reference_manager_(std::move(reference_manager)),
          info_(std::move(info)),
          config_(std::move(config)) {}

    ContactForceDeviationBoundsConstraint(
        const ContactForceDeviationBoundsConstraint& other)
        : ocs2::StateInputConstraint(other),
          reference_manager_(other.reference_manager_),
          info_(other.info_),
          config_(other.config_) {}

    ContactForceDeviationBoundsConstraint* clone() const override {
        return new ContactForceDeviationBoundsConstraint(*this);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override { return 12; }

    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        const ocs2::vector_t nominal = make_weight_compensating_input(
            info_, flags_from_reference_manager(reference_manager_, time));
        ocs2::vector_t value(12);
        const double limit = config_.max_contact_force_delta;
        for (int i = 0; i < 6; i++) {
            const double deviation = input[i] - nominal[i];
            value[2 * i] = limit - deviation;
            value[2 * i + 1] = limit + deviation;
        }
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(12, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(12, input.size());
        for (int i = 0; i < 6; i++) {
            approximation.dfdu(2 * i, i) = -1.0;
            approximation.dfdu(2 * i + 1, i) = 1.0;
        }
        return approximation;
    }

   private:
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
    ocs2::CentroidalModelInfo info_;
    CentroidalMpcConfig config_;
};

class JointVelocityBoundsConstraint final : public ocs2::StateInputConstraint {
   public:
    JointVelocityBoundsConstraint(ocs2::CentroidalModelInfo info,
                                  double velocity_limit)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          info_(std::move(info)),
          velocity_limit_(std::max(velocity_limit, 0.0)) {}

    JointVelocityBoundsConstraint(const JointVelocityBoundsConstraint& other)
        : ocs2::StateInputConstraint(other),
          info_(other.info_),
          velocity_limit_(other.velocity_limit_) {}

    JointVelocityBoundsConstraint* clone() const override {
        return new JointVelocityBoundsConstraint(*this);
    }

    bool isActive(ocs2::scalar_t) const override {
        return velocity_limit_ > 0.0;
    }

    size_t getNumConstraints(ocs2::scalar_t) const override {
        return 2 * info_.actuatedDofNum;
    }

    ocs2::vector_t getValue(ocs2::scalar_t,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        ocs2::vector_t value(2 * info_.actuatedDofNum);
        const int joint_velocity_offset = 3 * info_.numThreeDofContacts +
                                          6 * info_.numSixDofContacts;
        for (size_t i = 0; i < info_.actuatedDofNum; i++) {
            const double velocity =
                input[joint_velocity_offset + static_cast<int>(i)];
            value[2 * static_cast<int>(i)] = velocity_limit_ - velocity;
            value[2 * static_cast<int>(i) + 1] = velocity_limit_ + velocity;
        }
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx =
            ocs2::matrix_t::Zero(getNumConstraints(time), state.size());
        approximation.dfdu =
            ocs2::matrix_t::Zero(getNumConstraints(time), input.size());
        const int joint_velocity_offset = 3 * info_.numThreeDofContacts +
                                          6 * info_.numSixDofContacts;
        for (size_t i = 0; i < info_.actuatedDofNum; i++) {
            const int row = 2 * static_cast<int>(i);
            const int col = joint_velocity_offset + static_cast<int>(i);
            approximation.dfdu(row, col) = -1.0;
            approximation.dfdu(row + 1, col) = 1.0;
        }
        return approximation;
    }

   private:
    ocs2::CentroidalModelInfo info_;
    double velocity_limit_ = 0.0;
};

class JointPositionBoundsConstraint final : public ocs2::StateInputConstraint {
   public:
    JointPositionBoundsConstraint(ocs2::CentroidalModelInfo info,
                                  ocs2::vector_t lower,
                                  ocs2::vector_t upper)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          info_(std::move(info)),
          lower_(std::move(lower)),
          upper_(std::move(upper)) {}

    JointPositionBoundsConstraint(const JointPositionBoundsConstraint& other)
        : ocs2::StateInputConstraint(other),
          info_(other.info_),
          lower_(other.lower_),
          upper_(other.upper_) {}

    JointPositionBoundsConstraint* clone() const override {
        return new JointPositionBoundsConstraint(*this);
    }

    bool isActive(ocs2::scalar_t) const override {
        return lower_.size() == static_cast<int>(info_.actuatedDofNum) &&
               upper_.size() == static_cast<int>(info_.actuatedDofNum);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override {
        return 2 * info_.actuatedDofNum;
    }

    ocs2::vector_t getValue(ocs2::scalar_t,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)input;
        ocs2::vector_t value(2 * info_.actuatedDofNum);
        const int joint_position_offset = 12;
        for (size_t i = 0; i < info_.actuatedDofNum; i++) {
            const int index = joint_position_offset + static_cast<int>(i);
            const double position = state[index];
            value[2 * static_cast<int>(i)] = position - lower_[i];
            value[2 * static_cast<int>(i) + 1] = upper_[i] - position;
        }
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx =
            ocs2::matrix_t::Zero(getNumConstraints(time), state.size());
        approximation.dfdu =
            ocs2::matrix_t::Zero(getNumConstraints(time), input.size());
        const int joint_position_offset = 12;
        for (size_t i = 0; i < info_.actuatedDofNum; i++) {
            const int row = 2 * static_cast<int>(i);
            const int col = joint_position_offset + static_cast<int>(i);
            approximation.dfdx(row, col) = 1.0;
            approximation.dfdx(row + 1, col) = -1.0;
        }
        return approximation;
    }

   private:
    ocs2::CentroidalModelInfo info_;
    ocs2::vector_t lower_;
    ocs2::vector_t upper_;
};

class ContactFrictionConeConstraint final : public ocs2::StateInputConstraint {
   public:
    ContactFrictionConeConstraint(
        std::shared_ptr<ocs2::ReferenceManager> reference_manager,
        ocs2::CentroidalModelInfo info,
        CentroidalMpcConfig config,
        size_t contact_index)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Quadratic),
          reference_manager_(std::move(reference_manager)),
          info_(std::move(info)),
          config_(std::move(config)),
          contact_index_(contact_index) {}

    ContactFrictionConeConstraint(const ContactFrictionConeConstraint& other)
        : ocs2::StateInputConstraint(other),
          reference_manager_(other.reference_manager_),
          info_(other.info_),
          config_(other.config_),
          contact_index_(other.contact_index_) {}

    ContactFrictionConeConstraint* clone() const override {
        return new ContactFrictionConeConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return foot_in_contact(
            flags_from_reference_manager(reference_manager_, time),
            contact_index_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override { return 1; }

    ocs2::vector_t getValue(ocs2::scalar_t,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        const Eigen::Vector3d force =
            ocs2::centroidal_model::getContactForces(
                input, contact_index_, info_);
        return (ocs2::vector_t(1) << cone_value(force)).finished();
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(1, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(1, input.size());
        const Eigen::Vector3d force =
            ocs2::centroidal_model::getContactForces(
                input, contact_index_, info_);
        approximation.dfdu.block<1, 3>(0, force_offset(contact_index_)) =
            cone_gradient(force).transpose();
        return approximation;
    }

    ocs2::VectorFunctionQuadraticApproximation getQuadraticApproximation(
        ocs2::scalar_t time,
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionQuadraticApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(1, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(1, input.size());
        approximation.dfdxx.emplace_back(
            ocs2::matrix_t::Zero(state.size(), state.size()));
        approximation.dfduu.emplace_back(
            ocs2::matrix_t::Zero(input.size(), input.size()));
        approximation.dfdux.emplace_back(
            ocs2::matrix_t::Zero(input.size(), state.size()));
        const Eigen::Vector3d force =
            ocs2::centroidal_model::getContactForces(
                input, contact_index_, info_);
        approximation.dfdu.block<1, 3>(0, force_offset(contact_index_)) =
            cone_gradient(force).transpose();
        approximation.dfduu.front().block<3, 3>(
            force_offset(contact_index_), force_offset(contact_index_)) =
            cone_hessian(force);
        return approximation;
    }

   private:
    double cone_value(const Eigen::Vector3d& force) const {
        const double tangent_square =
            force.x() * force.x() + force.y() * force.y() +
            config_.friction_regularization;
        const double tangent_norm =
            std::sqrt(std::max(tangent_square, 1.0e-12));
        return config_.friction_coefficient * force.z() - tangent_norm;
    }

    Eigen::Vector3d cone_gradient(const Eigen::Vector3d& force) const {
        const double tangent_square =
            force.x() * force.x() + force.y() * force.y() +
            config_.friction_regularization;
        const double tangent_norm =
            std::sqrt(std::max(tangent_square, 1.0e-12));
        return Eigen::Vector3d(-force.x() / tangent_norm,
                               -force.y() / tangent_norm,
                               config_.friction_coefficient);
    }

    Eigen::Matrix3d cone_hessian(const Eigen::Vector3d& force) const {
        const double x2 = force.x() * force.x();
        const double y2 = force.y() * force.y();
        const double tangent_square =
            x2 + y2 + config_.friction_regularization;
        const double tangent_norm =
            std::sqrt(std::max(tangent_square, 1.0e-12));
        const double denom =
            std::max(tangent_norm * tangent_square, 1.0e-12);
        Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
        hessian(0, 0) =
            -(y2 + config_.friction_regularization) / denom;
        hessian(0, 1) = force.x() * force.y() / denom;
        hessian(1, 0) = hessian(0, 1);
        hessian(1, 1) =
            -(x2 + config_.friction_regularization) / denom;
        return hessian;
    }

    std::shared_ptr<ocs2::ReferenceManager> reference_manager_;
    ocs2::CentroidalModelInfo info_;
    CentroidalMpcConfig config_;
    size_t contact_index_ = 0;
};

}  // namespace

struct Ocs2CentroidalMpc::SolveRequest {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CentroidalMpcInput input;
    ocs2::vector_t state;
    ocs2::ModeSchedule mode_schedule;
    ocs2::TargetTrajectories target_trajectories;
    double time = 0.0;
    uint64_t sequence = 0;
};

struct Ocs2CentroidalMpc::PolicySnapshot {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ocs2::PrimalSolution solution;
    double time = 0.0;
    uint64_t sequence = 0;
    int iterations = 0;
    double objective = 0.0;
};

Ocs2CentroidalMpc::Ocs2CentroidalMpc(CentroidalMpcConfig config)
    : config_(std::move(config)) {
    Ocs2CentroidalModelConfig model_config;
    model_config.urdf_path = config_.model_path;
    model_config.left_foot_frame = config_.left_foot_frame;
    model_config.right_foot_frame = config_.right_foot_frame;
    model_config.joint_order = config_.joint_order;
    model_config.nominal_joint_angles = config_.nominal_joint_angles;
    model_ = create_ocs2_centroidal_model(model_config);
    if (!config_.nominal_joint_angles.empty()) {
        Eigen::VectorXd configured_nominal(config_.nominal_joint_angles.size());
        for (size_t i = 0; i < config_.nominal_joint_angles.size(); i++) {
            configured_nominal[static_cast<int>(i)] = config_.nominal_joint_angles[i];
        }
        nominal_joint_position_ =
            model_->model_order_joint_vector(configured_nominal);
    } else {
        nominal_joint_position_ =
            ocs2::vector_t::Zero(model_->info().actuatedDofNum);
    }
    if (!config_.joint_position_limits.empty()) {
        if (config_.joint_position_limits.size() !=
            2 * config_.joint_order.size()) {
            throw std::runtime_error(
                "OCS2 centroidal joint_position_limits size mismatch");
        }
        Eigen::VectorXd configured_lower(config_.joint_order.size());
        Eigen::VectorXd configured_upper(config_.joint_order.size());
        for (size_t i = 0; i < config_.joint_order.size(); i++) {
            configured_lower[static_cast<int>(i)] =
                config_.joint_position_limits[2 * i];
            configured_upper[static_cast<int>(i)] =
                config_.joint_position_limits[2 * i + 1];
        }
        joint_position_lower_ =
            model_->model_order_joint_vector(configured_lower);
        joint_position_upper_ =
            model_->model_order_joint_vector(configured_upper);
    }
    last_input_ = ocs2::vector_t::Zero(model_->info().inputDim);
    configure_solver();
    if (config_.mrt_enabled) {
        start_mrt_worker();
    }
}

Ocs2CentroidalMpc::~Ocs2CentroidalMpc() {
    stop_mrt_worker();
}

void Ocs2CentroidalMpc::reset() {
    const bool restart_mrt = config_.mrt_enabled && mrt_worker_running_;
    if (restart_mrt) {
        stop_mrt_worker();
    }
    time_ = 0.0;
    has_last_input_ = false;
    mrt_next_sequence_ = 1;
    if (last_input_.size() == static_cast<int>(model_->info().inputDim)) {
        last_input_.setZero();
    }
    {
        std::lock_guard<std::mutex> lock(mrt_policy_mutex_);
        mrt_policy_.reset();
    }
    if (mpc_) {
        std::lock_guard<std::mutex> lock(solver_mutex_);
        mpc_->reset();
    }
    if (restart_mrt) {
        start_mrt_worker();
    }
}

void Ocs2CentroidalMpc::validate_input(const CentroidalMpcInput& input) const {
    if (input.joint_position.size() !=
            static_cast<int>(config_.joint_order.size()) ||
        input.joint_velocity.size() !=
            static_cast<int>(config_.joint_order.size())) {
        throw std::runtime_error(
            "OCS2 centroidal MPC input joint vectors do not match joint_order");
    }
}

ocs2::vector_t Ocs2CentroidalMpc::make_state(
    const CentroidalMpcInput& input) {
    validate_input(input);
    const auto& info = model_->info();
    const Eigen::VectorXd joint_position =
        model_->model_order_joint_vector(input.joint_position);
    const Eigen::VectorXd joint_velocity =
        model_->model_order_joint_vector(input.joint_velocity);

    ocs2::vector_t rbd_state(2 * info.generalizedCoordinatesNum);
    rbd_state.setZero();
    rbd_state.segment<3>(0) = input.base_orientation_zyx;
    rbd_state.segment<3>(3) = input.base_position;
    rbd_state.segment(6, info.actuatedDofNum) = joint_position;
    rbd_state.segment<3>(info.generalizedCoordinatesNum) =
        input.base_angular_velocity;
    rbd_state.segment<3>(info.generalizedCoordinatesNum + 3) =
        input.base_linear_velocity;
    rbd_state.segment(info.generalizedCoordinatesNum + 6,
                      info.actuatedDofNum) = joint_velocity;
    return model_->rbd_conversions().computeCentroidalStateFromRbdModel(
        rbd_state);
}

ocs2::matrix_t Ocs2CentroidalMpc::state_weight(bool terminal) const {
    const auto& info = model_->info();
    ocs2::matrix_t Q = ocs2::matrix_t::Zero(info.stateDim, info.stateDim);
    const double scale =
        terminal ? std::max(config_.terminal_weight_scale, 0.0) : 1.0;

    Q(0, 0) = scale * config_.com_velocity_weight;
    Q(1, 1) = scale * config_.com_velocity_weight;
    Q(2, 2) = scale * 0.1 * config_.com_velocity_weight;
    Q(3, 3) = scale * config_.angular_rate_weight;
    Q(4, 4) = scale * config_.angular_rate_weight;
    Q(5, 5) = scale * 0.1 * config_.angular_rate_weight;

    Q(6, 6) = scale * config_.com_weight;
    Q(7, 7) = scale * config_.com_weight;
    Q(8, 8) = scale * config_.base_height_weight;
    Q(9, 9) = scale * config_.yaw_weight;
    Q(10, 10) = scale * config_.orientation_weight;
    Q(11, 11) = scale * config_.orientation_weight;

    for (size_t i = 0; i < info.actuatedDofNum; i++) {
        Q(12 + i, 12 + i) = scale * config_.joint_angle_weight;
    }
    return Q;
}

ocs2::matrix_t Ocs2CentroidalMpc::input_weight() const {
    const auto& info = model_->info();
    ocs2::matrix_t R = ocs2::matrix_t::Zero(info.inputDim, info.inputDim);
    const double force_weight = std::max(config_.force_weight, 1.0e-9);
    const double joint_velocity_weight =
        std::max(config_.joint_velocity_weight, 1.0e-9);
    for (int i = 0; i < 6; i++) {
        R(i, i) = force_weight;
    }
    for (size_t i = 0; i < info.actuatedDofNum; i++) {
        const int index = 6 + static_cast<int>(i);
        R(index, index) = joint_velocity_weight;
    }
    return R;
}

ocs2::vector_t Ocs2CentroidalMpc::nominal_input(
    bool left_contact,
    bool right_contact) const {
    return make_weight_compensating_input(model_->info(),
                                          ContactFlags{left_contact,
                                                       right_contact});
}

ocs2::TargetTrajectories Ocs2CentroidalMpc::make_target_trajectories(
    const CentroidalMpcInput& input,
    double time,
    const ocs2::vector_t& current_state) const {
    const auto& info = model_->info();
    const int knot_count = std::max(config_.horizon, 1) + 1;
    ocs2::scalar_array_t times;
    ocs2::vector_array_t states;
    ocs2::vector_array_t inputs;
    times.reserve(static_cast<size_t>(knot_count));
    states.reserve(static_cast<size_t>(knot_count));
    inputs.reserve(static_cast<size_t>(knot_count));

    for (int k = 0; k < knot_count; k++) {
        const double relative_time = static_cast<double>(k) * config_.dt;
        ocs2::vector_t target_state = ocs2::vector_t::Zero(info.stateDim);
        Eigen::Vector3d target_base_position = input.base_position;
        ContactFlags flags{input.left_contact, input.right_contact};
        if (config_.contact_schedule_enabled && input.contact_schedule.enabled) {
            const ContactScheduleSample sample =
                input.contact_schedule.sample_at(relative_time);
            flags = ContactFlags{sample.left_contact, sample.right_contact};
            const Eigen::Vector2d support_shift =
                (sample.support_center - input.support_center).head<2>();
            target_base_position.head<2>() += support_shift;
        }

        target_state.segment<3>(6) = target_base_position;
        target_state.segment<3>(9) = input.base_orientation_zyx;
        target_state[10] = config_.target_pitch;
        target_state[11] = config_.target_roll;
        if (nominal_joint_position_.size() ==
            static_cast<int>(info.actuatedDofNum)) {
            target_state.segment(12, info.actuatedDofNum) =
                nominal_joint_position_;
        } else if (current_state.size() == static_cast<int>(info.stateDim)) {
            target_state.segment(12, info.actuatedDofNum) =
                current_state.segment(12, info.actuatedDofNum);
        }

        times.push_back(time + relative_time);
        states.push_back(target_state);
        inputs.push_back(make_weight_compensating_input(info, flags));
    }
    return ocs2::TargetTrajectories(times, states, inputs);
}

Eigen::Vector3d Ocs2CentroidalMpc::project_foot_force(
    const Eigen::Vector3d& force,
    bool in_contact) const {
    if (!in_contact) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d output = force;
    output.z() =
        std::clamp(output.z(), config_.min_normal_force,
                   config_.max_normal_force);
    const double tangential_limit = config_.friction_coefficient * output.z();
    Eigen::Vector2d tangential = output.head<2>();
    tangential = clamp_norm(tangential, tangential_limit);
    output.x() = tangential.x();
    output.y() = tangential.y();
    return output;
}

ocs2::vector_t Ocs2CentroidalMpc::project_input(
    const ocs2::vector_t& control,
    bool left_contact,
    bool right_contact) const {
    const auto& info = model_->info();
    ocs2::vector_t output = control;
    if (output.size() != static_cast<int>(info.inputDim)) {
        throw std::runtime_error("OCS2 centroidal MPC returned invalid input size");
    }

    const ocs2::vector_t nominal =
        nominal_input(left_contact, right_contact);
    for (size_t contact_index = 0; contact_index < 2; contact_index++) {
        const bool in_contact =
            contact_index == kLeftContactIndex ? left_contact : right_contact;
        Eigen::Vector3d force =
            ocs2::centroidal_model::getContactForces(
                output, contact_index, info);
        Eigen::Vector3d nominal_force =
            ocs2::centroidal_model::getContactForces(
                nominal, contact_index, info);
        if (in_contact && config_.max_contact_force_delta > 0.0) {
            for (int i = 0; i < 3; i++) {
                force[i] = nominal_force[i] +
                           clamp_abs(force[i] - nominal_force[i],
                                     config_.max_contact_force_delta);
            }
        }
        force = project_foot_force(force, in_contact);
        if (in_contact && config_.max_contact_force_delta > 0.0) {
            for (int i = 0; i < 3; i++) {
                force[i] = nominal_force[i] +
                           clamp_abs(force[i] - nominal_force[i],
                                     config_.max_contact_force_delta);
            }
        }
        ocs2::centroidal_model::getContactForces(
            output, contact_index, info) = force;
    }
    return output;
}

void Ocs2CentroidalMpc::fill_output(
    const ocs2::vector_t& control,
    const ocs2::vector_t& full_state,
    const CentroidalMpcInput& input,
    int iterations,
    double objective,
    CentroidalMpcOutput& output) const {
    (void)full_state;
    const auto& info = model_->info();
    const ocs2::vector_t nominal =
        nominal_input(input.left_contact, input.right_contact);
    const Eigen::Vector3d left_force =
        ocs2::centroidal_model::getContactForces(
            control, kLeftContactIndex, info);
    const Eigen::Vector3d right_force =
        ocs2::centroidal_model::getContactForces(
            control, kRightContactIndex, info);
    const Eigen::Vector3d left_nominal =
        ocs2::centroidal_model::getContactForces(
            nominal, kLeftContactIndex, info);
    const Eigen::Vector3d right_nominal =
        ocs2::centroidal_model::getContactForces(
            nominal, kRightContactIndex, info);
    const Eigen::Vector3d net_contact_force = left_force + right_force;
    const Eigen::Vector3d net_contact_moment =
        (input.left_foot_position - input.com_position).cross(left_force) +
        (input.right_foot_position - input.com_position).cross(right_force);

    output.backend = name_;
    output.state << input.roll - config_.target_roll,
                    input.pitch - config_.target_pitch,
                    input.wx,
                    input.wy,
                    input.com_offset_error.x(),
                    input.com_offset_error.y(),
                    input.com_velocity.x(),
                    input.com_velocity.y();
    output.contact_force_delta.segment<3>(0) = left_force - left_nominal;
    output.contact_force_delta.segment<3>(3) = right_force - right_nominal;
    output.desired_left_contact_force = left_force;
    output.desired_right_contact_force = right_force;
    output.has_desired_contact_forces = true;
    const int joint_velocity_offset = 3 * info.numThreeDofContacts +
                                      6 * info.numSixDofContacts;
    const Eigen::VectorXd model_joint_velocity =
        control.segment(joint_velocity_offset, info.actuatedDofNum);
    output.desired_joint_velocity =
        model_->configured_order_joint_vector(model_joint_velocity);
    if (output.desired_joint_velocity.size() == input.joint_position.size()) {
        output.desired_joint_position =
            input.joint_position +
            config_.control_dt * output.desired_joint_velocity;
        output.has_desired_joint_command = true;
    }
    output.desired_com_acceleration.setZero();
    if (input.mass > 0.0) {
        output.desired_com_acceleration.x() = net_contact_force.x() / input.mass;
        output.desired_com_acceleration.y() = net_contact_force.y() / input.mass;
    }
    output.desired_com_acceleration.head<2>() =
        clamp_norm(output.desired_com_acceleration.head<2>(),
                   config_.max_com_accel);
    output.desired_com_acceleration.z() = 0.0;
    output.desired_angular_acceleration <<
        clamp_abs(net_contact_moment.x() /
                      std::max(input.roll_inertia, 1.0e-6),
                  config_.max_angular_accel),
        clamp_abs(net_contact_moment.y() /
                      std::max(input.pitch_inertia, 1.0e-6),
                  config_.max_angular_accel),
        0.0;
    output.control << output.desired_angular_acceleration.x(),
                      output.desired_angular_acceleration.y(),
                      output.desired_com_acceleration.x(),
                      output.desired_com_acceleration.y();
    output.solved = true;
    output.iterations = iterations;
    output.objective = objective;
}

void Ocs2CentroidalMpc::configure_solver() {
    const auto& info = model_->info();
    const ocs2::ModeSchedule initial_mode_schedule(
        std::vector<ocs2::scalar_t>{},
        std::vector<size_t>{kModeDoubleSupport});
    const ocs2::TargetTrajectories initial_target(
        ocs2::scalar_array_t{0.0},
        ocs2::vector_array_t{ocs2::vector_t::Zero(info.stateDim)},
        ocs2::vector_array_t{make_weight_compensating_input(
            info, ContactFlags{true, true})});
    SwingTrajectoryConfig swing_config;
    swing_config.lift_off_velocity = config_.swing_lift_off_velocity;
    swing_config.touch_down_velocity = config_.swing_touch_down_velocity;
    swing_config.swing_height = config_.swing_height;
    swing_config.swing_time_scale = config_.swing_time_scale;
    switched_reference_manager_ =
        std::make_shared<SwitchedModelReferenceManager>(
            initial_target, initial_mode_schedule, swing_config);
    reference_manager_ = switched_reference_manager_;

    problem_.dynamicsPtr =
        std::make_unique<PinocchioCentroidalSystemDynamics>(
            model_->pinocchio_interface(), info, config_);
    problem_.costPtr->add(
        "full_centroidal_tracking",
        std::make_unique<ocs2::QuadraticStateInputCost>(state_weight(false),
                                                        input_weight()));
    if (config_.terminal_cost_enabled) {
        problem_.finalCostPtr->add(
            "terminal_full_centroidal_tracking",
            std::make_unique<ocs2::QuadraticStateCost>(state_weight(true)));
    }
    if (config_.solver_constraints_enabled) {
        if (config_.zero_swing_force_constraint_enabled) {
            problem_.equalityConstraintPtr->add(
                "left_zero_swing_force",
                std::make_unique<SwingFootZeroForceConstraint>(
                    reference_manager_, info, kLeftContactIndex));
            problem_.equalityConstraintPtr->add(
                "right_zero_swing_force",
                std::make_unique<SwingFootZeroForceConstraint>(
                    reference_manager_, info, kRightContactIndex));
        }
        if (config_.normal_force_constraint_enabled) {
            problem_.inequalityConstraintPtr->add(
                "left_normal_force_bounds",
                std::make_unique<ContactFootNormalForceBoundsConstraint>(
                    reference_manager_, info, config_, kLeftContactIndex));
            problem_.inequalityConstraintPtr->add(
                "right_normal_force_bounds",
                std::make_unique<ContactFootNormalForceBoundsConstraint>(
                    reference_manager_, info, config_, kRightContactIndex));
        }
        if (config_.delta_force_constraint_enabled) {
            problem_.inequalityConstraintPtr->add(
                "contact_force_deviation_bounds",
                std::make_unique<ContactForceDeviationBoundsConstraint>(
                    reference_manager_, info, config_));
        }
        if (config_.joint_velocity_limit > 0.0) {
            problem_.inequalityConstraintPtr->add(
                "joint_velocity_bounds",
                std::make_unique<JointVelocityBoundsConstraint>(
                    info, config_.joint_velocity_limit));
        }
        if (joint_position_lower_.size() ==
                static_cast<int>(info.actuatedDofNum) &&
            joint_position_upper_.size() ==
                static_cast<int>(info.actuatedDofNum)) {
            problem_.inequalityConstraintPtr->add(
                "joint_position_bounds",
                std::make_unique<JointPositionBoundsConstraint>(
                    info, joint_position_lower_, joint_position_upper_));
        }
        if (config_.friction_cone_constraint_enabled &&
            config_.friction_coefficient > 0.0) {
            const ocs2::RelaxedBarrierPenalty::Config barrier_config(
                config_.friction_barrier_mu, config_.friction_barrier_delta);
            problem_.softConstraintPtr->add(
                "left_friction_cone",
                std::make_unique<ocs2::StateInputSoftConstraint>(
                    std::make_unique<ContactFrictionConeConstraint>(
                        reference_manager_, info, config_, kLeftContactIndex),
                    std::make_unique<ocs2::RelaxedBarrierPenalty>(
                        barrier_config)));
            problem_.softConstraintPtr->add(
                "right_friction_cone",
                std::make_unique<ocs2::StateInputSoftConstraint>(
                    std::make_unique<ContactFrictionConeConstraint>(
                        reference_manager_, info, config_, kRightContactIndex),
                    std::make_unique<ocs2::RelaxedBarrierPenalty>(
                        barrier_config)));
        }
        if (switched_reference_manager_ &&
            (config_.stance_zero_velocity_constraint_enabled ||
             config_.swing_normal_velocity_constraint_enabled ||
             (config_.swing_position_constraint_enabled &&
              config_.swing_position_weight > 0.0))) {
            auto left_kinematics = make_foot_kinematics(
                *model_, info, config_, config_.left_foot_frame,
                "roboparty_left_foot_kinematics");
            auto right_kinematics = make_foot_kinematics(
                *model_, info, config_, config_.right_foot_frame,
                "roboparty_right_foot_kinematics");
            if (config_.stance_zero_velocity_constraint_enabled) {
                problem_.equalityConstraintPtr->add(
                    "left_stance_zero_velocity",
                    std::make_unique<FootZeroVelocityConstraint>(
                        switched_reference_manager_, *left_kinematics,
                        kLeftContactIndex));
                problem_.equalityConstraintPtr->add(
                    "right_stance_zero_velocity",
                    std::make_unique<FootZeroVelocityConstraint>(
                        switched_reference_manager_, *right_kinematics,
                        kRightContactIndex));
            }
            if (config_.swing_normal_velocity_constraint_enabled) {
                problem_.equalityConstraintPtr->add(
                    "left_swing_normal_velocity",
                    std::make_unique<SwingFootNormalVelocityConstraint>(
                        switched_reference_manager_, *left_kinematics,
                        kLeftContactIndex));
                problem_.equalityConstraintPtr->add(
                    "right_swing_normal_velocity",
                    std::make_unique<SwingFootNormalVelocityConstraint>(
                        switched_reference_manager_, *right_kinematics,
                        kRightContactIndex));
            }
            if (config_.swing_position_constraint_enabled &&
                config_.swing_position_weight > 0.0) {
                problem_.softConstraintPtr->add(
                    "left_swing_position",
                    std::make_unique<ocs2::StateInputSoftConstraint>(
                        std::make_unique<SwingFootPositionConstraint>(
                            switched_reference_manager_, *left_kinematics,
                            kLeftContactIndex),
                        std::make_unique<ocs2::QuadraticPenalty>(
                            config_.swing_position_weight)));
                problem_.softConstraintPtr->add(
                    "right_swing_position",
                    std::make_unique<ocs2::StateInputSoftConstraint>(
                        std::make_unique<SwingFootPositionConstraint>(
                            switched_reference_manager_, *right_kinematics,
                            kRightContactIndex),
                        std::make_unique<ocs2::QuadraticPenalty>(
                            config_.swing_position_weight)));
            }
        }
    }
    problem_.targetTrajectoriesPtr =
        &reference_manager_->getTargetTrajectories();

    initializer_ =
        std::make_unique<WeightCompensatingInitializer>(info, reference_manager_);

    ocs2::mpc::Settings mpc_settings;
    mpc_settings.timeHorizon_ =
        static_cast<ocs2::scalar_t>(std::max(config_.horizon, 1)) * config_.dt;
    mpc_settings.solutionTimeWindow_ = config_.control_dt;
    mpc_settings.debugPrint_ = false;
    mpc_settings.coldStart_ = false;
    mpc_settings.mpcDesiredFrequency_ = 1.0 / config_.control_dt;
    mpc_settings.mrtDesiredFrequency_ = 1.0 / config_.control_dt;

    ocs2::sqp::Settings sqp_settings;
    sqp_settings.dt = config_.dt;
    sqp_settings.sqpIteration =
        static_cast<size_t>(std::max(config_.qp_iterations, 1));
    sqp_settings.useFeedbackPolicy = true;
    sqp_settings.createValueFunction = false;
    sqp_settings.projectStateInputEqualityConstraints = false;
    sqp_settings.printSolverStatus = false;
    sqp_settings.printSolverStatistics = false;
    sqp_settings.printLinesearch = false;
    sqp_settings.enableLogging = false;
    sqp_settings.nThreads = 1;
    sqp_settings.hpipmSettings.iter_max =
        std::max(config_.qp_iterations, 1);

    mpc_ = std::make_unique<ocs2::SqpMpc>(
        mpc_settings, sqp_settings, problem_, *initializer_);
    mpc_->getSolverPtr()->setReferenceManager(reference_manager_);
}

Ocs2CentroidalMpc::SolveRequest Ocs2CentroidalMpc::make_solve_request(
    const CentroidalMpcInput& input,
    const ocs2::vector_t& state,
    double time) const {
    const ContactFlags current_flags{input.left_contact, input.right_contact};
    SolveRequest request;
    request.input = input;
    request.state = state;
    request.time = time;
    request.mode_schedule = to_ocs2_mode_schedule(
        input.contact_schedule, time, current_flags);
    request.target_trajectories =
        make_target_trajectories(input, time, state);
    return request;
}

bool Ocs2CentroidalMpc::run_solver_iteration(const SolveRequest& request,
                                             PolicySnapshot& policy) {
    if (!mpc_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(solver_mutex_);
    try {
        const ContactFlags current_flags{request.input.left_contact,
                                         request.input.right_contact};
        if (switched_reference_manager_) {
            switched_reference_manager_->set_reference_input(
                request.input.contact_schedule, request.time, current_flags,
                request.input.left_foot_position,
                request.input.right_foot_position);
        }
        reference_manager_->setModeSchedule(request.mode_schedule);
        reference_manager_->setTargetTrajectories(request.target_trajectories);

        const bool updated = mpc_->run(request.time, request.state);
        if (!updated) {
            return false;
        }
        policy.solution = mpc_->getSolverPtr()->primalSolution(
            mpc_->getSolverPtr()->getFinalTime());
        policy.time = request.time;
        policy.sequence = request.sequence;
        policy.iterations =
            static_cast<int>(mpc_->getSolverPtr()->getNumIterations());
        policy.objective =
            mpc_->getSolverPtr()->getPerformanceIndeces().cost;
        return true;
    } catch (const std::exception&) {
        mpc_->reset();
        return false;
    }
}

bool Ocs2CentroidalMpc::compute_control_from_policy(
    const ocs2::PrimalSolution& solution,
    double time,
    const ocs2::vector_t& state,
    ocs2::vector_t& control) const {
    if (solution.controllerPtr_ && !solution.controllerPtr_->empty()) {
        control = solution.controllerPtr_->computeInput(time, state);
        return true;
    }
    if (!solution.inputTrajectory_.empty()) {
        control = solution.inputTrajectory_.front();
        return true;
    }
    return false;
}

bool Ocs2CentroidalMpc::try_get_policy_snapshot(
    double time,
    PolicySnapshot& policy) const {
    std::lock_guard<std::mutex> lock(mrt_policy_mutex_);
    if (!mrt_policy_) {
        return false;
    }
    if (config_.mrt_max_policy_age > 0.0 &&
        time - mrt_policy_->time > config_.mrt_max_policy_age) {
        return false;
    }
    if (mrt_policy_->solution.timeTrajectory_.empty()) {
        return false;
    }
    if (time >
        mrt_policy_->solution.timeTrajectory_.back() + config_.control_dt) {
        return false;
    }
    policy = *mrt_policy_;
    return true;
}

void Ocs2CentroidalMpc::publish_policy_snapshot(PolicySnapshot policy) {
    std::lock_guard<std::mutex> lock(mrt_policy_mutex_);
    if (!mrt_policy_ || policy.sequence >= mrt_policy_->sequence) {
        mrt_policy_ = std::make_unique<PolicySnapshot>(std::move(policy));
    }
}

void Ocs2CentroidalMpc::start_mrt_worker() {
    if (!config_.mrt_enabled || mrt_worker_.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mrt_request_mutex_);
        mrt_stop_requested_ = false;
        mrt_request_pending_ = false;
        mrt_worker_running_ = true;
        mrt_pending_request_.reset();
    }
    mrt_worker_ = std::thread(&Ocs2CentroidalMpc::mrt_worker_loop, this);
}

void Ocs2CentroidalMpc::stop_mrt_worker() {
    {
        std::lock_guard<std::mutex> lock(mrt_request_mutex_);
        mrt_stop_requested_ = true;
        mrt_request_pending_ = false;
        mrt_pending_request_.reset();
    }
    mrt_request_cv_.notify_all();
    if (mrt_worker_.joinable()) {
        mrt_worker_.join();
    }
    {
        std::lock_guard<std::mutex> lock(mrt_request_mutex_);
        mrt_worker_running_ = false;
        mrt_stop_requested_ = false;
    }
}

void Ocs2CentroidalMpc::enqueue_mrt_request(SolveRequest request) {
    if (!config_.mrt_enabled) {
        return;
    }
    request.sequence = mrt_next_sequence_++;
    {
        std::lock_guard<std::mutex> lock(mrt_request_mutex_);
        if (!mrt_worker_running_) {
            return;
        }
        mrt_pending_request_ =
            std::make_unique<SolveRequest>(std::move(request));
        mrt_request_pending_ = true;
    }
    mrt_request_cv_.notify_one();
}

void Ocs2CentroidalMpc::mrt_worker_loop() {
    while (true) {
        std::unique_ptr<SolveRequest> request;
        {
            std::unique_lock<std::mutex> lock(mrt_request_mutex_);
            mrt_request_cv_.wait(lock, [&]() {
                return mrt_stop_requested_ || mrt_request_pending_;
            });
            if (mrt_stop_requested_) {
                return;
            }
            request = std::move(mrt_pending_request_);
            mrt_request_pending_ = false;
        }
        if (!request) {
            continue;
        }
        PolicySnapshot policy;
        if (run_solver_iteration(*request, policy)) {
            publish_policy_snapshot(std::move(policy));
        }
    }
}

CentroidalMpcOutput Ocs2CentroidalMpc::solve(
    const CentroidalMpcInput& input) {
    const ocs2::vector_t state = make_state(input);
    SolveRequest request = make_solve_request(input, state, time_);

    CentroidalMpcOutput output;
    output.backend = name_;
    output.state << input.roll - config_.target_roll,
                    input.pitch - config_.target_pitch,
                    input.wx,
                    input.wy,
                    input.com_offset_error.x(),
                    input.com_offset_error.y(),
                    input.com_velocity.x(),
                    input.com_velocity.y();

    if (!mpc_) {
        return output;
    }

    const auto apply_control =
        [&](const ocs2::vector_t& control, int iterations,
            double objective) {
        if (control.size() != static_cast<int>(model_->info().inputDim)) {
            throw std::runtime_error(
                "OCS2 full centroidal MPC returned invalid input size");
        }

        ocs2::vector_t bounded_control =
            project_input(control, input.left_contact, input.right_contact);
        if (!has_last_input_) {
            last_input_ = nominal_input(input.left_contact, input.right_contact);
            has_last_input_ = true;
        }
        if (config_.input_smoothing_enabled &&
            config_.input_smooth_weight > 0.0) {
            const double alpha = 1.0 / (1.0 + config_.input_smooth_weight);
            bounded_control =
                alpha * bounded_control + (1.0 - alpha) * last_input_;
            bounded_control =
                project_input(bounded_control, input.left_contact,
                              input.right_contact);
        }

        fill_output(bounded_control, state, input,
                    iterations, objective, output);
        last_input_ = bounded_control;
    };

    const auto apply_last_valid_control = [&]() {
        if (!has_last_input_ ||
            last_input_.size() != static_cast<int>(model_->info().inputDim)) {
            return;
        }

        const ocs2::vector_t bounded_control =
            project_input(last_input_, input.left_contact, input.right_contact);
        fill_output(bounded_control, state, input, 0, 0.0, output);
        output.solved = false;
        output.iterations = 0;
        output.objective = 0.0;
    };

    try {
        if (config_.mrt_enabled) {
            start_mrt_worker();
            PolicySnapshot policy;
            bool has_policy = try_get_policy_snapshot(time_, policy);
            if (!has_policy && config_.mrt_first_solve_blocking) {
                request.sequence = mrt_next_sequence_++;
                has_policy = run_solver_iteration(request, policy);
                if (has_policy) {
                    publish_policy_snapshot(policy);
                }
            } else {
                enqueue_mrt_request(request);
            }

            if (has_policy) {
                ocs2::vector_t control;
                if (compute_control_from_policy(
                        policy.solution, time_, state, control)) {
                    apply_control(control, policy.iterations,
                                  policy.objective);
                }
            }
        } else {
            request.sequence = mrt_next_sequence_++;
            PolicySnapshot policy;
            if (run_solver_iteration(request, policy)) {
                ocs2::vector_t control;
                if (compute_control_from_policy(
                        policy.solution, time_, state, control)) {
                    apply_control(control, policy.iterations,
                                  policy.objective);
                }
            }
        }
    } catch (const std::exception&) {
        if (!config_.mrt_enabled) {
            std::lock_guard<std::mutex> lock(solver_mutex_);
            if (mpc_) {
                mpc_->reset();
            }
        }
        has_last_input_ = false;
        last_input_ = nominal_input(input.left_contact, input.right_contact);
    }

    if (!output.has_desired_contact_forces) {
        apply_last_valid_control();
    }

    time_ += config_.control_dt;
    if (!std::isfinite(time_) || time_ > 3600.0) {
        time_ = 0.0;
        if (mpc_) {
            std::lock_guard<std::mutex> lock(solver_mutex_);
            mpc_->reset();
        }
    }
    return output;
}

}  // namespace whole_body_mpc
