// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/mpc/ocs2_centroidal_mpc.hpp"

#include "whole_body_mpc/model/centroidal_model_info.hpp"
#include "whole_body_mpc/reference/mode_schedule_adapter.hpp"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamicsAD.h>
#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>

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
    last_input_ = ocs2::vector_t::Zero(model_->info().inputDim);
    configure_solver();
}

Ocs2CentroidalMpc::~Ocs2CentroidalMpc() = default;

void Ocs2CentroidalMpc::reset() {
    time_ = 0.0;
    has_last_input_ = false;
    if (last_input_.size() == static_cast<int>(model_->info().inputDim)) {
        last_input_.setZero();
    }
    if (mpc_) {
        mpc_->reset();
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
    reference_manager_ = std::make_shared<ocs2::ReferenceManager>(
        initial_target, initial_mode_schedule);

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

CentroidalMpcOutput Ocs2CentroidalMpc::solve(
    const CentroidalMpcInput& input) {
    const ocs2::vector_t state = make_state(input);

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

    const ContactFlags current_flags{input.left_contact, input.right_contact};
    reference_manager_->setModeSchedule(to_ocs2_mode_schedule(
        input.contact_schedule, time_, current_flags));
    reference_manager_->setTargetTrajectories(
        make_target_trajectories(input, time_, state));

    try {
        const bool updated = mpc_->run(time_, state);
        if (!updated) {
            return output;
        }

        const ocs2::PrimalSolution solution =
            mpc_->getSolverPtr()->primalSolution(
                mpc_->getSolverPtr()->getFinalTime());
        ocs2::vector_t control;
        if (solution.controllerPtr_ && !solution.controllerPtr_->empty()) {
            control = solution.controllerPtr_->computeInput(time_, state);
        } else if (!solution.inputTrajectory_.empty()) {
            control = solution.inputTrajectory_.front();
        } else {
            return output;
        }
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

        const double objective =
            mpc_->getSolverPtr()->getPerformanceIndeces().cost;
        fill_output(bounded_control, state, input,
                    static_cast<int>(mpc_->getSolverPtr()->getNumIterations()),
                    objective, output);
        last_input_ = bounded_control;
    } catch (const std::exception&) {
        mpc_->reset();
        has_last_input_ = false;
        last_input_ = nominal_input(input.left_contact, input.right_contact);
    }

    time_ += config_.control_dt;
    if (!std::isfinite(time_) || time_ > 3600.0) {
        time_ = 0.0;
        if (mpc_) {
            mpc_->reset();
        }
    }
    return output;
}

}  // namespace whole_body_mpc
