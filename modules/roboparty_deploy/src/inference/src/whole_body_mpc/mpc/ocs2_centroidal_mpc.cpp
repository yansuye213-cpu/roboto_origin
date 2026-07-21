// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "whole_body_mpc/mpc/ocs2_centroidal_mpc.hpp"

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace whole_body_mpc {

namespace {

constexpr double kGravity = 9.80665;

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

Eigen::Matrix<double, 8, 8> continuous_state_matrix() {
    Eigen::Matrix<double, 8, 8> A = Eigen::Matrix<double, 8, 8>::Zero();
    A(0, 2) = 1.0;
    A(1, 3) = 1.0;
    A(4, 6) = 1.0;
    A(5, 7) = 1.0;
    return A;
}

}  // namespace

struct Ocs2CentroidalMpcData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double mass = 1.0;
    double roll_inertia = 1.0;
    double pitch_inertia = 1.0;
    double time_origin = 0.0;
    Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
    bool left_contact = true;
    bool right_contact = true;
    bool use_contact_schedule = true;
    ContactSchedule contact_schedule;
};

namespace {

enum class Foot {
    Left,
    Right,
};

ContactScheduleSample sample_for_time(const Ocs2CentroidalMpcData& data,
                                      ocs2::scalar_t t) {
    ContactScheduleSample sample;
    sample.left_contact = data.left_contact;
    sample.right_contact = data.right_contact;
    sample.left_foot_position = data.left_foot_position;
    sample.right_foot_position = data.right_foot_position;
    sample.support_center = 0.5 * (data.left_foot_position + data.right_foot_position);
    if (data.use_contact_schedule && data.contact_schedule.enabled) {
        sample = data.contact_schedule.sample_at(t - data.time_origin);
    }
    return sample;
}

bool foot_in_contact(const ContactScheduleSample& sample, Foot foot) {
    return foot == Foot::Left ? sample.left_contact : sample.right_contact;
}

int contact_foot_count(const ContactScheduleSample& sample) {
    return (sample.left_contact ? 1 : 0) + (sample.right_contact ? 1 : 0);
}

int input_offset(Foot foot) {
    return foot == Foot::Left ? 0 : 3;
}

Eigen::Vector3d nominal_force_for_foot(const Ocs2CentroidalMpcData& data,
                                       const ContactScheduleSample& sample,
                                       Foot foot) {
    const int contact_count = contact_foot_count(sample);
    if (!foot_in_contact(sample, foot) || contact_count <= 0 || data.mass <= 0.0) {
        return Eigen::Vector3d::Zero();
    }
    return Eigen::Vector3d(
        0.0, 0.0, data.mass * kGravity / static_cast<double>(contact_count));
}

Eigen::Vector3d actual_force_for_foot(const Ocs2CentroidalMpcData& data,
                                      const ContactScheduleSample& sample,
                                      Foot foot,
                                      const ocs2::vector_t& input) {
    return nominal_force_for_foot(data, sample, foot) +
           input.segment<3>(input_offset(foot));
}

class ContactForceCentroidalDynamics final : public ocs2::SystemDynamicsBase {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit ContactForceCentroidalDynamics(
        std::shared_ptr<Ocs2CentroidalMpcData> data)
        : data_(std::move(data)) {}

    ContactForceCentroidalDynamics(const ContactForceCentroidalDynamics& other)
        : ocs2::SystemDynamicsBase(other), data_(other.data_) {}

    ContactForceCentroidalDynamics* clone() const override {
        return new ContactForceCentroidalDynamics(*this);
    }

    ocs2::vector_t computeFlowMap(ocs2::scalar_t t, const ocs2::vector_t& x,
                                  const ocs2::vector_t& u,
                                  const ocs2::PreComputation&) override {
        return state_matrix() * x + input_matrix(t) * u + nominal_flow(t);
    }

    ocs2::VectorFunctionLinearApproximation linearApproximation(
        ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
        const ocs2::PreComputation&) override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = computeFlowMap(t, x, u, ocs2::PreComputation());
        approximation.dfdx = state_matrix();
        approximation.dfdu = input_matrix(t);
        return approximation;
    }

   private:
    ocs2::matrix_t state_matrix() const {
        return continuous_state_matrix();
    }

    ocs2::vector_t nominal_flow(ocs2::scalar_t t) const {
        ocs2::vector_t flow = ocs2::vector_t::Zero(8);
        if (!data_) {
            return flow;
        }
        const ContactScheduleSample sample = sample_for_time(*data_, t);
        const double ix = std::max(std::abs(data_->roll_inertia), 1.0e-6);
        const double iy = std::max(std::abs(data_->pitch_inertia), 1.0e-6);
        const Eigen::Vector3d left_moment =
            (sample.left_foot_position - data_->com_position)
                .cross(nominal_force_for_foot(*data_, sample, Foot::Left));
        const Eigen::Vector3d right_moment =
            (sample.right_foot_position - data_->com_position)
                .cross(nominal_force_for_foot(*data_, sample, Foot::Right));
        const Eigen::Vector3d moment = left_moment + right_moment;
        flow(2) = moment.x() / ix;
        flow(3) = moment.y() / iy;
        return flow;
    }

    ocs2::matrix_t input_matrix(ocs2::scalar_t t) const {
        ocs2::matrix_t B = ocs2::matrix_t::Zero(8, 6);
        if (!data_) {
            return B;
        }

        const ContactScheduleSample sample = sample_for_time(*data_, t);

        const double mass = std::max(data_->mass, 1.0e-6);
        const double ix = std::max(std::abs(data_->roll_inertia), 1.0e-6);
        const double iy = std::max(std::abs(data_->pitch_inertia), 1.0e-6);

        const auto add_foot = [&](int offset, bool contact,
                                  const Eigen::Vector3d& foot_position) {
            if (!contact) {
                return;
            }
            const Eigen::Vector3d r = foot_position - data_->com_position;
            B(2, offset + 1) += -r.z() / ix;
            B(2, offset + 2) += r.y() / ix;
            B(3, offset + 0) += r.z() / iy;
            B(3, offset + 2) += -r.x() / iy;
            B(6, offset + 0) += 1.0 / mass;
            B(7, offset + 1) += 1.0 / mass;
        };

        add_foot(0, sample.left_contact, sample.left_foot_position);
        add_foot(3, sample.right_contact, sample.right_foot_position);
        return B;
    }

    std::shared_ptr<Ocs2CentroidalMpcData> data_;
};

class SwingFootZeroForceConstraint final : public ocs2::StateInputConstraint {
   public:
    SwingFootZeroForceConstraint(std::shared_ptr<Ocs2CentroidalMpcData> data,
                                 Foot foot)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          data_(std::move(data)),
          foot_(foot) {}

    SwingFootZeroForceConstraint(const SwingFootZeroForceConstraint& other)
        : ocs2::StateInputConstraint(other),
          data_(other.data_),
          foot_(other.foot_) {}

    SwingFootZeroForceConstraint* clone() const override {
        return new SwingFootZeroForceConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return data_ && !foot_in_contact(sample_for_time(*data_, time), foot_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override {
        return 3;
    }

    ocs2::vector_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        if (!data_) {
            return ocs2::vector_t::Zero(3);
        }
        ocs2::vector_t value(3);
        value = actual_force_for_foot(*data_, sample_for_time(*data_, time),
                                      foot_, input);
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        (void)pre_computation;
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(3, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(3, input.size());
        approximation.dfdu.block<3, 3>(0, input_offset(foot_)).setIdentity();
        return approximation;
    }

   private:
    std::shared_ptr<Ocs2CentroidalMpcData> data_;
    Foot foot_;
};

class ContactFootNormalForceBoundsConstraint final : public ocs2::StateInputConstraint {
   public:
    ContactFootNormalForceBoundsConstraint(
        std::shared_ptr<Ocs2CentroidalMpcData> data,
        CentroidalMpcConfig config, Foot foot)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          data_(std::move(data)),
          config_(std::move(config)),
          foot_(foot) {}

    ContactFootNormalForceBoundsConstraint(
        const ContactFootNormalForceBoundsConstraint& other)
        : ocs2::StateInputConstraint(other),
          data_(other.data_),
          config_(other.config_),
          foot_(other.foot_) {}

    ContactFootNormalForceBoundsConstraint* clone() const override {
        return new ContactFootNormalForceBoundsConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return data_ && foot_in_contact(sample_for_time(*data_, time), foot_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override {
        return 2;
    }

    ocs2::vector_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        if (!data_) {
            return ocs2::vector_t::Zero(2);
        }
        const ContactScheduleSample sample = sample_for_time(*data_, time);
        const double fz = actual_force_for_foot(*data_, sample, foot_, input).z();
        ocs2::vector_t value(2);
        value << fz - config_.min_normal_force,
                 config_.max_normal_force - fz;
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        (void)pre_computation;
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(2, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(2, input.size());
        const int z_index = input_offset(foot_) + 2;
        approximation.dfdu(0, z_index) = 1.0;
        approximation.dfdu(1, z_index) = -1.0;
        return approximation;
    }

   private:
    std::shared_ptr<Ocs2CentroidalMpcData> data_;
    CentroidalMpcConfig config_;
    Foot foot_;
};

class ContactForceDeltaBoundsConstraint final : public ocs2::StateInputConstraint {
   public:
    explicit ContactForceDeltaBoundsConstraint(CentroidalMpcConfig config)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          config_(std::move(config)) {}

    ContactForceDeltaBoundsConstraint(
        const ContactForceDeltaBoundsConstraint& other)
        : ocs2::StateInputConstraint(other), config_(other.config_) {}

    ContactForceDeltaBoundsConstraint* clone() const override {
        return new ContactForceDeltaBoundsConstraint(*this);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override {
        return 12;
    }

    ocs2::vector_t getValue(ocs2::scalar_t, const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        ocs2::vector_t value(12);
        const double limit = config_.max_contact_force_delta;
        for (int i = 0; i < 6; i++) {
            value[2 * i] = limit - input[i];
            value[2 * i + 1] = limit + input[i];
        }
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const ocs2::vector_t& state,
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
    CentroidalMpcConfig config_;
};

class ContactFrictionConeConstraint final : public ocs2::StateInputConstraint {
   public:
    ContactFrictionConeConstraint(std::shared_ptr<Ocs2CentroidalMpcData> data,
                                  CentroidalMpcConfig config, Foot foot)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Quadratic),
          data_(std::move(data)),
          config_(std::move(config)),
          foot_(foot) {}

    ContactFrictionConeConstraint(const ContactFrictionConeConstraint& other)
        : ocs2::StateInputConstraint(other),
          data_(other.data_),
          config_(other.config_),
          foot_(other.foot_) {}

    ContactFrictionConeConstraint* clone() const override {
        return new ContactFrictionConeConstraint(*this);
    }

    bool isActive(ocs2::scalar_t time) const override {
        return data_ && foot_in_contact(sample_for_time(*data_, time), foot_);
    }

    size_t getNumConstraints(ocs2::scalar_t) const override {
        return 1;
    }

    ocs2::vector_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation&) const override {
        (void)state;
        if (!data_) {
            return ocs2::vector_t::Zero(1);
        }
        const Eigen::Vector3d force = actual_force_for_foot(
            *data_, sample_for_time(*data_, time), foot_, input);
        return (ocs2::vector_t(1) << cone_value(force)).finished();
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocs2::PreComputation& pre_computation) const override {
        ocs2::VectorFunctionLinearApproximation approximation;
        approximation.f = getValue(time, state, input, pre_computation);
        approximation.dfdx = ocs2::matrix_t::Zero(1, state.size());
        approximation.dfdu = ocs2::matrix_t::Zero(1, input.size());
        if (!data_) {
            return approximation;
        }
        approximation.dfdu.block<1, 3>(0, input_offset(foot_)) =
            cone_gradient(actual_force_for_foot(
                *data_, sample_for_time(*data_, time), foot_, input)).transpose();
        return approximation;
    }

    ocs2::VectorFunctionQuadraticApproximation getQuadraticApproximation(
        ocs2::scalar_t time, const ocs2::vector_t& state,
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
        if (!data_) {
            return approximation;
        }

        const Eigen::Vector3d force = actual_force_for_foot(
            *data_, sample_for_time(*data_, time), foot_, input);
        approximation.dfdu.block<1, 3>(0, input_offset(foot_)) =
            cone_gradient(force).transpose();
        approximation.dfduu.front().block<3, 3>(
            input_offset(foot_), input_offset(foot_)) = cone_hessian(force);
        return approximation;
    }

   private:
    double cone_value(const Eigen::Vector3d& force) const {
        const double tangent_square =
            force.x() * force.x() + force.y() * force.y() +
            config_.friction_regularization;
        const double tangent_norm = std::sqrt(std::max(tangent_square, 1.0e-12));
        return config_.friction_coefficient * force.z() - tangent_norm;
    }

    Eigen::Vector3d cone_gradient(const Eigen::Vector3d& force) const {
        const double tangent_square =
            force.x() * force.x() + force.y() * force.y() +
            config_.friction_regularization;
        const double tangent_norm = std::sqrt(std::max(tangent_square, 1.0e-12));
        return Eigen::Vector3d(-force.x() / tangent_norm,
                               -force.y() / tangent_norm,
                               config_.friction_coefficient);
    }

    Eigen::Matrix3d cone_hessian(const Eigen::Vector3d& force) const {
        const double x2 = force.x() * force.x();
        const double y2 = force.y() * force.y();
        const double tangent_square =
            x2 + y2 + config_.friction_regularization;
        const double tangent_norm = std::sqrt(std::max(tangent_square, 1.0e-12));
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

    std::shared_ptr<Ocs2CentroidalMpcData> data_;
    CentroidalMpcConfig config_;
    Foot foot_;
};

}  // namespace

Ocs2CentroidalMpc::Ocs2CentroidalMpc(CentroidalMpcConfig config)
    : config_(std::move(config)),
      dynamics_data_(std::make_shared<Ocs2CentroidalMpcData>()),
      initializer_(kInputDim) {
    configure_solver();
}

void Ocs2CentroidalMpc::reset() {
    time_ = 0.0;
    last_control_.setZero();
    if (mpc_) {
        mpc_->reset();
    }
}

Eigen::Matrix<double, Ocs2CentroidalMpc::kStateDim, 1>
Ocs2CentroidalMpc::make_state(const CentroidalMpcInput& input) const {
    Eigen::Matrix<double, kStateDim, 1> state;
    state << input.roll - config_.target_roll,
             input.pitch - config_.target_pitch,
             input.wx,
             input.wy,
             input.com_offset_error.x(),
             input.com_offset_error.y(),
             input.com_velocity.x(),
             input.com_velocity.y();
    return state;
}

ocs2::matrix_t Ocs2CentroidalMpc::state_weight(bool terminal) const {
    ocs2::matrix_t Q = ocs2::matrix_t::Zero(kStateDim, kStateDim);
    const double scale = terminal ? std::max(config_.terminal_weight_scale, 0.0) : 1.0;
    Q(0, 0) = scale * config_.orientation_weight;
    Q(1, 1) = scale * config_.orientation_weight;
    Q(2, 2) = scale * config_.angular_rate_weight;
    Q(3, 3) = scale * config_.angular_rate_weight;
    Q(4, 4) = scale * config_.com_weight;
    Q(5, 5) = scale * config_.com_weight;
    Q(6, 6) = scale * config_.com_velocity_weight;
    Q(7, 7) = scale * config_.com_velocity_weight;
    return Q;
}

ocs2::matrix_t Ocs2CentroidalMpc::input_weight() const {
    return std::max(config_.force_weight, 1.0e-9) *
           ocs2::matrix_t::Identity(kInputDim, kInputDim);
}

ocs2::TargetTrajectories Ocs2CentroidalMpc::make_target_trajectories(
    const CentroidalMpcInput& input, double time) const {
    const int knot_count = std::max(config_.horizon, 1) + 1;
    ocs2::scalar_array_t times;
    ocs2::vector_array_t states;
    ocs2::vector_array_t inputs;
    times.reserve(static_cast<size_t>(knot_count));
    states.reserve(static_cast<size_t>(knot_count));
    inputs.reserve(static_cast<size_t>(knot_count));
    for (int k = 0; k < knot_count; k++) {
        const double relative_time = static_cast<double>(k) * config_.dt;
        ocs2::vector_t target_state = ocs2::vector_t::Zero(kStateDim);
        if (config_.contact_schedule_enabled && input.contact_schedule.enabled) {
            const ContactScheduleSample sample =
                input.contact_schedule.sample_at(relative_time);
            const Eigen::Vector2d support_shift =
                (sample.support_center - input.support_center).head<2>();
            target_state(4) = support_shift.x();
            target_state(5) = support_shift.y();
        }
        times.push_back(time + relative_time);
        states.push_back(target_state);
        inputs.push_back(ocs2::vector_t::Zero(kInputDim));
    }
    return ocs2::TargetTrajectories(times, states, inputs);
}

Eigen::Vector3d Ocs2CentroidalMpc::nominal_foot_force(
    bool in_contact, int contact_feet, const CentroidalMpcInput& input) const {
    if (!in_contact || contact_feet <= 0 || input.mass <= 0.0) {
        return Eigen::Vector3d::Zero();
    }
    return Eigen::Vector3d(
        0.0, 0.0, input.mass * kGravity / static_cast<double>(contact_feet));
}

Eigen::Vector3d Ocs2CentroidalMpc::project_foot_force(
    const Eigen::Vector3d& force, bool in_contact) const {
    if (!in_contact) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d output = force;
    output.z() =
        std::clamp(output.z(), config_.min_normal_force, config_.max_normal_force);
    const double tangential_limit = config_.friction_coefficient * output.z();
    Eigen::Vector2d tangential = output.head<2>();
    tangential = clamp_norm(tangential, tangential_limit);
    output.x() = tangential.x();
    output.y() = tangential.y();
    return output;
}

Eigen::Matrix<double, Ocs2CentroidalMpc::kInputDim, 1>
Ocs2CentroidalMpc::clamp_force_delta(
    const Eigen::Matrix<double, kInputDim, 1>& control,
    const CentroidalMpcInput& input) const {
    Eigen::Matrix<double, kInputDim, 1> output = control;
    const int contact_feet =
        (input.left_contact ? 1 : 0) + (input.right_contact ? 1 : 0);
    const Eigen::Vector3d left_nominal =
        nominal_foot_force(input.left_contact, contact_feet, input);
    const Eigen::Vector3d right_nominal =
        nominal_foot_force(input.right_contact, contact_feet, input);

    Eigen::Vector3d left_delta = output.segment<3>(0);
    Eigen::Vector3d right_delta = output.segment<3>(3);
    for (int i = 0; i < 3; i++) {
        left_delta[i] = clamp_abs(left_delta[i], config_.max_contact_force_delta);
        right_delta[i] = clamp_abs(right_delta[i], config_.max_contact_force_delta);
    }

    const Eigen::Vector3d left_force =
        project_foot_force(left_nominal + left_delta, input.left_contact);
    const Eigen::Vector3d right_force =
        project_foot_force(right_nominal + right_delta, input.right_contact);
    output.segment<3>(0) = left_force - left_nominal;
    output.segment<3>(3) = right_force - right_nominal;
    return output;
}

void Ocs2CentroidalMpc::fill_output(
    const Eigen::Matrix<double, kInputDim, 1>& control,
    const Eigen::Matrix<double, kStateDim, 1>& state,
    const CentroidalMpcInput& input,
    int iterations,
    double objective,
    CentroidalMpcOutput& output) const {
    const int contact_feet =
        (input.left_contact ? 1 : 0) + (input.right_contact ? 1 : 0);
    const Eigen::Vector3d left_nominal =
        nominal_foot_force(input.left_contact, contact_feet, input);
    const Eigen::Vector3d right_nominal =
        nominal_foot_force(input.right_contact, contact_feet, input);
    const Eigen::Vector3d left_force =
        project_foot_force(left_nominal + control.segment<3>(0), input.left_contact);
    const Eigen::Vector3d right_force =
        project_foot_force(right_nominal + control.segment<3>(3), input.right_contact);
    const Eigen::Vector3d left_delta = left_force - left_nominal;
    const Eigen::Vector3d right_delta = right_force - right_nominal;
    const Eigen::Vector3d net_contact_force = left_force + right_force;
    const Eigen::Vector3d net_contact_moment =
        (input.left_foot_position - input.com_position).cross(left_force) +
        (input.right_foot_position - input.com_position).cross(right_force);

    output.backend = name_;
    output.state = state;
    output.contact_force_delta.segment<3>(0) = left_delta;
    output.contact_force_delta.segment<3>(3) = right_delta;
    output.desired_left_contact_force = left_force;
    output.desired_right_contact_force = right_force;
    output.has_desired_contact_forces = true;
    output.desired_com_acceleration.setZero();
    if (input.mass > 0.0) {
        output.desired_com_acceleration.x() = net_contact_force.x() / input.mass;
        output.desired_com_acceleration.y() = net_contact_force.y() / input.mass;
    }
    output.desired_com_acceleration.head<2>() =
        clamp_norm(output.desired_com_acceleration.head<2>(), config_.max_com_accel);
    output.desired_com_acceleration.z() = 0.0;
    output.desired_angular_acceleration << 
        clamp_abs(net_contact_moment.x() / std::max(input.roll_inertia, 1.0e-6),
                  config_.max_angular_accel),
        clamp_abs(net_contact_moment.y() / std::max(input.pitch_inertia, 1.0e-6),
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

void Ocs2CentroidalMpc::update_dynamics_data(const CentroidalMpcInput& input) {
    dynamics_data_->mass = std::max(input.mass, 1.0e-6);
    dynamics_data_->roll_inertia = std::max(input.roll_inertia, 1.0e-6);
    dynamics_data_->pitch_inertia = std::max(input.pitch_inertia, 1.0e-6);
    dynamics_data_->time_origin = time_;
    dynamics_data_->com_position = input.com_position;
    dynamics_data_->left_foot_position = input.left_foot_position;
    dynamics_data_->right_foot_position = input.right_foot_position;
    dynamics_data_->left_contact = input.left_contact;
    dynamics_data_->right_contact = input.right_contact;
    dynamics_data_->use_contact_schedule = config_.contact_schedule_enabled;
    dynamics_data_->contact_schedule = input.contact_schedule;
}

void Ocs2CentroidalMpc::configure_solver() {
    problem_.dynamicsPtr =
        std::make_unique<ContactForceCentroidalDynamics>(dynamics_data_);
    problem_.costPtr->add(
        "centroidal_force_tracking",
        std::make_unique<ocs2::QuadraticStateInputCost>(state_weight(false),
                                                        input_weight()));
    if (config_.terminal_cost_enabled) {
        problem_.finalCostPtr->add(
            "terminal_centroidal_force_tracking",
            std::make_unique<ocs2::QuadraticStateCost>(state_weight(true)));
    }
    if (config_.solver_constraints_enabled) {
        if (config_.zero_swing_force_constraint_enabled) {
            problem_.equalityConstraintPtr->add(
                "left_zero_swing_force",
                std::make_unique<SwingFootZeroForceConstraint>(
                    dynamics_data_, Foot::Left));
            problem_.equalityConstraintPtr->add(
                "right_zero_swing_force",
                std::make_unique<SwingFootZeroForceConstraint>(
                    dynamics_data_, Foot::Right));
        }
        if (config_.normal_force_constraint_enabled) {
            problem_.inequalityConstraintPtr->add(
                "left_normal_force_bounds",
                std::make_unique<ContactFootNormalForceBoundsConstraint>(
                    dynamics_data_, config_, Foot::Left));
            problem_.inequalityConstraintPtr->add(
                "right_normal_force_bounds",
                std::make_unique<ContactFootNormalForceBoundsConstraint>(
                    dynamics_data_, config_, Foot::Right));
        }
        if (config_.delta_force_constraint_enabled) {
            problem_.inequalityConstraintPtr->add(
                "contact_force_delta_bounds",
                std::make_unique<ContactForceDeltaBoundsConstraint>(config_));
        }
        if (config_.friction_cone_constraint_enabled &&
            config_.friction_coefficient > 0.0) {
            const ocs2::RelaxedBarrierPenalty::Config barrier_config(
                config_.friction_barrier_mu, config_.friction_barrier_delta);
            problem_.softConstraintPtr->add(
                "left_friction_cone",
                std::make_unique<ocs2::StateInputSoftConstraint>(
                    std::make_unique<ContactFrictionConeConstraint>(
                        dynamics_data_, config_, Foot::Left),
                    std::make_unique<ocs2::RelaxedBarrierPenalty>(
                        barrier_config)));
            problem_.softConstraintPtr->add(
                "right_friction_cone",
                std::make_unique<ocs2::StateInputSoftConstraint>(
                    std::make_unique<ContactFrictionConeConstraint>(
                        dynamics_data_, config_, Foot::Right),
                    std::make_unique<ocs2::RelaxedBarrierPenalty>(
                        barrier_config)));
        }
    }

    reference_manager_ = std::make_shared<ocs2::ReferenceManager>(
        make_target_trajectories(CentroidalMpcInput{}, 0.0));
    problem_.targetTrajectoriesPtr = &reference_manager_->getTargetTrajectories();

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
        mpc_settings, sqp_settings, problem_, initializer_);
    mpc_->getSolverPtr()->setReferenceManager(reference_manager_);
}

CentroidalMpcOutput Ocs2CentroidalMpc::solve(const CentroidalMpcInput& input) {
    const Eigen::Matrix<double, kStateDim, 1> state = make_state(input);
    ocs2::vector_t ocs2_state(kStateDim);
    ocs2_state = state;

    CentroidalMpcOutput output;
    output.backend = name_;
    output.state = state;

    if (!mpc_) {
        return output;
    }

    update_dynamics_data(input);
    reference_manager_->setTargetTrajectories(make_target_trajectories(input, time_));
    try {
        const bool updated = mpc_->run(time_, ocs2_state);
        if (!updated) {
            return output;
        }

        const ocs2::PrimalSolution solution =
            mpc_->getSolverPtr()->primalSolution(mpc_->getSolverPtr()->getFinalTime());
        ocs2::vector_t control;
        if (solution.controllerPtr_ && !solution.controllerPtr_->empty()) {
            control = solution.controllerPtr_->computeInput(time_, ocs2_state);
        } else if (!solution.inputTrajectory_.empty()) {
            control = solution.inputTrajectory_.front();
        } else {
            return output;
        }
        if (control.size() != kInputDim) {
            throw std::runtime_error("OCS2 centroidal MPC returned invalid input size");
        }

        Eigen::Matrix<double, kInputDim, 1> bounded_control = control;
        bounded_control = clamp_force_delta(bounded_control, input);
        if (config_.input_smoothing_enabled && config_.input_smooth_weight > 0.0) {
            const double alpha = 1.0 / (1.0 + config_.input_smooth_weight);
            bounded_control =
                alpha * bounded_control + (1.0 - alpha) * last_control_;
            bounded_control = clamp_force_delta(bounded_control, input);
        }

        const double objective = mpc_->getSolverPtr()->getPerformanceIndeces().cost;
        fill_output(bounded_control, state, input,
                    static_cast<int>(mpc_->getSolverPtr()->getNumIterations()),
                    objective, output);
        last_control_ = bounded_control;
    } catch (const std::exception&) {
        mpc_->reset();
        last_control_.setZero();
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
