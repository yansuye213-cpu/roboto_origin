// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace stand_control {
class JointQpStandController;
}

namespace whole_body_mpc {
class WholeBodyMpcController;
}

class StandingStabilizer {
   public:
    struct Config {
        int joint_num = 0;
        std::string control_backend = "joint_qp";
        int horizon = 20;
        float dt = 0.004f;
        float q_angle = 120.0f;
        float q_rate = 1.5f;
        float r_accel = 0.15f;
        float max_accel = 12.0f;
        float roll_gain = 0.0f;
        float pitch_gain = 0.1f;
        float max_joint_correction = 0.12f;
        float target_roll = 0.0f;
        float target_pitch = 0.0f;
        bool qp_enabled = false;
        int qp_iterations = 32;
        float qp_tracking_weight = 4.0f;
        float qp_shape_weight = 0.25f;
        float qp_regularization_weight = 0.02f;
        float qp_smooth_weight = 0.8f;
        float qp_max_joint_velocity = 4.0f;
        std::string whole_body_model_path;
        std::string whole_body_base_link;
        std::string whole_body_left_foot_link;
        std::string whole_body_right_foot_link;
        std::vector<std::string> whole_body_joint_order;
        bool validate_whole_body_model = false;
        bool wbc_torque_enabled = false;
        int wbc_qp_iterations = 24;
        float wbc_friction_coefficient = 0.45f;
        float wbc_min_normal_force = 0.0f;
        float wbc_max_normal_force = 250.0f;
        float wbc_force_tracking_weight = 1.0f;
        float wbc_moment_tracking_weight = 0.03f;
        float wbc_regularization_weight = 1.0e-4f;
        float wbc_smooth_weight = 0.02f;
        float wbc_com_kp = 30.0f;
        float wbc_com_kd = 4.0f;
        float wbc_max_com_accel = 2.0f;
        float wbc_roll_moment_kp = 8.0f;
        float wbc_roll_moment_kd = 1.0f;
        float wbc_pitch_moment_kp = 8.0f;
        float wbc_pitch_moment_kd = 1.0f;
        float wbc_max_body_moment = 8.0f;
        float wbc_max_joint_torque = 0.6f;
        std::vector<double> wbc_torque_joint_scale;
        std::vector<double> roll_joint_scale;
        std::vector<double> pitch_joint_scale;
        std::vector<double> joint_limits;
    };

    struct Measurement {
        float roll = 0.0f;
        float pitch = 0.0f;
        float wx = 0.0f;
        float wy = 0.0f;
        float gravity_z = -1.0f;
        float qw = 1.0f;
        float qx = 0.0f;
        float qy = 0.0f;
        float qz = 0.0f;
    };

    struct Correction {
        float roll_accel = 0.0f;
        float pitch_accel = 0.0f;
        float roll_correction = 0.0f;
        float pitch_correction = 0.0f;
        float roll_allocated = 0.0f;
        float pitch_allocated = 0.0f;
        float max_joint_delta = 0.0f;
        float wbc_left_normal_force = 0.0f;
        float wbc_right_normal_force = 0.0f;
        float wbc_roll_moment = 0.0f;
        float wbc_pitch_moment = 0.0f;
        float wbc_max_joint_torque = 0.0f;
        bool qp_used = false;
    };

    struct Command {
        std::vector<float> position;
        std::vector<float> velocity;
        std::vector<float> kp;
        std::vector<float> kd;
        std::vector<float> tau;
        Correction correction;
    };

    explicit StandingStabilizer(Config config);
    ~StandingStabilizer();

    const Config& config() const { return config_; }
    std::vector<std::string> diagnostics() const;

    void reset();
    Measurement measure(const std::vector<float>& quat, const std::vector<float>& angular_velocity) const;
    Command apply(const Measurement& measurement, float blend, const std::vector<float>& base_target,
                  const std::vector<float>& kp, const std::vector<float>& kd,
                  const std::vector<float>& current_joint_position,
                  const std::vector<float>& current_joint_velocity);

   private:
    bool uses_whole_body_mpc() const;

    Config config_;
    std::unique_ptr<stand_control::JointQpStandController> joint_qp_controller_;
    std::unique_ptr<whole_body_mpc::WholeBodyMpcController> whole_body_mpc_controller_;
};
