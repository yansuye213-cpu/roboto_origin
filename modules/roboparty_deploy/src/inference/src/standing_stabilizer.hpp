// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <vector>

class StandingStabilizer {
   public:
    struct Config {
        int joint_num = 0;
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
    };

    struct Correction {
        float roll_accel = 0.0f;
        float pitch_accel = 0.0f;
        float roll_correction = 0.0f;
        float pitch_correction = 0.0f;
    };

    explicit StandingStabilizer(Config config);

    const Config& config() const { return config_; }

    Measurement measure(const std::vector<float>& quat, const std::vector<float>& angular_velocity) const;
    Correction apply(const Measurement& measurement, float blend, std::vector<float>& target) const;

   private:
    float solve_axis(float angle, float rate, float target_angle) const;

    Config config_;
};
