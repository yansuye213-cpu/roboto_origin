#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include "utils/close_chain_mapping.hpp"
#include "utils/thread_pool.hpp"
#include "motor_driver.hpp"
#include "imu_driver.hpp"

class RobotInterface {
   public:
    RobotInterface(const std::string& config_file);
    ~RobotInterface() {
        deinit_motors();
        motors_.clear();
        imu_.reset();
    }
    struct IMUCfg{
        int imu_id_, baudrate_;
        std::string imu_type_, imu_interface_type_, imu_interface_;
    };
    struct MotorsCfg{
        int master_id_offset_;
        std::vector<std::string> motor_type_;
        std::vector<std::string> motor_interface_type_;
        std::vector<std::string> motor_interface_;
        std::vector<long int> motor_id_, motor_model_, motor_num_;
        std::vector<double> motor_zero_offset_;
    };
    struct RobotCfg{
        std::vector<long int> close_chain_motor_idx_, motor_sign_, urdf2motor_;
        std::vector<double> kp_, kd_, extrinsic_R_;
    };

    void apply_action(std::vector<float> p,
                      std::vector<float> v  = {},
                      std::vector<float> kp = {},
                      std::vector<float> kd = {},
                      std::vector<float> tau = {});
    void init_motors();
    void deinit_motors();
    void reset_joints(std::vector<double> joint_default_angle);
    void set_zeros();
    void clear_errors();
    void refresh_joints();
    std::vector<float> sample_joint_q();
    std::string joint_motor_label(size_t joint_idx) const;
    std::vector<float> get_joint_q() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_q_;
    }
    std::vector<float> get_joint_vel() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_vel_;
    }
    std::vector<float> get_joint_tau() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_tau_;
    }
    const std::vector<float>& get_quat() {
        if (!imu_) {
            throw std::runtime_error("IMU not initialized");
        }
        auto raw = imu_->get_quat();  // w, x, y, z
        Eigen::Quaternionf raw_q(raw[0], raw[1], raw[2], raw[3]);
        if (raw_q.norm() > 1.0e-6f) {
            q_body_ = raw_q * extrinsic_q_inv_;
            q_body_.normalize();
        } else {
            q_body_ = Eigen::Quaternionf::Identity();
        }
        const auto now = std::chrono::steady_clock::now();
        if (has_last_quat_) {
            const float dt = std::chrono::duration<float>(now - last_quat_time_).count();
            if (dt > 1.0e-4f && dt < 1.0f) {
                Eigen::Quaternionf delta = last_q_body_.conjugate() * q_body_;
                delta.normalize();
                if (delta.w() < 0.0f) {
                    delta.coeffs() *= -1.0f;
                }
                const Eigen::Vector3f v(delta.x(), delta.y(), delta.z());
                const float s = v.norm();
                Eigen::Vector3f omega = Eigen::Vector3f::Zero();
                if (s > 1.0e-6f) {
                    const float angle = 2.0f * std::atan2(s, delta.w());
                    omega = (angle / dt) * (v / s);
                } else {
                    omega = (2.0f / dt) * v;
                }
                quat_ang_vel_buf_[0] = omega.x();
                quat_ang_vel_buf_[1] = omega.y();
                quat_ang_vel_buf_[2] = omega.z();
            }
        }
        last_q_body_ = q_body_;
        last_quat_time_ = now;
        has_last_quat_ = true;
        quat_buf_[0] = q_body_.w();
        quat_buf_[1] = q_body_.x();
        quat_buf_[2] = q_body_.y(); 
        quat_buf_[3] = q_body_.z();
        return quat_buf_;
    }
    const std::vector<float>& get_ang_vel() {
        if (!imu_) {
            throw std::runtime_error("IMU not initialized");
        }
        auto raw = imu_->get_ang_vel();  // in IMU frame
        Eigen::Map<const Eigen::Vector3f> omega_imu(raw.data());
        Eigen::Map<Eigen::Vector3f>(ang_vel_buf_.data()) = extrinsic_R_mat_ * omega_imu;
        if (std::abs(ang_vel_buf_[0]) < 1.0e-6f &&
            std::abs(ang_vel_buf_[1]) < 1.0e-6f &&
            std::abs(ang_vel_buf_[2]) < 1.0e-6f) {
            ang_vel_buf_ = quat_ang_vel_buf_;
        }
        return ang_vel_buf_;
    }

    std::atomic<bool> is_init_{false};

   private:
    std::shared_ptr<IMUCfg> imu_cfg_;
    std::shared_ptr<MotorsCfg> motors_cfg_;
    std::shared_ptr<RobotCfg> robot_cfg_;
    int offline_threshold_ = 25;
    std::shared_ptr<IMUDriver> imu_;
    std::shared_ptr<Decouple> ankle_decouple_;
    Eigen::Matrix3f extrinsic_R_mat_ = Eigen::Matrix3f::Identity();
    Eigen::Quaternionf extrinsic_q_inv_ = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf q_body_;
    Eigen::Quaternionf last_q_body_ = Eigen::Quaternionf::Identity();
    std::chrono::steady_clock::time_point last_quat_time_;
    bool has_last_quat_ = false;
    std::vector<float> quat_buf_{0.f, 0.f, 0.f, 0.f};
    std::vector<float> ang_vel_buf_{0.f, 0.f, 0.f};
    std::vector<float> quat_ang_vel_buf_{0.f, 0.f, 0.f};
    std::vector<std::shared_ptr<MotorDriver>> motors_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::vector<float> cached_ankle_action_;
    std::vector<float> last_ankle_joint_target_;

    std::mutex motors_mutex_, joint_mutex_;
    std::vector<float> joint_q_, joint_vel_, joint_tau_;
    std::vector<float> motor_pos_target_, motor_vel_target_, motor_kp_target_, motor_kd_target_, motor_tau_target_;
    std::vector<int> close_chain_joint_idx_, motor2urdf_;

    void setup_motors();
    void setup_imu();

    void exec_motors_parallel(const std::function<void(std::shared_ptr<MotorDriver>&, int)>& cmd_func);
    void motors_mit_cmd();
    void forward_close_chain();
};
