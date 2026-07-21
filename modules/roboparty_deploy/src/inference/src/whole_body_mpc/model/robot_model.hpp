// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

namespace whole_body_mpc {

class RobotModel {
   public:
    struct Config {
        std::string urdf_path;
        std::string base_link;
        std::string left_foot_frame;
        std::string right_foot_frame;
        std::vector<std::string> joint_order;
        bool floating_base = true;
    };

    struct Kinematics {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d com_velocity = Eigen::Vector3d::Zero();
        Eigen::Isometry3d left_foot_pose = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d right_foot_pose = Eigen::Isometry3d::Identity();
        Eigen::Vector3d left_foot_velocity = Eigen::Vector3d::Zero();
        Eigen::Vector3d right_foot_velocity = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, Eigen::Dynamic> base_jacobian;
        Eigen::Matrix<double, 6, Eigen::Dynamic> left_foot_jacobian;
        Eigen::Matrix<double, 6, Eigen::Dynamic> right_foot_jacobian;
        Eigen::Matrix<double, 6, 1> base_jacobian_dot_v = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> left_foot_jacobian_dot_v = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> right_foot_jacobian_dot_v = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::MatrixXd mass_matrix;
        Eigen::VectorXd nonlinear_effects;
    };

    explicit RobotModel(Config config);
    ~RobotModel();

    bool is_available() const;
    const Config& config() const { return config_; }
    const std::vector<std::string>& model_joint_order() const { return model_joint_order_; }
    const std::vector<std::string>& configured_joint_order() const { return config_.joint_order; }
    int nq() const;
    int nv() const;
    double total_mass() const;

    Eigen::VectorXd neutral_configuration() const;
    Eigen::VectorXd zero_velocity() const;
    Eigen::VectorXd make_configuration(const std::vector<float>& joint_position,
                                       const Eigen::Vector3d& base_position,
                                       const Eigen::Quaterniond& base_orientation) const;
    Eigen::VectorXd make_velocity(const std::vector<float>& joint_velocity,
                                  const Eigen::Vector3d& base_linear_velocity,
                                  const Eigen::Vector3d& base_angular_velocity) const;
    Kinematics compute_kinematics(const Eigen::VectorXd& q, const Eigen::VectorXd& v) const;
    Eigen::VectorXd nonlinear_effects(const Eigen::VectorXd& q, const Eigen::VectorXd& v) const;
    std::vector<double> configured_joint_torques(const Eigen::VectorXd& generalized_tau) const;
    const std::vector<int>& configured_joint_velocity_indices() const;

   private:
    struct Impl;

    void validate_config() const;

    Config config_;
    std::vector<std::string> model_joint_order_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace whole_body_mpc
