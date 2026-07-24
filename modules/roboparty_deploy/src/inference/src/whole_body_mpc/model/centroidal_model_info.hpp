// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

namespace whole_body_mpc {

struct Ocs2CentroidalModelConfig {
    std::string urdf_path;
    std::string left_foot_frame;
    std::string right_foot_frame;
    std::vector<std::string> joint_order;
    std::vector<double> nominal_joint_angles;
};

class Ocs2CentroidalModel {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Ocs2CentroidalModel(ocs2::PinocchioInterface pinocchio_interface,
                        ocs2::CentroidalModelInfo info,
                        std::vector<int> model_joint_index_by_config_joint);

    const ocs2::PinocchioInterface& pinocchio_interface() const {
        return pinocchio_interface_;
    }
    const ocs2::CentroidalModelInfo& info() const { return info_; }
    ocs2::CentroidalModelRbdConversions& rbd_conversions() {
        return rbd_conversions_;
    }

    Eigen::VectorXd model_order_joint_vector(
        const Eigen::VectorXd& configured_order_vector) const;
    Eigen::VectorXd configured_order_joint_vector(
        const Eigen::VectorXd& model_order_vector) const;

   private:
    ocs2::PinocchioInterface pinocchio_interface_;
    ocs2::CentroidalModelInfo info_;
    ocs2::CentroidalModelRbdConversions rbd_conversions_;
    std::vector<int> model_joint_index_by_config_joint_;
};

std::unique_ptr<Ocs2CentroidalModel> create_ocs2_centroidal_model(
    const Ocs2CentroidalModelConfig& config);

}  // namespace whole_body_mpc
