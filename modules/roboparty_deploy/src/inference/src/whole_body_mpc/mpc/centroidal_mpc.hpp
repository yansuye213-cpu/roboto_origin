// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/mpc/centroidal_mpc_backend.hpp"

#include <memory>
#include <string>

namespace whole_body_mpc {

class CentroidalMpc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Config = CentroidalMpcConfig;
    using Input = CentroidalMpcInput;
    using Output = CentroidalMpcOutput;

    explicit CentroidalMpc(Config config);

    void reset();
    Output solve(const Input& input);
    const std::string& backend_name() const { return backend_name_; }

   private:
    static std::unique_ptr<CentroidalMpcBackend> make_backend(const Config& config);

    Config config_;
    std::unique_ptr<CentroidalMpcBackend> backend_;
    std::string backend_name_ = "disabled";
};

}  // namespace whole_body_mpc
