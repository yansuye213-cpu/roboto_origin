// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include "whole_body_mpc/mpc/centroidal_mpc_types.hpp"

#include <string>

namespace whole_body_mpc {

class CentroidalMpcBackend {
   public:
    virtual ~CentroidalMpcBackend() = default;

    virtual const std::string& name() const = 0;
    virtual void reset() = 0;
    virtual CentroidalMpcOutput solve(const CentroidalMpcInput& input) = 0;
};

}  // namespace whole_body_mpc
