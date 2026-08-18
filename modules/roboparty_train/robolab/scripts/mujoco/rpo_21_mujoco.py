#!/usr/bin/env python3
# Copyright (c) 2025-2026, The RoboLab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared 21-DoF RPO constants and helpers for MuJoCo sim2sim scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBO_LAB_ROOT = Path(__file__).resolve().parents[2]
if str(ROBO_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBO_LAB_ROOT))

RPO_ACTION_JOINT_NAMES = [
    "left_leg_pitch_joint",
    "right_leg_pitch_joint",
    "head_yaw_joint",
    "left_leg_roll_joint",
    "right_leg_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_leg_yaw_joint",
    "right_leg_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

RPO_MJCF_JOINT_NAMES = [
    "left_leg_pitch_joint",
    "left_leg_roll_joint",
    "left_leg_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_leg_pitch_joint",
    "right_leg_roll_joint",
    "right_leg_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
]

RPO_NUM_ACTIONS = len(RPO_ACTION_JOINT_NAMES)
RPO_ACTION_TO_MJCF = [RPO_MJCF_JOINT_NAMES.index(name) for name in RPO_ACTION_JOINT_NAMES]

RPO_DEFAULT_POS = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0,
        0.18, 0.06, 0.0, 0.78,
        0.18, -0.06, 0.0, 0.78,
    ],
    dtype=np.double,
)

RPO_KPS = np.array(
    [
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
        30,
        40, 40, 40, 30,
        40, 40, 40, 30,
    ],
    dtype=np.double,
)

RPO_KDS = np.array(
    [
        3.3, 3.3, 3.3, 5.0, 2.0, 2.0,
        3.3, 3.3, 3.3, 5.0, 2.0, 2.0,
        1.5,
        2.0, 2.0, 2.0, 1.5,
        2.0, 2.0, 2.0, 1.5,
    ],
    dtype=np.double,
)

RPO_TAU_LIMIT = np.array(
    [
        120, 120, 120, 120, 97, 97,
        120, 120, 120, 120, 97, 97,
        27,
        27, 27, 27, 27,
        27, 27, 27, 27,
    ],
    dtype=np.double,
)


def quat_rotate_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame."""
    w, x, y, z = quat
    q_vec = np.array([x, y, z], dtype=np.double)
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (uuv - w * uv)


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Return yaw from a wxyz quaternion."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def get_obs(data):
    """Extract base/joint observations from MuJoCo data."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data.astype(np.double)
    v = quat_rotate_inverse_wxyz(quat, data.qvel[:3].astype(np.double))
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], dtype=np.double))
    return q, dq, quat, v, omega, gvec


def fill_locomotion_obs(obs, omega, gvec, q_obs, dq_obs, action, commands=(0.0, 0.0, 0.0), start=0):
    """Fill [omega, gravity, command, q, dq, action] into an observation row."""
    num_actions = len(action)
    obs[0, start : start + 3] = omega
    obs[0, start + 3 : start + 6] = gvec
    obs[0, start + 6 : start + 9] = commands
    obs[0, start + 9 : start + 9 + num_actions] = q_obs
    obs[0, start + 9 + num_actions : start + 9 + 2 * num_actions] = dq_obs
    obs[0, start + 9 + 2 * num_actions : start + 9 + 3 * num_actions] = action
    return start + 9 + 3 * num_actions


def policy_to_mjcf_order(action_order_values: np.ndarray) -> np.ndarray:
    """Map policy/action-order values into MuJoCo joint order."""
    mjcf_values = np.zeros(RPO_NUM_ACTIONS, dtype=np.double)
    for action_idx, mjcf_idx in enumerate(RPO_ACTION_TO_MJCF):
        mjcf_values[mjcf_idx] = action_order_values[action_idx]
    return mjcf_values


def mjcf_to_policy_order(mjcf_values: np.ndarray) -> np.ndarray:
    """Map MuJoCo joint-order values into policy/action order."""
    return np.array([mjcf_values[mjcf_idx] for mjcf_idx in RPO_ACTION_TO_MJCF], dtype=np.double)


def assert_rpo_21_mujoco_model(model, model_path: str):
    """Fail early if a MuJoCo XML is not the current 21-DoF RPO model."""
    import mujoco

    if model.nu != RPO_NUM_ACTIONS:
        raise ValueError(f"{model_path} has {model.nu} actuators, expected {RPO_NUM_ACTIONS}.")
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    if actuator_names != RPO_MJCF_JOINT_NAMES:
        raise ValueError(
            "MuJoCo actuator order does not match the configured 21-DoF RPO order.\n"
            f"Expected: {RPO_MJCF_JOINT_NAMES}\n"
            f"Actual:   {actuator_names}"
        )
