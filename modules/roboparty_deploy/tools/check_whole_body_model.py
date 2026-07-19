#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = ROOT / "src/inference"
DEFAULT_CONFIG = INFERENCE_ROOT / "robots/rpo/configs/default.yaml"


def load_params(config_path):
    data = yaml.safe_load(config_path.read_text())
    try:
        return data["inference_node"]["ros__parameters"]
    except KeyError as exc:
        raise RuntimeError(f"Invalid ROS parameter YAML: {config_path}") from exc


def resolve_model_path(raw_path):
    model_path = Path(raw_path)
    if model_path.is_absolute():
        return model_path
    return (INFERENCE_ROOT / model_path).resolve()


def format_vector(values):
    arr = np.asarray(values).reshape(-1)
    return "[" + ", ".join(f"{float(v): .5f}" for v in arr) + "]"


def format_pose(pose):
    rpy = pin.rpy.matrixToRpy(pose.rotation)
    return f"xyz={format_vector(pose.translation)} rpy={format_vector(rpy)}"


def frame_id(model, name):
    idx = model.getFrameId(name)
    if idx >= len(model.frames):
        raise RuntimeError(f"Frame not found in URDF: {name}")
    return idx


def one_dof_joint_indices(model):
    indices = {}
    order = []
    for joint_id in range(1, len(model.joints)):
        joint = model.joints[joint_id]
        name = model.names[joint_id]
        if joint.nq == 1 and joint.nv == 1:
            order.append(name)
            indices[name] = (joint.idx_q, joint.idx_v)
    return order, indices


def apply_joint_positions(q, configured_order, joint_indices, joint_positions):
    if len(joint_positions) != len(configured_order):
        raise RuntimeError(
            "stand_joint_angle length does not match stand_whole_body_joint_order length"
        )
    for name, position in zip(configured_order, joint_positions):
        q[joint_indices[name][0]] = float(position)


def clamp(value, limit):
    return max(-limit, min(limit, value))


def main():
    parser = argparse.ArgumentParser(
        description="Check the stand whole-body URDF, joint order, CoM, foot frames, and Jacobians."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--use-stand-angle",
        action="store_true",
        help="Compute kinematics with stand_joint_angle instead of all-zero joint angles.",
    )
    args = parser.parse_args()

    params = load_params(args.config)
    urdf_path = resolve_model_path(params["stand_whole_body_model_path"])
    base_link = params["stand_whole_body_base_link"]
    left_foot = params["stand_whole_body_left_foot_link"]
    right_foot = params["stand_whole_body_right_foot_link"]
    configured_order = params["stand_whole_body_joint_order"]

    model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    data = model.createData()
    total_mass = sum(inertia.mass for inertia in model.inertias)
    model_order, joint_indices = one_dof_joint_indices(model)

    missing = [name for name in configured_order if name not in joint_indices]
    if missing:
        raise RuntimeError("Configured joints missing from URDF: " + ", ".join(missing))
    if len(set(configured_order)) != len(configured_order):
        raise RuntimeError("stand_whole_body_joint_order contains duplicate joint names")

    base_id = frame_id(model, base_link)
    left_id = frame_id(model, left_foot)
    right_id = frame_id(model, right_foot)

    q = pin.neutral(model)
    if args.use_stand_angle:
        apply_joint_positions(q, configured_order, joint_indices, params["stand_joint_angle"])
    v = np.zeros(model.nv)

    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q, v, False)
    pin.computeJointJacobians(model, data, q)

    left_jacobian = pin.getFrameJacobian(model, data, left_id, pin.LOCAL_WORLD_ALIGNED)
    right_jacobian = pin.getFrameJacobian(model, data, right_id, pin.LOCAL_WORLD_ALIGNED)
    left_pose = data.oMf[left_id]
    right_pose = data.oMf[right_id]
    foot_midpoint = 0.5 * (left_pose.translation + right_pose.translation)
    com_xy_offset = data.com[0][:2] - foot_midpoint[:2]

    print(f"config: {args.config}")
    print(f"urdf: {urdf_path}")
    print(f"model nq/nv: {model.nq}/{model.nv}")
    print(f"total mass: {total_mass:.5f} kg")
    print(f"base frame[{base_id}]: {base_link}")
    print(f"left foot frame[{left_id}]: {left_foot}")
    print(f"right foot frame[{right_id}]: {right_foot}")
    print(f"kinematics pose source: {'stand_joint_angle' if args.use_stand_angle else 'neutral zero joints'}")
    print(f"CoM: {format_vector(data.com[0])}")
    print(f"left foot: {format_pose(left_pose)}")
    print(f"right foot: {format_pose(right_pose)}")
    print(f"foot midpoint: {format_vector(foot_midpoint)}")
    print(f"CoM XY offset from foot midpoint: {format_vector(com_xy_offset)}")
    print(f"left foot Jacobian: {left_jacobian.shape[0]}x{left_jacobian.shape[1]}")
    print(f"right foot Jacobian: {right_jacobian.shape[0]}x{right_jacobian.shape[1]}")

    static_fz = total_mass * 9.80665 * 0.5
    left_wrench = np.zeros(6)
    right_wrench = np.zeros(6)
    left_wrench[:3] = [0.0, 0.0, static_fz]
    right_wrench[:3] = [0.0, 0.0, static_fz]
    nonlinear = pin.nonLinearEffects(model, data, q, v)
    generalized_tau = nonlinear - left_jacobian.T @ left_wrench - right_jacobian.T @ right_wrench
    raw_joint_tau = [float(generalized_tau[joint_indices[name][1]]) for name in configured_order]
    torque_scale = params.get("stand_wbc_torque_joint_scale", [0.0] * len(configured_order))
    max_joint_torque = float(params.get("stand_wbc_max_joint_torque", 0.0))
    preview_joint_tau = [
        clamp(raw_tau * float(scale), max_joint_torque)
        for raw_tau, scale in zip(raw_joint_tau, torque_scale)
    ]
    print(f"static support force per foot: {static_fz:.5f} N")
    print(
        "static WBC torque preview: "
        f"raw_max={max(abs(v) for v in raw_joint_tau):.5f} Nm "
        f"scaled_max={max(abs(v) for v in preview_joint_tau):.5f} Nm"
    )

    print("configured joint mapping:")
    for idx, name in enumerate(configured_order):
        q_idx, v_idx = joint_indices[name]
        print(
            f"  runtime[{idx:02d}] {name}: q[{q_idx}] v[{v_idx}] "
            f"static_tau={raw_joint_tau[idx]: .5f} scaled={preview_joint_tau[idx]: .5f}"
        )

    if configured_order != model_order:
        print("note: configured joint order differs from Pinocchio model order.")
        print("Pinocchio model order:")
        for idx, name in enumerate(model_order):
            print(f"  model[{idx:02d}] {name}")


if __name__ == "__main__":
    main()
