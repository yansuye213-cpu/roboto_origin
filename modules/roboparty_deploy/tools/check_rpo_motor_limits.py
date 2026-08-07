#!/usr/bin/env python3

"""Inspect RPO motor-space limits against the deploy joint-space contract."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOY_ROOT.parents[1]
DEFAULT_ROBOT_CONFIG = DEPLOY_ROOT / "src/inference/robots/rpo/robot.yaml"
DEFAULT_POLICY_CONFIG = DEPLOY_ROOT / "src/inference/robots/rpo/configs/default.yaml"
DEFAULT_DEPLOY_URDF = DEPLOY_ROOT / "src/inference/robots/rpo/description/urdf/Loobot722.urdf"
DEFAULT_TRAIN_URDF = (
    REPO_ROOT
    / "modules/roboparty_train/robolab/data/robots/roboparty/rpo/urdf/Loobot722.urdf"
)
POSE_KEYS = ("joint_default_angle", "reset_joint_angle", "stand_joint_angle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treat default.yaml joint_limits as candidate raw motor limits, apply "
            "robot.yaml motor_sign, and compare both interpretations against configured poses."
        )
    )
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--deploy-urdf", type=Path, default=DEFAULT_DEPLOY_URDF)
    parser.add_argument("--train-urdf", type=Path, default=DEFAULT_TRAIN_URDF)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing YAML file: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a YAML mapping in {path}")
    return data


def require_list(mapping: dict, key: str, size: int | None = None) -> list:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"{key} must be a list")
    if size is not None and len(value) != size:
        raise RuntimeError(f"{key} has {len(value)} values, expected {size}")
    return value


def motor_labels(motors: dict) -> list[str]:
    interfaces = require_list(motors, "motor_interface")
    counts = require_list(motors, "motor_num", len(interfaces))
    ids = require_list(motors, "motor_id", sum(int(count) for count in counts))
    labels: list[str] = []
    offset = 0
    for interface, count in zip(interfaces, counts):
        for index in range(int(count)):
            labels.append(f"{interface}/{ids[offset + index]}")
        offset += int(count)
    return labels


def pairs(values: list, expected_pairs: int, label: str) -> list[tuple[float, float]]:
    if len(values) != expected_pairs * 2:
        raise RuntimeError(
            f"{label} has {len(values)} values, expected {expected_pairs * 2}"
        )
    result = []
    for index in range(expected_pairs):
        lower = float(values[index * 2])
        upper = float(values[index * 2 + 1])
        if lower > upper:
            raise RuntimeError(f"{label}[{index}] has lower {lower} above upper {upper}")
        result.append((lower, upper))
    return result


def signed_limit(limit: tuple[float, float], sign: int) -> tuple[float, float]:
    lower, upper = limit
    if sign == 1:
        return lower, upper
    if sign == -1:
        signed_lower, signed_upper = -upper, -lower
        return (
            0.0 if signed_lower == 0.0 else signed_lower,
            0.0 if signed_upper == 0.0 else signed_upper,
        )
    raise RuntimeError(f"motor_sign must be +1 or -1, got {sign}")


def urdf_limits(path: Path) -> dict[str, tuple[float, float]]:
    if not path.is_file():
        raise RuntimeError(f"Missing URDF file: {path}")
    root = ET.parse(path).getroot()
    result = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"{path}: {joint.attrib['name']} has no limit")
        result[joint.attrib["name"]] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )
    return result


def compare_urdf(
    label: str,
    path: Path,
    names: list[str],
    configured: list[tuple[float, float]],
) -> bool:
    actual = urdf_limits(path)
    mismatches = []
    for name, expected in zip(names, configured):
        if name not in actual:
            mismatches.append(f"{name}: missing")
        elif actual[name] != expected:
            mismatches.append(f"{name}: URDF={actual[name]}, config={expected}")
    if mismatches:
        print(f"[WARN] {label} URDF differs from default.yaml:")
        for mismatch in mismatches:
            print(f"       {mismatch}")
        return False
    print(f"[OK] {label} URDF limits match default.yaml")
    return True


def pose_violations(
    params: dict,
    names: list[str],
    limits: list[tuple[float, float]],
) -> list[str]:
    violations = []
    for pose_key in POSE_KEYS:
        pose = require_list(params, pose_key, len(names))
        for name, angle, (lower, upper) in zip(names, pose, limits):
            value = float(angle)
            if value < lower or value > upper:
                violations.append(
                    f"{pose_key}: {name}={value:.6f} outside [{lower:.6f}, {upper:.6f}]"
                )
    return violations


def print_violations(label: str, violations: list[str]) -> None:
    if not violations:
        print(f"[OK] {label}: all configured poses are inside the limits")
        return
    print(f"[WARN] {label}: {len(violations)} configured pose value(s) are outside the limits")
    for violation in violations:
        print(f"       {violation}")


def print_yaml_limits(limits: list[tuple[float, float]]) -> None:
    print("\nCandidate joint_limits after motor_sign (deploy joint order):")
    print("joint_limits:")
    for index, (lower, upper) in enumerate(limits):
        suffix = "," if index + 1 < len(limits) else ""
        prefix = "    [" if index == 0 else "     "
        closing = "]" if index + 1 == len(limits) else ""
        print(f"{prefix}{lower:.6f}, {upper:.6f}{closing}{suffix}")


def main() -> int:
    args = parse_args()
    robot_config = load_yaml(args.robot_config)
    policy_config = load_yaml(args.policy_config)
    robot = robot_config.get("robot")
    motors = robot_config.get("motors")
    if not isinstance(robot, dict) or not isinstance(motors, dict):
        raise RuntimeError("robot.yaml must contain robot and motors mappings")

    node = policy_config.get("inference_node", {})
    params = node.get("ros__parameters") if isinstance(node, dict) else None
    if not isinstance(params, dict):
        raise RuntimeError("Policy YAML is missing inference_node.ros__parameters")

    joint_num = int(params.get("joint_num", 0))
    names = [str(name) for name in require_list(params, "stand_whole_body_joint_order", joint_num)]
    configured_limits = pairs(
        require_list(params, "joint_limits"), joint_num, "joint_limits"
    )
    urdf2motor = [int(value) for value in require_list(robot, "urdf2motor", joint_num)]
    if sorted(urdf2motor) != list(range(joint_num)):
        raise RuntimeError("urdf2motor must be a permutation of all motor indexes")

    signs = [int(value) for value in require_list(robot, "motor_sign", joint_num)]
    kp = [float(value) for value in require_list(robot, "kp", joint_num)]
    kd = [float(value) for value in require_list(robot, "kd", joint_num)]
    labels = motor_labels(motors)

    processed_limits = [
        signed_limit(configured_limits[joint_index], signs[motor_index])
        for joint_index, motor_index in enumerate(urdf2motor)
    ]

    print("Input joint_limits interpreted as raw motor-coordinate limits")
    print("Ankles are intentionally handled with motor_sign only; no closed-chain mapping is used.\n")
    print(
        f"{'idx':>3}  {'joint':28} {'motor':7} {'sign':>4} "
        f"{'input/raw limit':25} {'after motor_sign':25} {'Kp':>6} {'Kd':>5}"
    )
    for index, name in enumerate(names):
        motor_index = urdf2motor[index]
        sign = signs[motor_index]
        raw_lower, raw_upper = configured_limits[index]
        out_lower, out_upper = processed_limits[index]
        print(
            f"{index:3d}  {name:28} {labels[motor_index]:7} {sign:+4d} "
            f"[{raw_lower:9.6f}, {raw_upper:9.6f}] "
            f"[{out_lower:9.6f}, {out_upper:9.6f}] "
            f"{kp[motor_index]:6.1f} {kd[motor_index]:5.1f}"
        )

    print("\nStatic consistency checks:")
    compare_urdf("deploy", args.deploy_urdf, names, configured_limits)
    compare_urdf("train", args.train_urdf, names, configured_limits)
    configured_violations = pose_violations(params, names, configured_limits)
    processed_violations = pose_violations(params, names, processed_limits)
    print_violations("Current limits treated as joint-space values", configured_violations)
    print_violations("Current limits treated as raw values, then signed", processed_violations)

    if len(configured_violations) < len(processed_violations):
        print(
            "\n[RESULT] The current numbers are more consistent with already being in joint/URDF "
            "coordinates. This is evidence, not proof of how they were measured."
        )
    elif len(processed_violations) < len(configured_violations):
        print(
            "\n[RESULT] The motor_sign-transformed numbers are more consistent with the configured "
            "joint-space poses."
        )
    else:
        print(
            "\n[RESULT] The configured poses do not distinguish the two interpretations. "
            "Check the original measurement source."
        )

    print_yaml_limits(processed_limits)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ET.ParseError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
