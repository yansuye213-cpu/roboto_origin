#!/usr/bin/env python3
# Copyright (c) 2025-2026, The RoboLab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the RPO 21-DoF train/deploy interface without launching Isaac Sim."""

from __future__ import annotations

import argparse
import ast
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


ROBO_LAB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROBO_LAB_ROOT.parents[2]
RPO_ASSET_PY = ROBO_LAB_ROOT / "robolab/assets/robots/roboparty.py"
RPO_MUJOCO_PY = ROBO_LAB_ROOT / "scripts/mujoco/rpo_21_mujoco.py"
RPO_ONNX_SIM_PY = ROBO_LAB_ROOT / "scripts/mujoco/sim2sim_rpo_onnx.py"
RPO_URDF = ROBO_LAB_ROOT / "data/robots/roboparty/rpo/urdf/Loobot722.urdf"
RPO_RETARGET_CFG = ROBO_LAB_ROOT / "scripts/tools/retarget/config/rpo.yaml"
RPO_BM_DIR = ROBO_LAB_ROOT / "data/motions/rpo_bm"
RPO_LAB_DIR = ROBO_LAB_ROOT / "data/motions/rpo_lab"
RPO_MJCF_DIR = ROBO_LAB_ROOT / "data/robots/roboparty/rpo/mjcf"
RPO_21_MJCF_FILES = ("rpo_21.xml", "rpo_21_terrain.xml", "rpo_21_stairs.xml")
DEPLOY_CONFIG_DIR = REPO_ROOT / "modules/roboparty_deploy/src/inference/robots/rpo/configs"


class Reporter:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str):
        print(f"[OK] {message}")

    def warn(self, message: str):
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def error(self, message: str):
        self.errors.append(message)
        print(f"[ERROR] {message}")


def _python_list_constants(path: Path, names: set[str]) -> dict[str, list]:
    module = ast.parse(path.read_text())
    constants = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                constants[target.id] = ast.literal_eval(node.value)
    missing = names - constants.keys()
    if missing:
        raise RuntimeError(f"Missing constants in {path}: {sorted(missing)}")
    return constants


def _python_np_array_constant(path: Path, name: str) -> list[float]:
    module = ast.parse(path.read_text())
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call):
            values = ast.literal_eval(value.args[0])
        else:
            values = ast.literal_eval(value)
        return [float(value) for value in values]
    raise RuntimeError(f"Missing {name} in {path}")


def _urdf_effort_limits() -> dict[str, float]:
    root = ET.parse(RPO_URDF).getroot()
    limits = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"{joint.attrib['name']} is missing a URDF limit")
        limits[joint.attrib["name"]] = float(limit.attrib["effort"])
    return limits


def check_robot_assets(reporter: Reporter) -> tuple[list[str], list[str], list[str], list[str]]:
    constants = _python_list_constants(
        RPO_ASSET_PY,
        {"RPO_ACTION_JOINT_NAMES", "RPO_XML_JOINT_NAMES", "RPO_DEPLOY_JOINT_NAMES", "RPO_LINKS"},
    )
    action_names = constants["RPO_ACTION_JOINT_NAMES"]
    xml_names = constants["RPO_XML_JOINT_NAMES"]
    deploy_names = constants["RPO_DEPLOY_JOINT_NAMES"]
    link_names = constants["RPO_LINKS"]

    root = ET.parse(RPO_URDF).getroot()
    urdf_joints = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]
    urdf_links = [link.attrib["name"] for link in root.findall("link")]
    mesh_refs = [mesh.attrib["filename"] for mesh in root.findall(".//mesh")]

    if len(action_names) == 21 and len(set(action_names)) == 21:
        reporter.ok("RPO_ACTION_JOINT_NAMES has 21 unique joints")
    else:
        reporter.error("RPO_ACTION_JOINT_NAMES must have 21 unique joints")

    if urdf_joints == xml_names:
        reporter.ok("RPO_XML_JOINT_NAMES matches Loobot722.urdf joint order")
    else:
        reporter.error("RPO_XML_JOINT_NAMES does not match Loobot722.urdf joint order")

    for label, names in {"action": action_names, "deploy": deploy_names, "xml": xml_names}.items():
        missing = sorted(set(names) - set(urdf_joints))
        extra = sorted(set(urdf_joints) - set(names))
        if missing or extra:
            reporter.error(f"{label} joints differ from URDF joints: missing={missing}, extra={extra}")

    if sorted(link_names) == sorted(urdf_links):
        reporter.ok("RPO_LINKS matches Loobot722.urdf links")
    else:
        reporter.error(
            f"RPO_LINKS differs from URDF links: "
            f"missing={sorted(set(link_names) - set(urdf_links))}, "
            f"extra={sorted(set(urdf_links) - set(link_names))}"
        )

    package_refs = [ref for ref in mesh_refs if ref.startswith("package://")]
    if package_refs:
        reporter.error(f"Loobot722.urdf still contains package mesh refs: {sorted(set(package_refs))}")
    else:
        reporter.ok("Loobot722.urdf uses local mesh paths")

    missing_meshes = sorted({ref for ref in mesh_refs if not (RPO_URDF.parent / ref).resolve().exists()})
    if missing_meshes:
        reporter.error(f"Loobot722.urdf references missing meshes: {missing_meshes}")
    else:
        reporter.ok("all Loobot722.urdf mesh references exist")

    return action_names, deploy_names, xml_names, link_names


def check_deploy_configs(reporter: Reporter, action_names: list[str], deploy_names: list[str]):
    locomotion_constants = _python_list_constants(
        RPO_ONNX_SIM_PY, {"LOCOMOTION_POLICY_JOINT_NAMES"}
    )
    locomotion_action_names = locomotion_constants["LOCOMOTION_POLICY_JOINT_NAMES"]
    locomotion_default_pos = _python_np_array_constant(
        RPO_ONNX_SIM_PY, "LOCOMOTION_DEFAULT_POS_POLICY"
    )
    locomotion_policy_signs = _python_np_array_constant(
        RPO_ONNX_SIM_PY, "LOCOMOTION_POLICY_SIGNS"
    )
    if len(locomotion_action_names) != 21 or len(set(locomotion_action_names)) != 21:
        reporter.error("LOCOMOTION_POLICY_JOINT_NAMES must have 21 unique joints")
    elif set(locomotion_action_names) != set(deploy_names):
        reporter.error("LOCOMOTION_POLICY_JOINT_NAMES differs from deploy joints")
    else:
        reporter.ok("LOCOMOTION_POLICY_JOINT_NAMES has the same 21 joints as deploy")

    config_paths = sorted(DEPLOY_CONFIG_DIR.glob("*.yaml"))
    if not config_paths:
        reporter.warn(f"No deploy configs found at {DEPLOY_CONFIG_DIR}")
        return
    for path in config_paths:
        params = yaml.safe_load(path.read_text())["inference_node"]["ros__parameters"]
        name = path.name
        is_external_locomotion = name == "default.yaml"
        policy_joint_names = locomotion_action_names if is_external_locomotion else action_names
        expected_mapping = [deploy_names.index(joint_name) for joint_name in policy_joint_names]
        mapping_label = "external locomotion policy" if is_external_locomotion else "train action"
        if params.get("joint_num") != 21:
            reporter.error(f"{name}: joint_num is {params.get('joint_num')}, expected 21")
        else:
            reporter.ok(f"{name}: joint_num is 21")
        if params.get("usd2urdf") != expected_mapping:
            reporter.error(f"{name}: usd2urdf does not match {mapping_label}/deploy order")
        else:
            reporter.ok(f"{name}: usd2urdf matches {mapping_label}/deploy order")
        if is_external_locomotion:
            actual_policy_signs = params.get("policy_joint_signs", [1.0] * len(deploy_names))
            if len(actual_policy_signs) != len(locomotion_policy_signs) or not np.allclose(
                actual_policy_signs, locomotion_policy_signs, rtol=0.0, atol=0.0
            ):
                reporter.error("default.yaml: policy_joint_signs does not match Sim2Sim")
            else:
                reporter.ok("default.yaml: policy_joint_signs matches Sim2Sim")
            expected_default_pos = [0.0] * len(deploy_names)
            for policy_index, deploy_index in enumerate(expected_mapping):
                expected_default_pos[deploy_index] = (
                    locomotion_policy_signs[policy_index] * locomotion_default_pos[policy_index]
                )
            actual_default_pos = params.get("joint_default_angle", [])
            if len(actual_default_pos) != len(expected_default_pos) or not np.allclose(
                actual_default_pos, expected_default_pos, rtol=0.0, atol=1e-9
            ):
                reporter.error("default.yaml: joint_default_angle does not match external locomotion policy")
            else:
                reporter.ok("default.yaml: joint_default_angle matches external locomotion policy")
            reset_joint_angle = params.get("reset_joint_angle", actual_default_pos)
            joint_limits = params.get("joint_limits", [])
            if len(reset_joint_angle) != len(deploy_names):
                reporter.error("default.yaml: reset_joint_angle must contain 21 joints")
            elif len(joint_limits) != 2 * len(deploy_names):
                reporter.error("default.yaml: joint_limits must contain lower/upper values for 21 joints")
            else:
                out_of_range = []
                for index, (joint_name, angle) in enumerate(zip(deploy_names, reset_joint_angle)):
                    lower, upper = joint_limits[2 * index : 2 * index + 2]
                    if not lower <= angle <= upper:
                        out_of_range.append(f"{joint_name}={angle} outside [{lower}, {upper}]")
                if out_of_range:
                    reporter.error(
                        "default.yaml: reset_joint_angle violates mechanical limits: "
                        + "; ".join(out_of_range)
                    )
                else:
                    reporter.ok("default.yaml: reset_joint_angle is within mechanical limits")
        for layout in params.get("obs_layouts", []):
            bad_sources = re.findall(r"(?:dof_pos|dof_vel|last_action|motion_pos|motion_vel):(?!21\b)\d+", layout)
            if bad_sources:
                reporter.error(f"{name}: non-21 joint-sized obs fields: {bad_sources}")


def check_retarget_config(reporter: Reporter, action_names: list[str]):
    config = yaml.safe_load(RPO_RETARGET_CFG.read_text())
    lab_names = config.get("lab_dof_names", [])
    if lab_names == action_names:
        reporter.ok("retarget config lab_dof_names matches RPO action order")
    else:
        reporter.error("retarget config lab_dof_names does not match RPO action order")


def _load_pkl(path: Path):
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except Exception:
        try:
            import joblib
        except Exception as exc:
            raise RuntimeError("pickle.load failed and joblib is unavailable") from exc
        return joblib.load(path)


def check_motion_data(reporter: Reporter, action_names: list[str], body_names: list[str], check_lab: bool):
    num_joints = len(action_names)
    num_bodies = len(body_names)
    for path in sorted(RPO_BM_DIR.glob("*.npz")):
        data = np.load(path)
        joint_dims = (data["joint_pos"].shape[1], data["joint_vel"].shape[1])
        body_dims = (
            data["body_pos_w"].shape[1],
            data["body_quat_w"].shape[1],
            data["body_lin_vel_w"].shape[1],
            data["body_ang_vel_w"].shape[1],
        )
        if joint_dims != (num_joints, num_joints) or body_dims != (num_bodies,) * 4:
            reporter.error(f"{path.name}: joint dims={joint_dims}, body dims={body_dims}; expected {num_joints} joints and {num_bodies} bodies")
        else:
            reporter.ok(f"{path.name}: motion dimensions match 21-DoF RPO")
        if "joint_names" not in data.files:
            reporter.warn(f"{path.name}: missing joint_names metadata; regenerate it with csv_to_npz.py")
        elif [str(name) for name in data["joint_names"].tolist()] != action_names:
            reporter.error(f"{path.name}: joint_names metadata does not match RPO action order")
        else:
            reporter.ok(f"{path.name}: joint_names metadata matches RPO action order")
        if "body_names" not in data.files:
            reporter.warn(f"{path.name}: missing body_names metadata; regenerate it with csv_to_npz.py")
        elif [str(name) for name in data["body_names"].tolist()] != body_names:
            reporter.error(f"{path.name}: body_names metadata does not match Loobot722.urdf link order")
        else:
            reporter.ok(f"{path.name}: body_names metadata matches Loobot722.urdf link order")

    if not check_lab:
        reporter.warn("Skipping rpo_lab pickle checks; pass --check-lab-motions to include them")
        return

    for path in sorted(RPO_LAB_DIR.glob("*.pkl")):
        try:
            data = _load_pkl(path)
        except Exception as exc:
            reporter.warn(f"{path.name}: could not load pkl ({exc})")
            continue
        dof_pos = data.get("dof_pos")
        if dof_pos is None:
            reporter.error(f"{path.name}: missing dof_pos")
            continue
        if dof_pos.shape[1] != num_joints:
            reporter.error(f"{path.name}: dof_pos has {dof_pos.shape[1]} DoFs, expected {num_joints}")
        else:
            reporter.ok(f"{path.name}: dof_pos has {num_joints} DoFs")


def check_mujoco_assets(reporter: Reporter, num_joints: int, deploy_names: list[str]):
    effort_limits = _urdf_effort_limits()
    expected_tau_limits = [effort_limits[name] for name in deploy_names]
    mujoco_constants = _python_list_constants(RPO_MUJOCO_PY, {"RPO_MJCF_JOINT_NAMES"})
    if mujoco_constants["RPO_MJCF_JOINT_NAMES"] == deploy_names:
        reporter.ok("RPO_MJCF_JOINT_NAMES matches deploy/MJCF joint order")
    else:
        reporter.error("RPO_MJCF_JOINT_NAMES does not match deploy/MJCF joint order")

    tau_limits = _python_np_array_constant(RPO_MUJOCO_PY, "RPO_TAU_LIMIT")
    if len(tau_limits) != num_joints:
        reporter.error(f"RPO_TAU_LIMIT has {len(tau_limits)} entries, expected {num_joints}")
    else:
        mismatches = [
            f"{name}: tau={actual:g}, urdf={expected:g}"
            for name, actual, expected in zip(deploy_names, tau_limits, expected_tau_limits)
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-6)
        ]
        if mismatches:
            reporter.error("RPO_TAU_LIMIT differs from Loobot722.urdf effort limits: " + "; ".join(mismatches))
        else:
            reporter.ok("RPO_TAU_LIMIT matches Loobot722.urdf effort limits")

    expected_paths = [RPO_MJCF_DIR / name for name in RPO_21_MJCF_FILES]
    for path in expected_paths:
        if not path.is_file():
            reporter.error(f"{path.name}: missing 21-DoF MuJoCo XML")
            continue
        root = ET.parse(path).getroot()
        actuator_nodes = [node for node in root.findall(".//actuator/*") if "joint" in node.attrib]
        actuator_joints = [node.attrib["joint"] for node in actuator_nodes]
        if len(actuator_joints) != num_joints:
            reporter.error(f"{path.name}: has {len(actuator_joints)} actuators, expected {num_joints}")
        else:
            reporter.ok(f"{path.name}: actuator count matches {num_joints}")
        if actuator_joints == deploy_names:
            reporter.ok(f"{path.name}: actuator order matches deploy/MJCF joint order")
        else:
            reporter.error(f"{path.name}: actuator order does not match deploy/MJCF joint order")
        actuator_mismatches = []
        for node in actuator_nodes:
            joint_name = node.attrib["joint"]
            expected = effort_limits.get(joint_name)
            ctrlrange = node.attrib.get("ctrlrange")
            if expected is None or ctrlrange is None:
                continue
            lower, upper = [float(value) for value in ctrlrange.split()]
            if not (np.isclose(lower, -expected, rtol=0.0, atol=1e-6) and np.isclose(upper, expected, rtol=0.0, atol=1e-6)):
                actuator_mismatches.append(f"{joint_name}: ctrlrange={ctrlrange}, urdf=+-{expected:g}")
        if actuator_mismatches:
            reporter.error(f"{path.name}: actuator ctrlrange differs from Loobot722.urdf effort limits: " + "; ".join(actuator_mismatches))
        else:
            reporter.ok(f"{path.name}: actuator ctrlrange matches Loobot722.urdf effort limits")

    for path in sorted(RPO_MJCF_DIR.glob("rpo*.xml")):
        if path.name in RPO_21_MJCF_FILES:
            continue
        text = path.read_text(errors="ignore")
        actuators = re.findall(r"<(?:motor|position|velocity|general)\b[^>]*\bjoint=\"([^\"]+)\"", text)
        if len(actuators) != num_joints:
            reporter.warn(f"{path.name}: legacy MuJoCo XML has {len(actuators)} actuators; current sim2sim uses rpo_21*.xml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-motion-data", action="store_true", help="Validate rpo_bm npz motion files.")
    parser.add_argument("--check-lab-motions", action="store_true", help="Also validate rpo_lab pickle files.")
    args = parser.parse_args()

    reporter = Reporter()
    action_names, deploy_names, _, link_names = check_robot_assets(reporter)
    check_deploy_configs(reporter, action_names, deploy_names)
    check_retarget_config(reporter, action_names)
    if args.check_motion_data or args.check_lab_motions:
        check_motion_data(reporter, action_names, link_names, args.check_lab_motions)
    else:
        reporter.warn("Skipping motion data checks; pass --check-motion-data to include rpo_bm npz files")
    check_mujoco_assets(reporter, len(action_names), deploy_names)

    print(f"\nSummary: {len(reporter.errors)} error(s), {len(reporter.warnings)} warning(s)")
    return 1 if reporter.errors else 0


if __name__ == "__main__":
    sys.exit(main())
