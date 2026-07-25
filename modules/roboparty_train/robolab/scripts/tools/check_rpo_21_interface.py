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
RPO_URDF = ROBO_LAB_ROOT / "data/robots/roboparty/rpo/urdf/rpo_21.urdf"
RPO_RETARGET_CFG = ROBO_LAB_ROOT / "scripts/tools/retarget/config/rpo.yaml"
RPO_BM_DIR = ROBO_LAB_ROOT / "data/motions/rpo_bm"
RPO_LAB_DIR = ROBO_LAB_ROOT / "data/motions/rpo_lab"
RPO_MJCF_DIR = ROBO_LAB_ROOT / "data/robots/roboparty/rpo/mjcf"
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
        reporter.ok("RPO_XML_JOINT_NAMES matches rpo_21.urdf joint order")
    else:
        reporter.error("RPO_XML_JOINT_NAMES does not match rpo_21.urdf joint order")

    for label, names in {"action": action_names, "deploy": deploy_names, "xml": xml_names}.items():
        missing = sorted(set(names) - set(urdf_joints))
        extra = sorted(set(urdf_joints) - set(names))
        if missing or extra:
            reporter.error(f"{label} joints differ from URDF joints: missing={missing}, extra={extra}")

    if sorted(link_names) == sorted(urdf_links):
        reporter.ok("RPO_LINKS matches rpo_21.urdf links")
    else:
        reporter.error(
            f"RPO_LINKS differs from URDF links: "
            f"missing={sorted(set(link_names) - set(urdf_links))}, "
            f"extra={sorted(set(urdf_links) - set(link_names))}"
        )

    package_refs = [ref for ref in mesh_refs if ref.startswith("package://")]
    if package_refs:
        reporter.error(f"rpo_21.urdf still contains package mesh refs: {sorted(set(package_refs))}")
    else:
        reporter.ok("rpo_21.urdf uses local mesh paths")

    missing_meshes = sorted({ref for ref in mesh_refs if not (RPO_URDF.parent / ref).resolve().exists()})
    if missing_meshes:
        reporter.error(f"rpo_21.urdf references missing meshes: {missing_meshes}")
    else:
        reporter.ok("all rpo_21.urdf mesh references exist")

    return action_names, deploy_names, xml_names, link_names


def check_deploy_configs(reporter: Reporter, action_names: list[str], deploy_names: list[str]):
    expected_mapping = [deploy_names.index(name) for name in action_names]
    config_paths = sorted(DEPLOY_CONFIG_DIR.glob("*.yaml"))
    if not config_paths:
        reporter.warn(f"No deploy configs found at {DEPLOY_CONFIG_DIR}")
        return
    for path in config_paths:
        params = yaml.safe_load(path.read_text())["inference_node"]["ros__parameters"]
        name = path.name
        if params.get("joint_num") != 21:
            reporter.error(f"{name}: joint_num is {params.get('joint_num')}, expected 21")
        else:
            reporter.ok(f"{name}: joint_num is 21")
        if params.get("usd2urdf") != expected_mapping:
            reporter.error(f"{name}: usd2urdf does not match train action/deploy order")
        else:
            reporter.ok(f"{name}: usd2urdf matches train action/deploy order")
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
            reporter.error(f"{path.name}: body_names metadata does not match rpo_21.urdf link order")
        else:
            reporter.ok(f"{path.name}: body_names metadata matches rpo_21.urdf link order")

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


def check_mujoco_assets(reporter: Reporter, num_joints: int):
    for path in sorted(RPO_MJCF_DIR.glob("rpo*.xml")):
        text = path.read_text(errors="ignore")
        actuators = re.findall(r"<(?:motor|position|velocity|general)\b[^>]*\bjoint=\"([^\"]+)\"", text)
        if len(actuators) != num_joints:
            reporter.warn(f"{path.name}: has {len(actuators)} actuators, expected {num_joints}; MuJoCo sim2sim is still legacy")
        else:
            reporter.ok(f"{path.name}: actuator count matches {num_joints}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-lab-motions", action="store_true", help="Also validate rpo_lab pickle files.")
    args = parser.parse_args()

    reporter = Reporter()
    action_names, deploy_names, _, link_names = check_robot_assets(reporter)
    check_deploy_configs(reporter, action_names, deploy_names)
    check_retarget_config(reporter, action_names)
    check_motion_data(reporter, action_names, link_names, args.check_lab_motions)
    check_mujoco_assets(reporter, len(action_names))

    print(f"\nSummary: {len(reporter.errors)} error(s), {len(reporter.warnings)} warning(s)")
    return 1 if reporter.errors else 0


if __name__ == "__main__":
    sys.exit(main())
