#!/usr/bin/env python3
# Copyright (c) 2025-2026, The RoboLab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate 21-DoF RPO MuJoCo XML from the current rpo_21 URDF."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path


ROBO_LAB_ROOT = Path(__file__).resolve().parents[2]
ASSET_PY = ROBO_LAB_ROOT / "robolab/assets/robots/roboparty.py"
URDF_PATH = ROBO_LAB_ROOT / "data/robots/roboparty/rpo/urdf/rpo_21.urdf"
MJCF_DIR = ROBO_LAB_ROOT / "data/robots/roboparty/rpo/mjcf"

MJCF_PRIMITIVE_GEOMS = {
    # MuJoCo rejects STL meshes with more than 200k faces. The Loobot614 base
    # mesh has 266106 faces, so use its bounding box for sim2sim collision.
    "base_link": {
        "type": "box",
        "pos": "0.00231503 0 0.08337592",
        "size": "0.08131504 0.22175001 0.20637409",
    },
}


def _fmt(values: str | list[str]) -> str:
    if isinstance(values, str):
        values = values.split()
    return " ".join(f"{float(value):.12g}" for value in values)


def _list_constant(name: str) -> list[str]:
    module = ast.parse(ASSET_PY.read_text())
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"Missing {name} in {ASSET_PY}")


def _child(element: ET.Element, path: str) -> ET.Element:
    child = element.find(path)
    if child is None:
        raise RuntimeError(f"Missing {path} in {element.attrib.get('name', element.tag)}")
    return child


def _link_info(link: ET.Element) -> dict[str, str]:
    inertial = _child(link, "inertial")
    inertia = _child(inertial, "inertia").attrib
    visual = link.find("visual")
    color = "0.89804 0.91765 0.92941 1"
    mesh_file = f"{link.attrib['name']}.STL"
    if visual is not None:
        mesh = visual.find("geometry/mesh")
        if mesh is not None:
            mesh_file = Path(mesh.attrib["filename"]).name
        color_node = visual.find("material/color")
        if color_node is not None:
            color = _fmt(color_node.attrib["rgba"])
    return {
        "mass": _child(inertial, "mass").attrib["value"],
        "pos": _fmt(_child(inertial, "origin").attrib.get("xyz", "0 0 0")),
        "fullinertia": _fmt(
            [
                inertia["ixx"],
                inertia["iyy"],
                inertia["izz"],
                inertia["ixy"],
                inertia["ixz"],
                inertia["iyz"],
            ]
        ),
        "mesh_file": mesh_file,
        "rgba": color,
    }


def _joint_info(joint: ET.Element) -> dict[str, str]:
    limit = _child(joint, "limit").attrib
    return {
        "parent": _child(joint, "parent").attrib["link"],
        "child": _child(joint, "child").attrib["link"],
        "pos": _fmt(_child(joint, "origin").attrib.get("xyz", "0 0 0")),
        "euler": _fmt(_child(joint, "origin").attrib.get("rpy", "0 0 0")),
        "axis": _fmt(_child(joint, "axis").attrib["xyz"]),
        "range": _fmt([limit["lower"], limit["upper"]]),
    }


def _joint_class(joint_name: str) -> str:
    if joint_name == "head_yaw_joint":
        return "head_joint_param"
    if "shoulder" in joint_name or "elbow" in joint_name:
        return "arm_joint_param"
    return "leg_joint_param"


def _tau_limit(joint_name: str) -> float:
    if "ankle" in joint_name or "shoulder" in joint_name or "elbow" in joint_name or joint_name == "head_yaw_joint":
        return 27.0
    return 120.0


def _add_common(parent: ET.Element):
    ET.SubElement(parent, "compiler", angle="radian", meshdir="../meshes/", eulerseq="zyx")
    option = ET.SubElement(parent, "option", timestep="0.001", iterations="100", solver="PGS", gravity="0 0 -9.81")
    ET.SubElement(option, "flag", frictionloss="enable")
    ET.SubElement(parent, "size", njmax="5000", nconmax="1000")
    visual = ET.SubElement(parent, "visual")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "map", znear="0.05")
    default = ET.SubElement(parent, "default")
    ET.SubElement(default, "joint", limited="true")
    ET.SubElement(default, "motor", ctrllimited="true")
    ET.SubElement(
        default,
        "geom",
        condim="4",
        contype="1",
        conaffinity="0",
        solref="0.001 2",
        friction="0.9 0.2 0.2",
        group="1",
    )
    ET.SubElement(default, "equality", solref="0.001 2")
    for class_name in ("leg_joint_param", "arm_joint_param", "head_joint_param"):
        class_default = ET.SubElement(default, "default", **{"class": class_name})
        ET.SubElement(class_default, "joint", damping="0.01", frictionloss="0.01", armature="0.01")


def _add_assets(parent: ET.Element, links: dict[str, dict[str, str]], include_terrain: bool, include_stairs: bool):
    asset = ET.SubElement(parent, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "texture",
        name="texplane",
        type="2d",
        builtin="checker",
        rgb1=".2 .3 .4",
        rgb2=".1 0.15 0.2",
        width="512",
        height="512",
        mark="cross",
        markrgb=".8 .8 .8",
    )
    ET.SubElement(asset, "material", name="matplane", reflectance="0.", texture="texplane", texrepeat="1 1", texuniform="true")
    if include_stairs:
        ET.SubElement(asset, "material", name="stairmat", rgba="0.65 0.65 0.7 1")
    for link_name, info in links.items():
        if link_name in MJCF_PRIMITIVE_GEOMS:
            continue
        ET.SubElement(asset, "mesh", name=link_name, file=info["mesh_file"])
    if include_terrain:
        ET.SubElement(asset, "hfield", name="terrain_hfield", file="../terrain_assets/terrain_hfield.png", size="10.0 10.0 3.4 2.0")


def _add_stairs(worldbody: ET.Element):
    tread_depth = 0.32
    tread_width = 1.2
    step_height = 0.08
    x0 = 0.7
    for i in range(8):
        height = step_height * (i + 1)
        x = x0 + i * tread_depth
        ET.SubElement(
            worldbody,
            "geom",
            name=f"rpo_21_stair_{i}",
            type="box",
            pos=f"{x:.3f} 0 {height / 2:.3f}",
            size=f"{tread_depth / 2:.3f} {tread_width / 2:.3f} {height / 2:.3f}",
            material="stairmat",
            condim="3",
            conaffinity="15",
            group="0",
        )


def _add_link_body(
    parent: ET.Element,
    link_name: str,
    links: dict[str, dict[str, str]],
    joints: dict[str, dict[str, str]],
    children_by_parent: dict[str, list[str]],
    joint_order: list[str],
    incoming_joint_name: str | None = None,
):
    info = links[link_name]
    ET.SubElement(parent, "inertial", pos=info["pos"], mass=info["mass"], fullinertia=info["fullinertia"])
    if link_name == "base_link":
        ET.SubElement(parent, "joint", name="floating_base_joint", type="free", limited="false")
    elif incoming_joint_name is not None:
        joint = joints[incoming_joint_name]
        ET.SubElement(
            parent,
            "joint",
            name=incoming_joint_name,
            pos="0 0 0",
            axis=joint["axis"],
            range=joint["range"],
            **{"class": _joint_class(incoming_joint_name)},
        )
    if link_name in MJCF_PRIMITIVE_GEOMS:
        ET.SubElement(parent, "geom", rgba=info["rgba"], **MJCF_PRIMITIVE_GEOMS[link_name])
    else:
        ET.SubElement(parent, "geom", type="mesh", rgba=info["rgba"], mesh=link_name)
    ordered_children = sorted(
        children_by_parent.get(link_name, []),
        key=lambda name: joint_order.index(name) if name in joint_order else len(joint_order),
    )
    for joint_name in ordered_children:
        joint = joints[joint_name]
        body = ET.SubElement(parent, "body", name=joint["child"], pos=joint["pos"])
        if joint["euler"] != "0 0 0":
            body.set("euler", joint["euler"])
        _add_link_body(body, joint["child"], links, joints, children_by_parent, joint_order, joint_name)


def _add_worldbody(
    parent: ET.Element,
    links: dict[str, dict[str, str]],
    joints: dict[str, dict[str, str]],
    children_by_parent: dict[str, list[str]],
    joint_order: list[str],
    include_terrain: bool,
    include_stairs: bool,
):
    worldbody = ET.SubElement(parent, "worldbody")
    ET.SubElement(worldbody, "light", directional="true", diffuse=".4 .4 .4", specular="0.1 0.1 0.1", pos="0 0 5.0", dir="0 0 -1", castshadow="false")
    ET.SubElement(worldbody, "light", directional="true", diffuse=".6 .6 .6", specular="0.2 0.2 0.2", pos="0 0 4", dir="0 0 -1")
    if include_terrain:
        ET.SubElement(worldbody, "geom", name="ground", type="hfield", hfield="terrain_hfield", pos="0 0 0", material="matplane", condim="1", conaffinity="15", group="0")
        base_pos = "0 0 2.75"
    else:
        ET.SubElement(worldbody, "geom", name="ground", type="plane", size="0 0 1", pos="0.001 0 0", quat="1 0 0 0", material="matplane", condim="1", conaffinity="15", group="0")
        base_pos = "0 0 0.75"
    if include_stairs:
        _add_stairs(worldbody)
    base = ET.SubElement(worldbody, "body", name="base_link", pos=base_pos)
    ET.SubElement(base, "site", name="imu", size="0.01", pos="0.0 0 0.0")
    _add_link_body(base, "base_link", links, joints, children_by_parent, joint_order)


def _add_actuators(parent: ET.Element, joint_order: list[str]):
    actuator = ET.SubElement(parent, "actuator")
    for joint_name in joint_order:
        limit = _tau_limit(joint_name)
        ET.SubElement(
            actuator,
            "motor",
            name=joint_name,
            joint=joint_name,
            gear="1",
            ctrllimited="true",
            ctrlrange=f"-{limit:g} {limit:g}",
        )


def _add_sensors(parent: ET.Element, joint_order: list[str]):
    sensor = ET.SubElement(parent, "sensor")
    for joint_name in joint_order:
        ET.SubElement(sensor, "actuatorpos", name=f"{joint_name}_p", actuator=joint_name, user="13")
    for joint_name in joint_order:
        ET.SubElement(sensor, "actuatorvel", name=f"{joint_name}_v", actuator=joint_name, user="13")
    for joint_name in joint_order:
        ET.SubElement(sensor, "actuatorfrc", name=f"{joint_name}_f", actuator=joint_name, user="13", noise="1e-3")
    ET.SubElement(sensor, "framequat", name="orientation", objtype="site", noise="0.001", objname="imu")
    ET.SubElement(sensor, "framepos", name="position", objtype="site", noise="0.001", objname="imu")
    ET.SubElement(sensor, "gyro", name="angular-velocity", site="imu", noise="0.005", cutoff="34.9")
    ET.SubElement(sensor, "velocimeter", name="linear-velocity", site="imu", noise="0.001", cutoff="30")
    ET.SubElement(sensor, "accelerometer", name="linear-acceleration", site="imu", noise="0.005", cutoff="157")
    ET.SubElement(sensor, "magnetometer", name="magnetometer", site="imu")


def generate(path: Path, include_terrain: bool, include_stairs: bool = False):
    root = ET.parse(URDF_PATH).getroot()
    joint_order = _list_constant("RPO_DEPLOY_JOINT_NAMES")
    links = {link.attrib["name"]: _link_info(link) for link in root.findall("link")}
    joints = {
        joint.attrib["name"]: _joint_info(joint)
        for joint in root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    }
    children_by_parent: dict[str, list[str]] = {}
    for joint_name, joint in joints.items():
        children_by_parent.setdefault(joint["parent"], []).append(joint_name)
    missing = sorted(set(joint_order) - set(joints))
    if missing:
        raise RuntimeError(f"Deploy joint order contains joints missing from URDF: {missing}")

    mujoco = ET.Element("mujoco", model="rpo_21")
    _add_common(mujoco)
    _add_assets(mujoco, links, include_terrain, include_stairs)
    _add_worldbody(mujoco, links, joints, children_by_parent, joint_order, include_terrain, include_stairs)
    _add_actuators(mujoco, joint_order)
    _add_sensors(mujoco, joint_order)
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.indent(mujoco, space="  ")
    tree = ET.ElementTree(mujoco)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    path.write_text(path.read_text() + "\n")


def main():
    generate(MJCF_DIR / "rpo_21.xml", include_terrain=False)
    generate(MJCF_DIR / "rpo_21_terrain.xml", include_terrain=True)
    generate(MJCF_DIR / "rpo_21_stairs.xml", include_terrain=False, include_stairs=True)


if __name__ == "__main__":
    main()
