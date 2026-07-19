#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/inference/robots/rpo/configs/default.yaml"

JOINT_INDEX = {
    "j0": {"first": 0, "second": 6},
    "j1": {"first": 1, "second": 7},
    "j2": {"first": 2, "second": 8},
    "j3": {"first": 3, "second": 9},
    "j4": {"first": 4, "second": 10},
    "j5": {"first": 5, "second": 11},
    "hip": {"first": 2, "second": 8},
    "knee": {"first": 3, "second": 9},
    "ankle": {"first": 4, "second": 10},
}


def format_vector(values):
    groups = [
        values[0:6],
        values[6:12],
        values[12:13],
        values[13:17],
        values[17:21],
    ]
    lines = []
    for i, group in enumerate(groups):
        prefix = "            [" if i == 0 else "             "
        suffix = "]" if i == len(groups) - 1 else ","
        lines.append(prefix + ", ".join(f"{v:.4f}" for v in group) + suffix)
    return "\n".join(lines) + "\n"


def replace_scalar(text, key, value):
    pattern = rf"(^\s*{re.escape(key)}:\s*)[-+0-9.eE]+"
    return re.sub(pattern, rf"\g<1>{value}", text, count=1, flags=re.MULTILINE)


def replace_bool(text, key, value):
    pattern = rf"(^\s*{re.escape(key)}:\s*)(true|false)"
    return re.sub(pattern, rf"\g<1>{'true' if value else 'false'}", text, count=1, flags=re.MULTILINE)


def replace_stand_joint_angle(text, values):
    replacement = "        stand_joint_angle:\n" + format_vector(values) + "        stand_transition_time:"
    pattern = r"        stand_joint_angle:\n(?:.*\n)*?        stand_transition_time:"
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not replace stand_joint_angle block")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Set a small stand-mode leg joint bias for hardware testing.")
    parser.add_argument("joint", choices=["reset", *JOINT_INDEX.keys()])
    parser.add_argument("value", nargs="?", type=float, default=0.0, help="Bias in radians, for example 0.03 or -0.03")
    parser.add_argument("--leg", choices=["both", "first", "second"], default="both")
    args = parser.parse_args()

    values = [0.0] * 21
    if args.joint != "reset":
        legs = ["first", "second"] if args.leg == "both" else [args.leg]
        for leg in legs:
            values[JOINT_INDEX[args.joint][leg]] = args.value

    text = CONFIG.read_text()
    text = replace_stand_joint_angle(text, values)
    text = replace_scalar(text, "stand_transition_time", "0.3")
    text = replace_bool(text, "stand_wbc_enable_torque", False)
    text = replace_scalar(text, "stand_wbc_max_joint_torque", "0.0")
    CONFIG.write_text(text)

    active = [(i, v) for i, v in enumerate(values) if abs(v) > 1e-9]
    print(f"Updated {CONFIG}")
    print("Active stand_joint_angle offsets:", active if active else "none")
    print("WBC torque disabled for this joint-bias test.")


if __name__ == "__main__":
    main()
