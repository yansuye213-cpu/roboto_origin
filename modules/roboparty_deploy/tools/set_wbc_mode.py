#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/inference/robots/rpo/configs/default.yaml"


def replace_string(text, key, value):
    pattern = rf"(^\s*{re.escape(key)}:\s*)\"[^\"]*\""
    updated, count = re.subn(pattern, rf'\g<1>"{value}"', text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not replace {key}")
    return updated


def replace_bool(text, key, value):
    pattern = rf"(^\s*{re.escape(key)}:\s*)(true|false)"
    updated, count = re.subn(pattern, rf"\g<1>{'true' if value else 'false'}", text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not replace {key}")
    return updated


def replace_number(text, key, value):
    pattern = rf"(^\s*{re.escape(key)}:\s*)[-+0-9.eE]+"
    updated, count = re.subn(pattern, rf"\g<1>{value}", text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not replace {key}")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Switch stand whole-body control testing mode.")
    parser.add_argument(
        "mode",
        choices=["validate", "dryrun", "torque"],
        help=(
            "validate: keep joint_qp and only load/check the whole-body model; "
            "dryrun: use whole_body_mpc but keep torque output disabled; "
            "torque: use whole_body_mpc and enable limited feedforward torque"
        ),
    )
    parser.add_argument(
        "--max-tau",
        type=float,
        default=0.2,
        help="Joint torque clamp in Nm for torque mode. Ignored by validate/dryrun.",
    )
    args = parser.parse_args()

    if args.max_tau < 0.0:
        raise RuntimeError("--max-tau must be non-negative")

    text = CONFIG.read_text()
    if args.mode == "validate":
        text = replace_string(text, "stand_control_backend", "joint_qp")
        text = replace_bool(text, "stand_validate_whole_body_model", True)
        text = replace_bool(text, "stand_wbc_enable_torque", False)
    elif args.mode == "dryrun":
        text = replace_string(text, "stand_control_backend", "whole_body_mpc")
        text = replace_bool(text, "stand_validate_whole_body_model", True)
        text = replace_bool(text, "stand_wbc_enable_torque", False)
    else:
        text = replace_string(text, "stand_control_backend", "whole_body_mpc")
        text = replace_bool(text, "stand_validate_whole_body_model", True)
        text = replace_bool(text, "stand_wbc_enable_torque", True)
        text = replace_number(text, "stand_wbc_max_joint_torque", f"{args.max_tau:.4f}")

    CONFIG.write_text(text)
    print(f"Updated {CONFIG}")
    print(f"stand WBC mode: {args.mode}")
    if args.mode == "torque":
        print(f"stand_wbc_max_joint_torque: {args.max_tau:.4f} Nm")
        print("Test suspended first. Ankle torque scale remains zero by default.")


if __name__ == "__main__":
    main()
