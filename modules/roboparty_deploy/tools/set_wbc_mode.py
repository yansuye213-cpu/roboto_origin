#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/inference/robots/rpo/configs/default.yaml"


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
            "validate: load/check the whole-body model with torque disabled; "
            "dryrun: run whole-body MPC/WBC but keep torque output disabled; "
            "torque: enable limited feedforward torque from whole-body WBC"
        ),
    )
    parser.add_argument(
        "--max-tau",
        type=float,
        default=0.2,
        help="Joint torque clamp in Nm for torque mode. Ignored by validate/dryrun.",
    )
    parser.add_argument(
        "--moment-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_moment_tracking_weight override.",
    )
    parser.add_argument(
        "--force-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_force_tracking_weight override.",
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_orientation_weight override.",
    )
    parser.add_argument(
        "--rate-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_angular_rate_weight override.",
    )
    parser.add_argument(
        "--com-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_com_weight override.",
    )
    parser.add_argument(
        "--com-vel-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_com_velocity_weight override.",
    )
    parser.add_argument(
        "--control-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_control_weight override.",
    )
    parser.add_argument(
        "--max-ang-accel",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_max_angular_accel override.",
    )
    parser.add_argument(
        "--max-com-accel",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_max_com_accel override.",
    )
    args = parser.parse_args()

    if args.max_tau < 0.0:
        raise RuntimeError("--max-tau must be non-negative")
    if args.moment_weight is not None and args.moment_weight < 0.0:
        raise RuntimeError("--moment-weight must be non-negative")
    if args.force_weight is not None and args.force_weight < 0.0:
        raise RuntimeError("--force-weight must be non-negative")
    if args.orientation_weight is not None and args.orientation_weight < 0.0:
        raise RuntimeError("--orientation-weight must be non-negative")
    if args.rate_weight is not None and args.rate_weight < 0.0:
        raise RuntimeError("--rate-weight must be non-negative")
    if args.com_weight is not None and args.com_weight < 0.0:
        raise RuntimeError("--com-weight must be non-negative")
    if args.com_vel_weight is not None and args.com_vel_weight < 0.0:
        raise RuntimeError("--com-vel-weight must be non-negative")
    if args.control_weight is not None and args.control_weight <= 0.0:
        raise RuntimeError("--control-weight must be positive")
    if args.max_ang_accel is not None and args.max_ang_accel < 0.0:
        raise RuntimeError("--max-ang-accel must be non-negative")
    if args.max_com_accel is not None and args.max_com_accel < 0.0:
        raise RuntimeError("--max-com-accel must be non-negative")

    text = CONFIG.read_text()
    if args.mode == "validate":
        text = replace_bool(text, "stand_validate_whole_body_model", True)
        text = replace_bool(text, "stand_wbc_enable_torque", False)
    elif args.mode == "dryrun":
        text = replace_bool(text, "stand_validate_whole_body_model", True)
        text = replace_bool(text, "stand_wbc_enable_torque", False)
    else:
        text = replace_bool(text, "stand_validate_whole_body_model", True)
        text = replace_bool(text, "stand_wbc_enable_torque", True)
        text = replace_number(text, "stand_wbc_max_joint_torque", f"{args.max_tau:.4f}")
    if args.moment_weight is not None:
        text = replace_number(
            text, "stand_wbc_moment_tracking_weight", f"{args.moment_weight:.4f}")
    if args.force_weight is not None:
        text = replace_number(
            text, "stand_wbc_force_tracking_weight", f"{args.force_weight:.4f}")
    if args.orientation_weight is not None:
        text = replace_number(
            text, "stand_wbc_mpc_orientation_weight", f"{args.orientation_weight:.4f}")
    if args.rate_weight is not None:
        text = replace_number(
            text, "stand_wbc_mpc_angular_rate_weight", f"{args.rate_weight:.4f}")
    if args.com_weight is not None:
        text = replace_number(
            text, "stand_wbc_mpc_com_weight", f"{args.com_weight:.4f}")
    if args.com_vel_weight is not None:
        text = replace_number(
            text, "stand_wbc_mpc_com_velocity_weight", f"{args.com_vel_weight:.4f}")
    if args.control_weight is not None:
        text = replace_number(
            text, "stand_wbc_mpc_control_weight", f"{args.control_weight:.4f}")
    if args.max_ang_accel is not None:
        text = replace_number(
            text, "stand_wbc_mpc_max_angular_accel", f"{args.max_ang_accel:.4f}")
    if args.max_com_accel is not None:
        text = replace_number(
            text, "stand_wbc_mpc_max_com_accel", f"{args.max_com_accel:.4f}")

    CONFIG.write_text(text)
    print(f"Updated {CONFIG}")
    print(f"stand WBC mode: {args.mode}")
    if args.moment_weight is not None:
        print(f"stand_wbc_moment_tracking_weight: {args.moment_weight:.4f}")
    if args.force_weight is not None:
        print(f"stand_wbc_force_tracking_weight: {args.force_weight:.4f}")
    if args.orientation_weight is not None:
        print(f"stand_wbc_mpc_orientation_weight: {args.orientation_weight:.4f}")
    if args.rate_weight is not None:
        print(f"stand_wbc_mpc_angular_rate_weight: {args.rate_weight:.4f}")
    if args.com_weight is not None:
        print(f"stand_wbc_mpc_com_weight: {args.com_weight:.4f}")
    if args.com_vel_weight is not None:
        print(f"stand_wbc_mpc_com_velocity_weight: {args.com_vel_weight:.4f}")
    if args.control_weight is not None:
        print(f"stand_wbc_mpc_control_weight: {args.control_weight:.4f}")
    if args.max_ang_accel is not None:
        print(f"stand_wbc_mpc_max_angular_accel: {args.max_ang_accel:.4f}")
    if args.max_com_accel is not None:
        print(f"stand_wbc_mpc_max_com_accel: {args.max_com_accel:.4f}")
    if args.mode == "torque":
        print(f"stand_wbc_max_joint_torque: {args.max_tau:.4f} Nm")
        print("Test suspended first. Ankle torque scale remains zero by default.")


if __name__ == "__main__":
    main()
