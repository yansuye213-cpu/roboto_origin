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


def replace_string(text, key, value):
    pattern = rf"(^\s*{re.escape(key)}:\s*)\"[^\"]*\""
    updated, count = re.subn(pattern, rf'\g<1>"{value}"', text, count=1, flags=re.MULTILINE)
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
        "--max-normal-force",
        type=float,
        default=None,
        help="Optional stand_wbc_max_normal_force override in newtons per foot.",
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
    parser.add_argument(
        "--backend",
        choices=["ocs2", "disabled"],
        default=None,
        help="Optional stand_wbc_mpc_backend override.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional stand_wbc_mpc_horizon override.",
    )
    parser.add_argument(
        "--mpc-dt",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_dt override in seconds.",
    )
    parser.add_argument(
        "--mpc-iters",
        type=int,
        default=None,
        help="Optional stand_wbc_mpc_qp_iterations override.",
    )
    parser.add_argument(
        "--mpc-force-weight",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_force_weight override for OCS2 force MPC.",
    )
    parser.add_argument(
        "--max-force-delta",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_max_contact_force_delta override in newtons.",
    )
    parser.add_argument(
        "--mpc-constraints",
        action="store_true",
        help="Enable OCS2 solver-side contact constraints.",
    )
    parser.add_argument(
        "--no-mpc-constraints",
        action="store_true",
        help="Disable OCS2 solver-side contact constraints.",
    )
    parser.add_argument(
        "--mpc-zero-swing-force",
        action="store_true",
        help="Enable OCS2 swing-foot zero-force equality constraints.",
    )
    parser.add_argument(
        "--no-mpc-zero-swing-force",
        action="store_true",
        help="Disable OCS2 swing-foot zero-force equality constraints.",
    )
    parser.add_argument(
        "--mpc-normal-force",
        action="store_true",
        help="Enable OCS2 normal-force bound constraints.",
    )
    parser.add_argument(
        "--no-mpc-normal-force",
        action="store_true",
        help="Disable OCS2 normal-force bound constraints.",
    )
    parser.add_argument(
        "--mpc-delta-force",
        action="store_true",
        help="Enable OCS2 contact-force-delta bound constraints.",
    )
    parser.add_argument(
        "--no-mpc-delta-force",
        action="store_true",
        help="Disable OCS2 contact-force-delta bound constraints.",
    )
    parser.add_argument(
        "--mpc-friction-cone",
        action="store_true",
        help="Enable OCS2 friction-cone soft constraints.",
    )
    parser.add_argument(
        "--no-mpc-friction-cone",
        action="store_true",
        help="Disable OCS2 friction-cone soft constraints.",
    )
    parser.add_argument(
        "--friction-barrier-mu",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_friction_barrier_mu override.",
    )
    parser.add_argument(
        "--friction-barrier-delta",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_friction_barrier_delta override.",
    )
    parser.add_argument(
        "--friction-regularization",
        type=float,
        default=None,
        help="Optional stand_wbc_mpc_friction_regularization override.",
    )
    parser.add_argument(
        "--step-recovery",
        action="store_true",
        help="Enable stand_wbc_step_recovery_enabled.",
    )
    parser.add_argument(
        "--no-step-recovery",
        action="store_true",
        help="Disable stand_wbc_step_recovery_enabled.",
    )
    parser.add_argument(
        "--active-set-iters",
        type=int,
        default=None,
        help="Optional stand_wbc_active_set_iterations override.",
    )
    parser.add_argument(
        "--foot-half-length",
        type=float,
        default=None,
        help="Optional stand_wbc_foot_half_length override in meters.",
    )
    parser.add_argument(
        "--foot-half-width",
        type=float,
        default=None,
        help="Optional stand_wbc_foot_half_width override in meters.",
    )
    parser.add_argument(
        "--foot-center-x",
        type=float,
        default=None,
        help="Optional stand_wbc_foot_center_x override in the foot frame.",
    )
    parser.add_argument(
        "--foot-contact-z",
        type=float,
        default=None,
        help="Optional stand_wbc_foot_contact_z override in the foot frame.",
    )
    args = parser.parse_args()

    if args.max_tau < 0.0:
        raise RuntimeError("--max-tau must be non-negative")
    if args.moment_weight is not None and args.moment_weight < 0.0:
        raise RuntimeError("--moment-weight must be non-negative")
    if args.force_weight is not None and args.force_weight < 0.0:
        raise RuntimeError("--force-weight must be non-negative")
    if args.max_normal_force is not None and args.max_normal_force < 0.0:
        raise RuntimeError("--max-normal-force must be non-negative")
    if args.orientation_weight is not None and args.orientation_weight < 0.0:
        raise RuntimeError("--orientation-weight must be non-negative")
    if args.rate_weight is not None and args.rate_weight < 0.0:
        raise RuntimeError("--rate-weight must be non-negative")
    if args.com_weight is not None and args.com_weight < 0.0:
        raise RuntimeError("--com-weight must be non-negative")
    if args.com_vel_weight is not None and args.com_vel_weight < 0.0:
        raise RuntimeError("--com-vel-weight must be non-negative")
    if args.max_ang_accel is not None and args.max_ang_accel < 0.0:
        raise RuntimeError("--max-ang-accel must be non-negative")
    if args.max_com_accel is not None and args.max_com_accel < 0.0:
        raise RuntimeError("--max-com-accel must be non-negative")
    if args.horizon is not None and args.horizon <= 0:
        raise RuntimeError("--horizon must be positive")
    if args.mpc_dt is not None and args.mpc_dt <= 0.0:
        raise RuntimeError("--mpc-dt must be positive")
    if args.mpc_iters is not None and args.mpc_iters < 0:
        raise RuntimeError("--mpc-iters must be non-negative")
    if args.mpc_force_weight is not None and args.mpc_force_weight <= 0.0:
        raise RuntimeError("--mpc-force-weight must be positive")
    if args.max_force_delta is not None and args.max_force_delta < 0.0:
        raise RuntimeError("--max-force-delta must be non-negative")
    if args.mpc_constraints and args.no_mpc_constraints:
        raise RuntimeError("--mpc-constraints and --no-mpc-constraints conflict")
    if args.mpc_zero_swing_force and args.no_mpc_zero_swing_force:
        raise RuntimeError("--mpc-zero-swing-force and --no-mpc-zero-swing-force conflict")
    if args.mpc_normal_force and args.no_mpc_normal_force:
        raise RuntimeError("--mpc-normal-force and --no-mpc-normal-force conflict")
    if args.mpc_delta_force and args.no_mpc_delta_force:
        raise RuntimeError("--mpc-delta-force and --no-mpc-delta-force conflict")
    if args.mpc_friction_cone and args.no_mpc_friction_cone:
        raise RuntimeError("--mpc-friction-cone and --no-mpc-friction-cone conflict")
    if args.friction_barrier_mu is not None and args.friction_barrier_mu < 0.0:
        raise RuntimeError("--friction-barrier-mu must be non-negative")
    if args.friction_barrier_delta is not None and args.friction_barrier_delta <= 0.0:
        raise RuntimeError("--friction-barrier-delta must be positive")
    if args.friction_regularization is not None and args.friction_regularization <= 0.0:
        raise RuntimeError("--friction-regularization must be positive")
    if args.step_recovery and args.no_step_recovery:
        raise RuntimeError("--step-recovery and --no-step-recovery conflict")
    if args.active_set_iters is not None and args.active_set_iters <= 0:
        raise RuntimeError("--active-set-iters must be positive")
    if args.foot_half_length is not None and args.foot_half_length < 0.0:
        raise RuntimeError("--foot-half-length must be non-negative")
    if args.foot_half_width is not None and args.foot_half_width < 0.0:
        raise RuntimeError("--foot-half-width must be non-negative")

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
    if args.max_normal_force is not None:
        text = replace_number(
            text, "stand_wbc_max_normal_force", f"{args.max_normal_force:.4f}")
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
    if args.max_ang_accel is not None:
        text = replace_number(
            text, "stand_wbc_mpc_max_angular_accel", f"{args.max_ang_accel:.4f}")
    if args.max_com_accel is not None:
        text = replace_number(
            text, "stand_wbc_mpc_max_com_accel", f"{args.max_com_accel:.4f}")
    if args.backend is not None:
        text = replace_string(text, "stand_wbc_mpc_backend", args.backend)
    if args.horizon is not None:
        text = replace_number(text, "stand_wbc_mpc_horizon", str(args.horizon))
    if args.mpc_dt is not None:
        text = replace_number(text, "stand_wbc_mpc_dt", f"{args.mpc_dt:.4f}")
    if args.mpc_iters is not None:
        text = replace_number(text, "stand_wbc_mpc_qp_iterations", str(args.mpc_iters))
    if args.mpc_force_weight is not None:
        text = replace_number(
            text, "stand_wbc_mpc_force_weight", f"{args.mpc_force_weight:.4f}")
    if args.max_force_delta is not None:
        text = replace_number(
            text, "stand_wbc_mpc_max_contact_force_delta",
            f"{args.max_force_delta:.4f}")
    if args.mpc_constraints:
        text = replace_bool(text, "stand_wbc_mpc_solver_constraints_enabled", True)
    if args.no_mpc_constraints:
        text = replace_bool(text, "stand_wbc_mpc_solver_constraints_enabled", False)
    if args.mpc_zero_swing_force:
        text = replace_bool(text, "stand_wbc_mpc_zero_swing_force_constraint_enabled", True)
    if args.no_mpc_zero_swing_force:
        text = replace_bool(text, "stand_wbc_mpc_zero_swing_force_constraint_enabled", False)
    if args.mpc_normal_force:
        text = replace_bool(text, "stand_wbc_mpc_normal_force_constraint_enabled", True)
    if args.no_mpc_normal_force:
        text = replace_bool(text, "stand_wbc_mpc_normal_force_constraint_enabled", False)
    if args.mpc_delta_force:
        text = replace_bool(text, "stand_wbc_mpc_delta_force_constraint_enabled", True)
    if args.no_mpc_delta_force:
        text = replace_bool(text, "stand_wbc_mpc_delta_force_constraint_enabled", False)
    if args.mpc_friction_cone:
        text = replace_bool(text, "stand_wbc_mpc_friction_cone_constraint_enabled", True)
    if args.no_mpc_friction_cone:
        text = replace_bool(text, "stand_wbc_mpc_friction_cone_constraint_enabled", False)
    if args.friction_barrier_mu is not None:
        text = replace_number(
            text, "stand_wbc_mpc_friction_barrier_mu",
            f"{args.friction_barrier_mu:.4f}")
    if args.friction_barrier_delta is not None:
        text = replace_number(
            text, "stand_wbc_mpc_friction_barrier_delta",
            f"{args.friction_barrier_delta:.4f}")
    if args.friction_regularization is not None:
        text = replace_number(
            text, "stand_wbc_mpc_friction_regularization",
            f"{args.friction_regularization:.4f}")
    if args.step_recovery:
        text = replace_bool(text, "stand_wbc_step_recovery_enabled", True)
    if args.no_step_recovery:
        text = replace_bool(text, "stand_wbc_step_recovery_enabled", False)
    if args.active_set_iters is not None:
        text = replace_number(
            text, "stand_wbc_active_set_iterations", str(args.active_set_iters))
    if args.foot_half_length is not None:
        text = replace_number(
            text, "stand_wbc_foot_half_length", f"{args.foot_half_length:.4f}")
    if args.foot_half_width is not None:
        text = replace_number(
            text, "stand_wbc_foot_half_width", f"{args.foot_half_width:.4f}")
    if args.foot_center_x is not None:
        text = replace_number(
            text, "stand_wbc_foot_center_x", f"{args.foot_center_x:.4f}")
    if args.foot_contact_z is not None:
        text = replace_number(
            text, "stand_wbc_foot_contact_z", f"{args.foot_contact_z:.4f}")

    CONFIG.write_text(text)
    print(f"Updated {CONFIG}")
    print(f"stand WBC mode: {args.mode}")
    if args.moment_weight is not None:
        print(f"stand_wbc_moment_tracking_weight: {args.moment_weight:.4f}")
    if args.force_weight is not None:
        print(f"stand_wbc_force_tracking_weight: {args.force_weight:.4f}")
    if args.max_normal_force is not None:
        print(f"stand_wbc_max_normal_force: {args.max_normal_force:.4f}")
    if args.orientation_weight is not None:
        print(f"stand_wbc_mpc_orientation_weight: {args.orientation_weight:.4f}")
    if args.rate_weight is not None:
        print(f"stand_wbc_mpc_angular_rate_weight: {args.rate_weight:.4f}")
    if args.com_weight is not None:
        print(f"stand_wbc_mpc_com_weight: {args.com_weight:.4f}")
    if args.com_vel_weight is not None:
        print(f"stand_wbc_mpc_com_velocity_weight: {args.com_vel_weight:.4f}")
    if args.max_ang_accel is not None:
        print(f"stand_wbc_mpc_max_angular_accel: {args.max_ang_accel:.4f}")
    if args.max_com_accel is not None:
        print(f"stand_wbc_mpc_max_com_accel: {args.max_com_accel:.4f}")
    if args.backend is not None:
        print(f"stand_wbc_mpc_backend: {args.backend}")
    if args.horizon is not None:
        print(f"stand_wbc_mpc_horizon: {args.horizon}")
    if args.mpc_dt is not None:
        print(f"stand_wbc_mpc_dt: {args.mpc_dt:.4f}")
    if args.mpc_iters is not None:
        print(f"stand_wbc_mpc_qp_iterations: {args.mpc_iters}")
    if args.mpc_force_weight is not None:
        print(f"stand_wbc_mpc_force_weight: {args.mpc_force_weight:.4f}")
    if args.max_force_delta is not None:
        print(f"stand_wbc_mpc_max_contact_force_delta: {args.max_force_delta:.4f}")
    if args.mpc_constraints or args.no_mpc_constraints:
        print(f"stand_wbc_mpc_solver_constraints_enabled: {args.mpc_constraints}")
    if args.mpc_zero_swing_force or args.no_mpc_zero_swing_force:
        print(
            "stand_wbc_mpc_zero_swing_force_constraint_enabled: "
            f"{args.mpc_zero_swing_force}"
        )
    if args.mpc_normal_force or args.no_mpc_normal_force:
        print(f"stand_wbc_mpc_normal_force_constraint_enabled: {args.mpc_normal_force}")
    if args.mpc_delta_force or args.no_mpc_delta_force:
        print(f"stand_wbc_mpc_delta_force_constraint_enabled: {args.mpc_delta_force}")
    if args.mpc_friction_cone or args.no_mpc_friction_cone:
        print(f"stand_wbc_mpc_friction_cone_constraint_enabled: {args.mpc_friction_cone}")
    if args.friction_barrier_mu is not None:
        print(f"stand_wbc_mpc_friction_barrier_mu: {args.friction_barrier_mu:.4f}")
    if args.friction_barrier_delta is not None:
        print(f"stand_wbc_mpc_friction_barrier_delta: {args.friction_barrier_delta:.4f}")
    if args.friction_regularization is not None:
        print(f"stand_wbc_mpc_friction_regularization: {args.friction_regularization:.4f}")
    if args.step_recovery or args.no_step_recovery:
        print(f"stand_wbc_step_recovery_enabled: {args.step_recovery}")
    if args.active_set_iters is not None:
        print(f"stand_wbc_active_set_iterations: {args.active_set_iters}")
    if args.foot_half_length is not None:
        print(f"stand_wbc_foot_half_length: {args.foot_half_length:.4f}")
    if args.foot_half_width is not None:
        print(f"stand_wbc_foot_half_width: {args.foot_half_width:.4f}")
    if args.foot_center_x is not None:
        print(f"stand_wbc_foot_center_x: {args.foot_center_x:.4f}")
    if args.foot_contact_z is not None:
        print(f"stand_wbc_foot_contact_z: {args.foot_contact_z:.4f}")
    if args.mode == "torque":
        print(f"stand_wbc_max_joint_torque: {args.max_tau:.4f} Nm")
        print("Test suspended first. Ankle torque scale remains zero by default.")


if __name__ == "__main__":
    main()
