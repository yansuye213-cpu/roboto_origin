#!/usr/bin/env python3
"""Run guarded single-joint sine tracking tests on an RPO robot.

The robot must be suspended or otherwise restrained. The script holds the
initial measured pose, excites one joint, records control-loop snapshots, then
returns to the initial pose and disables all motors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_ROOT = SCRIPT_DIR.parent
DEFAULT_ROBOT_CONFIG = DEPLOY_ROOT / "src/inference/robots/rpo/robot.yaml"
DEFAULT_DEPLOY_CONFIG = DEPLOY_ROOT / "src/inference/robots/rpo/configs/default.yaml"
DEFAULT_URDF = DEPLOY_ROOT / "src/inference/robots/rpo/description/urdf/Loobot722.urdf"
DEFAULT_OUTPUT_DIR = DEPLOY_ROOT / "experiment_logs/joint_sine"
EXPECTED_JOINT_COUNT = 21
WALKING_JOINT_NAMES = (
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
)
CONTROL_PROCESS_MARKERS = (
    "inference_node",
    "motion_player.py",
    "motors_py_example.py",
    "set_zero.py",
    "check_motor_sign.py",
    "joint_sine_tracking.py",
)


@dataclass(frozen=True)
class JointInfo:
    index: int
    name: str
    lower: float
    upper: float


@dataclass
class Sample:
    elapsed_s: float
    scheduled_s: float
    command_done_s: float
    sample_done_s: float
    loop_period_s: float
    phase: str
    target_offset_rad: float
    q_target_rad: float
    q_actual_rad: float
    dq_actual_rad_s: float
    feedback_field: float
    imu_qw: float
    imu_qx: float
    imu_qy: float
    imu_qz: float
    imu_wx_rad_s: float
    imu_wy_rad_s: float
    imu_wz_rad_s: float
    q_target_all_rad: list[float]
    q_actual_all_rad: list[float]
    dq_actual_all_rad_s: list[float]
    feedback_field_all: list[float]


class SafetyAbort(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RPO 单关节正弦跟踪测试。机器人必须悬吊或可靠约束；其他关节会保持使能时的实测姿态。"
        )
    )
    parser.add_argument(
        "joint",
        nargs="?",
        help="完整关节名，或按部署/RobotInterface 顺序的 1-21 序号；使用 --list-joints 查看",
    )
    parser.add_argument(
        "--walking-joints", action="store_true",
        help="依次测试左右腿全部 12 个行走关节；此时不要提供 joint",
    )
    parser.add_argument("--amplitude", type=float, default=0.02, help="正弦幅值，rad（默认: 0.02）")
    parser.add_argument("--frequency", type=float, default=0.5, help="正弦频率，Hz（默认: 0.5）")
    parser.add_argument("--cycles", type=float, default=8.0, help="稳态正弦周期数（默认: 8）")
    parser.add_argument("--rate", type=float, default=250.0, help="目标控制/记录频率，Hz（默认: 250）")
    parser.add_argument("--settle", type=float, default=2.0, help="激励前保持时间，s（默认: 2）")
    parser.add_argument("--ramp", type=float, default=2.0, help="正弦包络进入/退出时间，s（默认: 2）")
    parser.add_argument("--recover", type=float, default=2.0, help="激励后中心保持时间，s（默认: 2）")
    parser.add_argument(
        "--limit-margin", type=float, default=0.03,
        help="目标距离 URDF 关节限位的最小余量，rad（默认: 0.03）",
    )
    parser.add_argument(
        "--feedback-limit-tolerance", type=float, default=0.01,
        help="实际反馈允许超出 URDF 限位的测量/跟踪容差，rad（默认: 0.01）",
    )
    parser.add_argument(
        "--max-tracking-error", type=float, default=0.15,
        help="被测关节允许的最大 |q_target-q|，rad（默认: 0.15）",
    )
    parser.add_argument(
        "--max-hold-error", type=float, default=0.25,
        help="其他关节相对起始保持目标的最大误差，rad（默认: 0.25）",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument(
        "--joint-order-config", type=Path, default=DEFAULT_DEPLOY_CONFIG,
        help="提供 RobotInterface 21 维关节顺序的部署配置",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, help="CSV 输出路径；默认写入 experiment_logs/joint_sine")
    parser.add_argument("--list-joints", action="store_true", help="列出可测关节后退出，不连接硬件")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只检查参数、配置和目标范围，不导入 robot_py 或连接硬件",
    )
    return parser.parse_args()


def load_joint_order(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        order = config["inference_node"]["ros__parameters"]["stand_whole_body_joint_order"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{config_path} 缺少 inference_node.ros__parameters.stand_whole_body_joint_order"
        ) from error
    if not isinstance(order, list) or len(order) != EXPECTED_JOINT_COUNT:
        raise ValueError("stand_whole_body_joint_order 必须包含 21 个关节名")
    names = [str(name) for name in order]
    if len(set(names)) != EXPECTED_JOINT_COUNT:
        raise ValueError("stand_whole_body_joint_order 包含重复关节名")
    return names


def load_joint_info(urdf_path: Path, joint_order: list[str]) -> list[JointInfo]:
    root = ET.parse(urdf_path).getroot()
    urdf_limits = {}
    for joint in root.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"关节 {joint.get('name')} 缺少 lower/upper limit")
        name = str(joint.get("name"))
        urdf_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
    if len(urdf_limits) != EXPECTED_JOINT_COUNT:
        raise ValueError(
            f"URDF 可动关节应为 {EXPECTED_JOINT_COUNT}，实际为 {len(urdf_limits)}"
        )
    missing = [name for name in joint_order if name not in urdf_limits]
    extra = [name for name in urdf_limits if name not in joint_order]
    if missing or extra:
        raise ValueError(f"部署关节顺序与 URDF 不一致: missing={missing}, extra={extra}")
    return [
        JointInfo(index=index, name=name, lower=urdf_limits[name][0], upper=urdf_limits[name][1])
        for index, name in enumerate(joint_order)
    ]


def select_joint(value: str | None, joints: list[JointInfo]) -> JointInfo:
    if value is None:
        raise ValueError("缺少 joint；使用 --list-joints 查看可用值")
    if value.isdigit() and 1 <= int(value) <= len(joints):
        return joints[int(value) - 1]
    for joint in joints:
        if joint.name == value:
            return joint
    raise ValueError(f"未知关节 {value!r}；使用 --list-joints 查看可用值")


def select_joints(args: argparse.Namespace, joints: list[JointInfo]) -> list[JointInfo]:
    if args.walking_joints:
        if args.joint is not None:
            raise ValueError("--walking-joints 与 positional joint 不能同时使用")
        by_name = {joint.name: joint for joint in joints}
        return [by_name[name] for name in WALKING_JOINT_NAMES]
    return [select_joint(args.joint, joints)]


def load_robot_config(path: Path) -> tuple[dict, list[float], list[float], list[int]]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    robot = config.get("robot") if isinstance(config, dict) else None
    motors = config.get("motors") if isinstance(config, dict) else None
    if not isinstance(robot, dict) or not isinstance(motors, dict):
        raise ValueError("robot.yaml 必须包含 robot 和 motors")
    kp = [float(value) for value in robot.get("kp", [])]
    kd = [float(value) for value in robot.get("kd", [])]
    urdf2motor = [int(value) for value in robot.get("urdf2motor", [])]
    motor_count = sum(int(value) for value in motors.get("motor_num", []))
    if motor_count != EXPECTED_JOINT_COUNT:
        raise ValueError(f"motor_num 总数应为 {EXPECTED_JOINT_COUNT}，实际为 {motor_count}")
    if len(kp) != motor_count or len(kd) != motor_count:
        raise ValueError("robot.kp/kd 长度必须与电机数量一致")
    if sorted(urdf2motor) != list(range(motor_count)):
        raise ValueError("robot.urdf2motor 必须是 0-20 的排列")
    return config, kp, kd, urdf2motor


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "amplitude": args.amplitude,
        "frequency": args.frequency,
        "cycles": args.cycles,
        "rate": args.rate,
        "ramp": args.ramp,
        "max-tracking-error": args.max_tracking_error,
        "max-hold-error": args.max_hold_error,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} 必须是有限正数")
    for name, value in (
        ("settle", args.settle),
        ("recover", args.recover),
        ("limit-margin", args.limit_margin),
        ("feedback-limit-tolerance", args.feedback_limit_tolerance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} 必须是有限非负数")
    if args.amplitude > 0.10:
        raise ValueError("首版工具拒绝 amplitude > 0.10 rad；请先完成小幅测试")
    if args.frequency > 5.0:
        raise ValueError("首版工具拒绝 frequency > 5 Hz")
    if args.rate > 500.0:
        raise ValueError("首版工具拒绝 rate > 500 Hz")


def find_control_processes() -> list[tuple[int, str]]:
    found = []
    current_pid = os.getpid()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == current_pid:
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(marker in command for marker in CONTROL_PROCESS_MARKERS):
            found.append((int(process_dir.name), command))
    return sorted(found)


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def waveform(elapsed: float, args: argparse.Namespace) -> tuple[str, float]:
    steady_duration = args.cycles / args.frequency
    settle_end = args.settle
    ramp_in_end = settle_end + args.ramp
    steady_end = ramp_in_end + steady_duration
    ramp_out_end = steady_end + args.ramp
    if elapsed < settle_end:
        return "settle", 0.0
    excitation_time = elapsed - settle_end
    sine = math.sin(2.0 * math.pi * args.frequency * excitation_time)
    if elapsed < ramp_in_end:
        envelope = smoothstep((elapsed - settle_end) / args.ramp)
        return "ramp_in", args.amplitude * envelope * sine
    if elapsed < steady_end:
        return "steady", args.amplitude * sine
    if elapsed < ramp_out_end:
        envelope = 1.0 - smoothstep((elapsed - steady_end) / args.ramp)
        return "ramp_out", args.amplitude * envelope * sine
    return "recover", 0.0


def total_duration(args: argparse.Namespace) -> float:
    return args.settle + 2.0 * args.ramp + args.cycles / args.frequency + args.recover


def default_output_path(joint: JointInfo, args: argparse.Namespace, run_stamp: str) -> Path:
    frequency = str(args.frequency).replace(".", "p")
    amplitude = str(args.amplitude).replace(".", "p")
    return DEFAULT_OUTPUT_DIR / f"{run_stamp}_{joint.name}_{frequency}hz_{amplitude}rad.csv"


def validate_target_range(center: float, joint: JointInfo, args: argparse.Namespace) -> None:
    safe_lower = joint.lower + args.limit_margin
    safe_upper = joint.upper - args.limit_margin
    target_lower = center - args.amplitude
    target_upper = center + args.amplitude
    if safe_lower >= safe_upper:
        raise SafetyAbort(f"{joint.name} 限位区间不足以应用 {args.limit_margin} rad margin")
    if target_lower < safe_lower or target_upper > safe_upper:
        raise SafetyAbort(
            f"目标范围 [{target_lower:.4f}, {target_upper:.4f}] 超出安全范围 "
            f"[{safe_lower:.4f}, {safe_upper:.4f}] rad"
        )


def validate_pose_limits(
    values: list[float], joints: list[JointInfo], margin: float, label: str,
) -> None:
    for joint, value in zip(joints, values):
        lower = joint.lower + margin
        upper = joint.upper - margin
        if value < lower or value > upper:
            raise SafetyAbort(
                f"{joint.name} {label}={value:.4f} rad 超出 "
                f"[{lower:.4f}, {upper:.4f}] rad"
            )


def check_feedback(
    q_target: list[float], q: list[float], dq: list[float], feedback: list[float],
    center: list[float], joint: JointInfo, joints: list[JointInfo], args: argparse.Namespace,
) -> None:
    vectors = {"q_target": q_target, "q": q, "dq": dq, "feedback": feedback}
    for name, values in vectors.items():
        if len(values) != EXPECTED_JOINT_COUNT:
            raise SafetyAbort(f"{name} 长度应为 21，实际为 {len(values)}")
        if not all(math.isfinite(float(value)) for value in values):
            raise SafetyAbort(f"{name} 出现 NaN/Inf")
    # RobotInterface exposes joint coordinates after applying motor_sign. It also
    # converts joint targets back to raw motor coordinates when transmitting, so
    # URDF limits must be checked directly against these q_target/q values.
    validate_pose_limits(q_target, joints, 0.0, "target")
    validate_pose_limits(q, joints, -args.feedback_limit_tolerance, "feedback")
    tracking_error = abs(q_target[joint.index] - q[joint.index])
    if tracking_error > args.max_tracking_error:
        raise SafetyAbort(
            f"{joint.name} tracking error {tracking_error:.4f} rad 超过 "
            f"{args.max_tracking_error:.4f} rad"
        )
    hold_errors = [abs(q[index] - center[index]) for index in range(len(q)) if index != joint.index]
    if max(hold_errors, default=0.0) > args.max_hold_error:
        index = max(
            (idx for idx in range(len(q)) if idx != joint.index),
            key=lambda idx: abs(q[idx] - center[idx]),
        )
        raise SafetyAbort(
            f"joint[{index}] hold error {abs(q[index] - center[index]):.4f} rad 超过 "
            f"{args.max_hold_error:.4f} rad"
        )


def sine_fit(samples: list[Sample], frequency: float) -> dict[str, float]:
    steady = [sample for sample in samples if sample.phase == "steady"]
    if len(steady) < 10:
        return {}
    t = np.asarray([sample.elapsed_s for sample in steady], dtype=float)
    target = np.asarray([sample.q_target_rad for sample in steady], dtype=float)
    actual = np.asarray([sample.q_actual_rad for sample in steady], dtype=float)
    omega = 2.0 * math.pi * frequency
    design = np.column_stack((np.ones_like(t), np.sin(omega * t), np.cos(omega * t)))
    target_coef = np.linalg.lstsq(design, target, rcond=None)[0]
    actual_coef = np.linalg.lstsq(design, actual, rcond=None)[0]
    target_amp = float(np.hypot(target_coef[1], target_coef[2]))
    actual_amp = float(np.hypot(actual_coef[1], actual_coef[2]))
    target_phase = math.atan2(float(target_coef[2]), float(target_coef[1]))
    actual_phase = math.atan2(float(actual_coef[2]), float(actual_coef[1]))
    phase_lag = (target_phase - actual_phase + math.pi) % (2.0 * math.pi) - math.pi
    residual = actual - design @ actual_coef
    tracking = target - actual
    return {
        "steady_samples": len(steady),
        "target_amplitude_rad": target_amp,
        "actual_amplitude_rad": actual_amp,
        "gain": actual_amp / target_amp if target_amp > 1e-9 else float("nan"),
        "phase_lag_deg": math.degrees(phase_lag),
        "phase_delay_ms": phase_lag / omega * 1000.0,
        "tracking_rmse_rad": float(np.sqrt(np.mean(tracking * tracking))),
        "fit_residual_rmse_rad": float(np.sqrt(np.mean(residual * residual))),
        "max_abs_tracking_error_rad": float(np.max(np.abs(tracking))),
    }


def write_outputs(
    output_path: Path, samples: list[Sample], metadata: dict, metrics: dict[str, float],
    joint_names: list[str],
) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有日志: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vector_fields = {
        "q_target_all_rad", "q_actual_all_rad", "dq_actual_all_rad_s", "feedback_field_all"
    }
    scalar_fields = [name for name in asdict(samples[0]) if name not in vector_fields]
    joint_fields = []
    for joint_name in joint_names:
        joint_fields.extend(
            (
                f"{joint_name}.q_target_rad",
                f"{joint_name}.q_actual_rad",
                f"{joint_name}.dq_actual_rad_s",
                f"{joint_name}.feedback_field",
            )
        )
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields + joint_fields)
        writer.writeheader()
        for sample in samples:
            raw = asdict(sample)
            row = {name: raw[name] for name in scalar_fields}
            for index, joint_name in enumerate(joint_names):
                row[f"{joint_name}.q_target_rad"] = sample.q_target_all_rad[index]
                row[f"{joint_name}.q_actual_rad"] = sample.q_actual_all_rad[index]
                row[f"{joint_name}.dq_actual_rad_s"] = sample.dq_actual_all_rad_s[index]
                row[f"{joint_name}.feedback_field"] = sample.feedback_field_all[index]
            writer.writerow(row)
    report = {**metadata, "metrics": metrics, "csv": str(output_path)}
    metadata_path = output_path.with_suffix(".metadata.yaml")
    with metadata_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)


def run_test(
    robot, joint: JointInfo, args: argparse.Namespace,
    joints: list[JointInfo], samples: list[Sample], center: list[float],
) -> None:
    period = 1.0 / args.rate
    print("读取使能后的当前姿态；不会调用 reset_joints()...")
    robot.refresh_joints()
    center.extend(float(value) for value in robot.get_joint_q())
    if len(center) != EXPECTED_JOINT_COUNT or not all(math.isfinite(value) for value in center):
        raise SafetyAbort("未取得有效的 21 维起始关节位置")
    validate_pose_limits(center, joints, -args.feedback_limit_tolerance, "initial feedback")
    validate_pose_limits(center, joints, 0.0, "initial hold target")
    validate_target_range(center[joint.index], joint, args)
    print(
        f"起始 {joint.name}={center[joint.index]:+.5f} rad，"
        f"目标范围=[{center[joint.index] - args.amplitude:+.5f}, "
        f"{center[joint.index] + args.amplitude:+.5f}] rad"
    )

    start = time.perf_counter()
    previous_start = start
    next_tick = start
    duration = total_duration(args)
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start
        if elapsed >= duration:
            break
        phase, offset = waveform(elapsed, args)
        target = center.copy()
        target[joint.index] += offset
        robot.apply_action(target)
        command_done = time.perf_counter()
        q = [float(value) for value in robot.get_joint_q()]
        dq = [float(value) for value in robot.get_joint_vel()]
        feedback = [float(value) for value in robot.get_joint_tau()]
        quat = [float(value) for value in robot.get_quat()]
        ang_vel = [float(value) for value in robot.get_ang_vel()]
        sample_done = time.perf_counter()
        if len(quat) != 4 or len(ang_vel) != 3:
            raise SafetyAbort("IMU feedback 维度错误")
        check_feedback(target, q, dq, feedback, center, joint, joints, args)
        samples.append(
            Sample(
                elapsed_s=elapsed,
                scheduled_s=next_tick - start,
                command_done_s=command_done - start,
                sample_done_s=sample_done - start,
                loop_period_s=loop_start - previous_start,
                phase=phase,
                target_offset_rad=offset,
                q_target_rad=target[joint.index],
                q_actual_rad=q[joint.index],
                dq_actual_rad_s=dq[joint.index],
                feedback_field=feedback[joint.index],
                imu_qw=quat[0], imu_qx=quat[1], imu_qy=quat[2], imu_qz=quat[3],
                imu_wx_rad_s=ang_vel[0], imu_wy_rad_s=ang_vel[1], imu_wz_rad_s=ang_vel[2],
                q_target_all_rad=target,
                q_actual_all_rad=q,
                dq_actual_all_rad_s=dq,
                feedback_field_all=feedback,
            )
        )
        previous_start = loop_start
        next_tick += period
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        elif -sleep_time > period:
            next_tick = time.perf_counter()

    # Hold the measured center briefly before the caller disables the motors.
    recover_end = time.perf_counter() + 0.5
    while time.perf_counter() < recover_end:
        robot.apply_action(center)
        time.sleep(period)


def calculate_metrics(samples: list[Sample], frequency: float) -> dict[str, float]:
    metrics = sine_fit(samples, frequency)
    loop_periods = np.asarray([sample.loop_period_s for sample in samples[1:]], dtype=float)
    if loop_periods.size:
        metrics.update(
            {
                "measured_loop_rate_hz": float(1.0 / np.mean(loop_periods)),
                "loop_period_p95_ms": float(np.percentile(loop_periods, 95) * 1000.0),
                "loop_period_max_ms": float(np.max(loop_periods) * 1000.0),
            }
        )
    return metrics


def save_joint_result(
    output_path: Path, samples: list[Sample], center: list[float], stop_reason: str,
    joint: JointInfo, joints: list[JointInfo], args: argparse.Namespace,
    kp: list[float], kd: list[float], urdf2motor: list[int],
) -> dict[str, float]:
    if not samples:
        return {}
    motor_index = urdf2motor[joint.index]
    metrics = calculate_metrics(samples, args.frequency)
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "stop_reason": stop_reason,
        "warning": (
            "feedback_field is the DM protocol 12-bit torque/current field decoded over TauMax; "
            "it is not independently calibrated torque or current"
        ),
        "timing_limit": (
            "timestamps are Python control-loop timestamps, not per-CAN-frame TX/RX hardware timestamps"
        ),
        "robot_config": str(args.config.expanduser().resolve()),
        "joint_order_config": str(args.joint_order_config.expanduser().resolve()),
        "urdf": str(args.urdf.expanduser().resolve()),
        "coordinate_contract": (
            "q_target and q_actual are RobotInterface joint coordinates after motor_sign; "
            "URDF limits are checked in the same coordinates"
        ),
        "joint": asdict(joint),
        "motor_index": motor_index,
        "kp": kp[motor_index],
        "kd": kd[motor_index],
        "center_pose_rad": center,
        "test": {
            "amplitude_rad": args.amplitude,
            "frequency_hz": args.frequency,
            "cycles": args.cycles,
            "target_rate_hz": args.rate,
            "settle_s": args.settle,
            "ramp_s": args.ramp,
            "recover_s": args.recover,
            "limit_margin_rad": args.limit_margin,
            "feedback_limit_tolerance_rad": args.feedback_limit_tolerance,
            "max_tracking_error_rad": args.max_tracking_error,
            "max_hold_error_rad": args.max_hold_error,
        },
    }
    try:
        write_outputs(output_path, samples, metadata, metrics, [item.name for item in joints])
    except (OSError, ValueError) as error:
        print(f"写日志失败: {error}", file=sys.stderr)
        raise RuntimeError(f"写日志失败: {error}") from error

    print(f"日志: {output_path.expanduser().resolve()}")
    print(f"元数据/指标: {output_path.expanduser().resolve().with_suffix('.metadata.yaml')}")
    if metrics:
        print(
            "结果: "
            f"gain={metrics.get('gain', float('nan')):.4f}, "
            f"phase_lag={metrics.get('phase_lag_deg', float('nan')):.2f} deg, "
            f"phase_delay={metrics.get('phase_delay_ms', float('nan')):.2f} ms, "
            f"tracking_RMSE={metrics.get('tracking_rmse_rad', float('nan')):.5f} rad, "
            f"loop_rate={metrics.get('measured_loop_rate_hz', float('nan')):.1f} Hz"
        )
    return metrics


def print_test(joint: JointInfo, args: argparse.Namespace, kp, kd, urdf2motor) -> None:
    motor_index = urdf2motor[joint.index]
    print(
        f"测试: {joint.name} (RobotInterface index {joint.index}, motor index {motor_index})\n"
        f"URDF limit=[{joint.lower:+.4f}, {joint.upper:+.4f}] rad\n"
        f"信号: A={args.amplitude:.4f} rad, f={args.frequency:.3f} Hz, "
        f"steady={args.cycles:.1f} cycles, rate={args.rate:.1f} Hz\n"
        f"MIT 参数: Kp={kp[motor_index]:.3f}, Kd={kd[motor_index]:.3f}, "
        "dq_target=0, tau_ff=0"
    )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        joint_order = load_joint_order(args.joint_order_config)
        joints = load_joint_info(args.urdf, joint_order)
        _, kp, kd, urdf2motor = load_robot_config(args.config)
        if args.list_joints:
            for joint in joints:
                motor_index = urdf2motor[joint.index]
                walking = " walking" if joint.name in WALKING_JOINT_NAMES else ""
                print(
                    f"{joint.index + 1:2d}  {joint.name:32} "
                    f"limit=[{joint.lower:+.4f}, {joint.upper:+.4f}] "
                    f"Kp={kp[motor_index]:.2f} Kd={kd[motor_index]:.2f}{walking}"
                )
            return 0
        selected_joints = select_joints(args, joints)
        if args.walking_joints and args.output is not None:
            raise ValueError("--walking-joints 会生成 12 个日志，不能同时指定单个 --output")
    except (OSError, ET.ParseError, ValueError, yaml.YAMLError) as error:
        print(f"参数/配置错误: {error}", file=sys.stderr)
        return 2

    for joint in selected_joints:
        print_test(joint, args, kp, kd, urdf2motor)
    expected_duration = total_duration(args) * len(selected_joints)
    if args.dry_run:
        print(
            f"Dry run 通过；将顺序测试 {len(selected_joints)} 个关节，"
            f"总激励时间约 {expected_duration:.2f} s（不含状态刷新）。未连接硬件。"
        )
        return 0
    if not sys.stdin.isatty():
        print("错误: 实机测试必须在交互式终端运行", file=sys.stderr)
        return 2
    running = find_control_processes()
    if running:
        print("检测到其他可能控制电机的进程，拒绝继续：", file=sys.stderr)
        for pid, command in running:
            print(f"  PID {pid}: {command}", file=sys.stderr)
        return 2

    print("\n安全要求：机器人已悬吊或可靠约束，急停可用，运动范围内无人和障碍物。")
    print("每个关节单独激励；全部目标和反馈将按 motor_sign 处理后的 URDF 坐标检查限位。")
    confirmation = input("确认后输入 SUSPENDED：").strip()
    if confirmation != "SUSPENDED":
        print("未确认，退出。")
        return 1

    try:
        import robot_py
    except ImportError as error:
        print(
            "无法导入 robot_py。请先编译 roboparty_deploy 并 source install/setup.bash。\n"
            f"原始错误: {error}", file=sys.stderr,
        )
        return 2

    robot = None
    initialized = False
    last_center: list[float] = []
    overall_success = True
    run_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    try:
        robot = robot_py.RobotInterface(str(args.config.expanduser().resolve()))
        print("使能全部电机...")
        robot.init_motors()
        initialized = True
        time.sleep(0.3)
        for test_index, joint in enumerate(selected_joints, start=1):
            print(f"\n[{test_index}/{len(selected_joints)}]", end=" ")
            print_test(joint, args, kp, kd, urdf2motor)
            samples: list[Sample] = []
            center: list[float] = []
            stop_reason = "completed"
            try:
                run_test(robot, joint, args, joints, samples, center)
            except KeyboardInterrupt:
                stop_reason = "keyboard_interrupt"
                print("\n收到 Ctrl+C，停止全部测试。", file=sys.stderr)
            except SafetyAbort as error:
                stop_reason = f"safety_abort: {error}"
                print(f"\n安全停止: {error}", file=sys.stderr)
            except Exception as error:
                stop_reason = f"error: {type(error).__name__}: {error}"
                print(f"\n测试异常: {error}", file=sys.stderr)
            last_center = center
            output_path = args.output or default_output_path(joint, args, run_stamp)
            if samples:
                save_joint_result(
                    output_path, samples, center, stop_reason, joint, joints,
                    args, kp, kd, urdf2motor,
                )
            else:
                print(f"{joint.name} 没有采到数据，不写空日志。", file=sys.stderr)
            if stop_reason != "completed":
                overall_success = False
                break
    finally:
        if robot is not None and initialized:
            if last_center:
                try:
                    print("短暂保持最近一次起始姿态...")
                    end = time.perf_counter() + 0.5
                    while time.perf_counter() < end:
                        robot.apply_action(last_center)
                        time.sleep(1.0 / args.rate)
                except Exception as error:
                    print(f"保持起始姿态失败: {error}", file=sys.stderr)
            try:
                print("失能全部电机...")
                robot.deinit_motors()
            except Exception as error:
                overall_success = False
                print(f"电机失能命令失败: {error}", file=sys.stderr)
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
