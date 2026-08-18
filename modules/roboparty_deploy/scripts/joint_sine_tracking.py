#!/usr/bin/env python3
"""Run guarded pair-by-pair sine tracking tests on RPO leg motors.

The robot must be suspended or otherwise restrained. Each left/right motor pair
is created and enabled independently. The two motors hold their measured start
positions while the script excites one side at a time, then disables the pair.
All tests are written to one CSV and one metadata YAML file.
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
JOINT_PAIRS = {
    "ankle-roll": ("left_ankle_roll_joint", "right_ankle_roll_joint"),
    "ankle-pitch": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    "knee": ("left_knee_joint", "right_knee_joint"),
    "hip-yaw": ("left_leg_yaw_joint", "right_leg_yaw_joint"),
    "hip-roll": ("left_leg_roll_joint", "right_leg_roll_joint"),
    "hip-pitch": ("left_leg_pitch_joint", "right_leg_pitch_joint"),
}
CONTROL_PROCESS_MARKERS = (
    "inference_node",
    "motion_player.py",
    "motors_py_example.py",
    "set_zero.py",
    "check_motor_sign.py",
    "joint_sine_tracking.py",
)


@dataclass(frozen=True)
class MotorSpec:
    motor_index: int
    bus_index: int
    interface: str
    interface_type: str
    motor_type: str
    motor_id: int
    motor_model: int
    motor_zero_offset: float
    master_id_offset: int
    sign: int
    kp: float
    kd: float


@dataclass(frozen=True)
class JointInfo:
    index: int
    name: str
    lower: float
    upper: float
    motor: MotorSpec


@dataclass
class Sample:
    run_elapsed_s: float
    test_elapsed_s: float
    scheduled_s: float
    command_done_s: float
    sample_done_s: float
    loop_period_s: float
    pair_name: str
    side: str
    joint_name: str
    joint_index: int
    motor_index: int
    interface: str
    motor_id: int
    phase: str
    amplitude_rad: float
    frequency_hz: float
    kp: float
    kd: float
    target_offset_rad: float
    q_center_rad: float
    q_target_rad: float
    q_actual_rad: float
    dq_actual_rad_s: float
    feedback_field: float
    temperature_c: float
    error_id: int
    hold_joint_name: str
    hold_motor_index: int
    hold_q_center_rad: float
    hold_q_target_rad: float
    hold_q_actual_rad: float
    hold_dq_actual_rad_s: float
    hold_feedback_field: float
    hold_temperature_c: float
    hold_error_id: int


class SafetyAbort(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RPO 腿部电机逐对正弦测试。默认从脚向上测试；每完成一对的左右驱动后，"
            "询问是否继续下一对。全部结果写入一个 CSV。机器人必须悬吊或可靠约束。"
        )
    )
    parser.add_argument(
        "--pair", choices=tuple(JOINT_PAIRS),
        help="只测试指定的一组；不提供时自动测试全部六组",
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.12,
        help="正弦幅值，rad（默认: 0.12，约 6.88 度）",
    )
    parser.add_argument("--frequency", type=float, default=1.0, help="正弦频率，Hz（默认: 1）")
    parser.add_argument("--cycles", type=float, default=4.0, help="稳态正弦周期数（默认: 4）")
    parser.add_argument("--rate", type=float, default=250.0, help="目标控制/记录频率，Hz（默认: 250）")
    parser.add_argument("--settle", type=float, default=2.0, help="激励前保持时间，s（默认: 2）")
    parser.add_argument("--ramp", type=float, default=2.0, help="正弦包络进入/退出时间，s（默认: 2）")
    parser.add_argument("--recover", type=float, default=2.0, help="激励后保持时间，s（默认: 2）")
    parser.add_argument("--pair-pause", type=float, default=1.0, help="两组测试之间的失能等待，s（默认: 1）")
    parser.add_argument("--limit-margin", type=float, default=0.03, help="目标距 URDF 限位余量，rad")
    parser.add_argument("--feedback-limit-tolerance", type=float, default=0.01)
    parser.add_argument("--max-tracking-error", type=float, default=0.15)
    parser.add_argument("--max-hold-error", type=float, default=0.15)
    parser.add_argument("--max-velocity", type=float, default=5.0, help="反馈速度绝对值上限，rad/s")
    parser.add_argument("--max-temperature", type=float, default=75.0, help="电机温度上限，摄氏度")
    parser.add_argument("--config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--joint-order-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, help="单个汇总 CSV 路径")
    parser.add_argument("--list-pairs", action="store_true", help="列出测试顺序后退出")
    parser.add_argument("--dry-run", action="store_true", help="检查配置和测试计划，不连接硬件")
    return parser.parse_args()


def require_list(mapping: dict, key: str, count: int | None = None) -> list:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须是列表")
    if count is not None and len(value) != count:
        raise ValueError(f"配置项 {key} 应包含 {count} 项，实际为 {len(value)}")
    return value


def value_for_bus(value, bus_index: int, key: str):
    if isinstance(value, list):
        if bus_index >= len(value):
            raise ValueError(f"{key} 缺少 bus index {bus_index}")
        return value[bus_index]
    return value


def load_joint_order(path: Path) -> list[str]:
    with path.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        order = config["inference_node"]["ros__parameters"]["stand_whole_body_joint_order"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} 缺少 stand_whole_body_joint_order") from error
    if not isinstance(order, list) or len(order) != EXPECTED_JOINT_COUNT:
        raise ValueError("stand_whole_body_joint_order 必须包含 21 个关节")
    names = [str(name) for name in order]
    if len(set(names)) != EXPECTED_JOINT_COUNT:
        raise ValueError("stand_whole_body_joint_order 包含重复关节")
    return names


def load_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(path.expanduser()).getroot()
    limits = {}
    for joint in root.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"关节 {joint.get('name')} 缺少 lower/upper limit")
        limits[str(joint.get("name"))] = (
            float(limit.get("lower")), float(limit.get("upper")),
        )
    return limits


def load_configuration(args: argparse.Namespace) -> tuple[list[JointInfo], dict]:
    with args.config.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    motors = config.get("motors") if isinstance(config, dict) else None
    robot = config.get("robot") if isinstance(config, dict) else None
    if not isinstance(motors, dict) or not isinstance(robot, dict):
        raise ValueError("robot.yaml 必须包含 motors 和 robot")

    motor_num = [int(value) for value in require_list(motors, "motor_num")]
    motor_count = sum(motor_num)
    if motor_count != EXPECTED_JOINT_COUNT:
        raise ValueError(f"电机数量应为 21，实际为 {motor_count}")
    motor_ids = [int(value) for value in require_list(motors, "motor_id", motor_count)]
    motor_models = [int(value) for value in require_list(motors, "motor_model", motor_count)]
    zero_offsets = [float(value) for value in require_list(motors, "motor_zero_offset", motor_count)]
    interfaces = require_list(motors, "motor_interface", len(motor_num))
    interface_types = require_list(motors, "motor_interface_type", len(motor_num))
    motor_types = require_list(motors, "motor_type", len(motor_num))
    master_id_offset = int(motors.get("master_id_offset", 0))

    urdf2motor = [int(value) for value in require_list(robot, "urdf2motor", motor_count)]
    signs = [int(value) for value in require_list(robot, "motor_sign", motor_count)]
    kp = [float(value) for value in require_list(robot, "kp", motor_count)]
    kd = [float(value) for value in require_list(robot, "kd", motor_count)]
    if sorted(urdf2motor) != list(range(motor_count)):
        raise ValueError("robot.urdf2motor 必须是 0-20 的排列")
    if any(sign not in {-1, 1} for sign in signs):
        raise ValueError("robot.motor_sign 只能包含 -1 或 +1")
    if robot.get("close_chain_motor_idx"):
        raise ValueError(
            "此脚本按原始并联驱动通道直接测试，要求 close_chain_motor_idx 为空"
        )

    motor_bus = []
    for bus_index, count in enumerate(motor_num):
        motor_bus.extend([bus_index] * count)
    order = load_joint_order(args.joint_order_config)
    limits = load_joint_limits(args.urdf)
    if set(order) != set(limits):
        raise ValueError("部署关节顺序与 URDF 可动关节不一致")

    joints = []
    for joint_index, name in enumerate(order):
        motor_index = urdf2motor[joint_index]
        bus_index = motor_bus[motor_index]
        spec = MotorSpec(
            motor_index=motor_index,
            bus_index=bus_index,
            interface=str(value_for_bus(interfaces, bus_index, "motor_interface")),
            interface_type=str(value_for_bus(interface_types, bus_index, "motor_interface_type")),
            motor_type=str(value_for_bus(motor_types, bus_index, "motor_type")),
            motor_id=motor_ids[motor_index],
            motor_model=motor_models[motor_index],
            motor_zero_offset=zero_offsets[motor_index],
            master_id_offset=master_id_offset,
            sign=signs[motor_index],
            kp=kp[motor_index],
            kd=kd[motor_index],
        )
        joints.append(JointInfo(joint_index, name, *limits[name], spec))
    return joints, config


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "amplitude": args.amplitude,
        "frequency": args.frequency,
        "cycles": args.cycles,
        "rate": args.rate,
        "ramp": args.ramp,
        "max-tracking-error": args.max_tracking_error,
        "max-hold-error": args.max_hold_error,
        "max-velocity": args.max_velocity,
        "max-temperature": args.max_temperature,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} 必须是有限正数")
    for name, value in (
        ("settle", args.settle),
        ("recover", args.recover),
        ("pair-pause", args.pair_pause),
        ("limit-margin", args.limit_margin),
        ("feedback-limit-tolerance", args.feedback_limit_tolerance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} 必须是有限非负数")
    if args.amplitude > 0.12:
        raise ValueError("拒绝 amplitude > 0.12 rad")
    if args.frequency > 5.0 or args.rate > 500.0:
        raise ValueError("拒绝 frequency > 5 Hz 或 rate > 500 Hz")


def selected_pairs(args: argparse.Namespace, joints: list[JointInfo]):
    by_name = {joint.name: joint for joint in joints}
    names = [args.pair] if args.pair else list(JOINT_PAIRS)
    return [
        (pair_name, tuple(by_name[name] for name in JOINT_PAIRS[pair_name]))
        for pair_name in names
    ]


def print_plan(pairs, args: argparse.Namespace) -> None:
    print("测试计划（每组只使能以下两个物理电机，先左后右）：")
    for number, (pair_name, pair) in enumerate(pairs, start=1):
        descriptions = []
        for joint in pair:
            descriptions.append(
                f"{joint.name}={joint.motor.interface}/ID{joint.motor.motor_id}"
            )
        print(f"  {number}. {pair_name:12} " + " | ".join(descriptions))
    per_joint = total_duration(args)
    total = per_joint * 2 * len(pairs) + args.pair_pause * max(len(pairs) - 1, 0)
    print(
        f"信号: A={args.amplitude:.4f} rad, f={args.frequency:.3f} Hz, "
        f"cycles={args.cycles:.1f}, rate={args.rate:.1f} Hz"
    )
    print(f"预计总时间约 {total:.1f} s（不含硬件状态刷新）。")


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
        return "ramp_in", args.amplitude * smoothstep((elapsed - settle_end) / args.ramp) * sine
    if elapsed < steady_end:
        return "steady", args.amplitude * sine
    if elapsed < ramp_out_end:
        envelope = 1.0 - smoothstep((elapsed - steady_end) / args.ramp)
        return "ramp_out", args.amplitude * envelope * sine
    return "recover", 0.0


def total_duration(args: argparse.Namespace) -> float:
    return args.settle + 2.0 * args.ramp + args.cycles / args.frequency + args.recover


def create_motor(motors_py, joint: JointInfo):
    spec = joint.motor
    return motors_py.MotorDriver.create_motor(
        motor_id=spec.motor_id,
        interface_type=spec.interface_type,
        interface=spec.interface,
        motor_type=spec.motor_type,
        motor_model=spec.motor_model,
        master_id_offset=spec.master_id_offset,
        motor_zero_offset=spec.motor_zero_offset,
    )


def joint_position(motor, joint: JointInfo) -> float:
    return float(motor.get_motor_pos()) * joint.motor.sign


def joint_velocity(motor, joint: JointInfo) -> float:
    return float(motor.get_motor_spd()) * joint.motor.sign


def joint_feedback(motor, joint: JointInfo) -> float:
    return float(motor.get_motor_current()) * joint.motor.sign


def send_joint_target(motor, joint: JointInfo, target: float) -> None:
    motor.motor_mit_cmd(
        target * joint.motor.sign,
        0.0,
        joint.motor.kp,
        joint.motor.kd,
        0.0,
    )


def validate_target(center: float, joint: JointInfo, args: argparse.Namespace) -> None:
    lower = joint.lower + args.limit_margin
    upper = joint.upper - args.limit_margin
    if center - args.amplitude < lower or center + args.amplitude > upper:
        raise SafetyAbort(
            f"{joint.name} 目标范围 [{center - args.amplitude:.4f}, "
            f"{center + args.amplitude:.4f}] 超出 [{lower:.4f}, {upper:.4f}] rad"
        )


def validate_feedback(
    motor, joint: JointInfo, target: float, actual: float, velocity: float,
    center: float, args: argparse.Namespace, is_hold: bool,
) -> None:
    values = (target, actual, velocity, float(motor.get_motor_current()))
    if not all(math.isfinite(value) for value in values):
        raise SafetyAbort(f"{joint.name} 反馈出现 NaN/Inf")
    if actual < joint.lower - args.feedback_limit_tolerance or actual > joint.upper + args.feedback_limit_tolerance:
        raise SafetyAbort(f"{joint.name} 反馈 {actual:.4f} rad 超出 URDF 限位")
    allowed_error = args.max_hold_error if is_hold else args.max_tracking_error
    reference = center if is_hold else target
    if abs(reference - actual) > allowed_error:
        label = "hold" if is_hold else "tracking"
        raise SafetyAbort(f"{joint.name} {label} error {abs(reference - actual):.4f} rad")
    if abs(velocity) > args.max_velocity:
        raise SafetyAbort(f"{joint.name} 速度 {velocity:.3f} rad/s 超限")
    temperature = float(motor.get_motor_temperature())
    if math.isfinite(temperature) and temperature > args.max_temperature:
        raise SafetyAbort(f"{joint.name} 温度 {temperature:.1f} C 超限")
    error_id = int(motor.get_error_id())
    if error_id not in {0, 1}:
        raise SafetyAbort(f"{joint.name} 电机错误码 0x{error_id:02x}")
    if int(motor.get_response_count()) > 25:
        raise SafetyAbort(f"{joint.name} 电机疑似离线")


def enable_pair(motors, pair, args: argparse.Namespace) -> list[float]:
    for motor in motors:
        motor.refresh_motor_status()
    time.sleep(0.1)
    centers = [joint_position(motor, joint) for motor, joint in zip(motors, pair)]
    for center, joint in zip(centers, pair):
        if not math.isfinite(center):
            raise SafetyAbort(f"{joint.name} 未取得有效起始位置")
        validate_target(center, joint, args)
    initialized = []
    try:
        for motor, joint, center in zip(motors, pair, centers):
            motor.init_motor()
            initialized.append(motor)
            send_joint_target(motor, joint, center)
        time.sleep(0.3)
        return centers
    except BaseException:
        for motor in initialized:
            try:
                motor.deinit_motor()
            except Exception:
                pass
        raise


def disable_pair(motors) -> None:
    first_error = None
    for _ in range(3):
        for motor in motors:
            try:
                motor.deinit_motor()
            except Exception as error:
                first_error = first_error or error
        time.sleep(0.1)
    if first_error is not None:
        raise first_error


def run_joint_test(
    motors, pair_name: str, pair, centers: list[float], active_index: int,
    args: argparse.Namespace, samples: list[Sample], run_start: float,
) -> dict:
    active_joint = pair[active_index]
    hold_index = 1 - active_index
    hold_joint = pair[hold_index]
    period = 1.0 / args.rate
    print(
        f"  激励 {active_joint.name}: center={centers[active_index]:+.5f} rad, "
        f"range=[{centers[active_index] - args.amplitude:+.5f}, "
        f"{centers[active_index] + args.amplitude:+.5f}]"
    )
    local_samples = []
    start = time.perf_counter()
    previous_start = start
    next_tick = start
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start
        if elapsed >= total_duration(args):
            break
        phase, offset = waveform(elapsed, args)
        targets = centers.copy()
        targets[active_index] += offset
        send_joint_target(motors[active_index], active_joint, targets[active_index])
        send_joint_target(motors[hold_index], hold_joint, targets[hold_index])
        command_done = time.perf_counter()

        actual = [joint_position(motor, joint) for motor, joint in zip(motors, pair)]
        velocity = [joint_velocity(motor, joint) for motor, joint in zip(motors, pair)]
        feedback = [joint_feedback(motor, joint) for motor, joint in zip(motors, pair)]
        temperature = [float(motor.get_motor_temperature()) for motor in motors]
        error_id = [int(motor.get_error_id()) for motor in motors]
        sample_done = time.perf_counter()
        validate_feedback(
            motors[active_index], active_joint, targets[active_index], actual[active_index],
            velocity[active_index], centers[active_index], args, False,
        )
        validate_feedback(
            motors[hold_index], hold_joint, targets[hold_index], actual[hold_index],
            velocity[hold_index], centers[hold_index], args, True,
        )

        sample = Sample(
            run_elapsed_s=loop_start - run_start,
            test_elapsed_s=elapsed,
            scheduled_s=next_tick - start,
            command_done_s=command_done - start,
            sample_done_s=sample_done - start,
            loop_period_s=loop_start - previous_start,
            pair_name=pair_name,
            side="left" if active_index == 0 else "right",
            joint_name=active_joint.name,
            joint_index=active_joint.index,
            motor_index=active_joint.motor.motor_index,
            interface=active_joint.motor.interface,
            motor_id=active_joint.motor.motor_id,
            phase=phase,
            amplitude_rad=args.amplitude,
            frequency_hz=args.frequency,
            kp=active_joint.motor.kp,
            kd=active_joint.motor.kd,
            target_offset_rad=offset,
            q_center_rad=centers[active_index],
            q_target_rad=targets[active_index],
            q_actual_rad=actual[active_index],
            dq_actual_rad_s=velocity[active_index],
            feedback_field=feedback[active_index],
            temperature_c=temperature[active_index],
            error_id=error_id[active_index],
            hold_joint_name=hold_joint.name,
            hold_motor_index=hold_joint.motor.motor_index,
            hold_q_center_rad=centers[hold_index],
            hold_q_target_rad=targets[hold_index],
            hold_q_actual_rad=actual[hold_index],
            hold_dq_actual_rad_s=velocity[hold_index],
            hold_feedback_field=feedback[hold_index],
            hold_temperature_c=temperature[hold_index],
            hold_error_id=error_id[hold_index],
        )
        samples.append(sample)
        local_samples.append(sample)
        previous_start = loop_start
        next_tick += period
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        elif -sleep_time > period:
            next_tick = time.perf_counter()

    for _ in range(max(1, int(0.5 * args.rate))):
        for motor, joint, center in zip(motors, pair, centers):
            send_joint_target(motor, joint, center)
        time.sleep(period)
    return calculate_metrics(local_samples, args.frequency)


def calculate_metrics(samples: list[Sample], frequency: float) -> dict[str, float]:
    steady = [sample for sample in samples if sample.phase == "steady"]
    metrics = {}
    if len(steady) >= 10:
        t = np.asarray([sample.test_elapsed_s for sample in steady], dtype=float)
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
        tracking = target - actual
        residual = actual - design @ actual_coef
        metrics.update({
            "steady_samples": len(steady),
            "target_amplitude_rad": target_amp,
            "actual_amplitude_rad": actual_amp,
            "gain": actual_amp / target_amp if target_amp > 1e-9 else float("nan"),
            "phase_lag_deg": math.degrees(phase_lag),
            "phase_delay_ms": phase_lag / omega * 1000.0,
            "tracking_rmse_rad": float(np.sqrt(np.mean(tracking * tracking))),
            "fit_residual_rmse_rad": float(np.sqrt(np.mean(residual * residual))),
            "max_abs_tracking_error_rad": float(np.max(np.abs(tracking))),
        })
    periods = np.asarray([sample.loop_period_s for sample in samples[1:]], dtype=float)
    if periods.size:
        metrics.update({
            "measured_loop_rate_hz": float(1.0 / np.mean(periods)),
            "loop_period_p95_ms": float(np.percentile(periods, 95) * 1000.0),
            "loop_period_max_ms": float(np.max(periods) * 1000.0),
        })
    return metrics


def default_output_path(args: argparse.Namespace, stamp: str) -> Path:
    frequency = str(args.frequency).replace(".", "p")
    amplitude = str(args.amplitude).replace(".", "p")
    scope = args.pair or "all_leg_motors"
    return DEFAULT_OUTPUT_DIR / f"{stamp}_{scope}_{frequency}hz_{amplitude}rad.csv"


def write_outputs(output_path: Path, samples: list[Sample], report: dict) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有日志: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(samples[0])))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    metadata_path = output_path.with_suffix(".metadata.yaml")
    with metadata_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(
            {**report, "csv": str(output_path)}, stream,
            allow_unicode=True, sort_keys=False,
        )
    print(f"\n汇总 CSV: {output_path}")
    print(f"汇总指标: {metadata_path}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        joints, _ = load_configuration(args)
        pairs = selected_pairs(args, joints)
    except (OSError, ET.ParseError, ValueError, yaml.YAMLError) as error:
        print(f"参数/配置错误: {error}", file=sys.stderr)
        return 2

    print_plan(pairs, args)
    if args.list_pairs or args.dry_run:
        if args.dry_run:
            print("Dry run 通过；未导入 motors_py，未连接硬件。")
        return 0
    if not sys.stdin.isatty():
        print("错误: 实机测试必须在交互式终端运行", file=sys.stderr)
        return 2
    running = find_control_processes()
    if running:
        print("检测到其他电机控制进程，拒绝继续：", file=sys.stderr)
        for pid, command in running:
            print(f"  PID {pid}: {command}", file=sys.stderr)
        return 2

    print("\n安全要求：机器人已悬吊或可靠约束，硬件急停可用，运动范围内无人和障碍物。")
    print("脚本从脚向上测试；每组结束后先失能该组，再询问是否进入下一组。")
    if input("确认后输入 SUSPENDED：").strip() != "SUSPENDED":
        print("未确认，退出。")
        return 1
    try:
        import motors_py
    except ImportError as error:
        print(
            "无法导入 motors_py；请 source ROS2 和 install/setup.bash。\n"
            f"原始错误: {error}", file=sys.stderr,
        )
        return 2

    samples: list[Sample] = []
    results = []
    run_start = time.perf_counter()
    run_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stop_reason = "completed"
    for pair_number, (pair_name, pair) in enumerate(pairs, start=1):
        motors = []
        pair_initialized = False
        try:
            print(f"\n[{pair_number}/{len(pairs)}] {pair_name}: 创建并使能这一对电机")
            motors = [create_motor(motors_py, joint) for joint in pair]
            centers = enable_pair(motors, pair, args)
            pair_initialized = True
            for active_index in (0, 1):
                metrics = run_joint_test(
                    motors, pair_name, pair, centers, active_index,
                    args, samples, run_start,
                )
                result = {
                    "pair_name": pair_name,
                    "side": "left" if active_index == 0 else "right",
                    "joint": asdict(pair[active_index]),
                    "hold_joint": pair[1 - active_index].name,
                    "center_rad": centers[active_index],
                    "hold_center_rad": centers[1 - active_index],
                    "metrics": metrics,
                }
                results.append(result)
                print(
                    f"    gain={metrics.get('gain', float('nan')):.4f}, "
                    f"delay={metrics.get('phase_delay_ms', float('nan')):.2f} ms, "
                    f"RMSE={metrics.get('tracking_rmse_rad', float('nan')):.5f} rad"
                )
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
            print("\n收到 Ctrl+C，停止后续测试。", file=sys.stderr)
        except Exception as error:
            stop_reason = f"error: {type(error).__name__}: {error}"
            print(f"\n安全停止: {error}", file=sys.stderr)
        finally:
            if pair_initialized:
                try:
                    print(f"  失能 {pair_name} 电机")
                    disable_pair(motors)
                except Exception as error:
                    stop_reason = f"deinit_error: {type(error).__name__}: {error}"
                    print(f"失能失败: {error}", file=sys.stderr)
        if stop_reason != "completed":
            break
        if pair_number < len(pairs):
            next_pair_name = pairs[pair_number][0]
            answer = input(
                f"\n{pair_name} 左右两侧已完成。是否继续下一组 {next_pair_name}？[y/N]："
            ).strip().lower()
            if answer not in {"y", "yes"}:
                stop_reason = "user_stopped_between_pairs"
                print("用户选择停止；将保存已经完成的测试数据。")
                break
            if args.pair_pause > 0.0:
                time.sleep(args.pair_pause)

    if not samples:
        print("没有采到数据，不生成空日志。", file=sys.stderr)
        return 1
    output_path = args.output or default_output_path(args, run_stamp)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "stop_reason": stop_reason,
        "warning": (
            "feedback_field is the DM protocol 12-bit torque/current field decoded over TauMax; "
            "it is not calibrated joint torque"
        ),
        "coordinate_contract": (
            "q_target/q_actual are raw parallel-drive motor positions after motor_sign; "
            "ankle labels do not imply virtual foot pitch/roll decoupling"
        ),
        "robot_config": str(args.config.expanduser().resolve()),
        "joint_order_config": str(args.joint_order_config.expanduser().resolve()),
        "urdf": str(args.urdf.expanduser().resolve()),
        "test": {
            "amplitude_rad": args.amplitude,
            "frequency_hz": args.frequency,
            "cycles": args.cycles,
            "target_rate_hz": args.rate,
            "settle_s": args.settle,
            "ramp_s": args.ramp,
            "recover_s": args.recover,
        },
        "results": results,
    }
    try:
        write_outputs(output_path, samples, report)
    except (OSError, ValueError) as error:
        print(f"写日志失败: {error}", file=sys.stderr)
        return 1
    return 0 if stop_reason in {"completed", "user_stopped_between_pairs"} else 1


if __name__ == "__main__":
    sys.exit(main())
