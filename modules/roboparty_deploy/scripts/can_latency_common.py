#!/usr/bin/env python3
"""Shared configuration, safety, execution, and logging for CAN latency tests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
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
DEFAULT_OUTPUT_DIR = DEPLOY_ROOT / "experiment_logs/can_latency"
EXPECTED_JOINT_COUNT = 21
LEG_JOINT_NAMES = (
    "left_leg_pitch_joint", "left_leg_roll_joint", "left_leg_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_leg_pitch_joint", "right_leg_roll_joint", "right_leg_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
CONTROL_PROCESS_MARKERS = (
    "inference_node", "motion_player.py", "motors_py_example.py", "set_zero.py",
    "check_motor_sign.py", "joint_sine_tracking.py", "can_single_vs_bus.py",
    "can_send_order.py", "can_tx_rx_timestamps.py",
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


class SafetyAbort(RuntimeError):
    pass


def add_common_arguments(parser: argparse.ArgumentParser, *, sine: bool) -> None:
    parser.add_argument(
        "--joint", action="append", default=[], metavar="NAME",
        help="只测指定腿部关节，可重复；默认测试左右腿全部 12 个关节",
    )
    parser.add_argument("--rate", type=float, default=250.0, help="控制频率 Hz（默认 250）")
    if sine:
        parser.add_argument("--amplitude", type=float, default=0.08, help="正弦幅值 rad（默认 0.08）")
        parser.add_argument("--frequency", type=float, default=1.0, help="正弦频率 Hz（默认 1）")
        parser.add_argument("--cycles", type=float, default=4.0, help="稳态周期数（默认 4）")
        parser.add_argument("--settle", type=float, default=1.0, help="激励前保持时间 s")
        parser.add_argument("--ramp", type=float, default=1.0, help="正弦淡入/淡出时间 s")
        parser.add_argument("--recover", type=float, default=1.0, help="激励后保持时间 s")
    parser.add_argument("--case-pause", type=float, default=0.5, help="case 间失能等待 s")
    parser.add_argument("--limit-margin", type=float, default=0.03, help="目标距 URDF 限位余量 rad")
    parser.add_argument("--feedback-limit-tolerance", type=float, default=0.01)
    parser.add_argument("--max-tracking-error", type=float, default=0.18)
    parser.add_argument("--max-hold-error", type=float, default=0.15)
    parser.add_argument("--max-velocity", type=float, default=5.0)
    parser.add_argument("--max-temperature", type=float, default=75.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--joint-order-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, help="原始 CSV 输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和实验计划")


def validate_common_args(args: argparse.Namespace, *, sine: bool) -> None:
    positive = {
        "rate": args.rate,
        "max-tracking-error": args.max_tracking_error,
        "max-hold-error": args.max_hold_error,
        "max-velocity": args.max_velocity,
        "max-temperature": args.max_temperature,
    }
    if sine:
        positive.update({
            "amplitude": args.amplitude, "frequency": args.frequency,
            "cycles": args.cycles, "ramp": args.ramp,
        })
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} 必须是有限正数")
    nonnegative = [
        ("case-pause", args.case_pause), ("limit-margin", args.limit_margin),
        ("feedback-limit-tolerance", args.feedback_limit_tolerance),
    ]
    if sine:
        nonnegative.extend((("settle", args.settle), ("recover", args.recover)))
    for name, value in nonnegative:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} 必须是有限非负数")
    if args.rate > 500.0:
        raise ValueError("拒绝 --rate > 500 Hz")
    if sine and (args.amplitude > 0.12 or args.frequency > 5.0):
        raise ValueError("拒绝 --amplitude > 0.12 rad 或 --frequency > 5 Hz")


def _require_list(mapping: dict, key: str, count: int | None = None) -> list:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须是列表")
    if count is not None and len(value) != count:
        raise ValueError(f"配置项 {key} 应包含 {count} 项，实际 {len(value)}")
    return value


def _bus_value(value, bus_index: int, key: str):
    if isinstance(value, list):
        if bus_index >= len(value):
            raise ValueError(f"{key} 缺少 bus index {bus_index}")
        return value[bus_index]
    return value


def _load_joint_order(path: Path) -> list[str]:
    with path.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        order = config["inference_node"]["ros__parameters"]["stand_whole_body_joint_order"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} 缺少 stand_whole_body_joint_order") from error
    if not isinstance(order, list) or len(order) != EXPECTED_JOINT_COUNT:
        raise ValueError("stand_whole_body_joint_order 必须包含 21 个关节")
    return [str(name) for name in order]


def _load_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    limits = {}
    for joint in ET.parse(path.expanduser()).getroot().findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"关节 {joint.get('name')} 缺少 lower/upper limit")
        limits[str(joint.get("name"))] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def load_configuration(args: argparse.Namespace) -> tuple[list[JointInfo], dict]:
    with args.config.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    motors = config.get("motors") if isinstance(config, dict) else None
    robot = config.get("robot") if isinstance(config, dict) else None
    if not isinstance(motors, dict) or not isinstance(robot, dict):
        raise ValueError("robot.yaml 必须包含 motors 和 robot")

    motor_num = [int(v) for v in _require_list(motors, "motor_num")]
    count = sum(motor_num)
    if count != EXPECTED_JOINT_COUNT:
        raise ValueError(f"电机数量应为 21，实际 {count}")
    ids = [int(v) for v in _require_list(motors, "motor_id", count)]
    models = [int(v) for v in _require_list(motors, "motor_model", count)]
    offsets = [float(v) for v in _require_list(motors, "motor_zero_offset", count)]
    interfaces = _require_list(motors, "motor_interface", len(motor_num))
    interface_types = _require_list(motors, "motor_interface_type", len(motor_num))
    motor_types = _require_list(motors, "motor_type", len(motor_num))
    urdf2motor = [int(v) for v in _require_list(robot, "urdf2motor", count)]
    signs = [int(v) for v in _require_list(robot, "motor_sign", count)]
    kp = [float(v) for v in _require_list(robot, "kp", count)]
    kd = [float(v) for v in _require_list(robot, "kd", count)]
    if sorted(urdf2motor) != list(range(count)):
        raise ValueError("robot.urdf2motor 必须是 0-20 的排列")
    if any(sign not in {-1, 1} for sign in signs):
        raise ValueError("robot.motor_sign 只能是 -1/+1")
    if robot.get("close_chain_motor_idx"):
        raise ValueError("本测试直接控制物理电机，要求 close_chain_motor_idx 为空")
    bus_by_motor = []
    for bus_index, bus_count in enumerate(motor_num):
        bus_by_motor.extend([bus_index] * bus_count)

    order = _load_joint_order(args.joint_order_config)
    limits = _load_joint_limits(args.urdf)
    if set(order) != set(limits):
        raise ValueError("部署关节顺序与 URDF 可动关节不一致")
    result = []
    for joint_index, name in enumerate(order):
        motor_index = urdf2motor[joint_index]
        bus_index = bus_by_motor[motor_index]
        spec = MotorSpec(
            motor_index, bus_index,
            str(_bus_value(interfaces, bus_index, "motor_interface")),
            str(_bus_value(interface_types, bus_index, "motor_interface_type")),
            str(_bus_value(motor_types, bus_index, "motor_type")),
            ids[motor_index], models[motor_index], offsets[motor_index],
            int(motors.get("master_id_offset", 0)), signs[motor_index],
            kp[motor_index], kd[motor_index],
        )
        result.append(JointInfo(joint_index, name, *limits[name], spec))
    return result, config


def selected_leg_joints(joints: list[JointInfo], names: list[str]) -> list[JointInfo]:
    by_name = {joint.name: joint for joint in joints if joint.name in LEG_JOINT_NAMES}
    selected_names = names or list(LEG_JOINT_NAMES)
    unknown = [name for name in selected_names if name not in by_name]
    if unknown:
        raise ValueError("未知或非腿部关节: " + ", ".join(unknown))
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("--joint 不允许重复")
    return [by_name[name] for name in selected_names]


def leg_buses(joints: list[JointInfo]) -> dict[str, list[JointInfo]]:
    buses: dict[str, list[JointInfo]] = {}
    for joint in joints:
        if joint.name in LEG_JOINT_NAMES:
            buses.setdefault(joint.motor.interface, []).append(joint)
    for interface, members in buses.items():
        members.sort(key=lambda joint: joint.motor.motor_id)
        if len(members) != 6 or [j.motor.motor_id for j in members] != list(range(1, 7)):
            raise ValueError(f"{interface} 不是预期的 ID1-ID6 六电机腿部总线")
        if any(j.motor.interface_type != "can" for j in members):
            raise ValueError(f"{interface} 不是经典 CAN，当前实验不支持 CAN-FD")
    if len(buses) != 2:
        raise ValueError(f"预期两条腿部 CAN 总线，实际 {len(buses)}")
    return dict(sorted(buses.items()))


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


def confirm_suspended() -> None:
    if not os.isatty(0):
        raise RuntimeError("实机测试必须在交互式终端运行")
    running = find_control_processes()
    if running:
        details = "\n".join(f"  PID {pid}: {command}" for pid, command in running)
        raise RuntimeError(f"检测到其他电机控制进程：\n{details}")
    print("\n安全要求：机器人已悬吊并可靠约束，急停可用，运动范围内无人和障碍物。")
    print("脚本确认一次后自动完成所有 case；任一安全检查失败会立即失能。")
    if input("确认后输入 SUSPENDED：").strip() != "SUSPENDED":
        raise RuntimeError("未确认悬吊状态")


def import_motors_py():
    try:
        import motors_py
    except ImportError as error:
        raise RuntimeError(
            "无法导入 motors_py；请先 source ROS2 和本工作区 install/setup.bash"
        ) from error
    return motors_py


def create_motor(motors_py, joint: JointInfo):
    spec = joint.motor
    return motors_py.MotorDriver.create_motor(
        motor_id=spec.motor_id, interface_type=spec.interface_type,
        interface=spec.interface, motor_type=spec.motor_type,
        motor_model=spec.motor_model, master_id_offset=spec.master_id_offset,
        motor_zero_offset=spec.motor_zero_offset,
    )


def joint_position(motor, joint: JointInfo) -> float:
    return float(motor.get_motor_pos()) * joint.motor.sign


def joint_velocity(motor, joint: JointInfo) -> float:
    return float(motor.get_motor_spd()) * joint.motor.sign


def send_target(motor, joint: JointInfo, target: float) -> None:
    motor.motor_mit_cmd(target * joint.motor.sign, 0.0, joint.motor.kp, joint.motor.kd, 0.0)


def _validate_target(center: float, amplitude: float, joint: JointInfo, args: argparse.Namespace) -> None:
    safe_lower = joint.lower + args.limit_margin
    safe_upper = joint.upper - args.limit_margin
    if center - amplitude < safe_lower or center + amplitude > safe_upper:
        raise SafetyAbort(
            f"{joint.name} 目标 [{center-amplitude:.4f}, {center+amplitude:.4f}] "
            f"超出留余量限位 [{safe_lower:.4f}, {safe_upper:.4f}]"
        )


def validate_feedback(motor, joint: JointInfo, target: float, actual: float,
                      velocity: float, args: argparse.Namespace, *, active: bool) -> None:
    if not all(math.isfinite(v) for v in (target, actual, velocity, float(motor.get_motor_current()))):
        raise SafetyAbort(f"{joint.name} 反馈出现 NaN/Inf")
    if actual < joint.lower - args.feedback_limit_tolerance or actual > joint.upper + args.feedback_limit_tolerance:
        raise SafetyAbort(f"{joint.name} 反馈 {actual:.4f} rad 超出 URDF 限位")
    allowed = args.max_tracking_error if active else args.max_hold_error
    if abs(target - actual) > allowed:
        raise SafetyAbort(f"{joint.name} 跟踪误差 {abs(target-actual):.4f} rad 超限")
    if abs(velocity) > args.max_velocity:
        raise SafetyAbort(f"{joint.name} 速度 {velocity:.3f} rad/s 超限")
    temperature = float(motor.get_motor_temperature())
    if math.isfinite(temperature) and temperature > args.max_temperature:
        raise SafetyAbort(f"{joint.name} 温度 {temperature:.1f} C 超限")
    error_id = int(motor.get_error_id())
    if error_id not in {0, 1}:
        raise SafetyAbort(f"{joint.name} 电机错误码 0x{error_id:02x}")
    if int(motor.get_response_count()) > 25:
        raise SafetyAbort(f"{joint.name} 连续 25 帧未收到响应")


def enable_group(motors, joints: list[JointInfo], args: argparse.Namespace,
                 active_joint: JointInfo | None, amplitude: float) -> list[float]:
    for motor in motors:
        motor.refresh_motor_status()
    time.sleep(0.12)
    centers = [joint_position(motor, joint) for motor, joint in zip(motors, joints)]
    for center, joint in zip(centers, joints):
        if not math.isfinite(center):
            raise SafetyAbort(f"{joint.name} 未取得有效起始位置")
        _validate_target(center, amplitude if joint == active_joint else 0.0, joint, args)
    initialized = []
    try:
        for motor, joint, center in zip(motors, joints, centers):
            motor.init_motor()
            initialized.append(motor)
            send_target(motor, joint, center)
        time.sleep(0.3)
        return centers
    except BaseException:
        disable_group(initialized, suppress=True)
        raise


def disable_group(motors, *, suppress: bool = False) -> None:
    first_error = None
    for _ in range(3):
        for motor in motors:
            try:
                motor.deinit_motor()
            except Exception as error:
                first_error = first_error or error
        time.sleep(0.08)
    if first_error is not None and not suppress:
        raise first_error


def release_motors(motors) -> None:
    motors.clear()
    gc.collect()


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def sine_waveform(elapsed: float, args: argparse.Namespace) -> tuple[str, float]:
    steady_duration = args.cycles / args.frequency
    settle_end = args.settle
    ramp_in_end = settle_end + args.ramp
    steady_end = ramp_in_end + steady_duration
    ramp_out_end = steady_end + args.ramp
    if elapsed < settle_end:
        return "settle", 0.0
    sine = math.sin(2.0 * math.pi * args.frequency * (elapsed - settle_end))
    if elapsed < ramp_in_end:
        return "ramp_in", args.amplitude * smoothstep((elapsed-settle_end)/args.ramp) * sine
    if elapsed < steady_end:
        return "steady", args.amplitude * sine
    if elapsed < ramp_out_end:
        return "ramp_out", args.amplitude * (1.0-smoothstep((elapsed-steady_end)/args.ramp)) * sine
    return "recover", 0.0


def sine_duration(args: argparse.Namespace) -> float:
    return args.settle + 2.0 * args.ramp + args.cycles / args.frequency + args.recover


def run_sine_case(motors_py, group_joints: list[JointInfo], command_order: list[JointInfo],
                  active_joint: JointInfo, experiment: str, case: str,
                  args: argparse.Namespace) -> tuple[list[dict], dict]:
    by_name = {joint.name: i for i, joint in enumerate(group_joints)}
    motors = [create_motor(motors_py, joint) for joint in group_joints]
    rows: list[dict] = []
    try:
        centers = enable_group(motors, group_joints, args, active_joint, args.amplitude)
        motor_by_name = {joint.name: motor for joint, motor in zip(group_joints, motors)}
        center_by_name = {joint.name: center for joint, center in zip(group_joints, centers)}
        period = 1.0 / args.rate
        start = previous = next_tick = time.perf_counter()
        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start
            if elapsed >= sine_duration(args):
                break
            phase, offset = sine_waveform(elapsed, args)
            target_by_name = dict(center_by_name)
            target_by_name[active_joint.name] += offset
            active_call_ns = 0
            for joint in command_order:
                if joint.name == active_joint.name:
                    active_call_ns = time.perf_counter_ns()
                send_target(motor_by_name[joint.name], joint, target_by_name[joint.name])
            command_done = time.perf_counter()
            actual = [joint_position(motor, joint) for motor, joint in zip(motors, group_joints)]
            velocity = [joint_velocity(motor, joint) for motor, joint in zip(motors, group_joints)]
            for motor, joint, q, dq in zip(motors, group_joints, actual, velocity):
                validate_feedback(motor, joint, target_by_name[joint.name], q, dq, args,
                                  active=joint.name == active_joint.name)
            active_index = by_name[active_joint.name]
            active_motor = motors[active_index]
            rows.append({
                "experiment": experiment, "case": case,
                "interface": active_joint.motor.interface,
                "joint_name": active_joint.name, "motor_id": active_joint.motor.motor_id,
                "command_rank": next(i for i, j in enumerate(command_order) if j.name == active_joint.name),
                "elapsed_s": elapsed, "phase": phase, "loop_period_s": loop_start-previous,
                "active_call_s": (active_call_ns / 1e9) - start,
                "command_done_s": command_done-start,
                "q_center_rad": centers[active_index],
                "q_target_rad": target_by_name[active_joint.name],
                "q_actual_rad": actual[active_index], "dq_actual_rad_s": velocity[active_index],
                "feedback_field": float(active_motor.get_motor_current()) * active_joint.motor.sign,
                "temperature_c": float(active_motor.get_motor_temperature()),
                "error_id": int(active_motor.get_error_id()),
            })
            previous = loop_start
            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif -remaining > period:
                next_tick = time.perf_counter()
        for _ in range(max(1, int(0.3 * args.rate))):
            for joint in command_order:
                send_target(motor_by_name[joint.name], joint, center_by_name[joint.name])
            time.sleep(period)
        return rows, sine_metrics(rows, args.frequency)
    finally:
        disable_group(motors, suppress=True)
        release_motors(motors)
        time.sleep(args.case_pause)


def sine_metrics(rows: list[dict], frequency: float) -> dict:
    steady = [row for row in rows if row["phase"] == "steady"]
    metrics: dict[str, float | int] = {"steady_samples": len(steady)}
    if len(steady) >= 10:
        t = np.asarray([r["elapsed_s"] for r in steady], dtype=float)
        target = np.asarray([r["q_target_rad"] for r in steady], dtype=float)
        actual = np.asarray([r["q_actual_rad"] for r in steady], dtype=float)
        omega = 2.0 * math.pi * frequency
        design = np.column_stack((np.ones_like(t), np.sin(omega*t), np.cos(omega*t)))
        target_coef = np.linalg.lstsq(design, target, rcond=None)[0]
        actual_coef = np.linalg.lstsq(design, actual, rcond=None)[0]
        target_amp = float(np.hypot(target_coef[1], target_coef[2]))
        actual_amp = float(np.hypot(actual_coef[1], actual_coef[2]))
        target_phase = math.atan2(float(target_coef[2]), float(target_coef[1]))
        actual_phase = math.atan2(float(actual_coef[2]), float(actual_coef[1]))
        lag = (target_phase-actual_phase+math.pi) % (2.0*math.pi) - math.pi
        error = target-actual
        metrics.update({
            "target_amplitude_rad": target_amp, "actual_amplitude_rad": actual_amp,
            "gain": actual_amp/target_amp if target_amp > 1e-9 else float("nan"),
            "phase_lag_deg": math.degrees(lag), "phase_delay_ms": lag/omega*1000.0,
            "tracking_rmse_rad": float(np.sqrt(np.mean(error*error))),
            "max_abs_tracking_error_rad": float(np.max(np.abs(error))),
        })
    periods = np.asarray([r["loop_period_s"] for r in rows[1:]], dtype=float)
    if periods.size:
        metrics.update({
            "measured_loop_rate_hz": float(1.0/np.mean(periods)),
            "loop_period_p95_ms": float(np.percentile(periods, 95)*1000.0),
            "loop_period_max_ms": float(np.max(periods)*1000.0),
        })
    return metrics


def percentile_metrics(values_ns: list[int], prefix: str) -> dict[str, float]:
    if not values_ns:
        return {}
    values_ms = np.asarray(values_ns, dtype=float) / 1e6
    return {
        f"{prefix}_median_ms": float(np.median(values_ms)),
        f"{prefix}_p95_ms": float(np.percentile(values_ms, 95)),
        f"{prefix}_p99_ms": float(np.percentile(values_ms, 99)),
        f"{prefix}_max_ms": float(np.max(values_ms)),
    }


def default_output_path(experiment: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{stamp}_{experiment}.csv"


def write_outputs(output: Path, rows: list[dict], report: dict) -> tuple[Path, Path]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有日志: {output}")
    if not rows:
        raise ValueError("没有可写入的采样")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("CSV 行字段不一致")
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = output.with_suffix(".metadata.yaml")
    with metadata.open("x", encoding="utf-8") as stream:
        yaml.safe_dump({**report, "csv": str(output)}, stream, allow_unicode=True, sort_keys=False)
    print(f"\n原始 CSV: {output}\n汇总 YAML: {metadata}")
    return output, metadata


def write_partial_outputs(output: Path, rows: list[dict], experiment: str,
                          configuration: dict, results: list[dict], reason: object) -> None:
    if not rows:
        return
    try:
        write_outputs(output, rows, {
            "experiment": experiment, "status": "aborted",
            "stop_reason": str(reason) or type(reason).__name__,
            "configuration": configuration, "completed_results": results,
        })
        print("已保存中止前完成的 case。", file=sys.stderr)
    except Exception as write_error:
        print(f"保存部分日志失败: {write_error}", file=sys.stderr)


def config_metadata(args: argparse.Namespace) -> dict:
    return {
        "robot_config": str(args.config.expanduser().resolve()),
        "joint_order_config": str(args.joint_order_config.expanduser().resolve()),
        "urdf": str(args.urdf.expanduser().resolve()),
        "rate_hz": args.rate, "limit_margin_rad": args.limit_margin,
    }
