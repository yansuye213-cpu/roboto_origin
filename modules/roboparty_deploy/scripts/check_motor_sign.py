#!/usr/bin/env python3
"""Interactively verify RPO motor signs using raw motor position deltas."""

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import statistics
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = DEPLOY_ROOT / "src/inference/robots/rpo/robot.yaml"
DEFAULT_URDF = (
    DEPLOY_ROOT
    / "src/inference/robots/rpo/description/urdf/Loobot722.urdf"
)
JOINT_COUNT = 21
ANKLE_JOINTS = {
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
}
CONTROL_PROCESS_MARKERS = (
    "inference_node",
    "motion_player.py",
    "motors_py_example.py",
    "set_zero.py",
)


@dataclass(frozen=True)
class JointCheck:
    name: str
    label: str
    urdf_axis: tuple
    movement: str
    expected_urdf_delta: int


# Directions assume the robot coordinate system used by Loobot722.urdf:
# +X forward, +Y left, +Z upward. The instructions deliberately use movements
# that are easy to recognize from the robot's own point of view; some of them
# are negative URDF motion.
JOINT_CHECKS = (
    JointCheck(
        "left_leg_pitch_joint", "左髋俯仰", (0.0, 1.0, 0.0),
        "把左侧大腿向机器人正前方抬/摆（像向前迈左腿）。", -1,
    ),
    JointCheck(
        "left_leg_roll_joint", "左髋侧摆", (1.0, 0.0, 0.0),
        "把左腿向机器人左侧张开，远离身体中线。", 1,
    ),
    JointCheck(
        "left_leg_yaw_joint", "左髋旋转", (0.0, 0.0, 1.0),
        "从机器人自己低头看的方向，把左膝和左脚尖转向左侧（向外转）。", 1,
    ),
    JointCheck(
        "left_knee_joint", "左膝", (0.0, 1.0, 0.0),
        "弯曲左膝，让左小腿和脚跟向机器人后方移动。", 1,
    ),
    JointCheck(
        "left_ankle_pitch_joint", "左踝俯仰映射电机", (0.0, 1.0, 0.0),
        "直接摆左脚，让左脚尖向上、靠近小腿（只按该原始电机读数判断）。", -1,
    ),
    JointCheck(
        "left_ankle_roll_joint", "左踝侧摆映射电机", (1.0, 0.0, 0.0),
        "直接摆左脚，抬起左脚掌外侧（机器人左侧边缘），内侧边缘向下。", 1,
    ),
    JointCheck(
        "right_leg_pitch_joint", "右髋俯仰", (0.0, 1.0, 0.0),
        "把右侧大腿向机器人正前方抬/摆（像向前迈右腿）。", -1,
    ),
    JointCheck(
        "right_leg_roll_joint", "右髋侧摆", (1.0, 0.0, 0.0),
        "把右腿向机器人右侧张开，远离身体中线。", -1,
    ),
    JointCheck(
        "right_leg_yaw_joint", "右髋旋转", (0.0, 0.0, 1.0),
        "从机器人自己低头看的方向，把右膝和右脚尖转向右侧（向外转）。", -1,
    ),
    JointCheck(
        "right_knee_joint", "右膝", (0.0, 1.0, 0.0),
        "弯曲右膝，让右小腿和脚跟向机器人后方移动。", 1,
    ),
    JointCheck(
        "right_ankle_pitch_joint", "右踝俯仰映射电机", (0.0, 1.0, 0.0),
        "直接摆右脚，让右脚尖向上、靠近小腿（只按该原始电机读数判断）。", -1,
    ),
    JointCheck(
        "right_ankle_roll_joint", "右踝侧摆映射电机", (1.0, 0.0, 0.0),
        "直接摆右脚，抬起右脚掌外侧（机器人右侧边缘），内侧边缘向下。", -1,
    ),
    JointCheck(
        "head_yaw_joint", "头部偏航", (0.0, 0.0, 1.0),
        "从机器人自己低头看的方向，把头转向左侧。", 1,
    ),
    JointCheck(
        "left_shoulder_pitch_joint", "左肩俯仰", (0.0, 1.0, 0.0),
        "保持左臂较直，把整条左臂向机器人正前方抬/摆。", -1,
    ),
    JointCheck(
        "left_shoulder_roll_joint", "左肩侧摆", (1.0, 0.0, 0.0),
        "保持左臂较直，把左臂向机器人左侧抬起，远离身体。", 1,
    ),
    JointCheck(
        "left_shoulder_yaw_joint", "左肩旋转", (0.0, 0.0, 1.0),
        "先弯肘让左前臂大致朝前，再绕上臂轴转动，使左前臂朝机器人左侧转。", 1,
    ),
    JointCheck(
        "left_elbow_pitch_joint", "左肘", (0.0, 1.0, 0.0),
        "保持左上臂大致下垂，弯左肘，让左前臂末端向机器人正前方抬起。", -1,
    ),
    JointCheck(
        "right_shoulder_pitch_joint", "右肩俯仰", (0.0, 1.0, 0.0),
        "保持右臂较直，把整条右臂向机器人正前方抬/摆。", -1,
    ),
    JointCheck(
        "right_shoulder_roll_joint", "右肩侧摆", (1.0, 0.0, 0.0),
        "保持右臂较直，把右臂向机器人右侧抬起，远离身体。", -1,
    ),
    JointCheck(
        "right_shoulder_yaw_joint", "右肩旋转", (0.0, 0.0, 1.0),
        "先弯肘让右前臂大致朝前，再绕上臂轴转动，使右前臂朝机器人右侧转。", -1,
    ),
    JointCheck(
        "right_elbow_pitch_joint", "右肘", (0.0, 1.0, 0.0),
        "保持右上臂大致下垂，弯右肘，让右前臂末端向机器人正前方抬起。", -1,
    ),
)


class UserQuit(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "逐关节读取 motor_sign 处理前的位置变化，并按当前 URDF 方向检查 "
            "RPO 的 motor_sign。不会修改 robot.yaml。"
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"电机与映射配置（默认: {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--urdf", type=Path, default=DEFAULT_URDF,
        help=f"用于核对关节轴的 URDF（默认: {DEFAULT_URDF}）",
    )
    parser.add_argument(
        "--only", action="append", metavar="JOINT",
        help="只检查指定关节；可重复使用，值为完整关节名或 1-21 的显示序号",
    )
    parser.add_argument(
        "--damping-kd", type=float, default=1.0,
        help="手动摆动时的纯阻尼 Kd（默认: 1.0）",
    )
    parser.add_argument(
        "--min-delta", type=float, default=0.05,
        help="接受测量所需的最小绝对位置变化，单位 rad（默认: 0.05）",
    )
    parser.add_argument(
        "--output", type=Path,
        help="可选的 YAML 测量报告路径；不会写回机器人配置",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许覆盖已经存在的 --output 报告",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="只显示动作提示及电机映射，不连接硬件",
    )
    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def require_list(mapping, key, expected_length=None):
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} 必须是列表")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(
            f"{key} 长度应为 {expected_length}，实际为 {len(value)}"
        )
    return value


def validate_config(config):
    motors = config.get("motors")
    robot = config.get("robot")
    if not isinstance(motors, dict) or not isinstance(robot, dict):
        raise ValueError("配置必须包含 motors 和 robot")

    motor_ids = require_list(motors, "motor_id", JOINT_COUNT)
    motor_num = require_list(motors, "motor_num")
    if sum(int(value) for value in motor_num) != JOINT_COUNT:
        raise ValueError("motor_num 总数必须为 21")
    require_list(motors, "motor_model", JOINT_COUNT)
    require_list(motors, "motor_zero_offset", JOINT_COUNT)

    urdf2motor = [int(value) for value in require_list(
        robot, "urdf2motor", JOINT_COUNT
    )]
    if sorted(urdf2motor) != list(range(JOINT_COUNT)):
        raise ValueError("urdf2motor 必须是 0-20 的排列")

    signs = [int(value) for value in require_list(
        robot, "motor_sign", JOINT_COUNT
    )]
    invalid_signs = [value for value in signs if value not in (-1, 1)]
    if invalid_signs:
        raise ValueError(f"motor_sign 只能为 +1 或 -1，发现 {invalid_signs}")
    return motors, urdf2motor, signs, motor_ids


def parse_axis(text):
    values = tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"无效关节轴: {text!r}")
    return values


def validate_urdf(path):
    root = ET.parse(path).getroot()
    axes = {}
    for joint in root.findall("joint"):
        axis = joint.find("axis")
        if axis is not None and axis.get("xyz"):
            axes[joint.get("name")] = parse_axis(axis.get("xyz"))

    for check in JOINT_CHECKS:
        actual = axes.get(check.name)
        if actual is None:
            raise ValueError(f"URDF 中缺少关节或关节轴: {check.name}")
        if any(
            not math.isclose(got, expected, abs_tol=1e-6)
            for got, expected in zip(actual, check.urdf_axis)
        ):
            raise ValueError(
                f"{check.name} 的轴已变化: 脚本预期 {check.urdf_axis}，"
                f"URDF 实际 {actual}；请先更新动作定义"
            )


def select_checks(only):
    if not only:
        return list(enumerate(JOINT_CHECKS))

    indexes = []
    names = {check.name: index for index, check in enumerate(JOINT_CHECKS)}
    for item in only:
        if item.isdigit() and 1 <= int(item) <= JOINT_COUNT:
            index = int(item) - 1
        elif item in names:
            index = names[item]
        else:
            raise ValueError(f"未知关节 {item!r}；请使用 --list 查看可用值")
        if index not in indexes:
            indexes.append(index)
    return [(index, JOINT_CHECKS[index]) for index in indexes]


def bus_for_motor(motors_config, motor_index):
    start = 0
    for bus_index, count in enumerate(motors_config["motor_num"]):
        end = start + int(count)
        if start <= motor_index < end:
            return bus_index
        start = end
    raise ValueError(f"电机索引 {motor_index} 不属于任何接口")


def value_for_bus(value, bus_index, key):
    if isinstance(value, list):
        if bus_index >= len(value):
            raise ValueError(f"{key} 没有接口 {bus_index} 的配置")
        return value[bus_index]
    return value


def motor_descriptor(motors_config, motor_index):
    bus_index = bus_for_motor(motors_config, motor_index)
    interface = value_for_bus(
        motors_config["motor_interface"], bus_index, "motor_interface"
    )
    return {
        "motor_index": motor_index,
        "bus_index": bus_index,
        "interface": interface,
        "interface_type": value_for_bus(
            motors_config["motor_interface_type"],
            bus_index,
            "motor_interface_type",
        ),
        "motor_type": value_for_bus(
            motors_config["motor_type"], bus_index, "motor_type"
        ),
        "motor_id": int(motors_config["motor_id"][motor_index]),
        "motor_model": int(motors_config["motor_model"][motor_index]),
        "motor_zero_offset": float(
            motors_config["motor_zero_offset"][motor_index]
        ),
    }


def print_mapping(selected, motors_config, urdf2motor, signs):
    print("\n坐标约定: 机器人自身 +X 向前、+Y 向左、+Z 向上。")
    print("动作中的左右均指机器人自己的左右，不是面对机器人时操作者的左右。")
    print("raw 指 get_motor_pos() 返回但尚未乘 motor_sign 的位置；零点偏移不影响差值。")
    print("URDFΔ 表示提示动作在当前 URDF 坐标中应产生的正/负变化。\n")
    print(
        f"{'序号':>4}  {'关节':32} {'CAN/ID':10} "
        f"{'当前sign':>8} {'URDFΔ':>7}  动作"
    )
    print("-" * 128)
    for joint_index, check in selected:
        motor_index = urdf2motor[joint_index]
        desc = motor_descriptor(motors_config, motor_index)
        bus_id = f"{desc['interface']}/{desc['motor_id']}"
        expected = "+" if check.expected_urdf_delta > 0 else "-"
        print(
            f"{joint_index + 1:>4}  {check.name:32} {bus_id:10} "
            f"{signs[motor_index]:>+8d} {expected:>7}  {check.movement}"
        )


def find_control_processes():
    found = []
    current_pid = os.getpid()
    proc_root = Path("/proc")
    for process_dir in proc_root.iterdir():
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


def read_key(timeout=0.05):
    import select

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def create_motor(motors_py, motors_config, motor_index):
    desc = motor_descriptor(motors_config, motor_index)
    motor = motors_py.MotorDriver.create_motor(
        motor_id=desc["motor_id"],
        interface_type=desc["interface_type"],
        interface=desc["interface"],
        motor_type=desc["motor_type"],
        motor_model=desc["motor_model"],
        master_id_offset=int(motors_config.get("master_id_offset", 0)),
        motor_zero_offset=desc["motor_zero_offset"],
    )
    return motor, desc


def update_damping(motor, damping_kd, samples):
    motor.motor_mit_cmd(0.0, 0.0, 0.0, damping_kd, 0.0)
    position = float(motor.get_motor_pos())
    if math.isfinite(position):
        samples.append(position)
    return position


def stable_position(samples, fallback):
    if samples:
        return float(statistics.median(samples))
    if math.isfinite(fallback):
        return float(fallback)
    raise RuntimeError("没有取得有效的电机位置")


def wait_for_start(motor, check, damping_kd):
    samples = deque(maxlen=10)
    position = float("nan")
    print("\n先把关节放在舒适且留有摆动空间的起始位置。")
    print("[Enter] 记录起点  [s] 跳过  [q] 结束全部检查")
    while True:
        position = update_damping(motor, damping_kd, samples)
        print(
            f"\r处理前电机位置: {position:+.6f} rad | 错误码: "
            f"{motor.get_error_id()}    ",
            end="",
            flush=True,
        )
        key = read_key()
        if key in ("\r", "\n"):
            print()
            return stable_position(samples, position)
        if key in ("s", "S"):
            print()
            return None
        if key in ("q", "Q"):
            print()
            raise UserQuit
        if key == "\x03":
            raise KeyboardInterrupt
        time.sleep(0.01)


def wait_for_end(motor, check, start, damping_kd, min_delta):
    samples = deque(maxlen=10)
    position = start
    print(f"\n请按下面方向缓慢摆动并保持：\n  {check.movement}")
    print("[Enter] 记录终点  [r] 重选起点  [s] 跳过  [q] 结束全部检查")
    while True:
        position = update_damping(motor, damping_kd, samples)
        delta = position - start
        print(
            f"\r处理前电机位置: {position:+.6f} rad | Δraw: {delta:+.6f} rad "
            f"| 错误码: {motor.get_error_id()}    ",
            end="",
            flush=True,
        )
        key = read_key()
        if key in ("\r", "\n"):
            end = stable_position(samples, position)
            delta = end - start
            if abs(delta) < min_delta:
                print(
                    f"\n动作只有 {abs(delta):.6f} rad，小于阈值 "
                    f"{min_delta:.6f} rad；请继续沿同一方向摆动后再按 Enter。"
                )
                continue
            print()
            return "measured", end
        if key in ("r", "R"):
            print()
            return "restart", None
        if key in ("s", "S"):
            print()
            return "skipped", None
        if key in ("q", "Q"):
            print()
            raise UserQuit
        if key == "\x03":
            raise KeyboardInterrupt
        time.sleep(0.01)


def measure_joint(
    motors_py,
    motors_config,
    joint_index,
    check,
    motor_index,
    current_sign,
    damping_kd,
    min_delta,
):
    motor, desc = create_motor(motors_py, motors_config, motor_index)
    print("\n" + "=" * 78)
    print(f"[{joint_index + 1}/{JOINT_COUNT}] {check.label} ({check.name})")
    print(
        f"motor[{motor_index}] = {desc['interface']}/ID {desc['motor_id']} | "
        f"当前 motor_sign={current_sign:+d}"
    )
    if check.name in ANKLE_JOINTS:
        print("踝部模式: 原始映射电机直读，不调用闭链正运动学或解耦。")

    initialized = False
    try:
        print("使能该电机并进入纯阻尼模式（Kp=0）...")
        initialized = True
        motor.init_motor()
        time.sleep(0.3)
        motor.set_motor_control_mode(motors_py.MotorControlMode.MIT)
        time.sleep(0.05)
        motor.motor_mit_cmd(0.0, 0.0, 0.0, damping_kd, 0.0)
        time.sleep(0.05)

        while True:
            start = wait_for_start(motor, check, damping_kd)
            if start is None:
                return {
                    "status": "skipped",
                    "joint_index": joint_index,
                    "joint_name": check.name,
                    **desc,
                    "current_motor_sign": current_sign,
                }
            status, end = wait_for_end(
                motor, check, start, damping_kd, min_delta
            )
            if status == "restart":
                continue
            if status == "skipped":
                return {
                    "status": "skipped",
                    "joint_index": joint_index,
                    "joint_name": check.name,
                    **desc,
                    "current_motor_sign": current_sign,
                }
            break

        raw_delta = end - start
        raw_direction = 1 if raw_delta > 0 else -1
        recommended_sign = check.expected_urdf_delta * raw_direction
        processed_delta = raw_delta * current_sign
        matches = current_sign == recommended_sign
        verdict = "PASS，当前符号一致" if matches else "MISMATCH，建议反转"
        print(
            f"起点={start:+.6f}, 终点={end:+.6f}, Δraw={raw_delta:+.6f} rad"
        )
        print(
            f"当前处理后 Δ={processed_delta:+.6f} rad；提示动作应为 URDF "
            f"{'正' if check.expected_urdf_delta > 0 else '负'}方向"
        )
        print(
            f"结果: {verdict} | 当前={current_sign:+d}, "
            f"建议={recommended_sign:+d}"
        )
        return {
            "status": "pass" if matches else "mismatch",
            "joint_index": joint_index,
            "joint_name": check.name,
            "joint_label": check.label,
            **desc,
            "movement": check.movement,
            "expected_urdf_delta": check.expected_urdf_delta,
            "start_raw_position": start,
            "end_raw_position": end,
            "raw_delta": raw_delta,
            "processed_delta_with_current_sign": processed_delta,
            "current_motor_sign": current_sign,
            "recommended_motor_sign": recommended_sign,
            "ankle_direct_raw": check.name in ANKLE_JOINTS,
        }
    finally:
        if initialized:
            try:
                print("失能该电机...")
                motor.deinit_motor()
                time.sleep(0.2)
            except Exception as error:
                print(f"警告: 电机失能命令失败: {error}", file=sys.stderr)


def print_results(results, current_signs):
    print("\n" + "=" * 78)
    print("检查结果（motor_sign 按 motor index 排列）")
    print("=" * 78)
    if not results:
        print("没有完成任何测量。")
        return

    print(
        f"{'关节':32} {'CAN/ID':10} {'Δraw(rad)':>11} "
        f"{'当前':>6} {'建议':>6} {'结果':>10}"
    )
    print("-" * 84)
    recommendations = list(current_signs)
    measured_motor_indexes = set()
    for result in results:
        desc = f"{result['interface']}/{result['motor_id']}"
        if result["status"] == "skipped":
            print(
                f"{result['joint_name']:32} {desc:10} {'--':>11} "
                f"{result['current_motor_sign']:+6d} {'--':>6} {'SKIPPED':>10}"
            )
            continue
        recommendations[result["motor_index"]] = result["recommended_motor_sign"]
        measured_motor_indexes.add(result["motor_index"])
        print(
            f"{result['joint_name']:32} {desc:10} {result['raw_delta']:+11.6f} "
            f"{result['current_motor_sign']:+6d} "
            f"{result['recommended_motor_sign']:+6d} "
            f"{result['status'].upper():>10}"
        )

    print("\n基于已测关节合并后的候选 motor_sign（未测项保留当前值）:")
    print("  [" + ", ".join(f"{value:2d}" for value in recommendations) + "]")
    if len(measured_motor_indexes) != JOINT_COUNT:
        print("注意: 存在未测项，上述完整数组不能当作全量验证结果。")
    print("脚踝结论仅来自原始映射电机增量，没有经过闭链正运动学。")


def write_report(path, overwrite, config_path, urdf_path, args, results, current_signs):
    resolved_path = path.expanduser().resolve()
    protected_paths = {
        config_path.expanduser().resolve(),
        urdf_path.expanduser().resolve(),
    }
    if resolved_path in protected_paths:
        raise ValueError("--output 不能指向机器人配置或 URDF")
    if resolved_path.exists() and not overwrite:
        raise FileExistsError(
            f"报告已存在: {resolved_path}；如需覆盖请增加 --overwrite"
        )

    recommendations = list(current_signs)
    for result in results:
        if "recommended_motor_sign" in result:
            recommendations[result["motor_index"]] = result[
                "recommended_motor_sign"
            ]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path.expanduser().resolve()),
        "urdf": str(urdf_path.expanduser().resolve()),
        "coordinate_convention": "+X forward, +Y left, +Z up (robot view)",
        "ankle_mode": "direct_raw_motor_delta_without_closed_chain_kinematics",
        "damping_kd": args.damping_kd,
        "minimum_delta_rad": args.min_delta,
        "current_motor_sign": current_signs,
        "candidate_motor_sign_unmeasured_kept_current": recommendations,
        "results": results,
    }
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)
    print(f"\n报告已写入: {resolved_path}")


def main():
    args = parse_args()
    if args.damping_kd <= 0:
        print("错误: --damping-kd 必须大于 0", file=sys.stderr)
        return 2
    if args.min_delta <= 0:
        print("错误: --min-delta 必须大于 0", file=sys.stderr)
        return 2
    if not sys.stdin.isatty() and not args.list:
        print("错误: 实机检查必须在交互式终端中运行", file=sys.stderr)
        return 2

    try:
        config = load_yaml(args.config)
        motors_config, urdf2motor, signs, _ = validate_config(config)
        validate_urdf(args.urdf)
        selected = select_checks(args.only)
    except (OSError, ET.ParseError, ValueError, yaml.YAMLError) as error:
        print(f"配置检查失败: {error}", file=sys.stderr)
        return 2

    print_mapping(selected, motors_config, urdf2motor, signs)
    if args.list:
        return 0

    running = find_control_processes()
    if running:
        print("\n检测到可能正在控制电机的进程，拒绝继续：", file=sys.stderr)
        for pid, command in running:
            print(f"  PID {pid}: {command}", file=sys.stderr)
        print("请先正常停止这些进程，再重新运行本工具。", file=sys.stderr)
        return 2

    print("\n安全要求:")
    print("  1. 机器人必须断开推理/动作播放，并可靠吊起或固定，不能自行站立测试。")
    print("  2. 每次只使能一个电机；该电机 Kp=0，只提供阻尼，不会保持姿态。")
    print("  3. 手动托住被测肢体，缓慢小幅摆动，远离机械限位和夹点。")
    print("  4. Ctrl+C、q 或异常退出时，脚本会尝试立即失能当前电机。")
    confirmation = input("\n确认满足以上条件后，输入 CHECK 并按 Enter: ").strip()
    if confirmation != "CHECK":
        print("未确认，已退出且未使能电机。")
        return 1

    try:
        import motors_py
    except ImportError as error:
        print(
            "无法导入 motors_py。请先编译工作空间并执行 "
            "source install/setup.bash。",
            file=sys.stderr,
        )
        print(f"详细错误: {error}", file=sys.stderr)
        return 2

    results = []
    interrupted = False
    try:
        for joint_index, check in selected:
            motor_index = urdf2motor[joint_index]
            result = measure_joint(
                motors_py,
                motors_config,
                joint_index,
                check,
                motor_index,
                signs[motor_index],
                args.damping_kd,
                args.min_delta,
            )
            results.append(result)
    except UserQuit:
        print("\n用户结束检查。")
        interrupted = True
    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C，正在安全退出。")
        interrupted = True
    except Exception as error:
        print(f"\n检查过程中出错: {error}", file=sys.stderr)
        interrupted = True

    print_results(results, signs)
    if args.output:
        try:
            write_report(
                args.output,
                args.overwrite,
                args.config,
                args.urdf,
                args,
                results,
                signs,
            )
        except (OSError, ValueError) as error:
            print(f"写入报告失败: {error}", file=sys.stderr)
            return 2
    return 1 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
