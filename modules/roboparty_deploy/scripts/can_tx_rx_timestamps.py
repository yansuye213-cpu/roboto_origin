#!/usr/bin/env python3
"""Measure SocketCAN enqueue, write, and receive-dispatch timing."""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

import can_latency_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TX 入队、SocketCAN write、RX 回调时间戳对照")
    common.add_common_arguments(parser, sine=False)
    parser.add_argument("--samples", type=int, default=2500, help="每个 case 的记录周期数（默认 2500）")
    parser.add_argument("--warmup", type=int, default=250, help="记录前预热周期数（默认 250）")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    common.validate_common_args(args, sine=False)
    if args.samples < 1000:
        raise ValueError("--samples 至少为 1000，避免用过少样本判断尾延迟")
    if args.warmup < 0:
        raise ValueError("--warmup 必须为非负整数")


def send_and_check(motors, joints, centers, args) -> None:
    for motor, joint, center in zip(motors, joints, centers):
        common.send_target(motor, joint, center)
    for motor, joint, center in zip(motors, joints, centers):
        q = common.joint_position(motor, joint)
        dq = common.joint_velocity(motor, joint)
        common.validate_feedback(motor, joint, center, q, dq, args, active=False)


def run_timed_case(motors_py, joints, case: str, args) -> tuple[dict, dict]:
    interface = joints[0].motor.interface
    motors = [common.create_motor(motors_py, joint) for joint in joints]
    timing_enabled = False
    snapshot = None
    try:
        centers = common.enable_group(motors, joints, args, None, 0.0)
        period = 1.0 / args.rate
        next_tick = time.perf_counter()
        for _ in range(args.warmup):
            send_and_check(motors, joints, centers, args)
            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif -remaining > period:
                next_tick = time.perf_counter()

        motors_py.set_can_timing_enabled(interface, True)
        timing_enabled = True
        next_tick = time.perf_counter()
        for _ in range(args.samples):
            send_and_check(motors, joints, centers, args)
            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif -remaining > period:
                next_tick = time.perf_counter()
        time.sleep(max(0.02, 2.0 * period))
        motors_py.set_can_timing_enabled(interface, False)
        timing_enabled = False
        snapshot = dict(motors_py.drain_can_timing(interface))
        return snapshot, {
            "case": case, "interface": interface, "controlled_motor_count": len(joints),
            "expected_cycles_per_motor": args.samples,
            "tx_queue_drops": int(snapshot["tx_queue_drops"]),
            "timing_event_drops": int(snapshot["timing_event_drops"]),
        }
    finally:
        if timing_enabled:
            try:
                motors_py.set_can_timing_enabled(interface, False)
                snapshot = dict(motors_py.drain_can_timing(interface))
            except Exception:
                pass
        common.disable_group(motors, suppress=True)
        common.release_motors(motors)
        time.sleep(args.case_pause)


def match_motor_events(snapshot: dict, joint: common.JointInfo, case: str,
                       expected_samples: int) -> tuple[list[dict], dict]:
    motor_id = joint.motor.motor_id
    response_id = motor_id + joint.motor.master_id_offset
    tx_events = sorted(
        (dict(event) for event in snapshot["tx_events"] if int(event["can_id"]) == motor_id),
        key=lambda event: int(event["sequence"]),
    )
    rx_times = sorted(
        int(event["rx_ns"]) for event in snapshot["rx_events"]
        if int(event["can_id"]) == response_id
    )
    successful_write_times = [
        int(event["write_ns"]) for event in tx_events
        if bool(event["write_success"]) and int(event["write_ns"]) > 0
    ]
    rows = []
    rx_index = 0
    success_index = 0
    for event in tx_events:
        success = bool(event["write_success"]) and int(event["write_ns"]) > 0
        enqueue_ns = int(event["enqueue_ns"])
        write_ns = int(event["write_ns"])
        rx_ns = 0
        if success:
            next_write_ns = (
                successful_write_times[success_index + 1]
                if success_index + 1 < len(successful_write_times) else math.inf
            )
            while rx_index < len(rx_times) and rx_times[rx_index] < write_ns:
                rx_index += 1
            if rx_index < len(rx_times) and rx_times[rx_index] < next_write_ns:
                rx_ns = rx_times[rx_index]
                rx_index += 1
            success_index += 1
        rows.append({
            "experiment": "tx_rx_timestamps", "case": case,
            "interface": joint.motor.interface, "joint_name": joint.name,
            "motor_id": motor_id, "response_can_id": response_id,
            "sequence": int(event["sequence"]), "enqueue_ns": enqueue_ns,
            "write_ns": write_ns, "rx_dispatch_ns": rx_ns,
            "queue_depth": int(event["queue_depth"]),
            "write_success": success, "write_errno": int(event["write_errno"]),
            "matched_within_control_period": rx_ns > 0,
            "enqueue_to_write_ms": (write_ns-enqueue_ns)/1e6 if success else float("nan"),
            "write_to_rx_ms": (rx_ns-write_ns)/1e6 if rx_ns else float("nan"),
            "enqueue_to_rx_ms": (rx_ns-enqueue_ns)/1e6 if rx_ns else float("nan"),
        })

    enqueue_write = [int((r["write_ns"]-r["enqueue_ns"])) for r in rows if r["write_success"]]
    write_rx = [int(r["rx_dispatch_ns"]-r["write_ns"]) for r in rows if r["matched_within_control_period"]]
    enqueue_rx = [int(r["rx_dispatch_ns"]-r["enqueue_ns"]) for r in rows if r["matched_within_control_period"]]
    successful = sum(bool(row["write_success"]) for row in rows)
    matched = sum(bool(row["matched_within_control_period"]) for row in rows)
    queue_depths = np.asarray([row["queue_depth"] for row in rows], dtype=float)
    result = {
        "case": case, "interface": joint.motor.interface, "joint": joint.name,
        "motor_id": motor_id, "expected_samples": expected_samples,
        "observed_tx_events": len(rows), "successful_writes": successful,
        "write_failures_or_queue_drops": expected_samples-successful,
        "matched_responses": matched,
        "response_deadline_miss_rate": (successful-matched)/successful if successful else float("nan"),
        "max_queue_depth": int(np.max(queue_depths)) if queue_depths.size else 0,
        **common.percentile_metrics(enqueue_write, "enqueue_to_write"),
        **common.percentile_metrics(write_rx, "write_to_rx"),
        **common.percentile_metrics(enqueue_rx, "enqueue_to_rx"),
    }
    return rows, result


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        joints, _ = common.load_configuration(args)
        selected = common.selected_leg_joints(joints, args.joint)
        buses = common.leg_buses(joints)
    except Exception as error:
        print(f"参数/配置错误: {error}", file=sys.stderr)
        return 2

    selected_by_bus = {
        interface: [joint for joint in selected if joint.motor.interface == interface]
        for interface in buses
    }
    selected_by_bus = {interface: members for interface, members in selected_by_bus.items() if members}
    case_count = len(selected) + len(selected_by_bus)
    duration = case_count * (args.warmup + args.samples) / args.rate
    print("实验 3：驱动入队 -> SocketCAN write -> RX 分发时间戳。")
    print(f"共 {case_count} 个 case：{len(selected)} 个 single，加 {len(selected_by_bus)} 个 six_motor；记录约 {duration:.1f} s。")
    print("电机保持起始位置，不施加正弦，避免机械相位混入通信时延。")
    if args.dry_run:
        print("Dry run 通过；未连接硬件。")
        return 0

    rows, results, case_summaries = [], [], []
    output = args.output or common.default_output_path("tx_rx_timestamps")
    try:
        common.confirm_suspended()
        motors_py = common.import_motors_py()
        if not all(hasattr(motors_py, name) for name in ("set_can_timing_enabled", "drain_can_timing")):
            raise RuntimeError("motors_py 尚未包含时间戳 API；请重新构建并 source install/setup.bash")

        case_number = 0
        for joint in selected:
            case_number += 1
            print(f"\n[{case_number}/{case_count}] single: {joint.name}")
            snapshot, summary = run_timed_case(motors_py, [joint], "single", args)
            case_rows, metrics = match_motor_events(snapshot, joint, "single", args.samples)
            rows.extend(case_rows)
            results.append(metrics)
            case_summaries.append({**summary, "joint": joint.name})

        for interface, selected_members in selected_by_bus.items():
            case_number += 1
            print(f"\n[{case_number}/{case_count}] six_motor: {interface}")
            snapshot, summary = run_timed_case(motors_py, buses[interface], "six_motor", args)
            for joint in selected_members:
                case_rows, metrics = match_motor_events(snapshot, joint, "six_motor", args.samples)
                rows.extend(case_rows)
                results.append(metrics)
            case_summaries.append(summary)

        by_key = {(result["joint"], result["case"]): result for result in results}
        comparisons = []
        for joint in selected:
            single = by_key[(joint.name, "single")]
            loaded = by_key[(joint.name, "six_motor")]
            comparisons.append({
                "joint": joint.name,
                "enqueue_to_write_p95_increase_ms": loaded.get("enqueue_to_write_p95_ms", float("nan")) - single.get("enqueue_to_write_p95_ms", float("nan")),
                "write_to_rx_p95_increase_ms": loaded.get("write_to_rx_p95_ms", float("nan")) - single.get("write_to_rx_p95_ms", float("nan")),
                "enqueue_to_rx_p95_increase_ms": loaded.get("enqueue_to_rx_p95_ms", float("nan")) - single.get("enqueue_to_rx_p95_ms", float("nan")),
                "deadline_miss_rate_increase": loaded["response_deadline_miss_rate"] - single["response_deadline_miss_rate"],
            })
        common.write_outputs(output, rows, {
            "experiment": "socketcan_enqueue_write_rx_timestamps",
            "configuration": {**common.config_metadata(args), "samples": args.samples, "warmup": args.warmup},
            "matching_method": (
                "DM MIT frames have no sequence field. For each motor, the first RX after a successful "
                "write and before that motor's next write is matched; later RX is a control-period deadline miss."
            ),
            "case_summaries": case_summaries, "results": results,
            "comparisons": comparisons,
        })
        return 0
    except (KeyboardInterrupt, common.SafetyAbort) as error:
        print(f"\n安全中止: {error}", file=sys.stderr)
        common.write_partial_outputs(
            output, rows, "socketcan_enqueue_write_rx_timestamps",
            common.config_metadata(args), results, error,
        )
        return 1
    except Exception as error:
        print(f"实验失败: {error}", file=sys.stderr)
        common.write_partial_outputs(
            output, rows, "socketcan_enqueue_write_rx_timestamps",
            common.config_metadata(args), results, error,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
