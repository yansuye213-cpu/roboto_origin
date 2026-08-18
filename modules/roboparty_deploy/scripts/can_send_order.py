#!/usr/bin/env python3
"""Compare ascending and descending motor command order on each leg CAN bus."""

from __future__ import annotations

import argparse
import sys

import can_latency_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ID1->ID6 vs ID6->ID1 发送顺序正弦对照")
    common.add_common_arguments(parser, sine=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        common.validate_common_args(args, sine=True)
        joints, _ = common.load_configuration(args)
        selected = common.selected_leg_joints(joints, args.joint)
        buses = common.leg_buses(joints)
    except Exception as error:
        print(f"参数/配置错误: {error}", file=sys.stderr)
        return 2

    print("实验 2：ID 正序 vs 逆序发送；两种 case 都使能并控制同总线六电机。")
    print(f"将测试 {len(selected)} 个关节，每个关节自动完成 forward/reverse。")
    estimated = len(selected) * 2 * (common.sine_duration(args) + args.case_pause)
    print(f"预计自动运行约 {estimated:.1f} s（不含电机初始化）。")
    for joint in selected:
        print(f"  {joint.name}: {joint.motor.interface}/ID{joint.motor.motor_id}")
    if args.dry_run:
        print("Dry run 通过；未连接硬件。")
        return 0

    rows, results = [], []
    output = args.output or common.default_output_path("send_order")
    try:
        common.confirm_suspended()
        motors_py = common.import_motors_py()
        for number, active in enumerate(selected, 1):
            forward = buses[active.motor.interface]
            for case, order in (("forward", forward), ("reverse", list(reversed(forward)))):
                print(f"\n[{number}/{len(selected)}] {active.name}: {case}")
                case_rows, metrics = common.run_sine_case(
                    motors_py, forward, order, active, "send_order", case, args,
                )
                rows.extend(case_rows)
                results.append({
                    "joint": active.name, "interface": active.motor.interface,
                    "motor_id": active.motor.motor_id, "case": case,
                    "command_rank": next(i for i, joint in enumerate(order) if joint.name == active.name),
                    **metrics,
                })

        by_key = {(r["joint"], r["case"]): r for r in results}
        comparisons = []
        for joint in selected:
            forward = by_key[(joint.name, "forward")]
            reverse = by_key[(joint.name, "reverse")]
            comparisons.append({
                "joint": joint.name, "forward_rank": forward["command_rank"],
                "reverse_rank": reverse["command_rank"],
                "reverse_minus_forward_delay_ms": reverse.get("phase_delay_ms", float("nan")) - forward.get("phase_delay_ms", float("nan")),
                "reverse_minus_forward_gain": reverse.get("gain", float("nan")) - forward.get("gain", float("nan")),
            })
        common.write_outputs(output, rows, {
            "experiment": "forward_vs_reverse_send_order",
            "configuration": common.config_metadata(args), "results": results,
            "comparisons": comparisons,
        })
        return 0
    except (KeyboardInterrupt, common.SafetyAbort) as error:
        print(f"\n安全中止: {error}", file=sys.stderr)
        common.write_partial_outputs(
            output, rows, "forward_vs_reverse_send_order",
            common.config_metadata(args), results, error,
        )
        return 1
    except Exception as error:
        print(f"实验失败: {error}", file=sys.stderr)
        common.write_partial_outputs(
            output, rows, "forward_vs_reverse_send_order",
            common.config_metadata(args), results, error,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
