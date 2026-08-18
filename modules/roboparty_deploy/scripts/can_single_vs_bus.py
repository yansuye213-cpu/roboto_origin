#!/usr/bin/env python3
"""Compare one active motor on a CAN bus with all six motors commanded."""

from __future__ import annotations

import argparse
import sys

import can_latency_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单电机通信 vs 同总线六电机通信正弦对照")
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

    print("实验 1：单电机 vs 六电机同总线负载")
    print(f"将测试 {len(selected)} 个关节，每个关节自动完成 single 和 six_motor 两个 case。")
    estimated = len(selected) * 2 * (common.sine_duration(args) + args.case_pause)
    print(f"预计自动运行约 {estimated:.1f} s（不含电机初始化）。")
    for joint in selected:
        print(f"  {joint.name}: {joint.motor.interface}/ID{joint.motor.motor_id}")
    if args.dry_run:
        print("Dry run 通过；未连接硬件。")
        return 0

    rows, results = [], []
    output = args.output or common.default_output_path("single_vs_bus")
    try:
        common.confirm_suspended()
        motors_py = common.import_motors_py()
        for number, active in enumerate(selected, 1):
            bus = buses[active.motor.interface]
            print(f"\n[{number}/{len(selected)}] {active.name}: single")
            case_rows, metrics = common.run_sine_case(
                motors_py, [active], [active], active, "single_vs_bus", "single", args,
            )
            rows.extend(case_rows)
            results.append({"joint": active.name, "interface": active.motor.interface,
                            "motor_id": active.motor.motor_id, "case": "single", **metrics})

            print(f"[{number}/{len(selected)}] {active.name}: six_motor")
            case_rows, metrics = common.run_sine_case(
                motors_py, bus, bus, active, "single_vs_bus", "six_motor", args,
            )
            rows.extend(case_rows)
            results.append({"joint": active.name, "interface": active.motor.interface,
                            "motor_id": active.motor.motor_id, "case": "six_motor", **metrics})

        by_key = {(r["joint"], r["case"]): r for r in results}
        comparisons = []
        for joint in selected:
            single = by_key[(joint.name, "single")]
            loaded = by_key[(joint.name, "six_motor")]
            comparisons.append({
                "joint": joint.name,
                "phase_delay_increase_ms": loaded.get("phase_delay_ms", float("nan")) - single.get("phase_delay_ms", float("nan")),
                "gain_change": loaded.get("gain", float("nan")) - single.get("gain", float("nan")),
                "rmse_increase_rad": loaded.get("tracking_rmse_rad", float("nan")) - single.get("tracking_rmse_rad", float("nan")),
            })
        common.write_outputs(output, rows, {
            "experiment": "single_motor_vs_six_motor_bus_load",
            "configuration": common.config_metadata(args), "results": results,
            "comparisons": comparisons,
        })
        return 0
    except (KeyboardInterrupt, common.SafetyAbort) as error:
        print(f"\n安全中止: {error}", file=sys.stderr)
        common.write_partial_outputs(
            output, rows, "single_motor_vs_six_motor_bus_load",
            common.config_metadata(args), results, error,
        )
        return 1
    except Exception as error:
        print(f"实验失败: {error}", file=sys.stderr)
        common.write_partial_outputs(
            output, rows, "single_motor_vs_six_motor_bus_load",
            common.config_metadata(args), results, error,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
