#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${ROBOPARTY_AUTOSTART_CONFIG:-"$SCRIPT_DIR/roboparty-autostart.conf"}

if [ ! -r "$CONFIG_FILE" ]; then
    echo "无法读取自启动配置: $CONFIG_FILE" >&2
    exit 1
fi

# 配置文件与本脚本均由机器人管理员维护。
# shellcheck disable=SC1090
source "$CONFIG_FILE"

AUTOSTART_ENABLED=${AUTOSTART_ENABLED:-0}
BUILD_ON_START=${BUILD_ON_START:-0}
ROBOT=${ROBOT:-rpo}
POLICY=${POLICY:-default}
STARTUP_DELAY_SEC=${STARTUP_DELAY_SEC:-10}

case "$AUTOSTART_ENABLED" in
    0)
        echo "Roboparty 自启动开关已关闭，不启动机器人。"
        exit 0
        ;;
    1) ;;
    *)
        echo "AUTOSTART_ENABLED 只能是 0 或 1" >&2
        exit 1
        ;;
esac

case "$BUILD_ON_START" in
    0) build_option=--no-build ;;
    1) build_option=--build ;;
    *)
        echo "BUILD_ON_START 只能是 0 或 1" >&2
        exit 1
        ;;
esac

if [[ ! "$STARTUP_DELAY_SEC" =~ ^[0-9]+$ ]]; then
    echo "STARTUP_DELAY_SEC 必须是非负整数" >&2
    exit 1
fi

if [ "$STARTUP_DELAY_SEC" -gt 0 ]; then
    echo "等待 ${STARTUP_DELAY_SEC} 秒，让设备完成初始化..."
    sleep "$STARTUP_DELAY_SEC"
fi

exec "$SCRIPT_DIR/start_robot.sh" \
    --robot "$ROBOT" \
    --policy "$POLICY" \
    "$build_option"
