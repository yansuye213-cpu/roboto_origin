#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${ROBOPARTY_AUTOSTART_CONFIG:-"$SCRIPT_DIR/roboparty-autostart.conf"}

if [ ! -r "$CONFIG_FILE" ]; then
    echo "无法读取自启动配置: $CONFIG_FILE" >&2
    exit 1
fi

# 机器人和 Eye 服务共享配置文件，但分别校验和使用自己的配置项。
# shellcheck disable=SC1090
source "$CONFIG_FILE"

EYE_AUTOSTART_ENABLED=${EYE_AUTOSTART_ENABLED:-0}
EYE_PROGRAM_DIR=${EYE_PROGRAM_DIR:-/home/limrobot/Project/roboparty_xlong/roboto_origin/modules/roboparty_display/eyecontrol}
EYE_SCRIPT=${EYE_SCRIPT:-$EYE_PROGRAM_DIR/eye_ccontrol.py}
EYE_DISPLAY=${EYE_DISPLAY:-:0}
EYE_XAUTHORITY=${EYE_XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
EYE_STARTUP_TIMEOUT_SEC=${EYE_STARTUP_TIMEOUT_SEC:-60}

export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
export PYGAME_HIDE_SUPPORT_PROMPT=1

case "$EYE_AUTOSTART_ENABLED" in
    0)
        echo "Eye 自启动开关已关闭，不启动眼睛动画。"
        exit 0
        ;;
    1) ;;
    *)
        echo "EYE_AUTOSTART_ENABLED 只能是 0 或 1" >&2
        exit 1
        ;;
esac

echo "$(date --iso-8601=seconds) 准备启动眼睛动画"

if [[ ! "$EYE_STARTUP_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
    echo "EYE_STARTUP_TIMEOUT_SEC 必须是正整数" >&2
    exit 1
fi

if [ ! -r "$EYE_SCRIPT" ]; then
    echo "无法读取眼睛程序: $EYE_SCRIPT" >&2
    exit 1
fi

if ! /usr/bin/python3 -c 'import pygame'; then
    echo "Python 环境中未安装 pygame" >&2
    exit 1
fi

export DISPLAY="$EYE_DISPLAY"
export XAUTHORITY="$EYE_XAUTHORITY"

display_ready=0
for ((elapsed = 0; elapsed < EYE_STARTUP_TIMEOUT_SEC; elapsed++)); do
    if /usr/bin/timeout 2s /usr/bin/xset q >/dev/null 2>&1; then
        display_ready=1
        break
    fi
    sleep 1
done

if [ "$display_ready" -ne 1 ]; then
    if [ ! -r "$EYE_XAUTHORITY" ]; then
        echo "无法读取 X11 授权文件: $EYE_XAUTHORITY" >&2
    fi
    echo "${EYE_STARTUP_TIMEOUT_SEC} 秒内未能连接显示器 $EYE_DISPLAY" >&2
    exit 1
fi

cd "$EYE_PROGRAM_DIR"
echo "$(date --iso-8601=seconds) 显示器已就绪，启动 $EYE_SCRIPT"
exec /usr/bin/python3 -u "$EYE_SCRIPT"
