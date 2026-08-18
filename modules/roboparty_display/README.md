# Roboparty Display

Runtime display components for Roboparty robots.

## Eye display

The `eyecontrol/` directory contains the Pygame eye animation and its image
assets. The launcher and systemd unit live in `../roboparty_deploy/tools/` so
runtime code stays separate from deployment configuration.

Install the runtime dependency with:

```bash
sudo apt install python3-pygame x11-xserver-utils
```

Install and enable the service with:

```bash
sudo install -m 0644 ../roboparty_deploy/tools/eyecontrol.service /etc/systemd/system/eyecontrol.service
sudo systemctl daemon-reload
sudo systemctl enable eyecontrol.service
```

The service retries after startup failures, including when the display is not
connected yet. Inspect it with `systemctl status eyecontrol.service` and
`journalctl -u eyecontrol.service`.

## Autostart switches

The robot and Eye services read the same
`../roboparty_deploy/tools/roboparty-autostart.conf` file, but remain separate
systemd services. `AUTOSTART_ENABLED` controls only the robot, while
`EYE_AUTOSTART_ENABLED` controls only the Eye display.

After changing the Eye switch, apply it without restarting the robot:

```bash
sudo systemctl restart eyecontrol.service
```
