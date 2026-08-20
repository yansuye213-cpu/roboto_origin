#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - import-time failure is user-facing.
    raise SystemExit(
        "roboparty_camera_web.py requires python3-opencv and python3-numpy. "
        "Install them with: sudo apt install -y python3-opencv python3-numpy"
    ) from exc

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roboparty RealSense</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111318;
      --panel: #171a21;
      --panel-2: #1e232d;
      --text: #e8edf6;
      --muted: #9aa6b2;
      --accent: #67b7ff;
      --accent-2: #45d39e;
      --border: #2b3340;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #171a21 0%, #13161c 100%);
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
    }
    main {
      display: grid;
      gap: 16px;
      padding: 16px;
      max-width: 1280px;
      margin: 0 auto;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .frame {
      display: block;
      width: 100%;
      max-width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #000;
    }
    .bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      background: var(--panel-2);
      border-top: 1px solid var(--border);
      flex-wrap: wrap;
    }
    .status {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      color: #0b1118;
      background: var(--accent);
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.05); }
    button:active { transform: translateY(1px); }
    .hint {
      color: var(--muted);
      font-size: 12px;
    }
    .grid {
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(0, 1fr);
    }
    @media (min-width: 1100px) {
      .grid {
        grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
      }
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      word-break: break-all;
    }
  </style>
</head>
<body>
  <header>
    <h1>Roboparty RealSense</h1>
    <div class="status mono" id="status">connecting...</div>
  </header>
  <main>
    <div class="grid">
      <section class="panel">
        <img class="frame" src="/stream.mjpg" alt="Live RGB stream">
        <div class="bar">
          <div class="hint">Live RGB stream</div>
          <button id="capture-btn" type="button">Capture</button>
        </div>
      </section>
      <section class="panel">
        <div class="bar">
          <div class="hint">Snapshot bundle</div>
          <div class="status mono" id="save-dir">...</div>
        </div>
        <div style="padding: 16px">
          <p class="hint" style="margin-top:0">
            Press <span class="mono">C</span> or click <span class="mono">Capture</span> to save a local RGB + depth snapshot.
            The browser will download a zip bundle while the robot also keeps a copy on disk.
          </p>
          <p class="hint" style="margin-bottom:0">
            Stream endpoint: <span class="mono">/stream.mjpg</span><br>
            Snapshot endpoint: <span class="mono">/snapshot.zip</span>
          </p>
        </div>
      </section>
    </div>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const saveDirEl = document.getElementById('save-dir');
    const captureBtn = document.getElementById('capture-btn');

    async function updateStatus() {
      try {
        const response = await fetch('/status', { cache: 'no-store' });
        const payload = await response.json();
        const colorState = payload.color_ready ? 'rgb ready' : 'rgb waiting';
        const depthState = payload.depth_ready ? 'depth ready' : 'depth waiting';
        const captureState = payload.last_capture_id ? `last ${payload.last_capture_id}` : 'no capture yet';
        statusEl.textContent = `${colorState} | ${depthState} | ${captureState}`;
        saveDirEl.textContent = payload.save_dir || '';
      } catch (error) {
        statusEl.textContent = 'offline';
      }
    }

    async function captureSnapshot() {
      captureBtn.disabled = true;
      try {
        const response = await fetch('/snapshot.zip', {
          method: 'POST',
          cache: 'no-store',
        });
        if (!response.ok) {
          throw new Error(`capture failed: ${response.status}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const name = response.headers.get('X-Capture-Name') || 'roboparty_capture.zip';
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } catch (error) {
        console.error(error);
        alert(`Capture failed: ${error.message}`);
      } finally {
        captureBtn.disabled = false;
      }
    }

    captureBtn.addEventListener('click', captureSnapshot);
    document.addEventListener('keydown', (event) => {
      if (event.key.toLowerCase() === 'c' && !event.repeat) {
        captureSnapshot();
      }
    });

    updateStatus();
    setInterval(updateStatus, 1000);
  </script>
</body>
</html>
"""


@dataclass
class ColorFrame:
    seq: int = 0
    stamp_ns: int = 0
    encoding: str = ""
    width: int = 0
    height: int = 0
    jpeg: bytes | None = None


@dataclass
class DepthFrame:
    seq: int = 0
    stamp_ns: int = 0
    encoding: str = ""
    width: int = 0
    height: int = 0
    data: np.ndarray | None = None


class CameraWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], node: "RobopartyCameraWebNode"):
        super().__init__(server_address, handler_class)
        self.node = node


class CameraRequestHandler(BaseHTTPRequestHandler):
    server: CameraWebServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            payload = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/status":
            self._write_json(self.server.node.status_payload())
            return

        if path == "/stream.mjpg":
            self._stream_mjpeg()
            return

        if path == "/latest.jpg":
            frame = self.server.node.get_color_jpeg()
            if frame is None:
                self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No RGB frame available yet")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            return

        if path == "/snapshot.zip":
            self._send_snapshot()
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/snapshot.zip":
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > 0:
                self.rfile.read(length)
            self._send_snapshot()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _stream_mjpeg(self) -> None:
        boundary = b"frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_seq = -1
        while not self.server.node.shutdown_event.is_set():
            frame = self.server.node.wait_for_color_frame(last_seq, timeout=1.0)
            if frame is None:
                continue
            last_seq = frame.seq
            try:
                self.wfile.write(b"--" + boundary + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame.jpeg or b'')}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame.jpeg or b"")
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def _send_snapshot(self) -> None:
        try:
            capture = self.server.node.capture_snapshot()
        except RuntimeError as exc:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        payload = capture.zip_bytes
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{capture.download_name}"')
        self.send_header("X-Capture-Name", capture.download_name)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclass
class SnapshotBundle:
    download_name: str
    zip_bytes: bytes


class RobopartyCameraWebNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_web_server")

        self.declare_parameter("color_topic", "/camera/d455/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/d455/aligned_depth_to_color/image_raw")
        self.declare_parameter("bind_host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("save_dir", str(Path.home() / "Pictures" / "roboparty_camera"))
        self.declare_parameter("jpeg_quality", 80)

        self.color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.bind_host = self.get_parameter("bind_host").get_parameter_value().string_value
        self.port = int(self.get_parameter("port").value)
        self.save_dir = Path(self.get_parameter("save_dir").get_parameter_value().string_value).expanduser()
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._condition = threading.Condition()
        self._color = ColorFrame()
        self._depth = DepthFrame()
        self._last_capture_id = ""
        self._shutdown_event = threading.Event()
        self._rgb_encoding_warned = False
        self._depth_encoding_warned = False
        self._http_server = CameraWebServer((self.bind_host, self.port), CameraRequestHandler, self)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="camera_web_http",
            daemon=True,
        )
        self._http_thread.start()

        self.create_subscription(Image, self.color_topic, self._on_color_image, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, qos_profile_sensor_data)

        self.get_logger().info(
            f"Serving camera page on http://{self.bind_host}:{self.port}/ "
            f"(topics: {self.color_topic}, {self.depth_topic})"
        )
        self.get_logger().info(f"Snapshots are saved under {self.save_dir}")

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
        if self._http_thread.is_alive():
            self._http_thread.join(timeout=2.0)
        return super().destroy_node()

    @property
    def shutdown_event(self) -> threading.Event:
        return self._shutdown_event

    def _on_color_image(self, msg: Image) -> None:
        frame = self._image_to_bgr(msg)
        if frame is None:
            return

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warning("Failed to encode RGB frame to JPEG")
            return

        jpeg_bytes = encoded.tobytes()
        with self._condition:
            self._color.seq += 1
            self._color.stamp_ns = self._stamp_to_ns(msg.header.stamp)
            self._color.encoding = msg.encoding
            self._color.width = int(msg.width)
            self._color.height = int(msg.height)
            self._color.jpeg = jpeg_bytes
            self._condition.notify_all()

    def _on_depth_image(self, msg: Image) -> None:
        frame = self._image_to_depth(msg)
        if frame is None:
            return

        with self._condition:
            self._depth.seq += 1
            self._depth.stamp_ns = self._stamp_to_ns(msg.header.stamp)
            self._depth.encoding = msg.encoding
            self._depth.width = int(msg.width)
            self._depth.height = int(msg.height)
            self._depth.data = frame
            self._condition.notify_all()

    def get_color_jpeg(self) -> bytes | None:
        with self._condition:
            if self._color.jpeg is None:
                return None
            return bytes(self._color.jpeg)

    def wait_for_color_frame(self, last_seq: int, timeout: float) -> ColorFrame | None:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._shutdown_event.is_set() or self._color.seq != last_seq,
                timeout=timeout,
            )
            if not ready or self._shutdown_event.is_set() or self._color.jpeg is None:
                return None
            return ColorFrame(
                seq=self._color.seq,
                stamp_ns=self._color.stamp_ns,
                encoding=self._color.encoding,
                width=self._color.width,
                height=self._color.height,
                jpeg=bytes(self._color.jpeg),
            )

    def capture_snapshot(self) -> SnapshotBundle:
        with self._condition:
            if self._color.jpeg is None:
                raise RuntimeError("RGB frame is not ready yet")
            if self._depth.data is None:
                raise RuntimeError("Depth frame is not ready yet")

            color_frame = bytes(self._color.jpeg)
            depth_frame = self._depth.data.copy()
            capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            color_meta = {
                "seq": self._color.seq,
                "stamp_ns": self._color.stamp_ns,
                "encoding": self._color.encoding,
                "width": self._color.width,
                "height": self._color.height,
            }
            depth_meta = {
                "seq": self._depth.seq,
                "stamp_ns": self._depth.stamp_ns,
                "encoding": self._depth.encoding,
                "width": self._depth.width,
                "height": self._depth.height,
            }

        capture_dir = self.save_dir / capture_id
        capture_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = capture_dir / "rgb.jpg"
        depth_path = capture_dir / "depth.png"
        meta_path = capture_dir / "metadata.json"

        rgb_path.write_bytes(color_frame)

        ok, depth_png = cv2.imencode(".png", np.ascontiguousarray(depth_frame))
        if not ok:
            raise RuntimeError("Failed to encode depth frame")
        depth_bytes = depth_png.tobytes()
        depth_path.write_bytes(depth_bytes)

        metadata = {
            "capture_id": capture_id,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "color_topic": self.color_topic,
            "depth_topic": self.depth_topic,
            "color": color_meta,
            "depth": depth_meta,
            "jpeg_quality": self.jpeg_quality,
            "files": {
                "rgb": "rgb.jpg",
                "depth": "depth.png",
                "metadata": "metadata.json",
            },
        }
        meta_bytes = (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        meta_path.write_bytes(meta_bytes)

        zip_name = f"{capture_id}.zip"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("rgb.jpg", color_frame)
            archive.writestr("depth.png", depth_bytes)
            archive.writestr("metadata.json", meta_bytes)

        with self._condition:
            self._last_capture_id = capture_id

        self.get_logger().info(f"Saved snapshot to {capture_dir}")
        return SnapshotBundle(download_name=zip_name, zip_bytes=zip_buffer.getvalue())

    def status_payload(self) -> dict[str, Any]:
        with self._condition:
            return {
                "color_ready": self._color.jpeg is not None,
                "depth_ready": self._depth.data is not None,
                "color_seq": self._color.seq,
                "depth_seq": self._depth.seq,
                "last_capture_id": self._last_capture_id,
                "save_dir": str(self.save_dir),
                "color_topic": self.color_topic,
                "depth_topic": self.depth_topic,
                "port": self.port,
            }

    @staticmethod
    def _stamp_to_ns(stamp: Any) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _image_to_bgr(self, msg: Image) -> np.ndarray | None:
        encoding = msg.encoding.lower()
        if encoding not in {"rgb8", "bgr8"}:
            if not self._rgb_encoding_warned:
                self._rgb_encoding_warned = True
                self.get_logger().warning(
                    f"Unsupported RGB encoding '{msg.encoding}'. Expected rgb8 or bgr8."
                )
            return None

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            raw = raw[:, : msg.width * 3]
            frame = raw.reshape(msg.height, msg.width, 3)
        except ValueError:
            self.get_logger().warning("Failed to reshape RGB frame")
            return None

        if encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(frame)

    def _image_to_depth(self, msg: Image) -> np.ndarray | None:
        encoding = msg.encoding.lower()
        if encoding not in {"16uc1", "mono16"}:
            if not self._depth_encoding_warned:
                self._depth_encoding_warned = True
                self.get_logger().warning(
                    f"Unsupported depth encoding '{msg.encoding}'. Expected 16UC1."
                )
            return None

        if msg.step % 2 != 0:
            self.get_logger().warning("Depth frame step is not aligned to uint16 data")
            return None

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
            depth = raw[:, : msg.width]
        except ValueError:
            self.get_logger().warning("Failed to reshape depth frame")
            return None

        return np.ascontiguousarray(depth)


def main() -> None:
    rclpy.init()
    node = RobopartyCameraWebNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
