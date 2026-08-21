#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
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
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


@dataclass
class ColorFrame:
    seq: int = 0
    stamp_ns: int = 0
    encoding: str = ""
    width: int = 0
    height: int = 0
    jpeg: bytes | None = None


@dataclass
class SnapshotImage:
    download_name: str
    jpeg_bytes: bytes


class CameraWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        node: "RobopartyCameraWebNode",
    ) -> None:
        super().__init__(server_address, handler_class)
        self.node = node


class CameraRequestHandler(BaseHTTPRequestHandler):
    server: CameraWebServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._write_bytes(
                HTTPStatus.OK,
                "text/html; charset=utf-8",
                self.server.node.index_html,
                cache_control="no-cache",
            )
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
            self._write_bytes(HTTPStatus.OK, "image/jpeg", frame, cache_control="no-cache")
            return

        if path == "/snapshot.jpg":
            self._send_snapshot()
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/snapshot.jpg":
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
                jpeg = frame.jpeg or b""
                self.wfile.write(b"--" + boundary + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def _send_snapshot(self) -> None:
        try:
            capture = self.server.node.capture_snapshot()
        except RuntimeError as exc:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Disposition", f'attachment; filename="{capture.download_name}"')
        self.send_header("X-Capture-Name", capture.download_name)
        self.send_header("Content-Length", str(len(capture.jpeg_bytes)))
        self.end_headers()
        self.wfile.write(capture.jpeg_bytes)

    def _write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._write_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body, cache_control="no-cache")

    def _write_bytes(
        self,
        status: HTTPStatus,
        content_type: str,
        body: bytes,
        cache_control: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if cache_control is not None:
            self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self._write_bytes(status, "application/json; charset=utf-8", body)


class RobopartyCameraWebNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_web_server")

        package_share = Path(get_package_share_directory("roboparty_camera"))
        index_path = package_share / "web" / "index.html"
        try:
            self.index_html = index_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Failed to load camera web page: {index_path}") from exc

        self.declare_parameter("color_topic", "/camera/d455/color/image_raw")
        self.declare_parameter("bind_host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("save_dir", str(Path.home() / "图片" / "limrobot_camera"))
        self.declare_parameter("jpeg_quality", 80)

        self.color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        self.bind_host = self.get_parameter("bind_host").get_parameter_value().string_value
        self.port = int(self.get_parameter("port").value)
        self.save_dir = Path(self.get_parameter("save_dir").get_parameter_value().string_value).expanduser()
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._condition = threading.Condition()
        self._color = ColorFrame()
        self._last_capture_id = ""
        self._shutdown_event = threading.Event()
        self._rgb_encoding_warned = False
        self._http_server = CameraWebServer(
            (self.bind_host, self.port), CameraRequestHandler, self
        )
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="camera_web_http",
            daemon=True,
        )
        self._http_thread.start()

        self.create_subscription(
            Image, self.color_topic, self._on_color_image, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Serving camera page on http://{self.bind_host}:{self.port}/ "
            f"(topic: {self.color_topic})"
        )
        self.get_logger().info(f"Snapshots are saved under {self.save_dir}")

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
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
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            self.get_logger().warning("Failed to encode RGB frame to JPEG")
            return

        with self._condition:
            self._color.seq += 1
            self._color.stamp_ns = self._stamp_to_ns(msg.header.stamp)
            self._color.encoding = msg.encoding
            self._color.width = int(msg.width)
            self._color.height = int(msg.height)
            self._color.jpeg = encoded.tobytes()
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

    def capture_snapshot(self) -> SnapshotImage:
        with self._condition:
            if self._color.jpeg is None:
                raise RuntimeError("RGB frame is not ready yet")
            color_frame = bytes(self._color.jpeg)

        image_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = self.save_dir / image_name
        image_path.write_bytes(color_frame)

        with self._condition:
            self._last_capture_id = image_name

        self.get_logger().info(f"Saved snapshot to {image_path}")
        return SnapshotImage(download_name=image_name, jpeg_bytes=color_frame)

    def status_payload(self) -> dict[str, Any]:
        with self._condition:
            return {
                "color_ready": self._color.jpeg is not None,
                "color_seq": self._color.seq,
                "last_capture_id": self._last_capture_id,
                "save_dir": str(self.save_dir),
                "color_topic": self.color_topic,
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
            frame = raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        except ValueError:
            self.get_logger().warning("Failed to reshape RGB frame")
            return None

        if encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(frame)


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
