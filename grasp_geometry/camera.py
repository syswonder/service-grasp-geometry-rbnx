# SPDX-License-Identifier: MulanPSL-2.0
"""相机取帧：从 atlas 解析相机原语的话题，缓存"最新一帧"。

为什么服务自己订阅而不是让调用方传图：
- 服务要保证"量的是现在"，调用方传图容易拿到旧帧；
- 抓取技能就不必再碰相机原语（少一条跨包数据通路），相机坏了也只在一个地方坏。

话题名一律经 atlas 解析（`camera_provider_id` 可指定实例）；解析不到就报错，
**不退回写死的话题名**——写死的默认值在别的部署里十有八九对不上，挂上去是静默失联。
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from robonix_api import ATLAS

log = logging.getLogger("grasp_geometry.camera")

RGB_CONTRACT = "robonix/primitive/camera/rgb"
DEPTH_CONTRACT = "robonix/primitive/camera/depth"


class FrameUnavailable(RuntimeError):
    """取不到一帧可用的 RGB-D。"""


def _resolve_ros2_topic(service, contract_id: str, provider_id: str,
                        timeout_s: float = 60.0) -> str:
    """问 atlas 要一个 ROS2 话题名；解析不到就抛。"""
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            view = ATLAS.find_unique_capability(
                contract_id=contract_id, transport="ros2",
                provider_id=provider_id or "",
            )
            channel = service.connect_capability(view, contract_id, "ros2")
            endpoint = str(channel.endpoint or "")
            try:
                channel.close()
            except Exception:                       # noqa: BLE001
                pass
            if endpoint:
                return endpoint
            last_error = f"{contract_id}: atlas 给的端点为空"
        except Exception as error:                  # noqa: BLE001 —— 还没注册上就继续等
            last_error = f"{contract_id}: {type(error).__name__}: {error}"
        time.sleep(0.5)
    raise RuntimeError(f"atlas 上解析不到相机话题（{last_error}）")


def _image_to_numpy(message) -> np.ndarray:
    """sensor_msgs/Image → ndarray（支持 rgb8/bgr8/mono8/16UC1/32FC1）。"""
    height, width = int(message.height), int(message.width)
    encoding = str(message.encoding or "").lower()
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        image = raw.reshape(height, message.step // 3, 3)[:, :width, :]
        return image if encoding == "bgr8" else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "mono8":
        mono = raw.reshape(height, message.step)[:, :width]
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    if encoding in ("16uc1", "16sc1", "mono16"):
        return raw.view(np.uint16).reshape(height, message.step // 2)[:, :width]
    if encoding == "32fc1":
        return raw.view(np.float32).reshape(height, message.step // 4)[:, :width]
    raise FrameUnavailable(f"不支持的图像编码 {encoding!r}（支持 rgb8/bgr8/mono8/16UC1/32FC1）")


class CameraFeed:
    """订阅相机原语，维护"最新一帧 RGB + 深度"。线程安全。"""

    def __init__(self, service, config):
        self._service = service
        self._config = config
        self._lock = threading.Lock()
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._rgb_seen = 0.0
        self._depth_seen = 0.0
        self._frame_id = ""
        self._subscriptions: list = []

    # ------------------------------------------------------------ 生命周期

    def start(self) -> None:
        if self._config.rgb_topic and self._config.depth_topic:
            rgb_topic, depth_topic = self._config.rgb_topic, self._config.depth_topic
            log.info("相机话题来自配置：rgb=%s depth=%s", rgb_topic, depth_topic)
        else:
            provider = self._config.camera_provider_id
            rgb_topic = _resolve_ros2_topic(self._service, RGB_CONTRACT, provider)
            depth_topic = _resolve_ros2_topic(self._service, DEPTH_CONTRACT, provider)
            log.info("相机话题来自 atlas（provider=%r）：rgb=%s depth=%s",
                     provider or "<唯一>", rgb_topic, depth_topic)

        self._subscriptions.append(self._service.create_subscription(
            RGB_CONTRACT, topic=rgb_topic, msg_type="sensor_msgs/msg/Image",
            callback=self._on_rgb, qos="best_effort", declare=False,
        ))
        self._subscriptions.append(self._service.create_subscription(
            DEPTH_CONTRACT, topic=depth_topic, msg_type="sensor_msgs/msg/Image",
            callback=self._on_depth, qos="best_effort", declare=False,
        ))
        # 等第一帧（超时只告警：相机可能比 INIT 慢，locate 会如实报"没帧"）
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self._rgb is not None and self._depth is not None:
                log.info("首帧已到：rgb=%s depth=%s", self._rgb.shape, self._depth.shape)
                return
            time.sleep(0.1)
        log.warning("20 s 内没等到完整 RGB-D 帧（服务先起来，帧到了自动可用）")

    def stop(self) -> None:
        for sub in self._subscriptions:
            try:
                self._service_remove(sub)
            except Exception:                        # noqa: BLE001
                pass
        self._subscriptions.clear()

    def _service_remove(self, sub) -> None:
        # 框架没有暴露 destroy_subscription；进程退出时 node 一起回收。
        # 这里只把引用放开，避免重复持有。
        pass

    # ------------------------------------------------------------ 回调

    def _on_rgb(self, message) -> None:
        try:
            image = _image_to_numpy(message)
        except Exception:                            # noqa: BLE001
            return
        with self._lock:
            self._rgb = image
            self._rgb_seen = time.monotonic()
            if not self._frame_id:
                self._frame_id = str(getattr(message.header, "frame_id", "") or "")

    def _on_depth(self, message) -> None:
        try:
            image = _image_to_numpy(message)
        except Exception:                            # noqa: BLE001
            return
        with self._lock:
            self._depth = image
            self._depth_seen = time.monotonic()

    # ------------------------------------------------------------ 取帧

    def latest(self) -> tuple[np.ndarray, np.ndarray, str]:
        """取最新一帧（深拷贝）。没有/过期/尺寸不一致 → FrameUnavailable。"""
        with self._lock:
            rgb, depth = self._rgb, self._depth
            rgb_seen, depth_seen = self._rgb_seen, self._depth_seen
            frame_id = self._frame_id
        if rgb is None or depth is None:
            raise FrameUnavailable("还没有收到完整的 RGB-D 帧：检查相机原语是否在发数据")
        now = time.monotonic()
        stale = self._config.frame_stale_s
        if now - rgb_seen > stale:
            raise FrameUnavailable(
                f"RGB 帧已停更 {now - rgb_seen:.1f} s（阈值 {stale:.1f} s）")
        if now - depth_seen > stale:
            raise FrameUnavailable(
                f"深度帧已停更 {now - depth_seen:.1f} s（阈值 {stale:.1f} s）")
        rgb = np.array(rgb, copy=True)
        depth = np.array(depth, copy=True)
        if depth.shape[:2] != rgb.shape[:2]:
            raise FrameUnavailable(
                f"深度 {depth.shape[:2]} 与彩色 {rgb.shape[:2]} 尺寸不一致（需要对齐深度）")
        depth = _depth_to_mm(depth, self._config.depth_units_mm)
        return rgb, depth, frame_id or self._config.frame_id


def _depth_to_mm(depth: np.ndarray, units_mm: float) -> np.ndarray:
    """统一成 uint16 毫米（内部核心的约定）。"""
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        # 32FC1 约定是米
        mm = np.nan_to_num(depth.astype(np.float64), nan=0.0) * 1000.0
    else:
        mm = depth.astype(np.float64) * float(units_mm)
    mm = np.clip(mm, 0.0, 65535.0)
    return mm.astype(np.uint16)
