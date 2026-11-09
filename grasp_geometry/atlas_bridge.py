# SPDX-License-Identifier: MulanPSL-2.0
"""grasp_geometry 的 atlas 桥：能力声明 + 生命周期 + 一次 locate 的编排。

本包只做"感知服务"该做的事：从相机原语取最新一帧 RGB-D，跑深度几何，
把**相机系**的几何量交出去。它不碰机械臂、不碰底盘、不做像素→基座的
坐标变换（那一步由调用方乘一次 TF/标定）。

能力面：
    robonix/service/perception/grasp_geometry/locate   （MCP + gRPC）
生命周期：
    CMD_INIT      解析 config、解析相机话题、建订阅、等首帧
    CMD_ACTIVATE  幂等：已就绪就什么都不做
    CMD_SHUTDOWN  放订阅

与 service/perception/grasp_pose 的区别：那个吃 bbox、用 2D 单应 + 固定桌面高
给一个垂直抓取位姿；本服务吃"目标名/可选框"、用深度图分离承托面与物体顶面，
输出几何量，不依赖固定台面高度。
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np

from robonix_api import ATLAS, Err, Ok, Service

from grasp_geometry_mcp import Locate_Request, Locate_Response

from .camera import CameraFeed, FrameUnavailable
from .config import Config
from .core.detectors import VlmDetector
from .core.locate import locate as locate_core

log = logging.getLogger("grasp_geometry")

NAMESPACE = "robonix/service/perception/grasp_geometry"
LOCATE_ID = NAMESPACE + "/locate"

service = Service(id="grasp_geometry", namespace=NAMESPACE)

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _PKG_ROOT / "rbnx-build" / "data" / "grasp_geometry_logs"


class _Runtime:
    """进程内状态。只有 on_init 写它；locate 只读。"""

    config: Config | None = None
    camera: CameraFeed | None = None
    started: bool = False
    locate_lock: threading.Lock = threading.Lock()
    init_lock: threading.Lock = threading.Lock()


_state = _Runtime()


# ─────────────────────────────────────────────────────────── 生命周期


@service.on_init
def on_init(raw_config: dict) -> Ok | Err:
    """解析配置、绑定相机、开始缓存帧。可重入（幂等）。"""
    with _state.init_lock:
        if _state.started:
            log.info("on_init 重入：已就绪，只更新配置不重建订阅")
            return Ok()
        try:
            config = Config.from_dict(raw_config)
        except ValueError as error:
            log.error("配置非法：%s", error)
            return Err(f"配置非法：{error}")

        _state.config = config
        camera = CameraFeed(service, config)
        try:
            camera.start()
        except Exception as error:                   # noqa: BLE001 —— INIT 失败要如实报
            log.error("相机绑定失败：%s", error)
            return Err(f"相机绑定失败：{type(error).__name__}: {error}")
        _state.camera = camera
        _state.started = True
        log.info("grasp_geometry 就绪：top_point_source=%s depth_units_mm=%s log_dir=%s",
                 config.top_point_source, config.depth_units_mm,
                 str(log_dir(config)))
        return Ok()


@service.on_activate
def on_activate() -> Ok | Err:
    """服务在 INIT 里已经拿到热资源；ACTIVATE 幂等。"""
    if not _state.started:
        return Err("grasp_geometry 未初始化：先收到 CMD_INIT")
    return Ok()


@service.on_shutdown
def on_shutdown() -> None:
    if _state.camera is not None:
        try:
            _state.camera.stop()
        except Exception:                            # noqa: BLE001
            log.exception("停相机订阅时出错")
        _state.camera = None
    _state.started = False


# ─────────────────────────────────────────────────────────── locate


def _parse_box(box_text: str) -> list[float] | None:
    text = (box_text or "").strip()
    if not text:
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"box 不是合法 JSON：{error}") from error
    if not isinstance(raw, list) or len(raw) not in (4, 5):
        raise ValueError("box 要是 4 或 5 个数的数组 [x0,y0,x1,y1(,angle_deg)]")
    box = [float(v) for v in raw]
    if len(box) == 4:
        box.append(0.0)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"box 的 x1/y1 必须大于 x0/y0：{box}")
    return box


def _locate_impl(target: str, box_text: str) -> dict:
    """一次完整的 locate。返回 {"success","message","result_json"}。"""
    config = _state.config
    if config is None or not _state.started:
        return _failure("服务还没初始化（没有收到 CMD_INIT）")

    if not _state.locate_lock.acquire(blocking=False):
        return _failure("上一次 locate 还没处理完（服务一次只处理一个请求）")
    try:
        started = time.monotonic()
        try:
            box = _parse_box(box_text)
        except ValueError as error:
            return _failure(f"请求非法：{error}")

        target = (target or "").strip()
        if box is None and not target:
            return _failure("请求非法：target 和 box 至少给一个（不给 box 时用 target 做检测）")

        try:
            rgb, depth, frame_id = _state.camera.latest()
        except FrameUnavailable as error:
            return _failure(str(error))

        request_id = uuid.uuid4().hex[:12]
        detector = None
        if box is None:
            if not (config.vlm_base_url and config.vlm_api_key and config.vlm_model):
                return _failure(
                    "没给 box 且 VLM 凭据不全（config.vlm_base_url/vlm_api_key/vlm_model）："
                    "要么补全凭据让服务自己检测，要么调用时给 box")
            detector = VlmDetector(
                name=target, base_url=config.vlm_base_url,
                api_key=config.vlm_api_key, model=config.vlm_model,
                timeout_s=config.vlm_timeout_s,
            )

        sink: dict = {}
        try:
            measured = locate_core(
                rgb=rgb, depth=depth,
                camera_matrix=np.asarray(config.camera_matrix, dtype=float),
                dist_coeffs=np.asarray(config.dist_coeffs, dtype=float),
                target=target, box=box, detector=detector,
                top_point_source=config.top_point_source,
                debug_sink=sink,
            )
        except Exception as error:                   # noqa: BLE001 —— 算不出来要如实报
            detail = traceback.format_exc()
            _write_artifacts(request_id, config, rgb, depth, target, box,
                             result=None, detector=detector, error=detail)
            log.error("→ %s locate 抛异常：%s: %s", request_id, type(error).__name__, error)
            return _failure(f"{type(error).__name__}: {error}", request_id=request_id)

        elapsed_ms = (time.monotonic() - started) * 1000.0
        if not measured.get("ok"):
            why = str(measured.get("why") or "这一眼没量出来")
            _write_artifacts(request_id, config, rgb, depth, target, box,
                             result=measured, detector=detector)
            log.warning("→ %s 没量出来：%s（%.0f ms）", request_id, why, elapsed_ms)
            return _failure(why, request_id=request_id)

        payload = dict(measured)
        payload["frame_id"] = frame_id or config.frame_id
        payload["request_id"] = request_id
        payload["elapsed_ms"] = round(elapsed_ms, 1)
        payload["frame_shape"] = {"rgb": list(rgb.shape), "depth": list(depth.shape)}
        _write_artifacts(request_id, config, rgb, depth, target, box,
                         result=payload, detector=detector)
        log.info("→ %s ok 顶面=%s 宽=%s 用时=%.0fms",
                 request_id, _fmt_point(payload.get("top_point")),
                 payload.get("footprint_width"), elapsed_ms)
        return {
            "success": True,
            "message": "",
            "result_json": json.dumps(payload, ensure_ascii=False, default=_json_default),
        }
    finally:
        _state.locate_lock.release()


def _failure(message: str, request_id: str = "") -> dict:
    payload = {"ok": False, "why": message, "request_id": request_id}
    return {"success": False, "message": message,
            "result_json": json.dumps(payload, ensure_ascii=False, default=_json_default)}


# ─────────────────────────────────────────────────────────── 能力面


@service.mcp(LOCATE_ID)
def locate_mcp(request: Locate_Request) -> Locate_Response:
    """量一次目标物体的抓取几何（相机系）。

    服务自己从相机原语取最新一帧 RGB-D；`target` 是开放词表目标名（服务用 VLM
    检测），`box` 给了就跳过检测。返回 `result_json`：
    top_point / support{point,normal} / long_axis / closing / footprint_width /
    bbox / frame_id / request_id / diagnostics。

    结果是**相机系**；要落到基座系由调用方乘一次标定/TF。量不出来时 success=false，
    不会返回凑合的值——调用方应如实报错或换个观察位再看，不要拿旧值兜底。
    """
    result = _locate_impl(request.target, request.box)
    return Locate_Response(success=result["success"],
                           message=result["message"],
                           result_json=result["result_json"])


@service.grpc(LOCATE_ID)
def locate_grpc(request, context):                   # noqa: ARG001 —— gRPC 约定
    """同 locate_mcp，gRPC 传输。"""
    from grasp_geometry_pb2 import Locate_Response as PbLocateResponse

    result = _locate_impl(request.target, request.box)
    return PbLocateResponse(success=result["success"],
                            message=result["message"],
                            result_json=result["result_json"])


# ─────────────────────────────────────────────────────────── 留档


def log_dir(config: Config) -> Path:
    return Path(config.log_dir).expanduser() if config.log_dir else _DEFAULT_LOG_DIR


def _write_artifacts(request_id: str, config: Config, rgb, depth, target, box,
                     *, result: dict | None, detector, error: str | None = None) -> None:
    """每次请求留一份现场：请求参数 + 原始帧 + 结果/异常。失败不留空目录。"""
    if config.log_keep <= 0:
        return
    directory = log_dir(config) / request_id
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "request.json").write_text(json.dumps({
            "target": target,
            "box": box,
            "top_point_source": config.top_point_source,
            "frame_shape": {"rgb": list(rgb.shape), "depth": list(depth.shape)},
        }, ensure_ascii=False, indent=2))
        cv2.imwrite(str(directory / "rgb.jpg"), rgb)
        np.save(directory / "depth.npy", np.asarray(depth, dtype=np.uint16))
        if detector is not None and hasattr(detector, "diagnostics"):
            (directory / "detector.json").write_text(
                json.dumps(detector.diagnostics(), ensure_ascii=False, indent=2,
                           default=_json_default))
        if error is not None:
            (directory / "error.txt").write_text(error)
        (directory / "result.json").write_text(json.dumps(
            result, ensure_ascii=False, indent=2, default=_json_default))
    except Exception:                                # noqa: BLE001 —— 留档失败不阻断测量
        log.exception("写留档失败（不影响这一次结果）：%s", directory)
        return
    _prune_artifacts(log_dir(config), config.log_keep)


def _prune_artifacts(root: Path, keep: int) -> None:
    try:
        directories = sorted(
            (p for p in root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in directories[:-keep]:
            shutil.rmtree(stale, ignore_errors=True)
    except OSError:
        pass


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _fmt_point(point) -> str:
    try:
        return "%.4f,%.4f,%.4f" % tuple(float(v) for v in point[:3])
    except (TypeError, ValueError, IndexError):
        return str(point)


def main() -> int:
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
