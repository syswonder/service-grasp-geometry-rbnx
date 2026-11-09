# SPDX-License-Identifier: MulanPSL-2.0
"""grasp_geometry 的部署配置解析。

配置由 `rbnx boot` 经 `Driver(CMD_INIT, config_json)` 传进来，键的含义见
`config.spec`。原则和本仓库其他包一致：**缺项有默认、坏值报错**，不静默
退回一个"看起来能用"的值（尤其是内参和 VLM 凭据——错了会量歪/静默失败）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("grasp_geometry.config")

TOP_POINT_SOURCES = ("topface", "box_center")


def _as_float(value, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"config.{key} 要是数字，收到 {value!r}") from error


def _as_int(value, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"config.{key} 要是整数，收到 {value!r}") from error


def _as_matrix3(value, key: str) -> list[list[float]]:
    try:
        rows = [[float(v) for v in row] for row in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"config.{key} 要是 3x3 数字矩阵") from error
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"config.{key} 要是 3x3 数字矩阵，收到 {len(rows)} 行")
    return rows


@dataclass
class Config:
    """on_init 收到的 config 块（已校验）。"""

    # 相机绑定：provider id / 直连话题 / 内参
    camera_provider_id: str = ""
    rgb_topic: str = ""
    depth_topic: str = ""
    camera_matrix: list[list[float]] = field(default_factory=list)
    dist_coeffs: list[float] = field(default_factory=list)
    frame_id: str = ""
    depth_units_mm: float = 1.0

    # 顶面取法
    top_point_source: str = "topface"

    # 检测（只有调用方不给 box 时才需要）
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""
    vlm_timeout_s: float = 90.0

    # 运行时
    frame_stale_s: float = 2.0
    locate_timeout_s: float = 120.0
    log_dir: str = ""
    log_keep: int = 50

    # ---------------------------------------------------------------- 解析

    @classmethod
    def from_dict(cls, raw: dict | None) -> "Config":
        cfg = cls()
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config 要是对象，收到 {type(raw).__name__}")

        cfg.camera_provider_id = str(raw.get("camera_provider_id", "") or "").strip()
        cfg.rgb_topic = str(raw.get("rgb_topic", "") or "").strip()
        cfg.depth_topic = str(raw.get("depth_topic", "") or "").strip()

        if cfg.rgb_topic or cfg.depth_topic:
            if not (cfg.rgb_topic and cfg.depth_topic):
                raise ValueError(
                    "config.rgb_topic / depth_topic 要么都给（直连话题），要么都空（走 atlas 解析）")
        elif not cfg.camera_provider_id:
            log.warning(
                "config.camera_provider_id 为空：要求 atlas 上 camera/rgb + camera/depth 恰好各有一个提供方")

        if "camera_matrix" not in raw:
            raise ValueError(
                "config.camera_matrix 必填（3x3 内参）：服务不从 ROS 相机信息推内参，"
                "请把标定值放在 robot 部署仓库的 config: 里")
        cfg.camera_matrix = _as_matrix3(raw["camera_matrix"], "camera_matrix")

        dist = raw.get("dist_coeffs", []) or []
        try:
            cfg.dist_coeffs = [float(v) for v in dist]
        except (TypeError, ValueError) as error:
            raise ValueError("config.dist_coeffs 要是数字数组") from error
        if cfg.dist_coeffs and len(cfg.dist_coeffs) not in (4, 5, 8, 12, 14):
            raise ValueError(f"config.dist_coeffs 长度 {len(cfg.dist_coeffs)} 不常见（4/5/8/12/14）")

        cfg.frame_id = str(raw.get("frame_id", "") or "").strip()
        cfg.depth_units_mm = _as_float(raw.get("depth_units_mm", 1.0), "depth_units_mm")
        if cfg.depth_units_mm <= 0:
            raise ValueError("config.depth_units_mm 要 > 0")

        cfg.top_point_source = str(raw.get("top_point_source", "topface") or "topface").strip()
        if cfg.top_point_source not in TOP_POINT_SOURCES:
            raise ValueError(
                f"config.top_point_source 只能是 {TOP_POINT_SOURCES}，收到 {cfg.top_point_source!r}")

        cfg.vlm_base_url = str(raw.get("vlm_base_url", "") or "").strip()
        cfg.vlm_api_key = str(raw.get("vlm_api_key", "") or "").strip()
        cfg.vlm_model = str(raw.get("vlm_model", "") or "").strip()
        cfg.vlm_timeout_s = _as_float(raw.get("vlm_timeout_s", 90.0), "vlm_timeout_s")
        if cfg.vlm_timeout_s <= 0:
            raise ValueError("config.vlm_timeout_s 要 > 0")
        if not box_mode_ready(raw):
            log.warning(
                "VLM 凭据不全（vlm_base_url/vlm_api_key/vlm_model）："
                "不给 box 的 locate 调用会失败；要自检测请补全")

        cfg.frame_stale_s = _as_float(raw.get("frame_stale_s", 2.0), "frame_stale_s")
        if cfg.frame_stale_s <= 0:
            raise ValueError("config.frame_stale_s 要 > 0")
        cfg.locate_timeout_s = _as_float(raw.get("locate_timeout_s", 120.0), "locate_timeout_s")
        if cfg.locate_timeout_s <= 0:
            raise ValueError("config.locate_timeout_s 要 > 0")

        cfg.log_dir = str(raw.get("log_dir", "") or "").strip()
        cfg.log_keep = _as_int(raw.get("log_keep", 50), "log_keep")
        if cfg.log_keep < 0:
            raise ValueError("config.log_keep 要 >= 0")
        return cfg


def box_mode_ready(raw: dict) -> bool:
    """VLM 三件套是否齐全（不给 box 的调用需要）。"""
    return all(
        str(raw.get(k, "") or "").strip()
        for k in ("vlm_base_url", "vlm_api_key", "vlm_model")
    )
