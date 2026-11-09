# SPDX-License-Identifier: MulanPSL-2.0
"""config 解析/校验的单元测试（不依赖 ROS / atlas）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp_geometry.config import Config  # noqa: E402

VALID = {
    "camera_provider_id": "realsense_camera",
    "camera_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
    "dist_coeffs": [0.1, -0.2, 0.0, 0.0, 0.0],
    "top_point_source": "topface",
    "depth_units_mm": 1.0,
    "frame_stale_s": 2.0,
    "log_keep": 50,
}


class ConfigTest(unittest.TestCase):
    def test_valid(self) -> None:
        cfg = Config.from_dict(VALID)
        self.assertEqual(cfg.camera_provider_id, "realsense_camera")
        self.assertEqual(cfg.camera_matrix[0][0], 600.0)
        self.assertEqual(cfg.top_point_source, "topface")
        self.assertEqual(cfg.log_keep, 50)

    def test_camera_matrix_required(self) -> None:
        raw = dict(VALID)
        raw.pop("camera_matrix")
        with self.assertRaises(ValueError):
            Config.from_dict(raw)

    def test_camera_matrix_shape(self) -> None:
        raw = dict(VALID)
        raw["camera_matrix"] = [[1.0, 0.0], [0.0, 1.0]]
        with self.assertRaises(ValueError):
            Config.from_dict(raw)

    def test_top_point_source(self) -> None:
        raw = dict(VALID)
        raw["top_point_source"] = "magic"
        with self.assertRaises(ValueError):
            Config.from_dict(raw)

    def test_topic_pair_must_be_both_or_neither(self) -> None:
        raw = dict(VALID)
        raw["rgb_topic"] = "/cam/color"
        with self.assertRaises(ValueError):
            Config.from_dict(raw)
        raw["depth_topic"] = "/cam/depth"
        cfg = Config.from_dict(raw)
        self.assertEqual(cfg.depth_topic, "/cam/depth")

    def test_depth_units_positive(self) -> None:
        raw = dict(VALID)
        raw["depth_units_mm"] = 0
        with self.assertRaises(ValueError):
            Config.from_dict(raw)

    def test_defaults(self) -> None:
        raw = {"camera_matrix": VALID["camera_matrix"]}
        cfg = Config.from_dict(raw)
        self.assertEqual(cfg.top_point_source, "topface")
        self.assertEqual(cfg.log_keep, 50)
        self.assertEqual(cfg.frame_stale_s, 2.0)


if __name__ == "__main__":
    unittest.main()
