# SPDX-License-Identifier: MulanPSL-2.0
"""用真机样本跑一遍核心几何（不依赖 ROS / atlas / VLM）。

样本默认取本仓库的 `host-services/grasp-vision/samples/`；没有就跳过。
可以 `GRASP_GEOMETRY_SAMPLES=/path/to/samples python3 -m unittest discover tests` 指定。
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp_geometry.core.locate import locate  # noqa: E402

DEFAULT_SAMPLES = Path(
    "/home/dog/wfw/piper-standard/host-services/grasp-vision/samples")
SAMPLES = Path(os.environ.get("GRASP_GEOMETRY_SAMPLES", str(DEFAULT_SAMPLES)))


def _bbox_of(quad) -> list[float]:
    points = np.asarray(quad, dtype=float)
    return [
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()), 0.0,
    ]


@unittest.skipUnless(SAMPLES.is_dir(), f"没有样本目录 {SAMPLES}")
class SampleGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = []
        for directory in sorted(SAMPLES.iterdir()):
            meta_file = directory / "meta.json"
            if not meta_file.is_file():
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            rgb = cv2.imread(str(directory / meta["rgb"]))
            depth = cv2.imread(str(directory / meta["depth"]), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                continue
            cls.samples.append((directory.name, meta, rgb, depth))

    def test_samples_measure(self) -> None:
        self.assertTrue(self.samples, "样本目录里没有可读样本")
        for name, meta, rgb, depth in self.samples:
            for annotation in meta.get("annotations") or []:
                camera_matrix = np.asarray(meta["camera_matrix"], dtype=float)
                dist_coeffs = np.asarray(meta["dist_coeffs"], dtype=float)
                box = _bbox_of(annotation["quad_px"])
                with self.subTest(sample=name, annotation=annotation.get("what", "")):
                    result = locate(
                        rgb=rgb, depth=depth,
                        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                        target=str(annotation.get("what", "object")),
                        box=box, top_point_source="topface",
                    )
                    self.assertTrue(result.get("ok"), result.get("why"))
                    point = np.asarray(result["top_point"], dtype=float)
                    self.assertEqual(point.shape, (3,))
                    self.assertTrue(np.all(np.isfinite(point)))
                    # 相机系 z 必须为正（物体在相机前方），且量级合理（0.05~3 m）
                    self.assertGreater(point[2], 0.05)
                    self.assertLess(point[2], 3.0)
                    support = result["support"]
                    self.assertEqual(len(support["point"]), 3)
                    self.assertEqual(len(support["normal"]), 3)
                    self.assertEqual(len(result["closing"]), 3)


if __name__ == "__main__":
    unittest.main()
