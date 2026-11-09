# Description: 一帧彩色 + 一帧深度 → "物体在哪、多大、朝哪"。**一切在相机系。**
#
# 这段编排原来长在 `pipeline._look()` 里，跟机械臂、话题、坐标系变换混在一个方法里。
# 搬到这儿之后它只剩视觉：没有相机位姿、没有夹爪、没有 ROS。
#
# 相机系的意思：相机在原点，x 向右、y 向下、z 朝前（OpenCV 约定）。调用方要落到基座系
# 自己乘一次 `camera_base`，本模块一概不知道基座在哪。
import time

import numpy as np

from .vision import image_geometry
from .vision import plane as plane_mod
from .vision import topface

# 检测框**中心**那块的有效深度至少这么多，才算"这一眼测到物体本身"。
# 反光或太近时相机对物体一个数都不给，深度图上是一块空洞，框边上剩的数都是旁边桌面——
# 这时 robust_depth_at 给的中位数其实是桌面的距离，拿它当顶面高度会算成 0~1 mm 还不吭声
# （实测：46 mm 的方块报"物体太矮 1 mm"）。宁可这一眼就报"没量到"。
MIN_FILLED_FRACTION = 0.5

# 环带（用来拟合承托面那一圈）——与 topface 里同名常量取同值，这里是给下面的调用用。
RING_INNER_M = 0.010
RING_OUTER_M = 0.045
RING_MIN_INNER_M = 0.030

# "物体顶面在哪儿"有两条路，都留着、由 `top_point_source` 切（配置项，重启服务生效）。
# 名字用的是 topface 模块内部那两个（它本来就写着这两条路），右边是它们各自的叫法：
#     box_center —— 方案 A / "pure"：框中心像素 + 框内深度中位数 → 反投影
#     topface    —— 方案 B / "geometric"：从深度图分离物体、取顶面加权质心
# 两套都**只在"取哪一点当顶面"上不同**；检测、承托面拟合、长轴都一样。
# 宽度不一样：topface 那条路量得出"顶面那一层"，宽度就用它量；box_center 没有顶面层，
# 只能退回"框的四个角打到承托面上"那套（量出来明显偏大，见 image_geometry.top_extent）。
TOP_POINT_SOURCES = {"box_center": "pure", "topface": "geometric"}


def locate(*, rgb, depth, camera_matrix, dist_coeffs, target: str,
           box=None, detector=None, top_point_source: str = "box_center",
           debug_sink: dict | None = None) -> dict:
    """看一帧，把目标的位置/朝向/尺寸量出来。

    Args:
        rgb: (H, W, 3) uint8，BGR。
        depth: (H, W) uint16，毫米，0 = 无效。
        camera_matrix: 3x3 内参。调用方给——仿真和真机各是各的。
        dist_coeffs: 畸变系数。
        target: 目标名字。要检测时当提示词；`box` 给了就用不上。
        box: 可选 [x0, y0, x1, y1, angle]。给了就跳过检测（仿真传真值框走这条）。
        detector: 有 `detect(image) -> obb | None` 的检测器；`box` 没给时必须提供。
        top_point_source: "box_center"（方案 A）或 "topface"（方案 B），见上面那段。
        debug_sink: 给了就往里塞画图要用的中间量（`ring` / `object_pixels`）。**不塞进
            返回的 JSON**——那是给调用方的，掩码动辄上万个像素，不该走线。服务拿它落盘。

    Returns:
        {"ok", "why"?, "top_point", "support": {"point", "normal"},
         "long_axis", "closing", "footprint_width", "bbox", "diagnostics"}
        坐标一律相机系。量不出来时 ok=False，`why` 里写清原因——**不返回凑合的值**。
    """
    t_start = time.monotonic()
    steps_ms = {}
    camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1, 1)
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return _fail("彩色图要是 (H, W, 3)，收到 %s" % (rgb.shape,))
    if depth.shape[:2] != rgb.shape[:2]:
        return _fail("深度 %s 和彩色 %s 尺寸对不上" % (depth.shape, rgb.shape))

    # ---- 1. 目标在哪：给了框就用，没给就检测 ----
    t0 = time.monotonic()
    if box is not None:
        bbox = [float(v) for v in box[:5]]
        while len(bbox) < 5:
            bbox.append(0.0)
        obb = {"bbox": bbox,
               "center_px": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
               "source": "caller"}
    else:
        if detector is None:
            return _fail("没给 box、也没给检测器，这一帧不知道要找什么")
        obb = detector.detect(rgb)
    steps_ms["detect"] = (time.monotonic() - t0) * 1000.0
    if obb is None:
        return _fail("目标不在腕上相机的画面里")

    # 框贴到画面边上 = 物体没拍全（可能只拍到底座、或者框到了旁边的东西）。实测：方块被
    # 滑杆推到画面顶边时，检测器把整块斜坡板框了进去，后面拟合出来的"承托面"成了桌面，
    # 悬停位按错的方向算。贴边就明说，不拿这一眼往下走。
    x0, y0, x1, y1 = [float(v) for v in obb["bbox"][:4]]
    if x0 <= 1.0 or y0 <= 1.0 or x1 >= rgb.shape[1] - 2.0 or y1 >= rgb.shape[0] - 2.0:
        return _fail("目标框贴到画面边上了（%s），画面里没拍全——挪一下物体或换个姿势再拍"
                     % [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])

    # ---- 2. 框里的深度够不够 ----
    t0 = time.monotonic()
    distance = plane_mod.robust_depth_at(depth, obb["bbox"], 0.001)
    if distance is None:
        return _fail("目标框里没有有效深度")
    filled = plane_mod.center_filled_fraction(depth, obb["bbox"])
    steps_ms["depth_check"] = (time.monotonic() - t0) * 1000.0
    if filled < MIN_FILLED_FRACTION:
        return _fail("框中心那块几乎没有深度（有效 %.0f%%）——相机没测到物体本身"
                     "（多半是反光或者太近）。测不到就别量：挪一下物体、转个角度、"
                     "或者垫点哑光的东西再拍。" % (filled * 100))

    # ---- 3. 承托面：环带 + 稳健拟合 ----
    # 环带圆心用**物体自己的投影点**，不用框中心：框偏了（VLM 实测偏十几像素）圆心跟着偏，
    # 环带内边界就压到物体顶面上，取到的点一半是物体、一半是承托面。
    # 环带半径按**物理尺寸**换算（物体外 1~5 厘米那一圈）。按像素半径取的话，检测框给大了
    # 一倍（VLM 真给过），环带就跑到一倍远——那儿已经是桌面了，桌面的点比斜坡还多，投票
    # 一定选它（实测 632 个样本里 20% 法线误差 >5°、最坏 90°）。
    t0 = time.monotonic()
    point_camera = plane_mod.backproject(camera_matrix, dist_coeffs,
                                         [obb["center_px"]], [distance])[0]
    object_px = (camera_matrix @ point_camera) / point_camera[2]
    focal = float(camera_matrix[0, 0])
    half_px = max(x1 - x0, y1 - y0) / 2.0
    object_radius_m = half_px * distance / focal
    inner_px = max(object_radius_m + RING_INNER_M, RING_MIN_INNER_M) * focal / distance
    outer_px = (object_radius_m + RING_OUTER_M) * focal / distance
    ring = plane_mod.annulus_pixels(obb["bbox"], depth.shape[0], depth.shape[1],
                                    center=object_px[:2],
                                    inner_scale=inner_px / half_px,
                                    outer_scale=outer_px / half_px)
    # max_height_m：物体顶面到承托面的距离超过它，就认为"这个面不是它的承托面"，不要。
    # **原来写死 0.12，比系统里别处的物体高度上限还小**——topface 和 grasp 都用
    # MAX_OBJECT_HEIGHT_M = 0.20。后果是"高于 120 mm 的东西永远量不出来"：实测竖着站的
    # 一包纸巾高约 99~131 mm，正在那条线上下，直接报"平面没拟合出来"。
    # 现在取 0.25 = 上面那个 0.20 再加上一圈余量，别让两处口径打架。
    fitted = plane_mod.fit_plane(camera_matrix, dist_coeffs, depth, 0.001, ring,
                                 anchor=point_camera, max_height_m=0.25,
                                 near=obb["center_px"], near_radius_px=half_px)
    steps_ms["support_plane"] = (time.monotonic() - t0) * 1000.0
    if fitted is None:
        return _fail("平面没拟合出来（环带里有效深度太少）")
    normal = np.asarray(fitted["normal"], dtype=float)
    normal = normal / np.linalg.norm(normal)
    plane_point = np.asarray(fitted["point"], dtype=float)

    # ---- 4. 顶面中心：从深度图里把物体分出来，取顶面那一层的加权质心 ----
    # **不能**拿"框中心像素 + 框内深度中位数"反投影：那是拿**轮廓**的中心当物体中心。
    # 斜着看，轮廓里含可见侧面，中心落不到顶面中心上（实测平放斜视 25° 差 15.8 mm）；
    # 框一大一小它还跟着跑（外扩 1.4× 差 22.6 mm）——真机上实测就是这个量级，下爪会落到
    # 物体旁边。topface 用环带拟合承托面、在深度图上把物体分离出来，这两类误差都没有。
    t0 = time.monotonic()
    method = TOP_POINT_SOURCES.get(str(top_point_source).strip().lower())
    if method is None:
        return _fail("不认识的 top_point_source=%r（认 box_center / topface）" % top_point_source)
    topface_result = topface.estimate(depth, camera_matrix, dist_coeffs, obb["bbox"],
                                      method=method, support_hint=(plane_point, normal))
    steps_ms["topface"] = (time.monotonic() - t0) * 1000.0
    if not topface_result["ok"]:
        return _fail("顶面没量出来：" + topface_result["why"])
    top_point = np.asarray(topface_result["top_point"], dtype=float)

    # ---- 5. "图像上的框和角度"换算成三维方向与长度 ----
    # 这两样原来在 grasp.plan() 里算，而那儿要内参和外参——所以搬到这里，plan 那边只剩
    # 纯三维（这也是本次抽取的核心：plan 不该碰像素）。
    t0 = time.monotonic()
    top_plane = {"point": top_point, "normal": normal}
    # 长轴优先用**从物体自己量出来的**（topface 交出来的，最上面那一层的最小外接矩形长边）。
    # 框那个转角是检测器猜的，实测框转 15° 轴就跟着转 15°，爪子跟着瞄歪、宽度跟着变 7~11 mm；
    # 量的轴对框的缩放/平移/旋转都不动（≤1.2°）。只有拿不到那个轴时（方案 A 没有"那一层"）
    # 才退回用框的转角。
    measured_axis = topface_result.get("long_axis")
    if measured_axis is not None:
        long_axis = np.asarray(measured_axis, dtype=float)
    else:
        long_axis = image_geometry.long_axis_on_plane(
            camera_matrix, dist_coeffs, [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
            float(obb["bbox"][4]), top_plane)
    closing = _closing_from(long_axis, normal)
    # 宽度：先拿**顶面那一层自己的点**量（真的从深度里分出来的物体顶面）；量不出来再
    # 退回"框的四个角打到承托面上"那套。真机那一帧上两者是 48.9 mm 对 90.3 mm，实物
    # 50 mm——那套老做法量的是"框的影子"，框比物体大一圈、影子又比物体摊得开，而
    # grasp.plan 拿这个数当"夹得住夹不住"的硬判据，虚胖会把夹得住的判成夹不住。
    top_pixels = topface_result.get("top_pixels")
    footprint_width = None if top_pixels is None else image_geometry.footprint_extent(
        top_pixels, depth, camera_matrix, dist_coeffs, closing)
    if footprint_width is None:
        footprint_width = image_geometry.footprint_width(
            obb["bbox"], camera_matrix, dist_coeffs,
            {"point": plane_point, "normal": normal}, closing)
    steps_ms["axes"] = (time.monotonic() - t0) * 1000.0

    steps_ms["total"] = (time.monotonic() - t_start) * 1000.0
    diagnostics = {
        "bbox": [round(float(v), 1) for v in obb["bbox"]],
        "detect_source": obb.get("source", "detector"),
        # 这次用的是哪条路（两套算法都留着，开关在配置里）。A/B 对照时看这一项。
        "top_point_source": top_point_source,
        "topface_method": method,
        "top_point_m": np.round(top_point, 6).tolist(),
        "support_point_m": np.round(plane_point, 6).tolist(),
        "support_normal": np.round(normal, 6).tolist(),
        "long_axis": np.round(long_axis, 6).tolist(),
        "closing": np.round(closing, 6).tolist(),
        "footprint_width_mm": (None if footprint_width is None
                               else round(footprint_width * 1000, 3)),
        # 顶面是怎么来的：走的是哪条路、承托面和噪声尺度各是多少。光看 top_point 一个数，
        # 出了偏差没法判断是量错了还是算错了（吃过亏）。
        "topface": topface_result["report"],
        "plane_fit": {"points": fitted["points"],
                      "inlier_ratio": fitted["inlier_ratio"],
                      "rms_mm": fitted["rms_m"] * 1000},
        "center_filled": filled,
        "steps_ms": {k: round(v, 1) for k, v in steps_ms.items()},
    }
    if detector is not None and hasattr(detector, "diagnostics") and box is None:
        diagnostics["detector"] = detector.diagnostics()
    if debug_sink is not None:
        # 画图要用的：拟合承托面那圈环带、和顶面那一步分出来的物体像素。
        # 只给落盘用，不进返回——掩码上万个点，走线没意义。
        debug_sink["ring"] = np.asarray(ring, dtype=int)
        debug_sink["object_pixels"] = topface_result.get("object_pixels")
        debug_sink["center_px"] = tuple(obb["center_px"])

    return {
        "ok": True,
        "top_point": top_point.tolist(),
        "support": {"point": plane_point.tolist(), "normal": normal.tolist()},
        "long_axis": long_axis.tolist(),
        "closing": closing.tolist(),
        "footprint_width": footprint_width,
        "bbox": [float(v) for v in obb["bbox"]],
        "diagnostics": diagnostics,
    }


def _closing_from(long_axis, normal) -> np.ndarray:
    """闭合方向：落在承托面内、垂直于长轴。

    与 grasp.closing_direction 是同一个式子，放在这边是为了**只算一次**——调用方拿
    `long_axis` 和 `closing` 直接用，不用自己再推一遍（推两遍迟早会推歪一处）。
    """
    axis = np.asarray(long_axis, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    direction = np.cross(axis, normal)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("长轴与承托面法线平行，定不出闭合方向")
    return direction / norm


def _fail(why: str) -> dict:
    return {"ok": False, "why": why}
