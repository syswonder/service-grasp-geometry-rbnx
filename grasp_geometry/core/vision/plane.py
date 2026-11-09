# Description: 从深度图里取点、拟合目标所在的平面。
#
# 目标放在某个面上（桌面、斜面、坡道都算），要的是这个面的两个数：法线方向和它离基座系
# 原点多远。抓取姿态由法线定，下爪高度由"物体顶面离面多远"定，两者都从这里来。
#
# 取点范围不能只取目标周围几十像素：目标小的时候那点面积全是物体自己，拟出来的是物体的
# 顶面而不是它所靠的面。所以按物体外接框的等效尺寸缩放着取一圈环带，并且把环带里的
# "和种子点不在同一层"的点剔掉，避免把旁边的东西算进平面。
import cv2
import numpy as np


MIN_POINTS = 200          # 少于这么多点不拟合：小样本的平面法线抖得厉害
DIST_THRESH_M = 0.004     # RANSAC 内点阈值，米
REFINE_ITERS = 3          # 用内点重拟合的次数
DEFAULT_INNER = 1.2       # 环带内边界：相对物体等效半径
DEFAULT_OUTER = 2.0       # 环带外边界
FIT_NEAREST_FRAC = 0.35   # 投票只用最近这一部分点（环带取大是为了有足够的点可挑）
MIN_SPREAD_M = 0.005      # 内点在这两个方向上的展布小于这么多，当它是窄条/一小撮，不收
                          # （一小片点也能"拟合"出平面，但法线角度是碰出来的）


def normalized_coords(camera_matrix, dist_coeffs, pixels) -> np.ndarray:
    """像素 → 归一化像面坐标 (x/z, y/z)，N x 2。这是内参和畸变唯一该起作用的地方。"""
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, np.asarray(camera_matrix, dtype=float),
                               np.asarray(dist_coeffs, dtype=float).reshape(-1, 1)).reshape(-1, 2)


def ray_directions(camera_matrix, dist_coeffs, pixels) -> np.ndarray:
    """像素 → 相机系下的单位射线方向，N x 3。求射线与平面交点时用这个。"""
    normalized = normalized_coords(camera_matrix, dist_coeffs, pixels)
    directions = np.column_stack([normalized, np.ones(len(normalized))])
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def backproject(camera_matrix, dist_coeffs, pixels, depth_m) -> np.ndarray:
    """像素 + 深度 → 相机系三维点，N x 3。

    **深度是相机系的 z，不是沿视线的斜距**（RealSense 和 MuJoCo 都是这么给的）。
    所以还原出来的是 (x/z·d, y/z·d, d)，z 分量正好等于深度。当成斜距去乘单位射线的话，
    离画面中心越远的点被拉得越近，实测能差到 39 mm——那是斜的，拟合出来当然也是歪面。
    """
    normalized = normalized_coords(camera_matrix, dist_coeffs, pixels)
    depth_m = np.asarray(depth_m, dtype=float).reshape(-1)
    return np.column_stack([normalized[:, 0] * depth_m, normalized[:, 1] * depth_m, depth_m])


def robust_depth_at(depth, box, depth_scale: float = 0.001, near_frac: float = 0.6):
    """目标框里"这个物体有多远"的稳健估计（米）。

    不能只读框中心那一个像素：画面里常有比目标更近的东西压在它上面（相机自己的外壳、
    反光、飞虫、别人的手指），那一个像素读到的是那个东西的距离。这里取框内所有有效深度
    的中位数，并先把明显更近的那一撮剔掉（小于中位数的 near_frac 倍），再取中位数。
    """
    x0, y0, x1, y1 = [int(round(float(v))) for v in box[:4]]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(depth.shape[1], x1 + 1), min(depth.shape[0], y1 + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth[y0:y1, x0:x1].astype(float)
    valid = patch[patch > 0] * depth_scale
    if valid.size == 0:
        return None
    median = float(np.median(valid))
    keep = valid[valid >= median * near_frac]
    if keep.size == 0:
        keep = valid
    return float(np.median(keep))


def center_filled_fraction(depth, box, radius: int = 7) -> float:
    """检测框**中心一小块**里有效深度（非 0）占的比例。

    为什么盯中心、不盯整个框：物体反光或者太近时相机对它一个数都不给，深度图上是一块
    空洞，而框的边上往往还压着旁边的桌面（有数）。只看整框的比例会被那些边角骗过去
    （实测一次失败：框里 47% 有数，但全在边上、中心是空的）。中心那一小块是物体自己的
    位置——那儿没数，就是"这一眼没测到物体"。

    不判这一条会怎样：`robust_depth_at` 照样能返回一个"稳健"的中位数，那是框里那些**桌面**
    点的距离，拿它当物体顶面，高度算成 0~1 mm（实测：46 mm 的方块报"物体太矮 1 mm"）。
    """
    x0, y0, x1, y1 = [int(round(float(v))) for v in box[:4]]
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    left, top = max(0, center_x - radius), max(0, center_y - radius)
    right = min(depth.shape[1], center_x + radius + 1)
    bottom = min(depth.shape[0], center_y + radius + 1)
    patch = depth[top:bottom, left:right]
    return float((patch > 0).mean()) if patch.size else 0.0


def annulus_pixels(box, height, width, inner_scale: float = DEFAULT_INNER,
                   outer_scale: float = DEFAULT_OUTER, step: int = 2, center=None):
    """目标框外的一圈环带像素。

    内外半径按框的等效半径缩放：物体越大，环带跟着往外推，避免取到的全是物体自己。
    环带被画面边界裁掉的部分不算，不能为了凑数把边缘的坏点拉进来。

    center: 环带的圆心，默认取框中心。**检测框不准时**（比如 VLM 给的框偏十几个像素）
        要传物体自己的投影点：圆心跟着框跑的话，内边界会压到物体顶面上，一半点取到
        物体、一半取到承托面，拟合出来的面就在两者之间——法线歪十几度。
    """
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    center = (np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0]) if center is None
              else np.asarray(center, dtype=float).reshape(2))
    radius = max((x1 - x0), (y1 - y0)) / 2.0
    inner, outer = radius * inner_scale, radius * outer_scale
    grid_x, grid_y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    distance = np.hypot(grid_x - center[0], grid_y - center[1])
    keep = (distance >= inner) & (distance <= outer)
    return np.column_stack([grid_x[keep], grid_y[keep]]).astype(float)


def fit_plane(camera_matrix, dist_coeffs, depth, depth_scale, pixels,
              dist_thresh_m: float = DIST_THRESH_M, seed: int = 7,
              anchor=None, max_height_m: float = 0.12,
              near=None, near_radius_px: float = None) -> dict | None:
    """在一批像素上拟合平面。

    Args:
        pixels: 取哪些像素（一般是目标框外的一圈环带）。
        anchor: 物体顶面那个点（相机系）。给了就只认"从它脚下过去"的面——
            环带一大就会圈进旁边的桌面、地面，那些面点数更多，不拦的话会被选中。
            顶面到支撑面的距离最多 max_height_m，超过的候选直接不要。
        near / near_radius_px: 离目标多远的点算"紧挨着目标"。给这两个就按距离加权打分：
            紧挨着目标的点权重高。斜面摆在桌面上时，斜面离物体近、桌面离得远，两者高度
            只差几毫米，只靠高度分不开，靠距离权重才分得开。

    Returns:
        {"point": 面上一点（相机系）, "normal": 单位法线（指向相机一侧）,
         "points": 内点数, "rms_m": 内点残差均方根, "inlier_ratio": 内点占比}；
        点不够或拟合不出来时返回 None。
    """
    pixels = np.asarray(pixels, dtype=float)
    if len(pixels) == 0:
        return None
    u = np.round(pixels[:, 0]).astype(int)
    v = np.round(pixels[:, 1]).astype(int)
    inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    u, v = u[inside], v[inside]
    if len(u) < MIN_POINTS:
        return None
    raw = depth[v, u].astype(float)
    ok = raw > 0
    if int(ok.sum()) < MIN_POINTS:
        return None
    u, v, raw = u[ok], v[ok], raw[ok]
    points = backproject(camera_matrix, dist_coeffs, np.column_stack([u, v]), raw * depth_scale)
    anchor = None if anchor is None else np.asarray(anchor, dtype=float).reshape(3)
    if near is not None and near_radius_px:
        distance_px = np.hypot(u - float(near[0]), v - float(near[1]))
        weights = 1.0 / (1.0 + distance_px / float(near_radius_px))
    else:
        weights = np.ones(len(points))

    # 只拿"离目标最近的那一批"去投票：环带是取大一点好有足够的点，但边上那些点可能是
    # 旁边的桌面或地面。紧挨着物体的那圈才是它真正搁在上面的面。
    if near is not None and len(points) > MIN_POINTS * 2:
        order = np.argsort(np.hypot(u - float(near[0]), v - float(near[1])))
        keep = max(MIN_POINTS, int(len(points) * FIT_NEAREST_FRAC))
        fit_mask = np.zeros(len(points), dtype=bool)
        fit_mask[order[:keep]] = True
    else:
        fit_mask = np.ones(len(points), dtype=bool)

    rng = np.random.default_rng(seed)
    best_mask, best_score = None, 0.0
    candidates = np.flatnonzero(fit_mask)
    for _ in range(200):
        idx = rng.choice(candidates, 3, replace=False)
        sample = points[idx]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if anchor is not None:
            height = float((anchor - sample[0]) @ normal)
            if not (0.0 <= height <= max_height_m):
                continue
        distance = np.abs((points - sample[0]) @ normal)
        mask = distance < dist_thresh_m
        score = float(weights[mask & fit_mask].sum())
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None or int(best_mask.sum()) < MIN_POINTS:
        return None

    plane_point, normal = None, None
    for _ in range(REFINE_ITERS):
        inliers = points[best_mask]
        center = inliers.mean(axis=0)
        # full_matrices=False：只要 vt 那一行法线，默认的 True 会去分配一个 N×N 的 U
        # （几千个点就是几百 MB、每次精化白花半秒）。
        _, _, vt = np.linalg.svd(inliers - center, full_matrices=False)
        normal = vt[2]
        # 重算内点**只在候选集里长**（fit_mask）：投票时挑了"离物体最近的那部分 + 从物体
        # 脚下过"，精化这一步要是放开到全部点，面会一路爬到旁边的桌面上——那是另一个面，
        # 点数还更多。踩过：拟合出来的"承托面"是桌面，法线差了整整 30°。
        distance = np.abs((points - center) @ normal)
        best_mask = (distance < dist_thresh_m) & fit_mask
        plane_point = center
    if int(best_mask.sum()) < MIN_POINTS:
        return None
    # 面得"摊开"才算数：内点挤成一条线或一小片时，法线数学上算得出来、角度却定不准。
    spread = np.linalg.svd(points[best_mask] - plane_point, compute_uv=False)
    if spread[1] < MIN_SPREAD_M:
        return None                      # 窄条状的一撮点，法线不可信，宁可不给
    # 精化之后还得仍"从物体脚下过"：法线朝向任取，所以用绝对值卡高度
    if anchor is not None and abs(float((anchor - plane_point) @ normal)) > max_height_m:
        return None
    rms = float(np.sqrt(np.mean(((points[best_mask] - plane_point) @ normal) ** 2)))

    # 法线朝向相机：相机在原点，平面上的点朝相机的方向是 -point 方向
    if float(normal @ (plane_point - np.zeros(3))) > 0:
        normal = -normal
    return {"point": plane_point, "normal": normal, "points": int(best_mask.sum()),
            "rms_m": rms, "inlier_ratio": float(best_mask.mean())}

