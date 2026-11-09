# Description: 顶面提取 —— "2D 检测 + 深度图上做几何" 那套算法的成品版，独立一份。
#
# 它解决什么：原来 `pipeline._look` 把**检测框中心的那个像素**按**框内深度中位数**反投影，
# 拿它当物体的顶面中心。那是拿**轮廓**的中心当物体中心，两处都会偏：
#
#   一、斜着看就偏。轮廓里含可见的侧面，框中心落不到顶面中心上。实测平放斜视 25° 差
#       15.8 mm（30° 坡上只差 1.5 mm，所以一直没暴露）；相机再偏一点、物体再高一点就更大。
#   二、框一大一小就跟着跑。检测框是检测器给的，外扩/内收/平移都会直接把 top_point 带走。
#       实测框外扩 1.4× 时差 22.6 mm、平移 9 px 时差 12.4 mm。
#
# 改成：先用环带拟合承托面 → 在深度图上把"离面够高"的点分离成物体 → 取**顶面那一层的
# 加权质心**。轮廓那条路彻底不走了，所以上面两类误差都消掉（同样两个场景 0.5 / 1.0 mm）。
#
# 两条路都在这个文件里，`METHOD` 一个开关切换（也可以用环境变量 GRASP_TOPFACE 临场切）：
#     "geometric" 本模块那套；失败就如实报，不硬着头皮往下走（与仓库一贯的做法一致）
#     "pure"      原来的纯视觉做法（框中心 + 框内深度中位数），用于回退/对照
#
# 依赖只有 numpy + cv2（畸变校正）。**不 import 仓库里别的模块**，单独一份可搬。
#
# **一切在相机系里**（相机在原点）。原来那份要传 camera_pose 是为了把结果搬到基座系，
# 搬进本服务之后那层变换归调用方，所以这里的 camera_pose 参数已经拿掉了。
#
# 用法：
#     result = topface.estimate(depth_mm, K, dist, bbox)
#     result["ok"]        -> bool
#     result["top_point"] -> **相机系**三维点（要基座系由调用方乘一次 camera_base）
#     result["height_m"]  -> 顶面离承托面多高（只有 geometric 给）
#     result["report"]    -> 落进日志的中间量
import os

import cv2
import numpy as np

# ---------------------------------------------------------------- 开关

METHOD = os.environ.get("GRASP_TOPFACE", "geometric").strip().lower()

# ---------------------------------------------------------------- 参数（都在原型上扫过）
RING_INNER_M = 0.010        # 环带：物体边缘往外这么多开始取承托面
RING_OUTER_M = 0.045
RING_MIN_INNER_M = 0.030    # 环带内边界离物体中心至少这么远（框偏小时别落到物体顶面上）
REACH_FACTOR = 1.5          # 承托面最多长到环带外沿的这个倍数，再外面就不下判断
# 离承托面超过多少才算物体。**这个门槛必须站在承托面自己的噪声之上**，否则桌面会被当成
# 物体，连成一大片跟着物体跑（原来取 2 mm，正好卡在桌面残差的 p90 上，掩码里一半是桌面）。
#
# 但**不能写成一个固定的毫米数**：这个门槛取决于"这一帧里这个面有多不平、深度有多吵"，
# 而那个量随距离、随相机、随桌布在变——实测两帧的桌面残差尾部差 40%（p99 3.61 / 5.03 mm，
# 最大 6.52 / 9.76 mm），任何固定值都会在其中一帧上要么漏进桌面、要么把矮物体整个吞掉。
#
# 所以**从数据里推**：承托面刚拟合完，就去量物体外面那一圈**自由桌面点**的残差尺度，
# 拿它当尺子。两条都是踩过坑才定下来的：
#   - 要用**自由点**、不能用拟合内点：内点是被 Tukey 截断过的（截在 4.685σ），尾巴早就
#     被切掉了，拿它推会系统性低估。
#   - 要用 **MAD 这种尺度**、不能用 p99 这种分位数：采样圈里混进别人的东西是常态
#     （实测胡萝卜那一帧叶子占了圈里一成多，p99 被抬到 23 mm，阈值变成 46 mm，物体整个
#     漏掉）；MAD 崩溃点 50%，分位数崩溃点 1%。
# 倍数 5：残差不是纯高斯，实测"最坏点 / σ"到 4.4 倍（方块 6.52/1.48、胡萝卜 9.76/2.2），
# 5σ 刚好压在它上面。再用物体自己的半径封顶。
OBJECT_BAND_TAIL = 5.0      # 阈值 = 这个倍数 × 自由桌面点残差的 MAD 尺度
OBJECT_BAND_M = 0.008       # 兜底：自由点不够时用回这个固定值
FREE_SAMPLE_PX = 40         # 自由桌面点取在环带外沿再往外这么一圈里（贴着物体、又不含物体）
FREE_SAMPLE_MIN = 200       # 这一圈至少这么多点才敢拿它定尺子
MAX_OBJECT_HEIGHT_M = 0.20  # 离面比这还高的一律不算物体（与 grasp.MAX_OBJECT_HEIGHT_M 同值）
# "最上面那一层"的范围：**物体自己高度的顶部 TOP_FRACTION 那一段**，只往下取、不往上罚
# （物体之上没有别的东西，往上罚等于罚噪声）。
#
# **按物体的高度取，不按固定的毫米数取**——固定的毫米数里藏着一个"物体有平顶"的假设，
# 没有平顶的东西（胡萝卜、三棱锥、球）就被切成顶上那一小块，质心跟着跑到尖儿上去。
# 按比例取只问"物体自己最高的那一段在哪儿"，跟有没有平面无关。
#
# 这一段取多宽，是"两头夹"挤出来的：
#   - **太窄**：物体两头（尤其是细的那头）的曲面掉出这一段，层里只剩粗的那头，
#     质心被拽向粗头。实测胡萝卜：切在高度 90% 以上，中心差 5.1 px。
#   - **太宽**：吃进侧面的掠射区，那儿的深度不可信、点会横着甩出去。实测胡萝卜：
#     切在 50% 以上，中心差 23 px。
# 两帧一起扫"切在物体高度的百分之多少以上"（方块 50x50x50；胡萝卜，VLM 给的那个框）：
#       切在 60% 以上    方块 1.85 px   胡萝卜  6.02 px
#       切在 65% 以上    方块 1.83 px   胡萝卜  0.64 px
#       切在 70% 以上    方块 1.84 px   胡萝卜  0.92 px   <- 这个
#       切在 75% 以上    方块 1.85 px   胡萝卜  1.64 px
#       切在 80% 以上    方块 1.83 px   胡萝卜  2.93 px
#       切在 90% 以上    方块 2.36 px   胡萝卜  5.10 px
# 65%~75% 是一条平台（两头都是悬崖），取中间。方块对这个不敏感（它上下对称，多切少切
# 质心都不动），**约束全来自细的那头**——所以这个比例是按"一头粗一头细的物体"定的。
#
# **注意 ceiling 和这一段是两件事**：ceiling 是物体的真实最高处，grasp.plan 拿它当物体
# 高度用，一分不能少；这一段只决定"拿哪些点算水平中心"。混在一起会两头不讨好——实测
# 把 ceiling 压到 43 mm，像素差好看了（1.66→1.20 px），三维差却从 2.73 涨到 3.74 mm。
TOP_FRACTION = 0.30
TOP_BAND_MIN_M = 0.004
MIN_TOP_POINTS = 50         # 最上面那一层少于这么多点就不下判断（与 _in_box 的兜底阈值同量级）
COARSE_CELL_PX = 6          # 承托面粗网格的格边长
COARSE_MIN_POINTS = 8       # 一格至少这么多点才算"承托面"
GROW_STEPS = 6
MAX_SURFACE_SLOPE = 1.0     # 承托面高度场的坡度上限（1.0 = 45°），超了就是法线找错了
MAX_FRAME_M = 3.0           # 深度量程上限（米）
MIN_FRAME_M = 0.05


# ---------------------------------------------------------------- 基础

def _rays(camera_matrix, dist_coeffs, pixels):
    """像素 → 相机系归一化坐标 (x/z, y/z)。畸变只在这里起作用。"""
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, np.asarray(camera_matrix, dtype=float),
                               np.asarray(dist_coeffs, dtype=float).reshape(-1, 1)
                               ).reshape(-1, 2)


def backproject(camera_matrix, dist_coeffs, pixels, depths_m):
    """像素 + **相机系 z 深度** → 相机系三维点。口径同 plane.backproject（不是斜距）。"""
    normalized = _rays(camera_matrix, dist_coeffs, pixels)
    depths_m = np.asarray(depths_m, dtype=float).reshape(-1)
    return np.column_stack([normalized * depths_m[:, None], depths_m])


# 这里**没有坐标系变换**：本服务一律在相机系里进出（见包根的 README）。相机就在原点，
# 原来那份 `_to_frame(camera_pose, pts)` 在这儿是恒等变换，所以删掉；要落到基座系由调用方
# 自己乘一次 `camera_base` 负责。


def _in_box(pixels, center_px, angle_deg, half_w, half_h, margin):
    """像素在不在框里——**在框自己的坐标系里判**（框带角度，先把它转正）。

    [x0,y0,x1,y1] 给的是"转正之后框有多大"，直接跟图像坐标比就是拿另一个区域去卡物体。
    """
    angle = np.deg2rad(float(angle_deg))
    dx = pixels[:, 0] - center_px[0]
    dy = pixels[:, 1] - center_px[1]
    along = dx * np.cos(angle) + dy * np.sin(angle)          # 沿长轴
    across = -dx * np.sin(angle) + dy * np.cos(angle)        # 垂直于长轴
    return ((np.abs(along) <= half_w + margin)
            & (np.abs(across) <= half_h + margin))


def _pixels_in_box(shape, box):
    """矩形里所有像素（列优先），返回 (N,2) 的 (x, y)。"""
    x0 = max(0, int(np.floor(box[0])))
    y0 = max(0, int(np.floor(box[1])))
    x1 = min(shape[1], int(np.ceil(box[2])) + 1)
    y1 = min(shape[0], int(np.ceil(box[3])) + 1)
    grid = np.stack(np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)), axis=-1)
    return grid.reshape(-1, 2).astype(float)


# ---------------------------------------------------------------- 纯视觉那条路

def _estimate_pure(depth_m, valid_mm, camera_matrix, dist_coeffs, bbox):
    """原来的做法：框中心像素 + 框内深度中位数 → 反投影。

    中位数那一步和仓库 plane.robust_depth_at 等价（先剔掉明显更近的一撮再取中位数）。
    """
    height, width = depth_m.shape
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    px0, py0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    px1, py1 = min(width - 1, int(np.ceil(x1))), min(height - 1, int(np.ceil(y1)))
    patch = depth_m[py0:py1 + 1, px0:px1 + 1]
    keep = patch[valid_mm[py0:py1 + 1, px0:px1 + 1]]
    if keep.size == 0:
        return {"ok": False, "why": "框里没有有效深度"}
    median = float(np.median(keep))
    near = keep[keep >= median * 0.6]
    distance = float(np.median(near)) if near.size else median
    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    point = backproject(camera_matrix, dist_coeffs, [center], [distance])
    return {"ok": True, "top_point": point[0], "distance_m": distance,
            "height_m": None, "report": {"method": "pure", "distance_m": round(distance, 4)}}


# ---------------------------------------------------------------- 几何那条路

def label_components(xs, ys, mask, gap=2):
    """按图上邻域（允许 gap 像素的空隙）把掩码里的点分成连通块；返回每点的块号（-1=不在掩码里）。

    没有连通性时，任何落进范围、又在承托面之上的杂点都会变成"物体"，还能劫持顶面层
    （原型实测：1% 离群点把顶面从 30 mm 抬到 148 mm）。gap 留 2 px 是为了跨过局部缺失。
    """
    lookup = {}
    for index in np.flatnonzero(mask):
        lookup[(int(xs[index]), int(ys[index]))] = index
    labels = np.full(len(xs), -1)
    offsets = [(dx, dy) for dy in range(-gap, gap + 1) for dx in range(-gap, gap + 1)]
    current = 0
    for index in np.flatnonzero(mask):
        if labels[index] >= 0:
            continue
        labels[index] = current
        stack = [index]
        while stack:
            item = stack.pop()
            x, y = int(xs[item]), int(ys[item])
            for dx, dy in offsets:
                other = lookup.get((x + dx, y + dy))
                if other is not None and labels[other] < 0:
                    labels[other] = current
                    stack.append(other)
        current += 1
    return labels


def coarse_support(xs, ys, mask):
    """按粗网格归并，只留**最大的一块格子**——把承托面里的"鼓包"（石头）剔掉。

    石头和承托面只在接触圈上连着，图上连通性分不开；但换到 6 px 的粗网格，石头只占
    一两个格子，承托面是一大片连着的格子。没有这一道，拟合会被鼓包拽弯，而且会正反馈
    跑飞（拟合弯一点，更多石头点落进"贴面"判据，再弯）。原型实测：贴着物体的石头
    让外推偏 +16 mm。
    """
    # 归格这一步向量化：一轮生长要调它一次，逐点走 Python 循环的话一次调用就几百毫秒
    # （实测整个 estimate 767 ms，大头在这儿）。格数远小于点数，只对格做邻域遍历。
    inside = np.flatnonzero(mask)
    if inside.size == 0:
        return mask
    # 用一维编码再去重：np.unique(axis=0) 走的是 void 视图排序，实测在这一步占掉 17% 的
    # 总耗时；编成一维之后同样的去重快好几倍。（编码 x*4096+y，y 是图像行号，≤4096 够用）
    cell_x = xs[inside] // COARSE_CELL_PX
    cell_y = ys[inside] // COARSE_CELL_PX
    codes = cell_x.astype(np.int64) * 4096 + cell_y
    unique_codes, inverse, counts = np.unique(codes, return_inverse=True,
                                              return_counts=True)
    filled = unique_codes[counts >= COARSE_MIN_POINTS]
    if filled.size == 0:
        return mask
    filled_set = set(int(value) for value in filled)
    seen, best = set(), set()
    for start in filled_set:
        if start in seen:
            continue
        stack, block = [start], set()
        seen.add(start)
        while stack:
            current = stack.pop()
            cell_x, cell_y = divmod(current, 4096)
            block.add(current)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = (cell_x + dx) * 4096 + cell_y + dy
                    if other in filled_set and other not in seen:
                        seen.add(other)
                        stack.append(other)
        if len(block) > len(best):
            best = block
    if not best:
        return mask
    is_best = np.isin(codes, np.array(sorted(best)))        # 这一步也必须是向量化的
    out = np.zeros(len(xs), dtype=bool)
    out[inside[is_best]] = True
    return out


def seed_plane(points, camera_position, iterations=200, tolerance=0.004, sample=600,
               seed=0):
    """种子点里的**主平面**：RANSAC，不是直接 SVD。

    为什么不能直接 SVD：环带里可能混着别的东西——实测这一帧里就是夹爪那一团（很近、
    很密），仓库自己的 fit_plane 在同一圈上内点率只有 27%。整圈一起 SVD，主方向会被
    那团点带跑：实测求出来的法线和真法线差 56°，拟合直接崩（顶面量成 6 mm）。
    RANSAC 只认"内点最多"的那个平面，夹爪那团再密也只是另一组内点少的假设。
    """
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    pool = points if len(points) <= sample else points[rng.choice(len(points), sample,
                                                                 replace=False)]
    best_anchor, best_normal, best_count = None, None, -1
    for _ in range(iterations):
        picked = pool[rng.choice(len(pool), 3, replace=False)]
        normal = np.cross(picked[1] - picked[0], picked[2] - picked[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        count = int((np.abs((points - picked[0]) @ normal) < tolerance).sum())
        if count > best_count:
            best_anchor, best_normal, best_count = picked[0], normal, count
    if best_normal is None:
        return None
    selected = np.abs((points - best_anchor) @ best_normal) < tolerance
    inliers = points[selected] if selected.sum() >= 20 else points
    centered = inliers - inliers.mean(axis=0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][2]
    if normal @ (inliers.mean(axis=0) - np.asarray(camera_position)[:3]) > 0:
        normal = -normal
    return {"point": inliers.mean(axis=0), "normal": normal,
            "inlier_ratio": float(len(inliers)) / max(len(points), 1)}


def _surface_frame(points, normal_hint):
    origin = points.mean(axis=0)
    normal = np.asarray(normal_hint, dtype=float)
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(helper, normal)
    first = first / np.linalg.norm(first)
    return origin, first, np.cross(normal, first), normal


def _design(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def fit_surface(points, frame, rounds=4):
    """鲁棒最小二乘拟合高度场 h(u,v)（局部二次，带 Tukey 权重）。

    为什么要鲁棒：承托面的外侧可能是另一个面（斜坡板的边外面就是桌面，落差几十毫米），
    普通最小二乘会被它整片拽走。为什么要二次：平面在曲面上外推会偏——原型实测球冠
    R=0.25 m 时平面外推偏 2.2 mm，二次降到 0.15 mm。
    """
    origin, first, second, normal = frame
    local = points - origin
    x, y = local @ first, local @ second
    height = local @ normal
    weights = np.ones(len(points))
    coeffs, scale = None, 0.0
    for round_index in range(rounds):
        matrix = _design(x, y) * weights[:, None]
        coeffs, *_ = np.linalg.lstsq(matrix, height * weights, rcond=None)
        residual = height - _design(x, y) @ coeffs
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        # **尺度退火**：头几轮把尺度卡住，不许一次坏初拟合把截断撑开。
        # 不加这个会翻车：第一轮是无权重的普通最小二乘，环带里只要有 25% 的远处干扰
        # （实测是夹爪，比承托面近 340 mm），它的力臂就把初拟合拽成两边的折中；再拿这个
        # 坏初拟合算 MAD，尺度被撑到 25 mm、Tukey 截断变成 117 mm——**什么都拦不住**，
        # 于是越迭代越松，拟合出一条 62° 的"斜墙"（实测系数 −79/−115），整条链崩掉。
        # 退火从 3 mm 起、每轮翻倍，四轮封顶 24 mm：细看一遍再放宽，粗糙的承托面也容得下。
        scale = min(mad, 0.003 * (2.0 ** round_index))
        ratio = residual / (4.685 * scale)
        weights = np.where(np.abs(ratio) < 1.0, (1.0 - ratio ** 2) ** 2, 0.0)
    return {"coeffs": coeffs, "frame": frame, "scale_m": scale,
            "inliers": float((weights > 0).mean())}


def _height_above(surface, points):
    origin, first, second, normal = surface["frame"]
    local = np.asarray(points, dtype=float) - origin
    predicted = _design(local @ first, local @ second) @ surface["coeffs"]
    return local @ normal - predicted


def top_of_main_mass(residual, mask, bin_mm=1.0, fraction=0.25):
    """顶面在哪个高度：取**最高的那个"够厚"的高度格所在的平台重心**。

    不能取分位数：分位数对稀疏的高处点毫无抵抗力（σ=2 mm 噪声下 97 分位被噪声抬高，
    顶面层整层跟着上移）。也不能取平台上沿：上沿系统性偏高半个带宽。"够厚"这个门槛
    把稀疏点挡在外面，平台重心对对称噪声无偏。
    """
    values = residual[mask] * 1000.0
    low, high = float(values.min()), float(values.max())
    count = max(2, int((high - low) / bin_mm) + 1)
    counts, edges = np.histogram(values, bins=count, range=(low, high))
    threshold = max(5.0, fraction * float(counts.max()))
    top = len(counts) - 1
    while top >= 0 and counts[top] < threshold:
        top -= 1
    if top < 0:
        top = int(np.argmax(counts))
    low_index = top
    while low_index - 1 >= 0 and counts[low_index - 1] >= threshold:
        low_index -= 1
    weights = counts[low_index:top + 1].astype(float)
    centers = 0.5 * (edges[low_index:top + 1] + edges[low_index + 1:top + 2])
    return float((weights * centers).sum() / weights.sum()) / 1000.0


def _estimate_geometric(depth_m, valid_mm, camera_matrix, dist_coeffs, bbox,
                        support_hint=None, debug_out=None):
    height, width = depth_m.shape
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    # **框是带角度的**：检测器给的 angle 是"物体长轴相对图像 +x 轴的转角"，而
    # [x0,y0,x1,y1] 是框**在自己那个坐标系里**的范围，不是正着的矩形。凡是"这个点在不在
    # 框里"的判断都要先转到框自己的坐标系去（_in_box）。当正着的矩形用就是用错了区域：
    # 实测胡萝卜那一帧（角度 -60°）顶面层被框的上下沿横着切齐（y 恰好停在 195/317）。
    angle = float(bbox[4]) if len(bbox) > 4 else 0.0
    center_px = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    half_w, half_h = (x1 - x0) / 2.0, (y1 - y0) / 2.0

    # 先粗看一眼框附近的有效深度，好把"物体有多大"换算成像素（环带和范围都按它定）
    rough = _pixels_in_box((height, width), [x0 - 90.0, y0 - 90.0, x1 + 90.0, y1 + 90.0])
    rough_x, rough_y = rough[:, 0].astype(int), rough[:, 1].astype(int)
    rough_keep = valid_mm[rough_y, rough_x]
    if rough_keep.sum() < 200:
        return {"ok": False, "why": "框附近有效深度太少（%d）" % int(rough_keep.sum())}
    distance = float(np.median(depth_m[rough_y[rough_keep], rough_x[rough_keep]]))

    half_px = max(x1 - x0, y1 - y0) / 2.0
    focal = float(camera_matrix[0, 0])
    radius_m = half_px * distance / focal
    inner_px = max(radius_m + RING_INNER_M, RING_MIN_INNER_M) * focal / distance
    outer_px = (radius_m + RING_OUTER_M) * focal / distance
    reach_px = outer_px * REACH_FACTOR

    # 只在"承托面可能长到"的范围里取点：拿二次曲面在拟合孔径之外推一大片，边缘会直接崩
    region = _pixels_in_box((height, width),
                            [center_px[0] - reach_px - 2, center_px[1] - reach_px - 2,
                             center_px[0] + reach_px + 2, center_px[1] + reach_px + 2])
    xs, ys = region[:, 0].astype(int), region[:, 1].astype(int)
    keep = valid_mm[ys, xs]
    pixels, xs, ys = region[keep], xs[keep], ys[keep]
    depths = depth_m[ys, xs]
    if len(pixels) < 200:
        return {"ok": False, "why": "承托面范围里有效深度太少（%d）" % len(pixels)}
    points = backproject(camera_matrix, dist_coeffs, pixels, depths)
    offset = np.linalg.norm(pixels - center_px, axis=1)

    ring = (offset >= inner_px) & (offset <= outer_px)
    if ring.sum() < 100:
        return {"ok": False, "why": "环带里点太少（%d）" % int(ring.sum())}

    seed = points[ring]
    if support_hint is not None:
        # 调用方（流水线）已经在同一个环带上做过一次稳健的平面拟合了，直接借它的法线当
        # 起点：白捡一次 RANSAC，而且和它后面算物体高度用的那个面**同源**，不会一个歪一个正。
        normal_hint = np.asarray(support_hint[1], dtype=float)
        normal_hint = normal_hint / np.linalg.norm(normal_hint)
        seed_source = "hint"
    else:
        # 相机在原点（本服务一律相机系），"法线朝相机那一侧"就是 origin - 点均值
        plane = seed_plane(seed, np.zeros(3))
        if plane is not None:
            normal_hint = plane["normal"]
            seed_source = "ransac %.2f" % plane["inlier_ratio"]
        else:
            centered = seed - seed.mean(axis=0)
            normal_hint = np.linalg.svd(centered, full_matrices=False)[2][2]
            if normal_hint @ points.mean(axis=0) < 0:
                normal_hint = -normal_hint           # 法线朝相机那一侧＝朝上
            seed_source = "svd"
    frame = _surface_frame(seed, normal_hint)

    # 生长：外沿一圈圈推出去，每圈重拟合一次，并且每圈都过一遍粗网格（剔鼓包）
    history, support = [], ring.copy()
    for step in range(GROW_STEPS):
        bound = outer_px + (reach_px - outer_px) * (step + 1) / float(GROW_STEPS)
        surface = fit_surface(points[support], frame)
        frame = surface["frame"]
        residual = _height_above(surface, points)
        grown = (offset <= bound) & (np.abs(residual) < 0.0015)
        grown = coarse_support(xs, ys, grown)
        history.append({"step": step, "bound_px": round(float(bound), 1),
                        "points": int(grown.sum()), "inliers": round(surface["inliers"], 3)})
        if grown.sum() < 100:
            break
        support = grown

    surface = fit_surface(points[support], frame)
    residual = _height_above(surface, points)
    normal = surface["frame"][3]

    # ---- 承托面得像个承托面：法线方向对了，高度场的**坡度就该接近零** ----
    # 不管承托面多斜（60° 的坡也一样），只要拟合用的那个坐标系的 z 轴就是它的法线，
    # 梯度就接近 0。梯度一大只有一个解释：这个坐标系的法线找错了，拟合出来的是一面墙。
    # 实测崩掉那轮是 hypot(−79, −115) = 140（相当于 89.6°）；正常是 0.001 量级。
    slope = float(np.hypot(surface["coeffs"][1], surface["coeffs"][2]))
    if slope > MAX_SURFACE_SLOPE:
        return {"ok": False,
                "why": "承托面没拟合出来：面上坡度 %.2f（判据 %.2f），法线找错了"
                       % (slope, MAX_SURFACE_SLOPE),
                "report": {"method": "geometric", "slope": round(slope, 4),
                           "seed_normal_from": seed_source}}

    # 物体 = 框内、离面够高、又没高到离谱的点，取**离框心最近**的连通块。
    # 高度上界是必须的：压在手前面的东西（比如 tcp 那个小球，比物体近 270 mm）在图上也
    # 压着物体，连通性分不开，只能靠"它离面 270 mm 高"这一点排掉。
    # ---- 物体阈值：用承托面自己的噪声当尺子（见文件头那段） ----
    free = (offset > outer_px) & (offset <= outer_px + FREE_SAMPLE_PX)
    if int(free.sum()) >= FREE_SAMPLE_MIN:
        sample = residual[free]
        # 用 MAD 而不是分位数：**这一圈里混进别人的东西是常态**（实测胡萝卜那一帧，叶子
        # 落在采样圈里占一成多，p99 直接被抬到 23 mm，阈值跟着变成 46 mm，物体整个漏掉）。
        # MAD 的崩溃点是 50%，一两成的干扰动不了它；p99 的崩溃点是 1%，动一下就废。
        sigma = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) + 1e-9
        # 地板：面特别干净时别把阈值压到零，至少比拟合给出的噪声尺度大一截。
        # 封顶：不许超过物体自己的半径，兜住"旁边真有个高东西、且多到能带动 MAD"这种情况。
        object_band = min(OBJECT_BAND_TAIL * max(sigma, 2.0 * float(surface["scale_m"])),
                          max(radius_m, OBJECT_BAND_M))
        band_source = "自由桌面点 sigma %.2f mm" % (sigma * 1000)
    else:
        object_band = OBJECT_BAND_M
        band_source = "兜底固定值（自由点只有 %d 个）" % int(free.sum())

    raw = ((offset <= outer_px) & (residual > object_band)
           & (residual < MAX_OBJECT_HEIGHT_M))
    if raw.sum() < 20:
        return {"ok": False, "why": "物体点太少（%d）" % int(raw.sum())}
    labels = label_components(xs, ys, raw)
    candidates = []
    for index in range(int(labels.max()) + 1):
        pick = labels == index
        if pick.sum() >= 20:
            candidates.append((index, float(np.percentile(residual[pick], 95.0))))
    candidates = [item for item in candidates if 0.0 < item[1] < MAX_OBJECT_HEIGHT_M]
    if not candidates:
        return {"ok": False, "why": "没有高度说得过去的连通块（%d 块）" % len(candidates)}
    picked = min(candidates, key=lambda item: float(np.min(np.hypot(
        xs[labels == item[0]] - center_px[0], ys[labels == item[0]] - center_px[1]))))[0]
    object_mask = labels == picked

    # 顶面高度只在**框内**统计：框是检测器对"物体在哪"的提名，用它把旁边的石头排除掉是
    # 正当的（原型里那颗贴着物体的石头直径 54 mm、顶比 30 mm 的物体还高，不排除就把整个
    # 顶面层抢走，误差 7.3 mm）。框内点太少才退回整个掩码。
    core = object_mask & _in_box(pixels, center_px, angle, half_w, half_h, 3.0)
    ceiling = top_of_main_mass(residual, core if core.sum() >= 50 else object_mask)

    # 最上面那一层 = 物体高度的顶部 TOP_FRACTION 那一段（比例的来历见文件头常量那段）。
    #
    # **这里不再拿框去圈一遍**。以前收在"框内放宽 8 px"，是为了挡开贴着物体、侧腹正好
    # 落在同一高度上的石头。但框本身不可靠——它的**形状**和**中心**都是检测器给的：
    # 实测胡萝卜那一帧 VLM 给的框，中心离物体中心 59 px、尺寸是物体的两倍，拿它收紧
    # 区域等于把物体的**尖端切掉**（顶面层被框的沿向限位切齐），中心差从 0.92 px 涨到
    # 17.2 px。而挡石头这件事，"顶部这一段"本身已经做了大半（石头的顶没那么高）。
    # 代价：**贴着物体、且一样高**的东西会被一起算进来。真出现再加"在这一层里按连通性
    # 取离框心最近的那一块"，那比拿框圈区域稳——连通性不依赖框准不准。
    band = max(TOP_FRACTION * ceiling, TOP_BAND_MIN_M)
    allowed = object_mask & (residual >= ceiling - band)
    if int(allowed.sum()) < MIN_TOP_POINTS:
        return {"ok": False, "why": "最上面那一层点太少（%d）" % int(allowed.sum())}
    # **这一层里就当等权，不再按高度加权**。这一层是"按高度选出来的"，层里的点已经都
    # 属于物体的上部；再用高度加权，等于对横向位置没来由地再偏一次——物体哪儿厚、哪儿
    # 的点就多、权也大，质心就被拽向厚的那头（实测胡萝卜：带权 3.73 px，等权 1.22 px；
    # 方块 1.64 / 1.64）。等权就是这一层的面积质心，也就是"物体上面那一片的中心"。
    centroid = points[allowed].mean(axis=0)
    # 但面积质心**落在表面之下**——这一层是曲面上（或含一截侧面）的一片，质心在弦高下面：
    # 实测方块低 2.9 mm、胡萝卜低 7.3 mm。而 top_point 的语义是"物体最上面那一点"，
    # grasp.plan 又直接拿它跟承托面的高度差当**物体高度**用，矮了就把爪子停高。
    # 所以横向位置用质心、高度抬到物体的最高处。
    centroid_height = float(_height_above(surface, centroid.reshape(1, 3))[0])
    top_point = centroid + (ceiling - centroid_height) * normal

    # 长轴：**从这一层自己量**（面内最小外接矩形的长边），不用框给的转角。框那个转角是
    # 检测器猜的，实测框转 15°、轴就跟着转 15°；而闭合方向是垂直于长轴的，轴歪 15° 就是
    # 爪子瞄歪 15°，量出来的宽度也跟着变 7~11 mm。几何轴对框的缩放/平移/旋转都不动
    # （实测 ≤1.2°，连长宽比 1.04、本来就没有长短轴的方块也稳）。
    layer = points[allowed] - centroid
    layer = layer - (layer @ normal)[:, None] * normal          # 压到面内
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(helper, normal)
    first = first / np.linalg.norm(first)
    second = np.cross(normal, first)
    (_, _), (side_a, side_b), angle = cv2.minAreaRect(
        np.c_[layer @ first, layer @ second].astype(np.float32))
    theta = np.deg2rad(angle) + (np.pi / 2.0 if side_a < side_b else 0.0)
    long_axis = np.cos(theta) * first + np.sin(theta) * second
    long_axis = long_axis - float(long_axis @ normal) * normal
    long_axis = long_axis / np.linalg.norm(long_axis)

    if debug_out is not None:      # 调试用：把判据真正用的那两个量交出来
        debug_out["residual_mm"] = residual * 1000.0
        debug_out["pixels"] = pixels
        debug_out["object_mask"] = object_mask
        debug_out["outer_px"] = outer_px
        debug_out["ceiling_mm"] = ceiling * 1000.0

    return {
        "ok": True, "top_point": top_point, "height_m": ceiling,
        "long_axis": long_axis,
        "support": {"point": surface["frame"][0].copy(), "normal": normal.copy()},
        # 深度图里被判成"物体"的那些像素（就是上面那个 object_mask）。交出来是为了能跟
        # 2D 检测框摆在一起看：这一路是**从深度里把物体分出来的**，框只用来告诉它往哪儿
        # 找，所以框偏一点、大一圈，它给的点不跟着走（见包里那份扰动实测）。
        "object_pixels": pixels[object_mask].astype(int),
        # **顶面那一层**：真正参与了顶面质心的点（框内、且残差落在顶面高度附近）。
        # 与 object_pixels 的区别是它不含侧面——斜着看时侧面从 0 一直连到顶面高度，
        # 任何"高出几毫米"的阈值都挡不住它们，所以"物体"和"顶面"必须分开给。
        "top_pixels": pixels[allowed].astype(int),
        "report": {
            "method": "geometric", "distance_m": round(distance, 4),
            "seed_normal_from": seed_source,
            "object_band_mm": round(object_band * 1000, 2), "object_band_from": band_source,
            "object_points": int(object_mask.sum()), "core_points": int(core.sum()),
            "dropped_points": int(raw.sum() - object_mask.sum()),
            "ceiling_mm": round(ceiling * 1000, 2),
            "band_mm": round(band * 1000, 2),
            "noise_scale_mm": round(float(surface["scale_m"]) * 1000, 2),
            "coeffs": np.round(surface["coeffs"], 4).tolist(),
            "growth": history,
        }}


# ---------------------------------------------------------------- 对外的入口

def estimate(depth_mm, camera_matrix, dist_coeffs, bbox, method=None,
             support_hint=None, debug_out=None) -> dict:
    """给一帧深度、内参/外参和一个 2D 检测框，算物体的顶面中心。

    Args:
        depth_mm: 深度图，uint16 毫米、0=无效（与真机 D435i、与仓库 wrist_depth 同口径）。
        camera_matrix / dist_coeffs: 相机内参与畸变系数。
        （没有 camera_pose：本服务一律在相机系里进出，相机在原点。要基座系由调用方乘一次。）
        bbox: [x0, y0, x1, y1] 或带角度的 [x0, y0, x1, y1, angle]（只用前四个）。
        method: 'geometric' / 'pure'；不给就用模块级 METHOD。
        support_hint: 可选的 (point, normal)——调用方已经拟合好的承托面。给了就用它的
            法线当拟合起点（省一次 RANSAC，而且和调用方算高度用的面同源）。

    Returns:
        {"ok", "top_point", "height_m", "support", "why", "report"}
        ok 为假时 why 里写清原因——**不返回一个凑合的点**，由调用方决定怎么办。
    """
    chosen = (method or METHOD or "geometric").strip().lower()
    depth_mm = np.asarray(depth_mm)
    valid_mm = (depth_mm > MIN_FRAME_M * 1000.0) & (depth_mm < MAX_FRAME_M * 1000.0)
    depth_m = depth_mm.astype(np.float64) / 1000.0
    if chosen == "pure":
        return _estimate_pure(depth_m, valid_mm, camera_matrix, dist_coeffs, bbox)
    if chosen != "geometric":
        return {"ok": False, "why": "不认识的顶面方法 %r（认 geometric / pure）" % chosen}
    return _estimate_geometric(depth_m, valid_mm, camera_matrix, dist_coeffs,
                               bbox, support_hint, debug_out)
