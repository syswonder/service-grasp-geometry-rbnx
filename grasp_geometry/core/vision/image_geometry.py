# Description: "图像上的框/角度" → "三维方向/长度" 那一小段。从 grasp.py 搬过来的。
#
# 为什么单独一个文件：这三件事都要**内参 + 畸变 + 像素**，属于视觉，不属于抓取几何。
# 原来它们混在 grasp.py 里，而那儿的文件头写着"纯数学，不碰硬件、不读图像"——不成立。
# 搬进本服务之后 grasp.py 那句才成立。
#
# 一切在**相机系**：相机在原点，所以原来那些 `camera_pose[:3,3]` / `[:3,:3]` 在这儿
# 分别就是 `zeros(3)` 和单位阵，参数直接拿掉。要落到基座系由调用方乘一次 `camera_base`。
import cv2
import numpy as np

from . import plane as plane_mod


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("方向向量的长度是零")
    return vector / norm


def ray_plane_hit(origin, direction, plane) -> np.ndarray:
    """射线与平面的交点；射线与平面平行、或平面在射线背后时抛异常。"""
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = _unit(direction)
    normal = _unit(plane["normal"])
    denominator = float(direction @ normal)
    if abs(denominator) < 1e-9:
        raise ValueError("射线与平面平行，求不出交点")
    distance = float((np.asarray(plane["point"], dtype=float) - origin) @ normal) / denominator
    if distance <= 0:
        raise ValueError("平面在相机背后")
    return origin + distance * direction


def long_axis_on_plane(camera_matrix, dist_coeffs, center_px, angle_deg,
                       top_plane, probe_px: float = 30.0) -> np.ndarray:
    """OBB 的长轴方向，换算成物体顶面上的单位方向（相机系）。

    做法：在图像上沿 OBB 转角取中心两侧各一个点，两条射线打到"过顶面、与支撑面平行"
    的那个面上，两交点之差就是长轴方向。用射线与平面求交而不是直接读深度，是为了避开
    偏移点上可能落到桌面（另一个高度）的情况。
    """
    angle = np.deg2rad(float(angle_deg))
    offset = probe_px * np.array([np.cos(angle), np.sin(angle)])
    center = np.asarray(center_px, dtype=float).reshape(2)
    pixels = np.vstack([center - offset, center + offset])
    directions = plane_mod.ray_directions(camera_matrix, dist_coeffs, pixels)
    # 相机在原点：射线起点是零、方向不用转
    hits = [ray_plane_hit(np.zeros(3), direction, top_plane) for direction in directions]
    axis = hits[1] - hits[0]
    # 投影到顶面内：射线求交本身已经在那个面上，这里只是把数值误差抹掉
    normal = _unit(top_plane["normal"])
    axis = axis - float(axis @ normal) * normal
    return _unit(axis)


MIN_TOP_EXTENT_POINTS = 50      # 点少于这个数就不量了，退回用框


def footprint_extent(pixels, depth_mm, camera_matrix, dist_coeffs, direction):
    """给一层点，量它们**水平**方向上的跨度：沿 `direction` 投影，取最大减最小（米）。

    为什么不投到某个平面上：物体点各自在自己的高度上，把它们的射线投到某一个平面
    （承托面或顶面）上都会把影子摊开——实测方块的物体点投到承托面上是 65 mm、实物 50。
    沿 `direction`（在承托面内、垂直于长轴）算各点**自己**的位置，等于直接量水平轮廓，
    跟物体形状无关，也不用挑一个平面出来。

    为什么给的是"最上面那一层"而不是整个物体的点：物体下半部分的表面是**掠射**的，
    深度在那儿不可信，横向位置会被甩出去——按高度分层实测，方块低于 45 mm 的点能甩到
    真值中心外 39 mm，而它的半宽只有 25 mm。最上面那一层正对相机、深度可信，
    量出来最准。整层一起量是多少（真值：方块 50、胡萝卜 54）：
        方块 51.8   胡萝卜 53.5     ← 不腐蚀
        方块 50.6   胡萝卜 52.0     ← 腐蚀 1 px
    两个都在 2 mm 以内，就不再引入腐蚀这个旋钮了。

    量不出来（点太少）返回 None，由调用方决定怎么办。
    """
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(pixels) < MIN_TOP_EXTENT_POINTS:
        return None
    pixels = pixels[:, :2]
    px = pixels.astype(int)
    depth_m = np.asarray(depth_mm, dtype=float)[px[:, 1], px[:, 0]] / 1000.0
    good = depth_m > 0.0
    if int(good.sum()) < MIN_TOP_EXTENT_POINTS:
        return None
    points = plane_mod.backproject(camera_matrix, dist_coeffs, pixels[good], depth_m[good])
    offset = (points - points.mean(axis=0)) @ _unit(direction)
    return float(offset.max() - offset.min())


def footprint_width(box, camera_matrix, dist_coeffs, plane, closing):
    """目标外接框的四个角打到平面上，量它在闭合方向上的宽度。

    判据要的是"两指合起来要跨过多少"，所以量的是闭合方向上的跨度，不是框的对角线。
    只是个估计：框是图像上的框、物体还有高度，打到平面上的跨度会比实物略长，
    所以判据里留了余量。量不出来（射线打不到那个面）返回 None。
    """
    corners = np.array([[box[0], box[1]], [box[2], box[1]],
                        [box[2], box[3]], [box[0], box[3]]], dtype=float)
    directions = plane_mod.ray_directions(camera_matrix, dist_coeffs, corners)
    try:
        hits = np.array([ray_plane_hit(np.zeros(3), direction, plane)
                         for direction in directions])
    except ValueError:
        return None
    normal = _unit(plane["normal"])
    hits = hits - ((hits - np.asarray(plane["point"], dtype=float)) @ normal)[:, None] * normal
    extent = (hits - hits.mean(axis=0)) @ _unit(closing)
    return float(extent.max() - extent.min())
