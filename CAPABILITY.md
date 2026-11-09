---
description: 深度几何抓取位姿估计。取相机最新一帧 RGB-D，分离承托面与物体顶面，返回 top_point / support / long_axis / closing / footprint_width（相机系）；不依赖 2D 单应与固定桌面高度。
---

# 抓取几何（`robonix.service.grasp_geometry`）

一帧 RGB-D 进，"物体在哪、朝哪、多大、坐在什么面上"出。**一切在相机系**（OpenCV
约定：x 右、y 下、z 前）；要落到臂/底盘基座系由调用方自己乘一次标定或 TF。

目录里另一个同名能力是 `robonix.service.grasp_pose`：它吃 bbox、用 2D 单应 +
固定桌面高度，直接给一个垂直抓取位姿。本服务不同：

| | `grasp_pose` | `grasp_geometry`（本服务） |
|---|---|---|
| 输入 | 调用方给的 bbox | `target`（开放词表，服务自己检测）或可选 bbox |
| 数据 | RGB（深度是忽略的） | RGB + 对齐深度 |
| 输出 | `PoseStamped`（一个抓取位姿） | 几何量：顶面点、承托面、长轴、闭合方向、跨度 |
| 台面高度 | 配置写死 | 每帧从深度图拟合 |
| 适用 | 固定姿态 + 固定桌面高度 | 桌面高度/坡度/物体位置变化（移动抓取） |

## 能力面

| 契约 | 传输 | 说明 |
|---|---|---|
| `robonix/service/perception/grasp_geometry/locate` | MCP、gRPC | 量一次；服务取最新一帧，调用方只给 target/box |

生命周期由框架自动注册的共享 `robonix/lifecycle/driver` 承担（本包不单独声明 driver）。

## 运行时路径

```
调用方（技能/Agent）
  -> locate(target, box="")                        MCP / gRPC
  -> 服务：取相机原语最新一帧 RGB-D（自己订阅，不传图）
  -> 核心：检测(VLM，若没给 box) -> 承托面拟合 -> 顶面提取 -> 长轴/闭合/跨度
  -> result_json（相机系）
  -> 调用方：乘一次 camera->base 的标定/TF，再决定悬停/下爪
```

服务持续订阅相机原语（`robonix/primitive/camera/rgb` + `depth`，话题名经 atlas
解析，`camera_provider_id` 可指定实例）。**相机不再由抓取技能订阅**——谁做视觉谁
拿帧，数据通路只有一条。

## 配置

键与默认值见 `config.spec`，部署模板见 `config.example.yaml`。要点：

- `camera_provider_id`：用哪个相机原语实例；多相机部署必填。
- `camera_matrix` / `dist_coeffs`：**本体标定，必填**，放 robot 部署仓库。
- `depth_units_mm`：RealSense 对齐深度是 uint16 毫米，填 `1.0`。
- `top_point_source`：`topface`（推荐，从深度图分离物体顶面）或 `box_center`。
- `vlm_*`：不给 box 时自检测用；缺凭据时"给 box 的调用"仍可用。
- `frame_stale_s`：帧超过这么久没更新就拒绝测量（不拿旧帧量）。

## 构建与运行

```bash
bash scripts/build.sh     # rbnx codegen：proto_gen + robonix_mcp_types
bash scripts/start.sh     # exec python3 -m grasp_geometry.atlas_bridge
```

依赖：ROS 2 Humble 的 rclpy、`numpy`、`opencv-python`（见 `requirements.txt`），
以及 `rbnx` CLI 与 `robonix_api`（`rbnx path robonix-api`）。

## 布局

```
grasp_geometry/
  atlas_bridge.py   能力声明 + 生命周期 + locate 编排 + 留档
  camera.py         相机原语解析、Image->ndarray、最新帧缓存/新鲜度检查
  config.py         部署配置解析与校验
  core/
    locate.py       一帧 -> 几何量（核心算法，与 host-services/grasp-vision 同源）
    detectors.py    VLM 检测器（OpenAI 兼容 /chat/completions）
    vision/         承托面拟合(plane) / 顶面提取(topface) / 反投影与几何(image_geometry)
capabilities/
  perception/grasp_geometry/locate.v1.toml
  lib/grasp_geometry/srv/Locate.srv
```

## 留档与排障

每次 `locate` 在 `log_dir`（默认 `rbnx-build/data/grasp_geometry_logs/`）下留一个
以 `request_id` 命名的目录：`request.json`、`rgb.jpg`、`depth.npy`、`detector.json`
（用了 VLM 时）、`result.json`；失败时另有 `error.txt`。只保留最近 `log_keep` 次。
`result_json.request_id` 就是这个目录名——从调用方日志一行就能追到现场。

常见失败（`success=false`，`message` 写清原因）：

- `还没有收到完整的 RGB-D 帧` / `帧已停更`：相机原语没起或在掉帧；
- `目标不在腕上相机的画面里`：换个观察位再看（只有这一类值得换角度）；
- `框中心那块几乎没有深度`：反光/太近，相机没测到物体本身；
- `VLM 凭据不全`：补 `vlm_*` 或调用时给 `box`。

## 许可

MulanPSL-2.0。核心算法与仓库内 `host-services/grasp-vision` 同源（一份代码搬进包，
不再依赖宿主 HTTP 服务）。
