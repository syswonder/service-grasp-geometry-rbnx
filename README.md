# service-grasp-geometry-rbnx

Robonix 感知服务：**深度几何抓取位姿估计**。目录名
`robonix.service.grasp_geometry`，能力面 `robonix/service/perception/grasp_geometry/*`。

它把"一帧 RGB-D → 物体顶面点 / 承托面 / 长轴 / 闭合方向 / 夹持跨度（相机系）"
这一步从抓取技能里独立出来，成为一个机器人无关、可被任意抓取技能/Agent 复用的服务。
与目录里已有的 `robonix.service.grasp_pose`（bbox + 2D 单应 + 固定桌面高度 → 位姿）
是两条路线：本服务不依赖固定台面高度，适合移动抓取（底座位置、姿态、桌面高度会变）。

## 能力面

| Contract | Transport | Purpose |
|---|---|---|
| `robonix/service/perception/grasp_geometry/locate` | MCP + gRPC | 量一次；服务自己取相机最新一帧，调用方给 `target` 或可选 `box` |

返回 `result_json`（相机系，米）：
`top_point`、`support{point,normal}`、`long_axis`、`closing`、`footprint_width`、
`bbox`、`frame_id`、`request_id`、`diagnostics`。量不出来时 `success=false`，
`message` 写清原因，**不返回凑合值**。

## 运行时依赖

- **相机原语**：`robonix/primitive/camera/rgb` + `depth`（atlas 解析话题；
  `config.camera_provider_id` 指定实例，多相机部署必填）。
- **VLM 端点**（可选，但"不给 box 的调用"需要）：OpenAI 兼容 `/chat/completions`，
  凭据放部署配置/环境变量，不进包。
- `rclpy`、`numpy`、`opencv-python`；构建期需要 `rbnx` CLI。

## 构建与启动

```bash
bash scripts/build.sh    # rbnx codegen -p . --mcp
bash scripts/start.sh    # exec python3 -m grasp_geometry.atlas_bridge
```

`rbnx boot` 拉起本包进程后，框架自动注册共享 lifecycle Driver，并由
`CMD_INIT` 把部署清单里的 `config:` 交进来；`on_init` 解析相机、建订阅、等首帧。

## 启动顺序

本服务在 `CMD_INIT` 里解析相机话题并建订阅，所以部署清单里**必须排在相机原语之后**
（与 `object_detect` 对相机的要求相同）。相机比服务晚起时 INIT 会失败（不猜话题、
不退回写死值）——把相机实例放到前面即可。

## 配置

完整键表见 [`config.spec`](config.spec)，可直接复制的模板见
[`config.example.yaml`](config.example.yaml)。最小配置：

```yaml
- name: grasp_geometry
  url: https://github.com/syswonder/service-grasp-geometry-rbnx
  config:
    camera_provider_id: realsense_camera
    camera_matrix: [[902.2, 0.0, 657.8], [0.0, 902.0, 365.0], [0.0, 0.0, 1.0]]
    dist_coeffs: [0.120, -0.348, 0.0019, -0.00045, 0.245]
    top_point_source: topface
    vlm_base_url: ${VLM_BASE_URL}
    vlm_api_key: ${VLM_API_KEY}
    vlm_model: ${VLM_MODEL}
```

**本体参数（尺寸/内参/标定/VLM 凭据）属于 robot 部署仓库**，不要写回本包。

## 目录

```
grasp_geometry/
  atlas_bridge.py   能力声明 + 生命周期 + locate 编排 + 留档
  camera.py         相机原语解析、Image→ndarray、最新帧缓存与新鲜度检查
  config.py         配置解析与校验
  core/             算法核心（与 host-services/grasp-vision 同源）
    locate.py detectors.py vision/{plane,topface,image_geometry}
capabilities/
  perception/grasp_geometry/locate.v1.toml
  lib/grasp_geometry/srv/Locate.srv
scripts/{build,start,stop}.sh
config.spec  config.example.yaml
```

## 留档

每次请求在 `config.log_dir`（默认 `rbnx-build/data/grasp_geometry_logs/`）下留
`request_id` 目录：请求参数、原始 RGB/深度、VLM 原文、结果/异常。只保留最近
`log_keep` 次。

## 许可

MulanPSL-2.0。核心算法来自本仓库 `host-services/grasp-vision`（同一套实现）。
