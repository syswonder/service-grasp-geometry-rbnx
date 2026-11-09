# robonix.service.grasp_geometry 的配置说明（config.spec）。
#
# 这些键由部署清单的 `config:` 块经 Driver(CMD_INIT, config_json) 传进来；本文件
# 只做人/模型可读说明，不参与运行时校验（运行时以 atlas_bridge 的 on_init 校验为准）。
config:
  # string, default: "".
  # 相机 provider id（atlas 上的原语实例名，例如 wrist_camera / realsense_camera）。
  # 为空时要求 atlas 上 robonix/primitive/camera/{rgb,depth} 恰好各有一个提供方；
  # 多相机部署必须填。填了但找不到该 provider 时 INIT 失败（不猜）。
  camera_provider_id: ""

  # string, default: "".
  # 直接指定话题时**跳过 atlas 解析**，两个都要给；用于没有 atlas 的离线联调。
  rgb_topic: ""
  depth_topic: ""

  # 3x3 相机内参，行优先。必填：深度反投影要用，服务不从 ROS 相机信息推。
  # 单应/标定属于部署参数，放在 robot 部署仓库的 config: 里。
  camera_matrix:
    - [615.0, 0.0, 320.0]
    - [0.0, 615.0, 240.0]
    - [0.0, 0.0, 1.0]

  # float list，默认 []（等于无畸变）。OpenCV 畸变系数。
  dist_coeffs: []

  # string, default: "".
  # 结果 result_json 里的 frame_id；为空时沿用相机消息 header.frame_id。
  frame_id: ""

  # float, default: 1.0。
  # 深度像素值 → 毫米的换算系数。RealSense aligned depth 是 uint16 毫米，填 1.0；
  # 有些相机是 0.1 mm 单位，填 0.1。
  depth_units_mm: 1.0

  # string, default: "topface"。
  # "物体顶面点"取法：topface（从深度图分离物体、取顶面加权质心，推荐）或
  # box_center（检测框中心像素 + 框内深度中位数反投影，框偏了会跟着偏）。
  top_point_source: "topface"

  # 检测用视觉大模型（OpenAI 兼容 /chat/completions）。只有调用方不给 box 时用；
  # 三项里任何一项为空 → 不给 box 的 locate 调用直接失败（不会静默退回别的检测器）。
  # 这些都是凭据/部署值：不要写进包，放 robot 部署仓库的 config: 或环境变量注入。
  vlm_base_url: ""
  vlm_api_key: ""
  vlm_model: ""
  # float, default: 90.0；VLM 单次请求超时。
  vlm_timeout_s: 90.0

  # float, default: 2.0。
  # 一帧 RGB / 深度超过这么久没更新就当没有（防止拿几分钟前的旧帧量）。
  frame_stale_s: 2.0

  # float, default: 120.0；单次 locate 的预算（含 VLM 检测）。
  locate_timeout_s: 120.0

  # string, default: ""；每次请求的现场留档目录（原始帧 npy + VLM 原文 + 中间量）。
  # 为空 → 落在包目录 rbnx-build/data/grasp_geometry_logs/。
  log_dir: ""
  # int, default: 50；留档最多保留多少次请求（0 = 不留）。
  log_keep: 50
