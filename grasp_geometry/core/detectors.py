# Description: 从一帧彩色里找出目标在哪儿。搬自 skill 包 pipeline.py 的 VlmDetector。
#
# 为什么走 VLM：宿主那个 YOLO（host-services/yolo）是在真实蔬果照片上微调的，**认不出
# 仿真里渲染出来的几何体**（实测：把三样参数化网格摆好、拍清楚，检出 0 个；反而把红方块
# 认成 potato，置信度 0.99）。VLM 是"看图说话"，不挑形状，仿真和真机都能用。
#
# 提示词留在这个文件里、不挪进配置文件：它是要跟着算法一起迭代的东西，跟代码同版本更省事。
# 凭据（endpoint / key / model）在 `config.*.yaml` 里。
import base64
import json
import re
import urllib.request

import cv2


class VlmDetector:
    """给一帧彩色，回一个带角度的像素框；没找到回 None。"""

    PROMPT = (
        'You are a machine vision expert. Locate the "{name}" in this image.\n'
        'The image may show a tilted surface with objects on it. Find the object that is '
        'most clearly "{name}" and most graspable.\n'
        'Return PURE JSON, no markdown, with:\n'
        '{{"box_center_x": float, "box_center_y": float, "box_width": float, '
        '"box_height": float, "angle_deg": float, "failed": bool}}\n'
        'All coordinates are NORMALISED to 0..1 of the image width/height '
        '(box_center_* is the box centre, box_*_size is the full width/height).\n'
        'angle_deg is the in-plane rotation of the object\'s LONG axis, in degrees, '
        'measured from the image +x axis, in the range -90..90.\n'
        'If the object is not visible, set "failed": true.'
    )

    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 timeout_s: float = 90.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        # 最近一次的完整记录，供 locate 的诊断留档用；不含 API key。
        self.last_result = None

    def _record(self, *, request: dict, response_text: str, parsed: dict | None,
                error: str | None = None) -> None:
        self.last_result = {
            "model": self.model,
            "request": request,
            "response_text": response_text,
            "parsed": parsed,
            "error": error,
        }

    def diagnostics(self) -> dict:
        return self.last_result or {"model": self.model, "response_text": "", "parsed": None}

    def detect(self, image) -> dict | None:
        """把这一帧发给 VLM，拿回归一化框，换算成像素。没找到返回 None。

        发给 VLM 那一趟走 JPEG：那是给外部模型看的，跟本服务对调用方的**不压缩**传输
        是两回事。
        """
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("这一帧编不成 JPEG")
        payload_obj = {
            "model": self.model, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self.PROMPT.format(name=self.name)},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64,"
                           + base64.b64encode(encoded.tobytes()).decode()}}]}]}
        request_meta = {"model": self.model, "temperature": 0.0,
                        "target_name": self.name, "image_shape": list(image.shape)}
        payload = json.dumps(payload_obj).encode()
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                content = json.loads(response.read())["choices"][0]["message"]["content"]
        except Exception as error:
            self._record(request=request_meta, response_text="", parsed=None,
                         error=f"{type(error).__name__}: {error}")
            raise

        # 模型爱把 JSON 包在 ``` 里，从正文里抠第一个花括号块
        match = re.search(r"\{.*\}", content, re.S)
        if match is None:
            self._record(request=request_meta, response_text=content, parsed=None,
                         error="VLM 没给出 JSON")
            raise RuntimeError("VLM 没给出 JSON：%s" % content[:120])
        answer = json.loads(match.group(0))
        self._record(request=request_meta, response_text=content, parsed=answer)
        if answer.get("failed"):
            return None

        height, width = image.shape[:2]
        cx = float(answer["box_center_x"]) * width
        cy = float(answer["box_center_y"]) * height
        half_w = float(answer["box_width"]) * width / 2
        half_h = float(answer["box_height"]) * height / 2
        angle = float(answer.get("angle_deg") or 0.0)
        x0, y0 = max(0.0, cx - half_w), max(0.0, cy - half_h)
        x1, y1 = min(width - 1.0, cx + half_w), min(height - 1.0, cy + half_h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            self.last_result["error"] = "解析出的框太小"
            return None
        self.last_result["pixel_bbox"] = [x0, y0, x1, y1, angle]
        return {"bbox": [float(x0), float(y0), float(x1), float(y1), angle],
                "center_px": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                "corners_px": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "source": "vlm"}
