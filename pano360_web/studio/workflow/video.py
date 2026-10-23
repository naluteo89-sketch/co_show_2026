"""
360 영상: 사용자 노드 파일(comfy_nodes/video/*.json)을 그대로 쓰고 웹 값만 넣어 실행

노드 id 로 찾음:
  1131 원본 360 · 1128/1112 Positive/Negative · 1102 길이(초) · 1114 fps · 1132 360 LoRA · 1100 시드 · SaveVideo 전부
  마스크: 1162 LoadImage(칠한 마스크 그림) 가 있으면 그 그림을 넣고,
         없으면 사각 마스크(1134 SolidMask 크기 + 1136 MaskComposite 위치, 1133 캔버스 기준)에 칠한 범위의 좌표를 넣음.
  계산 노드(ComfyMathExpression)는 숫자로 바꿔 넣음.

크롭 생성 노드 파일 (event_cropgen_*.json):
  칠한 범위만 원본 360(8K)에서 잘라 일반 영상으로 고해상도 생성(2단계) → 칠한 곳 밖은 사진 그대로 → 4K 360 원래 자리에 붙임.
  장면마다 다른 값(자를 범위·생성 크기·붙일 자리)은 칠한 마스크로 계산해서 아래 노드에 넣음 (장면 값은 노드 파일에 두지 않음).
  1202 ImageCrop(원본) · 1203/1204 생성 크기(2단계/1단계) · 1208 CropMask(마스크 2048 좌표)
  1210/1212 마스크 크기(1단계/2단계) · 1219 4K 크기 · 1221 4K 붙일 자리 · 1218 4K 360 (4096x2048 고정)
"""

import copy

import numpy as np

from .core import random_seed, ui_node, ui_set, ui_to_api, ui_values

VIDEO_IDS = {"image": 1131, "positive": 1128, "negative": 1112, "seconds": 1102, "fps": 1114, "lora": 1132, "seed": 1100,
             "mask_image": 1162, "rect": 1134, "rect_pos": 1136, "rect_canvas": 1133, "rect_feather": 1135}
VIDEO_CROP_IDS = {"src_crop": 1202, "gen_hi": 1203, "gen_lo": 1204, "mask_crop": 1208, "mask_lo": 1210,
                  "mask_hi": 1212, "back_scale": 1219, "paste": 1221, "full": 1218}
CROP_MAX_PIXELS = 2560 * 1088          # 2단계 생성 크기 상한 (GPU 부담 기준)
OUT_W, OUT_H = 4096, 2048              # 최종 360 영상


def video_mask_mode(ui) -> str:
    if (n := ui_node(ui, VIDEO_IDS["mask_image"])) is not None and n["type"] == "LoadImage":
        return "image"
    if ui_node(ui, VIDEO_IDS["rect"]) is not None and ui_node(ui, VIDEO_IDS["rect_pos"]) is not None:
        return "rect"
    return "none"


def video_rect_canvas(ui) -> tuple[int, int]:
    nm = ui_values(ui, VIDEO_IDS["rect_canvas"])
    return int(nm.get("width", 2048)), int(nm.get("height", 1024))


def video_rect_feather(ui) -> int:
    """사각 마스크 경계 흐림(1135 FeatherMask) 의 가장 큰 값 (없으면 0)."""
    return int(max(ui_values(ui, VIDEO_IDS["rect_feather"]).values(), default=0))


def video_has_crop(ui) -> bool:
    n = ui_node(ui, VIDEO_CROP_IDS["src_crop"])
    return n is not None and n["type"] == "ImageCrop" and n.get("mode", 0) not in (2, 4)


def fill_video_node(template_ui: dict, *, image: str, positive: str, negative: str, seconds=4, lora_strength=None,
                    seed=None, prefix="VR360/flood", mask_image: str | None = None, rect: tuple | None = None,
                    crop: dict | None = None):
    """노드 파일(UI)에 값 넣기 → (채운 UI, 시드, 정보). rect = (x, y, w, h) — 1133 캔버스 좌표.
    crop = crop_plan() 결과 — 크롭 생성 노드 파일(video_has_crop)일 때 필요."""
    ui = copy.deepcopy(template_ui)
    ids = VIDEO_IDS
    seed = random_seed(seed)
    ui_set(ui, ids["image"], "image", image)
    ui_set(ui, ids["positive"], "text", positive)
    ui_set(ui, ids["negative"], "text", negative)
    secs = max(1, int(round(float(seconds))))
    ui_set(ui, ids["seconds"], "value", secs)
    if ui_node(ui, ids["seed"]) is not None:
        ui_set(ui, ids["seed"], "noise_seed", seed)
    if lora_strength is not None and ui_node(ui, ids["lora"]) is not None:
        ui_set(ui, ids["lora"], "strength_model", float(lora_strength))
    mode = video_mask_mode(ui)
    if mode == "image":
        if not mask_image:
            raise ValueError("마스크 그림이 없어요")
        ui_set(ui, ids["mask_image"], "image", mask_image)
    elif mode == "rect":
        if not rect:
            raise ValueError("마스크 범위가 없어요")
        x, y, w, h = (int(v) for v in rect)
        ui_set(ui, ids["rect"], "width", w)
        ui_set(ui, ids["rect"], "height", h)
        ui_set(ui, ids["rect_pos"], "x", x)
        ui_set(ui, ids["rect_pos"], "y", y)
        if ui_node(ui, ids["rect_feather"]) is not None:          # 경계 흐림이 사각형보다 크면 줄임
            lim = max(0, min(w, h) // 2 - 1)
            for k in ("left", "top", "right", "bottom"):
                if int(ui_values(ui, ids["rect_feather"]).get(k, 0)) > lim:
                    ui_set(ui, ids["rect_feather"], k, lim)
    else:
        raise ValueError("노드 파일에서 마스크 노드(1162 LoadImage 또는 1134 SolidMask + 1136 MaskComposite)를 못 찾았어요")
    if video_has_crop(ui):
        if not crop:
            raise ValueError("크롭 생성 노드 파일에는 잘라낼 범위(crop)가 필요해요")
        fill_crop(ui, crop)
    k = 0
    for n in ui["nodes"]:
        if n["type"] == "SaveVideo":
            k += 1
            ui_set(ui, n["id"], "filename_prefix", f"{prefix}_{k}")
    fps = int(ui_values(ui, ids["fps"]).get("value", 24))
    return ui, seed, {"mode": mode, "seconds": secs, "fps": fps, "frames": secs * fps + 1, "rect": rect, "crop": crop}


def video_node_api(filled_ui: dict) -> tuple[dict, list]:
    """채운 UI → API. 계산 노드는 숫자로. 반환 (그래프, 저장 노드 목록)."""
    api = ui_to_api(filled_ui)
    ids = VIDEO_IDS
    vals = {str(ids["seconds"]): int(api[str(ids["seconds"])]["inputs"]["value"]),
            str(ids["fps"]): int(api[str(ids["fps"])]["inputs"]["value"])} if str(ids["seconds"]) in api else {}

    def src_val(nid, slot):                      # 링크 출처 값 (PrimitiveInt · 계산 노드)
        nid = str(nid)
        if nid in vals:
            return vals[nid]
        n = api.get(nid)
        if n and n["class_type"] == "ComfyMathExpression":
            env = {}
            for k, v in n["inputs"].items():
                if k.startswith("values."):
                    env[k.split(".", 1)[1]] = src_val(v[0], v[1]) if isinstance(v, list) else v
            r = eval(n["inputs"]["expression"], {"__builtins__": {}}, env)   # 노드 파일의 간단한 산술식
            return float(r) if slot == 0 else int(r) if slot == 1 else bool(r)
        return None

    calc = [k for k, n in api.items() if n["class_type"] in ("ComfyMathExpression", "PrimitiveInt", "PrimitiveFloat")]
    for n in api.values():
        for k, v in list(n["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and v[0] in calc:
                r = src_val(v[0], v[1])
                if r is None:
                    raise ValueError(f"계산 노드 {v[0]} 값을 못 구했어요")
                n["inputs"][k] = r
    for k in calc:
        api.pop(k, None)
    saves = []
    for k, n in api.items():
        if n["class_type"] == "SaveVideo":
            src = api.get(str((n["inputs"].get("video") or [None])[0]), {})
            saves.append({"node": k, "audio": "audio" in src.get("inputs", {})})
    saves.sort(key=lambda s: (not s["audio"], s["node"]))          # 소리 있는 것 먼저
    return api, saves


def crop_plan(mask, src_size, *, margin=0.12, max_pixels=CROP_MAX_PIXELS, min_aspect=0.6, max_aspect=3.0):
    """칠한 마스크(PIL L, 보통 2048x1024) → 자를 범위·생성 크기·4K 붙일 자리.
    마스크 범위에 여유를 붙이고, 생성 크기(64 배수, 상한 max_pixels)와 같은 비율로 맞춤."""
    MW, MH = mask.size
    m = np.asarray(mask.convert("L")) > 127
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    if x0 <= 2 and x1 >= MW - 2:
        raise ValueError("칠한 범위가 360 좌우 끝에 걸쳐 있어요 — 한쪽으로 모아서 칠해 주세요")
    pad = int(round(margin * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(MW, x1 + pad), min(MH, y1 + pad)
    w, h = x1 - x0, y1 - y0
    aspect = min(max_aspect, max(min_aspect, w / h))
    gw = max(256, int((max_pixels * aspect) ** 0.5) // 64 * 64)
    gh = max(256, int(round(gw / aspect / 64)) * 64)
    while gw * gh > max_pixels:
        gw -= 64
    aspect = gw / gh                                   # 생성 크기 비율에 정확히 맞춤 (짧은 쪽을 넓힘)
    if w / h < aspect:
        w = min(MW, int(round(h * aspect)))
        h = min(MH, int(round(w / aspect)))
    else:
        h = min(MH, int(round(w / aspect)))
        w = min(MW, int(round(h * aspect)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x = int(round(min(max(0, cx - w / 2), MW - w)))
    y = int(round(min(max(0, cy - h / 2), MH - h)))
    sx, sy = src_size[0] / MW, src_size[1] / MH
    ox, oy = OUT_W / MW, OUT_H / MH
    return {"mask": [x, y, w, h],
            "src": [int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))],
            "gen": [gw, gh], "gen_lo": [gw // 2, gh // 2],
            "out": [int(round(x * ox)), int(round(y * oy)), int(round(w * ox)), int(round(h * oy))]}


def fill_crop(ui, plan):
    ids = VIDEO_CROP_IDS
    for nid, box in ((ids["src_crop"], plan["src"]), (ids["mask_crop"], plan["mask"])):
        for k, v in zip(("x", "y", "width", "height"), box):
            ui_set(ui, nid, k, v)
    for nid, (w, h) in ((ids["gen_hi"], plan["gen"]), (ids["mask_hi"], plan["gen"]),
                        (ids["gen_lo"], plan["gen_lo"]), (ids["mask_lo"], plan["gen_lo"]),
                        (ids["back_scale"], plan["out"][2:])):
        ui_set(ui, nid, "width", int(w))
        ui_set(ui, nid, "height", int(h))
    ui_set(ui, ids["paste"], "x", int(plan["out"][0]))
    ui_set(ui, ids["paste"], "y", int(plan["out"][1]))
