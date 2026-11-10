"""
ERP 컨트롤 이미지 (tools/erp_control_maker.html 과 같은 계산을 서버에서 수행) · 원본 보존 마스크 · Quest JPG

원본 사진을 핀홀(원근) 투영으로 W×H 등장방형 캔버스에 붙이고 빈 곳은 #00ff00 으로 채운다.
"""

import math

import numpy as np
from PIL import Image

from .projection import GREEN, RAD, edge_weight, erp_dirs, photo_coords, plane_uv, sample_plane, smooth, to_image


def make_control(img: Image.Image, yaw, pitch, hfov, vfov=None, W=2048, H=1024) -> Image.Image:
    hf = hfov * RAD
    iw, ih = img.size
    aspect = ih / iw
    th = math.tan(hf / 2)
    tv = math.tan(vfov * RAD / 2) if vfov else th * aspect

    # 파노라마 위 크기 정도로 미리 축소 (에일리어싱 방지) — html 과 동일
    target_w = min(iw, math.ceil(hf / (2 * math.pi) * W * 1.5))
    sw, sh = target_w, max(1, round(target_w * aspect))
    src = np.asarray(img.convert("RGBA").resize((sw, sh), Image.LANCZOS), np.float32)

    u, v, front = plane_uv(*erp_dirs(W, H), yaw, pitch, th, tv)
    out = np.broadcast_to(GREEN, (H, W, 3)).copy()
    m = front & (u >= -1) & (u <= 1) & (v >= -1) & (v <= 1)
    c = sample_plane(src, u[m], v[m])
    a = c[:, 3:4] / 255.0
    out[m] = c[:, :3] * a + GREEN * (1 - a)
    return to_image(out)


def keep_mask(photo_size, yaw, pitch, hfov, vfov=None, W=2048, H=1024, feather=0.04, floor=0.0) -> Image.Image:
    """생성 때 '다시 그릴 곳' 마스크 (흰색=생성, 검정=원본 사진 유지). 사진 가장자리 feather 만큼은 섞어 그림.
    floor > 0: 사진 안쪽도 floor 만큼은 다시 그림 (회색 마스크 · DifferentialDiffusion 과 함께 '약하게 다듬기')."""
    sw, sh = photo_size
    u, v, f = photo_coords(W, H, np.arange(H), yaw, pitch, hfov, sh / sw, vfov)
    w, _ = edge_weight(u, v, f, feather)
    return Image.fromarray(((1.0 - w * (1.0 - float(floor))) * 255).astype(np.uint8), "L").convert("RGB")


def edge_lock_mask(photo_size, yaw, pitch, hfov, vfov=None, W=2048, H=1024, band=0.24, keep_boxes=(), people_boxes=(),
                   keep_band=0.06, people_pad=0.15, soft=0.03) -> Image.Image:
    """1차 생성용 '다시 그릴 곳' 마스크 (흰색=생성, 검정=원본 유지, 회색=부분 · DifferentialDiffusion 과 함께).
    - 사진 안쪽은 원본 유지, 가장자리 band(정규화 거리, 0.24 = 각 변에서 사진 폭·높이의 12%) 만 다시 그림.
    - keep_boxes(랜드마크·간판, 사진 비율 상자): 가장자리에 걸쳐도 keep_band 까지만 다시 그림.
    - people_boxes(사람 상자): people_pad 만큼 넓혀 완전히 다시 그림 (사람 지우기, 다른 규칙보다 우선)."""
    sw, sh = photo_size
    u, v, f = photo_coords(W, H, np.arange(H), yaw, pitch, hfov, sh / sw, vfov)
    w, _ = edge_weight(u, v, f, band)
    gen = 1.0 - w
    inside = f & (np.abs(u) <= 1) & (np.abs(v) <= 1)
    fx, fy = (u + 1) / 2, (1 - v) / 2

    def box_w(b, pad):
        x0, y0, x1, y1 = [float(t) for t in b[:4]]
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
        d = np.minimum(np.minimum(fx - x0, x1 - fx), np.minimum(fy - y0, y1 - fy))
        return np.where(inside, smooth(d / max(soft, 1e-4)), 0.0)

    if keep_boxes:
        wk, _ = edge_weight(u, v, f, keep_band)
        gk = 1.0 - wk
        for b in keep_boxes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                s = box_w(b, 0.0)
                gen = gen * (1 - s) + np.minimum(gen, gk) * s
    for b in people_boxes or ():
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            gen = np.maximum(gen, box_w(b, people_pad))
    return Image.fromarray((np.clip(gen, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")


def _box_weight(fx, fy, inside, b, pad, soft):
    """사진 비율 상자 b=[x0,y0,x1,y1] (pad 만큼 넓힘) 안 1 → 가장자리 soft 동안 0."""
    x0, y0, x1, y1 = [float(t) for t in b[:4]]
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    d = np.minimum(np.minimum(fx - x0, x1 - fx), np.minimum(fy - y0, y1 - fy))
    return np.where(inside, smooth(d / max(soft, 1e-4)), 0.0)


def c_form_masks(photo_size, yaw, pitch, hfov, vfov=None, W=2048, H=1024, *, band=0.24, keep_boxes=(),
                 people_boxes=(), keep_band=0.06, people_pad=0.15, r1_extra=55.0, r2=140.0, overlap=8.0,
                 overlap_full=10.0, overlap_ramp=15.0):
    """C형태 2K 생성 마스크 3장 (흰색=그림, 검정=그대로, 회색=부분 · DifferentialDiffusion).
    1단계: 사진 좌우로 hfov/2 + r1_extra 도까지 (사진 안쪽 유지 · 가장자리 띠 · 사람 상자 = edge_lock_mask).
    2단계: 거기서 ±r2 도까지 (앞 단계와 overlap 도 겹쳐 섞음).
    3단계: 남은 뒤쪽 틈 — 반 바퀴 돌린 캔버스 기준 (0°/360° 가 가운데에서 한 번에 이어 그려짐)."""
    m1 = np.asarray(edge_lock_mask(photo_size, yaw, pitch, hfov, vfov, W, H, band=band, keep_boxes=keep_boxes,
                                   people_boxes=people_boxes, keep_band=keep_band, people_pad=people_pad))[..., 0] / 255.0
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 360.0
    d = np.abs(((lon - float(yaw) + 180.0) % 360.0) - 180.0)            # 정면(사진 방향)에서 떨어진 각도
    # 단계 사이: 앞 단계 쪽으로 overlap_full 도는 완전히 다시 그리고, 그 앞 overlap_ramp 도는 점점 약하게
    # (4스텝 모델은 부분 세기만으로는 잘 안 섞여 경계선이 남음 → 경계를 다시 그린 띠 안쪽으로)
    of, orp = max(0.0, float(overlap_full)), max(1.0, float(overlap_ramp))
    R1 = float(hfov) / 2 + float(r1_extra)
    R2 = max(R1 + of + orp + 10.0, float(r2))
    c1 = (d <= R1).astype(np.float64)
    c2 = smooth((d - (R1 - of - orp)) / orp) * (d <= R2)
    c3 = smooth((d - (R2 - of - orp)) / orp)
    m1 = m1 * c1[None, :]
    m2 = np.broadcast_to(c2[None, :], (H, W))
    m3 = np.roll(np.broadcast_to(c3[None, :], (H, W)), -(W // 2), axis=1)   # _roll_half 와 같은 방향
    img = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")
    return img(m1), img(m2), img(m3), {"R1": round(R1, 1), "R2": round(R2, 1)}


def refine_mask_8k(photo_size, yaw, pitch, hfov, vfov=None, W=8192, H=4096, *, band=0.24, photo_level=0.33,
                   text_boxes=(), text_pad=0.25, soft=0.03, pad_right=1536, chunk=256) -> Image.Image:
    """8K 리파인 노이즈 마스크 (흰색=리파인 그대로, 회색=약하게, 검정=손대지 않음).
    생성 영역 1 · 사진 안쪽 photo_level · 글자 상자 0. 리파인 이음새 패딩(pad_right)만큼 왼쪽 앞부분을 오른쪽에 이어 붙임."""
    sw, sh = photo_size
    out = np.zeros((H, W), np.uint8)
    for r0 in range(0, H, chunk):
        rows = np.arange(r0, min(H, r0 + chunk))
        u, v, f = photo_coords(W, H, rows, yaw, pitch, hfov, sh / sw, vfov)
        w, _ = edge_weight(u, v, f, band)
        val = 1.0 - w * (1.0 - float(photo_level))
        inside = f & (np.abs(u) <= 1) & (np.abs(v) <= 1)
        fx, fy = (u + 1) / 2, (1 - v) / 2
        for b in text_boxes or ():
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                s = _box_weight(fx, fy, inside, b, text_pad, soft)
                val = val * (1 - s)
        out[rows] = (np.clip(val, 0, 1) * 255).astype(np.uint8)
    if pad_right:
        out = np.concatenate([out, out[:, :int(pad_right)]], axis=1)
    return Image.fromarray(out, "L").convert("RGB")


def text_mask_8k(photo_size, yaw, pitch, hfov, vfov=None, W=8192, H=4096, *, text_boxes=(), pad=0.15, soft=0.02,
                 chunk=256) -> Image.Image:
    """8K 에서 글자 상자 자리 = 1 (원본 사진 원래 해상도 투영을 넣을 곳), 가장자리 soft 동안 0."""
    sw, sh = photo_size
    out = np.zeros((H, W), np.uint8)
    for r0 in range(0, H, chunk):
        rows = np.arange(r0, min(H, r0 + chunk))
        u, v, f = photo_coords(W, H, rows, yaw, pitch, hfov, sh / sw, vfov)
        inside = f & (np.abs(u) <= 1) & (np.abs(v) <= 1)
        fx, fy = (u + 1) / 2, (1 - v) / 2
        val = np.zeros(u.shape)
        for b in text_boxes or ():
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                val = np.maximum(val, _box_weight(fx, fy, inside, b, pad, soft))
        out[rows] = (np.clip(val, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out, "L").convert("RGB")

def green_people(ctrl: Image.Image, photo_size, yaw, pitch, hfov, vfov=None, people_boxes=(), pad=0.15) -> Image.Image:
    """컨트롤 이미지의 사람 상자 자리를 초록(#00ff00 = 비어 있음)으로 칠함 → 모델이 사람 정보 없이 배경으로 채움
    (참조 ReferenceLatent 에 사람이 남아 있으면 그 자리를 다시 그려도 사람을 다시 그림)."""
    if not people_boxes:
        return ctrl
    W, H = ctrl.size
    sw, sh = photo_size
    u, v, f = photo_coords(W, H, np.arange(H), yaw, pitch, hfov, sh / sw, vfov)
    inside = f & (np.abs(u) <= 1) & (np.abs(v) <= 1)
    fx, fy = (u + 1) / 2, (1 - v) / 2
    m = np.zeros(u.shape, bool)
    for b in people_boxes:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            m |= _box_weight(fx, fy, inside, b, pad, 1e-4) > 0.5
    a = np.asarray(ctrl.convert("RGB")).copy()
    a[m] = GREEN.astype(np.uint8) if hasattr(GREEN, "astype") else GREEN
    return Image.fromarray(a, "RGB")


def photo_erp(img: Image.Image, yaw, pitch, hfov, vfov=None, W=8192, H=4096, feather=0.06, inset=0.005,
              chunk=256) -> tuple[Image.Image, Image.Image]:
    """원본 사진을 원래 해상도로 W×H ERP 에 투영 (8K 단계에서 사진 자리 디테일을 원본에서 가져오기 위함).
    반환: (RGB 그림 · 사진 밖은 검정, 섞기 마스크 L · 사진 안 255 → 가장자리 feather 동안 0)."""
    src = np.asarray(img.convert("RGB"), np.float32)
    aspect = src.shape[0] / src.shape[1]
    rgb = np.zeros((H, W, 3), np.uint8)
    msk = np.zeros((H, W), np.uint8)
    for r0 in range(0, H, chunk):
        rows = np.arange(r0, min(H, r0 + chunk))
        u, v, f = photo_coords(W, H, rows, yaw, pitch, hfov, aspect, vfov)
        w, d = edge_weight(u, v, f, feather)
        w = np.where(d < inset, 0.0, w)                        # 사진 테두리 잡티 제외
        m = w > 0
        if not m.any():
            continue
        blk = np.zeros((len(rows), W, 3), np.float32)
        blk[m] = sample_plane(src, u[m], v[m])
        rgb[rows] = np.clip(np.rint(blk), 0, 255).astype(np.uint8)
        msk[rows] = np.clip(np.rint(w * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB"), Image.fromarray(msk, "L").convert("RGB")


def photo_point_dir(fx, fy, yaw, pitch, hfov, aspect, vfov=None):
    """사진 안 한 점 (fx, fy: 0~1, 왼쪽 위 0) → 360 방향 (yaw°, pitch°). photo_coords / plane_uv 의 역변환."""
    th = math.tan(math.radians(hfov) / 2)
    tv = math.tan(math.radians(vfov) / 2) if vfov else th * aspect
    u, v = fx * 2 - 1, 1 - fy * 2
    x, y, z = u * th, v * tv, 1.0
    n = math.sqrt(x * x + y * y + 1)
    x, y, z = x / n, y / n, z / n
    cy, sy, cp, sp = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    dy = y * cp + z * sp
    z1 = -y * sp + z * cp
    dx, dz = x * cy + z1 * sy, -x * sy + z1 * cy
    return math.degrees(math.atan2(dx, dz)), math.degrees(math.asin(max(-1.0, min(1.0, dy))))


def text_region_dirs(photo_size, regions, yaw, pitch, hfov, vfov=None, W=8192, H=4096):
    """글자 상자(사진 비율) → 360 위치: 가운데 yaw/pitch, 사진 정면 기준 좌우 차이, 각 크기, 8K ERP 픽셀 상자.
    HMD(Quest) 는 ERP 가운데(yaw 0)를 처음 정면으로 봄 → yaw 는 그 기준 (오른쪽 +, 위 +)."""
    sw, sh = photo_size
    aspect = sh / sw
    out = []
    for i, r in enumerate(regions or []):
        b = (r or {}).get("box") or []
        if len(b) != 4:
            continue
        x0, y0, x1, y1 = [float(t) for t in b]
        cyaw, cpit = photo_point_dir((x0 + x1) / 2, (y0 + y1) / 2, yaw, pitch, hfov, aspect, vfov)
        pts = [photo_point_dir(fx, fy, yaw, pitch, hfov, aspect, vfov) for fx in (x0, x1) for fy in (y0, y1)]
        ys = [p[0] for p in pts]
        if max(ys) - min(ys) > 180:                      # 360 경계(±180°)를 걸친 상자
            ys = [y + 360 if y < 0 else y for y in ys]
        ps = [p[1] for p in pts]
        px = [((y / 360 + 0.5) % 1.0) * W for y in (min(ys), max(ys))]
        py = [(0.5 - p / 180) * H for p in (max(ps), min(ps))]
        out.append({"index": i, **{k: r[k] for k in ("text", "script", "name", "role") if k in r},
                    "yaw": round(cyaw, 2), "pitch": round(cpit, 2), "yaw_from_photo": round(cyaw - yaw, 2),
                    "size_deg": [round(max(ys) - min(ys), 2), round(max(ps) - min(ps), 2)],
                    "erp_px": [int(px[0]), int(py[0]), int(px[1]), int(py[1])]})
    return out


def layout_360(photo_size, objects, yaw, pitch, hfov, vfov=None, W=8192):
    """360 배치 기록: 사진이 덮는 yaw 범위 + 사진 속 물체 위치 + 사진 밖(생성) 구역 yaw 범위 (HMD 정면 = yaw 0)."""
    sw, sh = photo_size
    left, _ = photo_point_dir(0.0, 0.5, yaw, pitch, hfov, sh / sw, vfov)
    right, _ = photo_point_dir(1.0, 0.5, yaw, pitch, hfov, sh / sw, vfov)
    top = photo_point_dir(0.5, 0.0, yaw, pitch, hfov, sh / sw, vfov)[1]
    bot = photo_point_dir(0.5, 1.0, yaw, pitch, hfov, sh / sw, vfov)[1]
    gap = 360 - (right - left)
    return {"front": "HMD 처음 정면 = yaw 0 (8K ERP 가운데 x=%d)" % (W // 2),
            "photo": {"yaw": [round(left, 1), round(right, 1)], "pitch": [round(bot, 1), round(top, 1)]},
            "generated": {"right_side": [round(right, 1), round(right + gap / 3, 1)],
                          "behind": [round(right + gap / 3, 1), round(right + 2 * gap / 3, 1)],
                          "left_side": [round(right + 2 * gap / 3 - 360, 1), round(left, 1)]},
            "objects": text_region_dirs(photo_size, objects, yaw, pitch, hfov, vfov, W=W, H=W // 2)}


def subject_front_yaw(photo_size, objects, pitch, hfov, vfov=None, limit=60.0):
    """주인공(main_subject) 상자 가운데가 HMD 정면(yaw 0)에 오도록 하는 사진 yaw. 없으면 None."""
    sw, sh = photo_size
    for o in objects or []:
        b = (o or {}).get("box") or []
        if o.get("role") == "main_subject" and len(b) == 4:
            off, _ = photo_point_dir((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0.0, pitch, hfov, sh / sw, vfov)
            return round(max(-limit, min(limit, -off)), 1) + 0.0
    return None


def photo_quality_ratio(photo_w, hfov, final_w=8192) -> float:
    """원본 사진의 1°당 픽셀 / 최종 ERP 의 1°당 픽셀 (1 이상이면 원본이 최종보다 선명)."""
    return (float(photo_w) / max(1e-3, float(hfov))) / (float(final_w) / 360.0)


def to_2to1(im: Image.Image) -> Image.Image:
    """가로:세로 2:1 이 아니면 폭 기준으로 맞춤."""
    w, h = im.size
    return im if w == 2 * h else im.resize((w, w // 2), Image.LANCZOS)


def save_quest_jpg(src_png, dst_jpg, quality=95, optimize=True):
    """Quest 뷰어용 JPG: 4:4:4 (색 번짐 없음), 2:1 확인.
    optimize 는 아주 큰 이미지에서 libjpeg 버퍼 오류가 날 수 있어 끌 수 있게 둠."""
    im = to_2to1(Image.open(src_png).convert("RGB"))
    im.save(dst_jpg, "JPEG", quality=int(quality), subsampling=0, optimize=optimize)
    return im.size
