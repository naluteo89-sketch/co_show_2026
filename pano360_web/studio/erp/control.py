"""
ERP 컨트롤 이미지 (tools/erp_control_maker.html 과 같은 계산을 서버에서 수행) · 원본 보존 마스크 · Quest JPG

원본 사진을 핀홀(원근) 투영으로 W×H 등장방형 캔버스에 붙이고 빈 곳은 #00ff00 으로 채운다.
"""

import math

import numpy as np
from PIL import Image

from .projection import GREEN, RAD, edge_weight, erp_dirs, photo_coords, plane_uv, sample_plane, to_image


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
