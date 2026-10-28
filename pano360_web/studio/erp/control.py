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


def keep_mask(photo_size, yaw, pitch, hfov, vfov=None, W=2048, H=1024, feather=0.04, strength=1.0) -> Image.Image:
    """생성 때 '다시 그릴 곳' 마스크 (흰색=생성, 검정=원본 사진 유지). 사진 가장자리 feather 만큼은 섞어 그림.
    strength < 1 이면 사진 영역이 회색(1 - strength) → 완전히 잠그지 않고 그만큼만 덜 바뀜 (글씨·형태 최대한 유지)."""
    sw, sh = photo_size
    u, v, f = photo_coords(W, H, np.arange(H), yaw, pitch, hfov, sh / sw, vfov)
    w, _ = edge_weight(u, v, f, feather)
    s = float(np.clip(strength, 0.0, 1.0))
    return Image.fromarray(((1.0 - s * w) * 255).astype(np.uint8), "L").convert("RGB")


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
