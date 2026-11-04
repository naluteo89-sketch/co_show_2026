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
