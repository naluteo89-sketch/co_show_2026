"""
ERP(등장방형) 투영 공통 계산 — 다른 erp 모듈이 모두 이 함수들을 쓴다.

yaw: +오른쪽, pitch: +위, hfov: 가로 화각, vfov: 비우면 사진 비율로 계산
"""

import math

import numpy as np
from PIL import Image, ImageOps

GREEN = np.array([0, 255, 0], np.float32)
RAD = math.pi / 180


def load_upright(path_or_file):
    im = Image.open(path_or_file)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGBA")


def exif_hfov(path_or_file):
    """EXIF 35mm 환산 초점거리로 가로 화각 계산. 없으면 None."""
    try:
        im = Image.open(path_or_file)
        exif = im.getexif()
        ifd = exif.get_ifd(0x8769)
        f35 = ifd.get(0xA405)
        if not f35:
            return None
        w, h = ImageOps.exif_transpose(im).size
        long_side = 36.0 if w >= h else 24.0     # 3:2 풀프레임 기준 근사
        return round(math.degrees(2 * math.atan(long_side / (2 * float(f35)))), 2)
    except Exception:
        return None


def pitch_from_horizon(horizon_y, hfov, img_w, img_h):
    """사진 속 수평선 위치(0=위,1=아래) → 카메라 pitch(°, +위)."""
    if horizon_y is None:
        return 0.0
    tan_v = math.tan(math.radians(hfov) / 2) * img_h / img_w
    return round(math.degrees(math.atan((float(horizon_y) - 0.5) * 2 * tan_v)), 2)


def to_image(a: np.ndarray) -> Image.Image:
    """float 배열 → RGB 그림 (반올림·0~255 자르기)."""
    return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8), "RGB")


def smooth(t):
    """smoothstep: 0..1 로 자른 뒤 t²(3-2t)."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def erp_dirs(W, H, rows=None, dtype=np.float64):
    """ERP 각 픽셀(또는 rows 행만)의 단위 방향 (dx, dy, dz)."""
    xs = (np.arange(W, dtype=dtype) + 0.5) / W
    ys = ((np.arange(H) if rows is None else rows).astype(dtype) + 0.5) / H
    lon, lat = np.meshgrid((xs - 0.5) * 2 * np.pi, (0.5 - ys) * np.pi)
    cl = np.cos(lat)
    return cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)


def to_camera(dx, dy, dz, yaw, pitch):
    """세계 방향 → (yaw, pitch) 를 바라보는 카메라 좌표 (x 오른쪽, y 위, z 앞)."""
    cy, sy, cp, sp = math.cos(yaw * RAD), math.sin(yaw * RAD), math.cos(pitch * RAD), math.sin(pitch * RAD)
    x1 = dx * cy - dz * sy
    z1 = dx * sy + dz * cy
    return x1, dy * cp - z1 * sp, dy * sp + z1 * cp


def plane_uv(dx, dy, dz, yaw, pitch, th, tv):
    """방향 → 카메라 평면 좌표 (u, v ∈ [-1,1] 이 화면 안) 와 앞쪽 여부. th/tv = tan(반화각)."""
    x1, y2, z2 = to_camera(dx, dy, dz, yaw, pitch)
    front = z2 > 1e-6
    zz = np.where(front, z2, 1.0)
    return x1 / (zz * th), y2 / (zz * tv), front


def sample_plane(src, u, v):
    """평면 그림 src(h,w,C float) 를 평면 좌표 (u, v) 에서 쌍선형 샘플 (가장자리는 끝값)."""
    sh, sw = src.shape[:2]
    px = (u + 1) / 2 * sw - 0.5
    py = (1 - v) / 2 * sh - 0.5
    x0 = np.clip(np.floor(px), 0, sw - 1).astype(np.int64)
    y0 = np.clip(np.floor(py), 0, sh - 1).astype(np.int64)
    x1 = np.minimum(sw - 1, x0 + 1)
    y1 = np.minimum(sh - 1, y0 + 1)
    fx = np.clip(px - x0, 0, 1)[:, None]
    fy = np.clip(py - y0, 0, 1)[:, None]
    return (src[y0, x0] * (1 - fx) + src[y0, x1] * fx) * (1 - fy) + (src[y1, x0] * (1 - fx) + src[y1, x1] * fx) * fy


def sample_erp(src, x, y, z):
    """방향 (x,y,z) 에서 ERP src(H,W,C float) 를 쌍선형 샘플 (좌우 wrap)."""
    H, W = src.shape[:2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1, 1))
    px = (lon / (2 * np.pi) + 0.5) * W - 0.5
    py = (0.5 - lat / np.pi) * H - 0.5
    x0 = np.floor(px).astype(np.int64)
    y0 = np.clip(np.floor(py).astype(np.int64), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (px - x0)[..., None]
    fy = np.clip(py - y0, 0, 1)[..., None]
    x0w, x1w = x0 % W, (x0 + 1) % W
    return ((src[y0, x0w] * (1 - fx) + src[y0, x1w] * fx) * (1 - fy) +
            (src[y1, x0w] * (1 - fx) + src[y1, x1w] * fx) * fy)


def photo_coords(W, H, rows, yaw, pitch, hfov, aspect, vfov=None):
    """ERP 의 rows 행들에 대해 사진 평면 좌표 (u,v ∈ [-1,1]) 와 앞쪽 여부."""
    th = math.tan(hfov * RAD / 2)
    tv = math.tan(vfov * RAD / 2) if vfov else th * aspect
    return plane_uv(*erp_dirs(W, H, rows, np.float32), yaw, pitch, th, tv)


def edge_weight(u, v, front, feather):
    """사진 안쪽 1, 가장자리에서 feather(정규화 거리) 동안 부드럽게 0."""
    d = np.minimum(1 - np.abs(u), 1 - np.abs(v))
    return np.where(front, smooth(d / feather), 0.0), np.where(front, d, -9.0)


def view_dirs(yaw, pitch, hfov, w, h):
    """평면 화면 (w×h, 가로 화각 hfov) 각 픽셀의 세계 방향 (dx, dy, dz)."""
    th = math.tan(hfov * RAD / 2)
    tv = th * h / w
    u = ((np.arange(w, dtype=np.float64) + 0.5) / w) * 2 - 1
    v = 1 - ((np.arange(h, dtype=np.float64) + 0.5) / h) * 2
    u, v = np.meshgrid(u, v)
    n = np.sqrt((u * th) ** 2 + (v * tv) ** 2 + 1)
    x1, y2, z2 = u * th / n, v * tv / n, 1 / n
    cy, sy, cp, sp = math.cos(yaw * RAD), math.sin(yaw * RAD), math.cos(pitch * RAD), math.sin(pitch * RAD)
    dy = y2 * cp + z2 * sp
    z1 = -y2 * sp + z2 * cp
    return x1 * cy + z1 * sy, dy, -x1 * sy + z1 * cy


def to_erp_xy(dx, dy, dz, W, H):
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1, 1))
    return (lon / (2 * np.pi) + 0.5) * W, (0.5 - lat / np.pi) * H


def roll_array(a: np.ndarray, shift: int, base_width: int = 2048) -> np.ndarray:
    """360 그림 배열을 가로로 돌림. shift 는 base_width 기준 픽셀이라 그림 폭에 맞게 비례."""
    return np.roll(a, int(round(shift * a.shape[1] / base_width)), axis=1)
