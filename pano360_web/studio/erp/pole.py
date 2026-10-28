"""
극점(발밑·하늘) 보정

360 이미지를 시선축 기준으로 90° 굴려(roll) 발밑은 왼쪽(u=0.25), 하늘 꼭대기는 오른쪽(u=0.75) 적도 위로 옮긴다.
거기서는 늘어짐 없이 보이므로 생성 모델이 자연스럽게 다시 그릴 수 있다. 끝나면 되돌려 극점 부분만 합성.
"""

import numpy as np
from PIL import Image

from .projection import erp_dirs, sample_erp, smooth, to_image


def _roll90(x, y, z, inverse=False):
    # 정방향: 출력 방향 d_out → 원본 방향 d_in = Rz(90°)·d_out  (x'=-y, y'=x)
    return (y, -x, z) if inverse else (-y, x, z)


def pole_cap_weight(lat_deg, nadir_cap, zenith_cap, feather):
    """원본 위도(°) → 보정 가중치 0..1 (극점 근처 1)."""
    wn = smooth((-(90 - nadir_cap) + feather - lat_deg) / (2 * feather)) if nadir_cap > 0 else 0
    wz = smooth((lat_deg - (90 - zenith_cap) + feather) / (2 * feather)) if zenith_cap > 0 else 0
    return np.maximum(wn, wz)


def pole_prepare(erp_img: Image.Image, nadir_cap=28, zenith_cap=22, feather=6, mask_margin=4):
    """→ (돌린 ERP 이미지, 인페인트 마스크 이미지[흰색=다시 그림])"""
    src = np.asarray(erp_img.convert("RGB"), np.float32)
    H, W = src.shape[:2]
    xi, yi, zi = _roll90(*erp_dirs(W, H))
    rot = sample_erp(src, xi, yi, zi)
    lat_in = np.degrees(np.arcsin(np.clip(yi, -1, 1)))
    m = pole_cap_weight(lat_in, nadir_cap + mask_margin, zenith_cap + mask_margin, feather)
    mask_img = Image.fromarray((m * 255).astype(np.uint8), "L").convert("RGB")
    return to_image(rot), mask_img


def pole_merge(erp_img: Image.Image, repaired_rot: Image.Image, nadir_cap=28, zenith_cap=22, feather=6) -> Image.Image:
    """다시 그린 '돌린 ERP' 를 원위치로 되돌려 극점 부분만 원본에 부드럽게 합성."""
    src = np.asarray(erp_img.convert("RGB"), np.float32)
    H, W = src.shape[:2]
    rep = np.asarray(repaired_rot.convert("RGB").resize((W, H), Image.LANCZOS), np.float32)
    x, y, z = erp_dirs(W, H)
    back = sample_erp(rep, *_roll90(x, y, z, inverse=True))
    lat = np.degrees(np.arcsin(np.clip(y, -1, 1)))
    w = pole_cap_weight(lat, nadir_cap, zenith_cap, feather)[..., None]
    return to_image(src * (1 - w) + back * w)


def _flat_weight(rows):
    """무늬 없는 곳(하늘 등) 1, 무늬 있는 곳(지붕·가지·바닥 돌) 0."""
    import cv2
    W = rows.shape[1]
    g = cv2.cvtColor(np.clip(rows, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = cv2.GaussianBlur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 3)), (0, 0), max(2.0, W / 330))
    t = smooth((6.0 - e) / 4.0).astype(np.float32)
    k = max(3, int(W / 264) | 1)
    return cv2.erode(t, np.ones((k, k), np.uint8))
