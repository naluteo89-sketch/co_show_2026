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


def _arc_at(a, arc_deg, full_deg, arc_pole=30.0, pole_deg=85.0):
    """흐림 각도: full_deg 에서 arc_deg → 극점 쪽(pole_deg)으로 갈수록 arc_pole 까지 (하늘 꼭대기의 넓은 부채꼴 색 띠까지 모이게)."""
    return arc_deg + (max(arc_pole, arc_deg) - arc_deg) * float(smooth((a - full_deg) / max(1e-3, pole_deg - full_deg)))


def _row_lowpass(rows, lat_deg, arc_deg, start_deg, full_deg):
    """위도별 가로(원형) 가우시안: sigma 를 구면상 같은 각도(arc_deg, 극점 쪽은 더 넓게)로 → 극점에 가까울수록 한 점으로 모임."""
    W = rows.shape[1]
    out = rows.copy()
    k = np.fft.rfftfreq(W)
    px_per_deg = W / 360.0
    for i, lat in enumerate(lat_deg):
        a = abs(lat)
        if a < start_deg:
            continue
        sig = _arc_at(a, arc_deg, full_deg) * px_per_deg / max(np.cos(np.radians(a)), 1e-4)
        g = np.exp(-2 * (np.pi * k * sig) ** 2)
        lp = np.fft.irfft(np.fft.rfft(rows[i], axis=0) * g[:, None], n=W, axis=0)
        t = float(smooth((a - start_deg) / (full_deg - start_deg)))
        out[i] = rows[i] * (1 - t) + lp * t
    return out


def _flat_weight(rows):
    """무늬 없는 곳(하늘 등) 1, 무늬 있는 곳(지붕·가지·바닥 돌) 0."""
    import cv2
    W = rows.shape[1]
    g = cv2.cvtColor(np.clip(rows, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = cv2.GaussianBlur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 3)), (0, 0), max(2.0, W / 330))
    t = smooth((6.0 - e) / 4.0).astype(np.float32)
    k = max(3, int(W / 264) | 1)
    return cv2.erode(t, np.ones((k, k), np.uint8))


def pole_converge(img_path, arc_deg=8.0, start_deg=50.0, full_deg=62.0) -> dict:
    """천정·발밑의 방사형 별무늬(극점에서 한 점으로 안 모임) 없애기 — 무늬 없는 곳만.
    하늘 픽셀만으로 평균(마스크 정규화)해서 지붕·나뭇가지 색이 하늘로 번지지 않음. 무늬 있는 바닥은 그대로.
    반환: 위·아래에서 실제로 바뀐 평균 정도(0~255)."""
    im = np.array(Image.open(img_path).convert("RGB"))      # uint8 그대로 (8K float 전체는 메모리가 큼)
    H = im.shape[0]
    lat = 90.0 - (np.arange(H) + 0.5) * 180.0 / H
    n = int(np.ceil((90.0 - start_deg) / 180.0 * H))
    pad = min(H // 8, n // 2 + 8)                           # 게이트 흐림이 띠 끝에서 잘리지 않게
    changed = {}
    for name, sl, core in (("zenith", slice(0, n + pad), slice(0, n)), ("nadir", slice(H - n - pad, H), slice(pad, n + pad))):
        rows = im[sl].astype(np.float32)
        w = _flat_weight(rows)[..., None]
        wa = w ** 4                                    # 평균에 쓰는 무게: 경계(지붕·가지 옆) 픽셀은 거의 안 씀 → 어두운 번짐 방지
        num = _row_lowpass(rows * wa, lat[sl], arc_deg, start_deg, full_deg)
        den = _row_lowpass(np.repeat(wa, 3, 2), lat[sl], arc_deg, start_deg, full_deg)
        lp = np.where(den > 0.2, num / np.maximum(den, 1e-3), rows)     # 하늘 픽셀이 너무 적은 곳은 손대지 않음
        new = rows * (1 - w) + lp * w
        changed[name] = round(float(np.abs(new[core] - rows[core]).mean()), 2)
        im[sl] = np.clip(new + 0.5, 0, 255).astype(np.uint8)   # 띠 아래(pad) 줄은 lowpass 가 0 이라 그대로
    Image.fromarray(im).save(img_path)
    return changed
