"""
원본 사진 되살리기 + 톤 맞춤

아웃페인트는 사진 부분까지 전부 다시 그려서 실사가 그림처럼 변한다.
최종 8K 위에 원본 사진(고해상도)을 같은 위치로 다시 투영해 붙이고, 생성된 주변은 사진 톤에 맞춘다.
"""

import numpy as np
from PIL import Image

from .control import keep_mask
from .projection import edge_weight, photo_coords, sample_plane, smooth, to_image


def _ycc(a):
    a = a / 255.0
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.stack([y, a[..., 2] - y, a[..., 0] - y], -1)


def _rgb(c):
    y, cb, cr = c[..., 0], c[..., 1], c[..., 2]
    r, b = cr + y, cb + y
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.stack([r, g, b], -1) * 255.0


def photo_composite(final_img: Image.Image, photo: Image.Image, yaw, pitch, hfov, vfov=None,
                    feather=0.06, inset=0.01, color_strength=0.8, chunk=256) -> Image.Image:
    """final_img(생성된 ERP) 에 원본 사진을 고해상도로 다시 붙이고, 생성 영역 톤을 사진에 맞춤."""
    out = np.asarray(final_img.convert("RGB"), np.float32).copy()
    H, W = out.shape[:2]
    ph = np.asarray(photo.convert("RGB"), np.float32)
    aspect = ph.shape[0] / ph.shape[1]

    # 1) 톤 통계 (작은 해상도로): 사진 가장자리 안쪽 띠 vs 생성 영역의 바로 바깥 띠
    sW, sH = 1024, 512
    small = np.asarray(final_img.convert("RGB").resize((sW, sH), Image.BILINEAR), np.float32)
    u, v, f = photo_coords(sW, sH, np.arange(sH), yaw, pitch, hfov, aspect, vfov)
    _, d = edge_weight(u, v, f, feather)
    ring_out = (d < -0.02) & (d > -0.35)            # 생성된 바로 바깥
    ph_small = np.asarray(photo.convert("RGB").resize((256, max(1, int(256 * aspect))), Image.BILINEAR), np.float32)

    if ring_out.sum() > 100:
        g = _ycc(small[ring_out])
        p = _ycc(ph_small.reshape(-1, 3))
        mu_g, sd_g = g.mean(0), g.std(0) + 1e-4
        mu_p, sd_p = p.mean(0), p.std(0) + 1e-4
        scale = np.clip(sd_p / sd_g, 0.5, 2.0)
    else:
        mu_g = mu_p = np.zeros(3)
        scale = np.ones(3)
        color_strength = 0.0

    # 2) 행 단위로: 생성 영역 톤 맞춤 + 사진 영역 원본으로 교체
    for r0 in range(0, H, chunk):
        rows = np.arange(r0, min(H, r0 + chunk))
        blk = out[rows]
        if color_strength > 0:
            c2 = (_ycc(blk) - mu_g) * scale + mu_p
            blk = blk * (1 - color_strength) + _rgb(c2) * color_strength
        u, v, f = photo_coords(W, H, rows, yaw, pitch, hfov, aspect, vfov)
        w, _ = edge_weight(u, v, f, feather)
        w = np.where(np.minimum(1 - np.abs(u), 1 - np.abs(v)) < inset, 0.0, w)   # 사진 테두리 1px 잡티 제외
        m = w > 0
        if m.any():
            ww = w[m][:, None]
            blk[m] = blk[m] * (1 - ww) + sample_plane(ph, u[m], v[m]) * ww
        out[rows] = blk
    return to_image(out)


def tone_match(final_img: Image.Image, photo: Image.Image, strength=0.6, chunk=512) -> Image.Image:
    """생성 결과 전체의 밝기·대비·색을 원본 사진 쪽으로 당김 (YCbCr 평균/표준편차 맞춤).
    아웃페인트 결과가 사진보다 훨씬 밝고 평평해서(예: 밝기 126→200) 그림처럼 보이는 것을 완화."""
    if strength <= 0:
        return final_img
    g = _ycc(np.asarray(final_img.convert("RGB").resize((1024, 512), Image.BILINEAR), np.float32).reshape(-1, 3))
    ps = photo.convert("RGB")
    ps.thumbnail((512, 512))
    p = _ycc(np.asarray(ps, np.float32).reshape(-1, 3))
    mu_g, sd_g = g.mean(0), g.std(0) + 1e-4
    mu_p, sd_p = p.mean(0), p.std(0) + 1e-4
    scale = np.clip(sd_p / sd_g, 0.6, 1.8)

    out = np.asarray(final_img.convert("RGB"), np.float32).copy()
    for r0 in range(0, out.shape[0], chunk):
        blk = out[r0:r0 + chunk]
        c2 = (_ycc(blk) - mu_g) * scale + mu_p
        out[r0:r0 + chunk] = blk * (1 - strength) + _rgb(c2) * strength
    return to_image(out)


def photo_paste_mode(final_w: int, photo_w: int, hfov: float, mode="auto") -> tuple[str, float]:
    """원본 사진이 최종 해상도보다 흐리면 'detail'(원본 색·형태 + 결과의 선명한 디테일), 충분히 선명하면 'original'."""
    final_ppd = final_w / 360.0
    photo_ppd = photo_w / max(1e-3, float(hfov))
    if mode in ("original", "detail"):
        return mode, final_ppd / photo_ppd
    return ("original" if photo_ppd >= 0.95 * final_ppd else "detail"), final_ppd / photo_ppd


def harmonize_paste(final_img: Image.Image, pasted: Image.Image, photo_size, yaw, pitch, hfov, vfov=None,
                    band=0.45, edge=0.04, sigma_frac=0.012) -> Image.Image:
    """붙인 원본 사진이 주변(생성 부분)과 '사각형으로 따로 노는' 것을 없앰 (2대역 합성).
    저주파(밝기·색·안개·하늘 톤): 사진 경계 양쪽 band 폭에 걸쳐 '바깥 생성 부분'과 '안쪽 사진'을 넓게 섞음.
      — 각자 자기 쪽 픽셀만으로 흐림(마스크 정규화)이라 경계의 단차가 섞이지 않음.
    고주파(디테일): 경계 바로 안쪽 edge 폭에서 생성 → 사진으로 바뀜. 사진 안쪽은 원본 그대로."""
    import cv2                                         # opencv 없는 파이썬에서도 서버는 뜨도록 여기서
    W, H = pasted.size
    sw, sh = photo_size
    sW, sH = 2048, 1024
    u, v, f = photo_coords(sW, sH, np.arange(sH), yaw, pitch, hfov, sh / sw, vfov)
    d = np.where(f, np.minimum(1 - np.abs(u), 1 - np.abs(v)), -9.0).astype(np.float32)
    if d.max() <= 0:
        return pasted
    w_lo = smooth((d + band) / (2 * band)).astype(np.float32)      # 바깥 band → 안쪽 band
    w_hi = smooth(d / edge).astype(np.float32)                     # 경계 → 안쪽 edge
    m_in = (d > 0).astype(np.float32)
    sig = max(1.0, sigma_frac * sW)
    F = np.asarray(final_img.convert("RGB"), np.float32)
    P = np.asarray(pasted.convert("RGB"), np.float32)
    Fs = cv2.resize(F, (sW, sH), interpolation=cv2.INTER_AREA)
    Ps = cv2.resize(P, (sW, sH), interpolation=cv2.INTER_AREA)

    def mblur(x, m):                                       # 마스크 안 픽셀만으로 흐림
        num = cv2.GaussianBlur(x * m[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(m, (0, 0), sig)[..., None]
        return num / np.maximum(den, 1e-3)

    LF, LP = mblur(Fs, 1 - m_in), mblur(Ps, m_in)
    L = LF * (1 - w_lo[..., None]) + LP * w_lo[..., None]
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    wh = up(w_hi)[..., None]
    out = up(L)
    out += (F - up(LF)) * (1 - wh)
    out += (P - up(LP)) * wh
    return to_image(out)


def paste_photo(final_img: Image.Image, photo: Image.Image, yaw, pitch, hfov, vfov=None, feather=0.05,
                mode="auto", detail=0.9) -> Image.Image:
    """최종 결과에 원본 사진을 다시 붙임 (사진에 있던 건 절대 안 사라짐).

    original: 원본 사진 픽셀 그대로 (사진이 최종 해상도만큼 선명할 때)
    detail:   원본 사진이 작아 늘리면 흐려질 때 — 저주파(색·형태)는 원본, 고주파(선명한 디테일)는 8K 업스케일 결과.
              결과의 사진 영역은 원래 사진에서 업스케일된 것이라 위치가 맞음. 사진 밖은 결과 그대로."""
    import cv2                                         # opencv 없는 파이썬에서도 서버는 뜨도록 여기서
    pasted = photo_composite(final_img, photo, yaw, pitch, hfov, vfov, feather=feather, color_strength=0.0)
    pasted = harmonize_paste(final_img, pasted, photo.size, yaw, pitch, hfov, vfov)
    m, scale = photo_paste_mode(final_img.width, photo.width, hfov, mode)
    if m == "original":
        return pasted
    sigma = max(0.8, 0.7 * scale)                      # 원본 사진에 없는 세밀함 = 이 반경보다 작은 것
    F = np.asarray(final_img.convert("RGB"), np.float32)
    P = np.asarray(pasted, np.float32)
    gF = cv2.GaussianBlur(F, (0, 0), sigma)
    out = F + (cv2.GaussianBlur(P, (0, 0), sigma) - gF)
    if detail != 1.0:                                  # 사진 영역의 AI 디테일 세기 (지어낸 결이 과하면 낮춤)
        wp = 1.0 - np.asarray(keep_mask(photo.size, yaw, pitch, hfov, vfov, feather=feather).convert("L")
                              .resize(final_img.size, Image.BILINEAR), np.float32)[..., None] / 255.0
        out += wp * (float(detail) - 1.0) * (F - gF)
    return to_image(out)
