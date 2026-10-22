"""
ERP 컨트롤 이미지 생성 (erp_control_maker.html 과 같은 계산을 서버에서 수행)

원본 사진을 핀홀(원근) 투영으로 W×H 등장방형 캔버스에 붙이고 빈 곳은 #00ff00 으로 채운다.
yaw: +오른쪽, pitch: +위, hfov: 가로 화각, vfov: 비우면 사진 비율로 계산
"""

import math

import numpy as np
from PIL import Image, ImageFilter, ImageOps

GREEN = np.array([0, 255, 0], np.float32)


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


def make_control(img: Image.Image, yaw, pitch, hfov, vfov=None, W=2048, H=1024) -> Image.Image:
    rad = math.pi / 180
    yaw_r, pitch_r, hf = yaw * rad, pitch * rad, hfov * rad
    iw, ih = img.size
    aspect = ih / iw
    th = math.tan(hf / 2)
    tv = math.tan(vfov * rad / 2) if vfov else th * aspect

    # 파노라마 위 크기 정도로 미리 축소 (에일리어싱 방지) — html 과 동일
    target_w = min(iw, math.ceil(hf / (2 * math.pi) * W * 1.5))
    sw, sh = target_w, max(1, round(target_w * aspect))
    src = np.asarray(img.convert("RGBA").resize((sw, sh), Image.LANCZOS), np.float32)

    xs = (np.arange(W, dtype=np.float64) + 0.5) / W
    ys = (np.arange(H, dtype=np.float64) + 0.5) / H
    lon = (xs - 0.5) * 2 * math.pi
    lat = (0.5 - ys) * math.pi
    lon, lat = np.meshgrid(lon, lat)
    cl, sl = np.cos(lat), np.sin(lat)
    dx, dy, dz = cl * np.sin(lon), sl, cl * np.cos(lon)

    cy, sy, cp, sp = math.cos(yaw_r), math.sin(yaw_r), math.cos(pitch_r), math.sin(pitch_r)
    x1 = dx * cy - dz * sy
    z1 = dx * sy + dz * cy
    y2 = dy * cp - z1 * sp
    z2 = dy * sp + z1 * cp

    out = np.broadcast_to(GREEN, (H, W, 3)).copy()
    front = z2 > 1e-6
    zz = np.where(front, z2, 1.0)
    u = x1 / (zz * th)
    v = y2 / (zz * tv)
    m = front & (u >= -1) & (u <= 1) & (v >= -1) & (v <= 1)

    px = (u[m] + 1) / 2 * sw - 0.5
    py = (1 - v[m]) / 2 * sh - 0.5
    x0 = np.clip(np.floor(px), 0, sw - 1).astype(np.int64)
    y0 = np.clip(np.floor(py), 0, sh - 1).astype(np.int64)
    xa = np.minimum(sw - 1, x0 + 1)
    ya = np.minimum(sh - 1, y0 + 1)
    fx = np.clip(px - x0, 0, 1)[:, None]
    fy = np.clip(py - y0, 0, 1)[:, None]
    c = (src[y0, x0] * (1 - fx) + src[y0, xa] * fx) * (1 - fy) + (src[ya, x0] * (1 - fx) + src[ya, xa] * fx) * fy
    a = c[:, 3:4] / 255.0
    out[m] = c[:, :3] * a + GREEN * (1 - a)

    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def save_quest_jpg(src_png, dst_jpg, quality=95):
    """Quest 뷰어용 JPG: 4:4:4 (색 번짐 없음), 2:1 확인."""
    im = Image.open(src_png).convert("RGB")
    w, h = im.size
    if w != 2 * h:
        im = im.resize((w, w // 2), Image.LANCZOS)
    im.save(dst_jpg, "JPEG", quality=int(quality), subsampling=0, optimize=True)
    return dst_jpg


# ----------------------------------------------------------------------------- 극점(발밑·하늘) 보정
# 360 이미지를 시선축 기준으로 90° 굴려(roll) 발밑은 왼쪽(u=0.25), 하늘 꼭대기는 오른쪽(u=0.75) 적도 위로 옮긴다.
# 거기서는 늘어짐 없이 보이므로 생성 모델이 자연스럽게 다시 그릴 수 있다. 끝나면 되돌려 극점 부분만 합성.

def _dirs(W, H):
    u = (np.arange(W, dtype=np.float64) + 0.5) / W
    v = (np.arange(H, dtype=np.float64) + 0.5) / H
    lon, lat = np.meshgrid((u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi)
    cl = np.cos(lat)
    return cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)


def _sample(src, x, y, z):
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


def _roll90(x, y, z, inverse=False):
    # 정방향: 출력 방향 d_out → 원본 방향 d_in = Rz(90°)·d_out  (x'=-y, y'=x)
    return (y, -x, z) if inverse else (-y, x, z)


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def pole_cap_weight(lat_deg, nadir_cap, zenith_cap, feather):
    """원본 위도(°) → 보정 가중치 0..1 (극점 근처 1)."""
    wn = _smooth((-(90 - nadir_cap) + feather - lat_deg) / (2 * feather)) if nadir_cap > 0 else 0
    wz = _smooth((lat_deg - (90 - zenith_cap) + feather) / (2 * feather)) if zenith_cap > 0 else 0
    return np.maximum(wn, wz)


def pole_prepare(erp_img: Image.Image, nadir_cap=28, zenith_cap=22, feather=6, mask_margin=4):
    """→ (돌린 ERP 이미지, 인페인트 마스크 이미지[흰색=다시 그림])"""
    src = np.asarray(erp_img.convert("RGB"), np.float32)
    H, W = src.shape[:2]
    x, y, z = _dirs(W, H)
    xi, yi, zi = _roll90(x, y, z)
    rot = _sample(src, xi, yi, zi)
    lat_in = np.degrees(np.arcsin(np.clip(yi, -1, 1)))
    m = pole_cap_weight(lat_in, nadir_cap + mask_margin, zenith_cap + mask_margin, feather)
    rot_img = Image.fromarray(np.clip(np.rint(rot), 0, 255).astype(np.uint8), "RGB")
    mask_img = Image.fromarray((m * 255).astype(np.uint8), "L").convert("RGB")
    return rot_img, mask_img


def pole_merge(erp_img: Image.Image, repaired_rot: Image.Image, nadir_cap=28, zenith_cap=22, feather=6) -> Image.Image:
    """다시 그린 '돌린 ERP' 를 원위치로 되돌려 극점 부분만 원본에 부드럽게 합성."""
    src = np.asarray(erp_img.convert("RGB"), np.float32)
    H, W = src.shape[:2]
    rep = np.asarray(repaired_rot.convert("RGB").resize((W, H), Image.LANCZOS), np.float32)
    x, y, z = _dirs(W, H)
    back = _sample(rep, *_roll90(x, y, z, inverse=True))
    lat = np.degrees(np.arcsin(np.clip(y, -1, 1)))
    w = pole_cap_weight(lat, nadir_cap, zenith_cap, feather)[..., None]
    out = src * (1 - w) + back * w
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


# ----------------------------------------------------------------------------- 원본 사진 되살리기 + 톤 맞춤
# 아웃페인트는 사진 부분까지 전부 다시 그려서 실사가 그림처럼 변한다.
# 최종 8K 위에 원본 사진(고해상도)을 같은 위치로 다시 투영해 붙이고, 생성된 주변은 사진 톤에 맞춘다.

def _photo_coords(W, H, rows, yaw, pitch, hfov, aspect, vfov=None):
    """ERP 의 rows 행들에 대해 사진 평면 좌표 (u,v ∈ [-1,1]) 와 앞쪽 여부."""
    rad = math.pi / 180
    th = math.tan(hfov * rad / 2)
    tv = math.tan(vfov * rad / 2) if vfov else th * aspect
    xs = (np.arange(W, dtype=np.float32) + 0.5) / W
    ys = (rows.astype(np.float32) + 0.5) / H
    lon = (xs - 0.5) * 2 * np.pi
    lat = (0.5 - ys) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    cl, sl = np.cos(lat), np.sin(lat)
    dx, dy, dz = cl * np.sin(lon), sl, cl * np.cos(lon)
    cy, sy, cp, sp = math.cos(yaw * rad), math.sin(yaw * rad), math.cos(pitch * rad), math.sin(pitch * rad)
    x1 = dx * cy - dz * sy
    z1 = dx * sy + dz * cy
    y2 = dy * cp - z1 * sp
    z2 = dy * sp + z1 * cp
    front = z2 > 1e-6
    zz = np.where(front, z2, 1.0)
    return x1 / (zz * th), y2 / (zz * tv), front


def _edge_weight(u, v, front, feather):
    """사진 안쪽 1, 가장자리에서 feather(정규화 거리) 동안 부드럽게 0."""
    d = np.minimum(1 - np.abs(u), 1 - np.abs(v))
    w = np.clip(d / feather, 0, 1)
    w = w * w * (3 - 2 * w)
    return np.where(front, w, 0.0), np.where(front, d, -9.0)


def photo_composite(final_img: Image.Image, photo: Image.Image, yaw, pitch, hfov, vfov=None,
                    feather=0.06, inset=0.01, color_strength=0.8, chunk=256) -> Image.Image:
    """final_img(생성된 ERP) 에 원본 사진을 고해상도로 다시 붙이고, 생성 영역 톤을 사진에 맞춤."""
    out = np.asarray(final_img.convert("RGB"), np.float32).copy()
    H, W = out.shape[:2]
    ph = np.asarray(photo.convert("RGB"), np.float32)
    sh, sw = ph.shape[:2]
    aspect = sh / sw

    # 1) 톤 통계 (작은 해상도로): 사진 가장자리 안쪽 띠 vs 생성 영역의 바로 바깥 띠
    sW, sH = 1024, 512
    small = np.asarray(final_img.convert("RGB").resize((sW, sH), Image.BILINEAR), np.float32)
    u, v, f = _photo_coords(sW, sH, np.arange(sH), yaw, pitch, hfov, aspect, vfov)
    _, d = _edge_weight(u, v, f, feather)
    ring_out = (d < -0.02) & (d > -0.35)            # 생성된 바로 바깥
    ph_small = np.asarray(photo.convert("RGB").resize((256, max(1, int(256 * aspect))), Image.BILINEAR), np.float32)

    def ycc(a):
        a = a / 255.0
        y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return np.stack([y, a[..., 2] - y, a[..., 0] - y], -1)

    def rgb(c):
        y, cb, cr = c[..., 0], c[..., 1], c[..., 2]
        r = cr + y
        b = cb + y
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        return np.stack([r, g, b], -1) * 255.0

    if ring_out.sum() > 100:
        g = ycc(small[ring_out])
        p = ycc(ph_small.reshape(-1, 3))
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
            c = ycc(blk)
            c2 = (c - mu_g) * scale + mu_p
            blk = blk * (1 - color_strength) + rgb(c2) * color_strength
        u, v, f = _photo_coords(W, H, rows, yaw, pitch, hfov, aspect, vfov)
        w, _ = _edge_weight(u, v, f, feather)
        w = np.where(np.minimum(1 - np.abs(u), 1 - np.abs(v)) < inset, 0.0, w)   # 사진 테두리 1px 잡티 제외
        m = w > 0
        if m.any():
            px = (u[m] + 1) / 2 * sw - 0.5
            py = (1 - v[m]) / 2 * sh - 0.5
            x0 = np.clip(np.floor(px), 0, sw - 1).astype(np.int64)
            y0 = np.clip(np.floor(py), 0, sh - 1).astype(np.int64)
            x1 = np.minimum(sw - 1, x0 + 1)
            y1 = np.minimum(sh - 1, y0 + 1)
            fx = np.clip(px - x0, 0, 1)[:, None]
            fy = np.clip(py - y0, 0, 1)[:, None]
            s = (ph[y0, x0] * (1 - fx) + ph[y0, x1] * fx) * (1 - fy) + (ph[y1, x0] * (1 - fx) + ph[y1, x1] * fx) * fy
            ww = w[m][:, None]
            sub = blk[m]
            blk[m] = sub * (1 - ww) + s * ww
        out[rows] = blk
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def tone_match(final_img: Image.Image, photo: Image.Image, strength=0.6, chunk=512) -> Image.Image:
    """생성 결과 전체의 밝기·대비·색을 원본 사진 쪽으로 당김 (YCbCr 평균/표준편차 맞춤).
    아웃페인트 결과가 사진보다 훨씬 밝고 평평해서(예: 밝기 126→200) 그림처럼 보이는 것을 완화."""
    if strength <= 0:
        return final_img

    def ycc(a):
        a = a / 255.0
        y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return np.stack([y, a[..., 2] - y, a[..., 0] - y], -1)

    def rgb(c):
        y, cb, cr = c[..., 0], c[..., 1], c[..., 2]
        r, b = cr + y, cb + y
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        return np.stack([r, g, b], -1) * 255.0

    g = ycc(np.asarray(final_img.convert("RGB").resize((1024, 512), Image.BILINEAR), np.float32).reshape(-1, 3))
    ps = photo.convert("RGB")
    ps.thumbnail((512, 512))
    p = ycc(np.asarray(ps, np.float32).reshape(-1, 3))
    mu_g, sd_g = g.mean(0), g.std(0) + 1e-4
    mu_p, sd_p = p.mean(0), p.std(0) + 1e-4
    scale = np.clip(sd_p / sd_g, 0.6, 1.8)

    out = np.asarray(final_img.convert("RGB"), np.float32).copy()
    for r0 in range(0, out.shape[0], chunk):
        blk = out[r0:r0 + chunk]
        c2 = (ycc(blk) - mu_g) * scale + mu_p
        out[r0:r0 + chunk] = blk * (1 - strength) + rgb(c2) * strength
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")



def wrap_seam_score(img_path) -> dict:
    """좌우 끝(360 이음새) 색 차이 vs 일반 이웃 픽셀 차이. ratio 가 2 이하면 눈에 거의 안 띔."""
    im = Image.open(img_path).convert("L")
    if im.width > 4096:
        im = im.resize((4096, 2048), Image.BILINEAR)
    a = np.asarray(im, np.float32)
    wrap = float(np.abs(a[:, 0] - a[:, -1]).mean())
    normal = float(np.median(np.abs(np.diff(a, axis=1)).mean(0))) + 1e-3
    return {"wrap": round(wrap, 1), "normal": round(normal, 1), "ratio": round(wrap / normal, 2)}



# ----------------------------------------------------------------------------- 원본 사진 보존
def keep_mask(photo_size, yaw, pitch, hfov, vfov=None, W=2048, H=1024, feather=0.04) -> Image.Image:
    """생성 때 '다시 그릴 곳' 마스크 (흰색=생성, 검정=원본 사진 유지). 사진 가장자리 feather 만큼은 섞어 그림."""
    sw, sh = photo_size
    u, v, f = _photo_coords(W, H, np.arange(H), yaw, pitch, hfov, sh / sw, vfov)
    w, _ = _edge_weight(u, v, f, feather)
    return Image.fromarray(((1.0 - w) * 255).astype(np.uint8), "L").convert("RGB")


def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    try:
        import cv2
        return cv2.GaussianBlur(a, (0, 0), sigma)
    except Exception:
        return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma)),
                          np.float32)


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
    import cv2
    W, H = pasted.size
    sw, sh = photo_size
    sW, sH = 2048, 1024
    u, v, f = _photo_coords(sW, sH, np.arange(sH), yaw, pitch, hfov, sh / sw, vfov)
    d = np.where(f, np.minimum(1 - np.abs(u), 1 - np.abs(v)), -9.0).astype(np.float32)
    if d.max() <= 0:
        return pasted
    def ss(x):
        x = np.clip(x, 0, 1)
        return (x * x * (3 - 2 * x)).astype(np.float32)
    w_lo = ss((d + band) / (2 * band))                     # 바깥 band → 안쪽 band
    w_hi = ss(d / edge)                                    # 경계 → 안쪽 edge
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
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def paste_photo(final_img: Image.Image, photo: Image.Image, yaw, pitch, hfov, vfov=None, feather=0.05,
                mode="auto", detail=0.9) -> Image.Image:
    """최종 결과에 원본 사진을 다시 붙임 (사진에 있던 건 절대 안 사라짐).

    original: 원본 사진 픽셀 그대로 (사진이 최종 해상도만큼 선명할 때)
    detail:   원본 사진이 작아 늘리면 흐려질 때 — 저주파(색·형태)는 원본, 고주파(선명한 디테일)는 8K 업스케일 결과.
              결과의 사진 영역은 원래 사진에서 업스케일된 것이라 위치가 맞음. 사진 밖은 결과 그대로."""
    pasted = photo_composite(final_img, photo, yaw, pitch, hfov, vfov, feather=feather, color_strength=0.0)
    pasted = harmonize_paste(final_img, pasted, photo.size, yaw, pitch, hfov, vfov)
    m, scale = photo_paste_mode(final_img.width, photo.width, hfov, mode)
    if m == "original":
        return pasted
    sigma = max(0.8, 0.7 * scale)                      # 원본 사진에 없는 세밀함 = 이 반경보다 작은 것
    F = np.asarray(final_img.convert("RGB"), np.float32)
    P = np.asarray(pasted, np.float32)
    gF = _gauss(F, sigma)
    out = F + (_gauss(P, sigma) - gF)
    if detail != 1.0:                                  # 사진 영역의 AI 디테일 세기 (지어낸 결이 과하면 낮춤)
        wp = 1.0 - np.asarray(keep_mask(photo.size, yaw, pitch, hfov, vfov, feather=feather).convert("L")
                              .resize(final_img.size, Image.BILINEAR), np.float32)[..., None] / 255.0
        out += wp * (float(detail) - 1.0) * (F - gF)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


# ----------------------------------------------------------------------------- 영상화: 움직일 영역
# yaw(+오른쪽)·pitch(+위) 는 make_control 과 같은 기준. 영역 = 그 방향을 바라본 평면(핀홀) 화면.

def _view_dirs(yaw, pitch, hfov, w, h):
    """평면 화면 (w×h, 가로 화각 hfov) 각 픽셀의 세계 방향 (dx, dy, dz)."""
    rad = math.pi / 180
    th = math.tan(hfov * rad / 2)
    tv = th * h / w
    u = ((np.arange(w, dtype=np.float64) + 0.5) / w) * 2 - 1
    v = 1 - ((np.arange(h, dtype=np.float64) + 0.5) / h) * 2
    u, v = np.meshgrid(u, v)
    n = np.sqrt((u * th) ** 2 + (v * tv) ** 2 + 1)
    x1, y2, z2 = u * th / n, v * tv / n, 1 / n
    cy, sy, cp, sp = math.cos(yaw * rad), math.sin(yaw * rad), math.cos(pitch * rad), math.sin(pitch * rad)
    dy = y2 * cp + z2 * sp
    z1 = -y2 * sp + z2 * cp
    return x1 * cy + z1 * sy, dy, -x1 * sy + z1 * cy


def view_size(orientation="landscape", long_side=1408, ratio=704 / 512):
    """영상 모델 학습 비율(704×512)에 맞춘 크기. 2배로 잘라 두고 모델 쪽에서 줄여 씀."""
    short = round(long_side / ratio)
    w, h = (long_side, short) if orientation != "portrait" else (short, long_side)
    return w - w % 16, h - h % 16


ASPECT_MIN, ASPECT_MAX = 0.5, 2.0


def region_aspect(r) -> float:
    """영역 화면 가로:세로. aspect 가 없으면 orientation(가로/세로 704:512)에서."""
    a = r.get("aspect")
    if not a:
        a = 704 / 512 if r.get("orientation", "landscape") != "portrait" else 512 / 704
    return float(min(ASPECT_MAX, max(ASPECT_MIN, float(a))))


def region_size(r, area=1408 * 1024, step=64) -> tuple[int, int]:
    """영역 비율대로, 픽셀 수는 학습 크기(704×512)의 4배 근처로. 영상 모델 조건(1단계 절반이 32 배수)에 맞춰 64 배수."""
    a = region_aspect(r)
    w = math.sqrt(area * a)
    h = w / a
    return max(step * 6, int(round(w / step)) * step), max(step * 6, int(round(h / step)) * step)


def perspective(erp_img: Image.Image, yaw, pitch, hfov, w, h) -> Image.Image:
    """360 이미지에서 (yaw, pitch) 방향을 가로 화각 hfov 의 평면 화면 w×h 로 잘라냄."""
    src = np.asarray(erp_img.convert("RGB"), np.float32)
    out = _sample(src, *_view_dirs(yaw, pitch, hfov, w, h))
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def _to_erp_xy(dx, dy, dz, W, H):
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1, 1))
    return (lon / (2 * np.pi) + 0.5) * W, (0.5 - lat / np.pi) * H


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)       # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


def grid_overlay(erp_img: Image.Image, W=2048) -> Image.Image:
    """LLM 이 위치를 숫자로 말할 수 있게 30° 격자와 yaw/pitch 값을 그린 ERP."""
    from PIL import ImageDraw
    H = W // 2
    im = erp_img.convert("RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for yaw in range(-180, 181, 30):
        x = min(W - 1, (yaw / 360 + 0.5) * W)
        d.line([(x, 0), (x, H)], fill=(255, 255, 0), width=4 if yaw == 0 else 2)
        d.text((min(W - 110, x + 6), H // 2 + 8), f"yaw {yaw}", fill=(255, 255, 0), font=_font(26),
               stroke_width=3, stroke_fill=(0, 0, 0))
    for pitch in range(-60, 61, 30):
        y = (0.5 - pitch / 180) * H
        d.line([(0, y), (W, y)], fill=(0, 255, 255), width=3 if pitch == 0 else 2)
        d.text((8, y + 6), f"pitch {pitch:+d}", fill=(0, 255, 255), font=_font(26), stroke_width=3, stroke_fill=(0, 0, 0))
    return im


def region_outline(yaw, pitch, hfov, w, h, W, H, n=40):
    """영역 화면 테두리를 ERP 좌표 선분들로 (좌우 끝을 넘는 곳은 끊음)."""
    m = max(2, round(n * h / w))
    dx, dy, dz = _view_dirs(yaw, pitch, hfov, n, m)
    idx = ([(0, j) for j in range(n)] + [(i, n - 1) for i in range(m)] +
           [(m - 1, j) for j in range(n - 1, -1, -1)] + [(i, 0) for i in range(m - 1, -1, -1)])
    rr, cc = zip(*idx)
    xs, ys = _to_erp_xy(dx[rr, cc], dy[rr, cc], dz[rr, cc], W, H)
    segs, cur = [], [(xs[0], ys[0])]
    for i in range(1, len(xs)):
        if abs(xs[i] - xs[i - 1]) > W / 2:
            segs.append(cur)
            cur = []
        cur.append((float(xs[i]), float(ys[i])))
    segs.append(cur)
    return [s for s in segs if len(s) > 1]


def region_mask(regions: list, W=2048) -> Image.Image:
    """움직일 영역 = 흰색, 정지 = 검정 (ERP)."""
    H = W // 2
    out = np.zeros((H, W), bool)
    dx, dy, dz = _dirs(W, H)
    rad = math.pi / 180
    for r in regions:
        w, h = region_size(r)
        th = math.tan(r["hfov"] * rad / 2)
        tv = th * h / w
        cy, sy = math.cos(r["yaw"] * rad), math.sin(r["yaw"] * rad)
        cp, sp = math.cos(r["pitch"] * rad), math.sin(r["pitch"] * rad)
        x1 = dx * cy - dz * sy
        z1 = dx * sy + dz * cy
        y2 = dy * cp - z1 * sp
        z2 = dy * sp + z1 * cp
        front = z2 > 1e-6
        zz = np.where(front, z2, 1.0)
        out |= front & (np.abs(x1 / (zz * th)) <= 1) & (np.abs(y2 / (zz * tv)) <= 1)
    return Image.fromarray((out * 255).astype(np.uint8), "L")


MOTION_COLORS = [(255, 90, 110), (90, 200, 255), (255, 200, 60), (120, 230, 140), (200, 140, 255)]


def motion_overlay(erp_img: Image.Image, regions: list, W=2048) -> Image.Image:
    """움직일 영역만 원래 밝기, 나머지(정지)는 어둡게 + 번호 테두리."""
    from PIL import ImageDraw
    H = W // 2
    full = erp_img.convert("RGB").resize((W, H), Image.LANCZOS)
    im = Image.blend(full, Image.new("RGB", (W, H), (0, 0, 0)), 0.55)
    im.paste(full, (0, 0), region_mask(regions, W))
    d = ImageDraw.Draw(im)
    for i, r in enumerate(regions):
        w, h = region_size(r)
        col = MOTION_COLORS[i % len(MOTION_COLORS)]
        for seg in region_outline(r["yaw"], r["pitch"], r["hfov"], w, h, W, H):
            d.line(seg, fill=col, width=4)
        x, y = (r["yaw"] / 360 + 0.5) * W, (0.5 - r["pitch"] / 180) * H
        d.ellipse([x - 20, y - 20, x + 20, y + 20], fill=col)
        d.text((x, y), str(i + 1), fill=(0, 0, 0), font=_font(26), anchor="mm")
    return im
