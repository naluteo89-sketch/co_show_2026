"""
좌우 이음새(360 뒤쪽 x=0) 후처리: 가는 세로선 지우기 · 양쪽 톤 맞추기 · 이음새 점수
"""

import numpy as np
from PIL import Image

from .projection import smooth as smoothstep


def _median1d(a: np.ndarray, size: int) -> np.ndarray:
    """열 방향(0축) 이동 중앙값 (가장자리는 끝값 반복)."""
    k = size // 2
    pad = np.concatenate([np.repeat(a[:1], k, 0), a, np.repeat(a[-1:], k, 0)], 0)
    win = np.lib.stride_tricks.sliding_window_view(pad, size, axis=0)
    return np.median(win, axis=-1)


def _wrap_window(rgb: np.ndarray, half: int) -> np.ndarray:
    """오른쪽 끝 half 열 + 왼쪽 앞 half 열을 이어 붙인 이음새 창 (float)."""
    W = rgb.shape[1]
    return np.concatenate([rgb[:, W - half:], rgb[:, :half]], 1).astype(np.float32)


def wrap_line_columns(rgb: np.ndarray, half=48, thr=4.0, segs=6) -> list:
    """좌우 이음새(x=0) ±24px 안에서 위아래 전체로 이어진 가는 선(밝기·색이 한 방향으로 튀는 열) 찾기.
    반환: 이음새 기준 열 번호 (음수 = 오른쪽 끝 쪽)."""
    H = rgb.shape[0]
    win = _wrap_window(rgb, half)
    R, G, B = win[..., 0], win[..., 1], win[..., 2]
    parts = np.array_split(np.arange(int(H * 0.05), int(H * 0.95)), segs)
    hit = np.zeros(win.shape[1], bool)
    for f in ((R + G + B) / 3, B - 0.5 * (R + G), R - 0.5 * (G + B)):
        dv = np.stack([f[p].mean(0) - _median1d(f[p].mean(0), 41) for p in parts])
        hit |= np.all(dv > thr, 0) | np.all(dv < -thr, 0)        # 모든 높이 구간에서 같은 방향 = 세로선
    near = np.zeros_like(hit)
    near[half - 24: half + 24] = True
    return [int(c - half) for c in np.nonzero(hit & near)[0]]


def fix_wrap_line(img_path, half=48, grow=4) -> list:
    """생성 결과의 좌우 이음새에 생긴 가는 세로선(1~2px@2K)을 지움: 선 자리를 양옆 결을 거울처럼 이어 서서히 섞음.
    고친 열 목록 반환 (없으면 [] — 파일 그대로)."""
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    cols = wrap_line_columns(rgb, half)
    if not cols:
        return []
    W = rgb.shape[1]
    win = _wrap_window(rgb, half)
    lo, hi = max(1, min(cols) + half - grow), min(2 * half - 1, max(cols) + half + grow + 1)
    n = hi - lo
    if lo - n < 0 or hi + n > 2 * half:
        return []
    left = win[:, lo - 1 - np.arange(n)]                 # i=0 → 왼쪽 경계 열
    right = win[:, hi + (n - 1 - np.arange(n))]          # i=n-1 → 오른쪽 경계 열
    w = smoothstep((np.arange(n) + 0.5) / n)[None, :, None]
    win[:, lo:hi] = left * (1 - w) + right * w
    win = np.clip(win + 0.5, 0, 255).astype(np.uint8)
    out = rgb.copy()
    out[:, W - half:] = win[:, :half]
    out[:, :half] = win[:, half:]
    Image.fromarray(out).save(img_path)
    return list(range(lo - half, hi - half))


def match_wrap_tone(img_path, width=None, probe=None, smooth=None) -> float:
    """좌우 이음새 양쪽 밝기·색 차이를 줄임: 줄마다(세로로 부드럽게) 양쪽 평균 차이를 구해
    이음새에서 width px 까지 서서히 줄어드는 보정을 반씩 나눠 더함. 반환: 보정 전 평균 차이(0~255).
    폭·높이 값은 2K(2048) 기준 384/48/96 → 이미지 크기에 비례 (8K 에서 384px 면 이음새 옆에 밝기 띠가 남음)."""
    import cv2                                         # opencv 없는 파이썬에서도 서버는 뜨도록 여기서
    im = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    W = im.shape[1]
    k = W / 2048.0
    width = int(width or round(384 * k))               # 8K: 1536px ≈ 67°
    width = min(width, W // 4)
    probe = int(probe or max(8, round(48 * k)))
    smooth = float(smooth or 96 * k)
    L = im[:, W - probe:].mean(1)                     # 오른쪽 끝 (H, 3)
    R = im[:, :probe].mean(1)                         # 왼쪽 앞
    D = cv2.GaussianBlur((L - R)[:, None, :], (1, 0), sigmaX=0.1, sigmaY=smooth)[:, 0, :]   # 세로로 부드럽게
    gap = float(np.abs(D).mean())
    if gap < 1.0:
        return gap
    x = np.arange(width, dtype=np.float32)
    ramp = (0.5 * (1 + np.cos(np.pi * x / width)))[None, :, None]       # 이음새 1 → width 0
    im[:, :width] += 0.5 * D[:, None, :] * ramp                          # 왼쪽 앞은 오른쪽 끝 쪽으로
    im[:, W - width:] -= 0.5 * D[:, None, :] * ramp[:, ::-1]             # 오른쪽 끝은 왼쪽 앞 쪽으로
    Image.fromarray(np.clip(im + 0.5, 0, 255).astype(np.uint8)).save(img_path)
    return gap


def wrap_seam_score(img_path) -> dict:
    """좌우 끝(360 이음새) 색 차이 vs 일반 이웃 픽셀 차이. ratio 가 2 이하면 눈에 거의 안 띔."""
    im = Image.open(img_path).convert("L")
    if im.width > 4096:
        im = im.resize((4096, 2048), Image.BILINEAR)
    a = np.asarray(im, np.float32)
    wrap = float(np.abs(a[:, 0] - a[:, -1]).mean())
    normal = float(np.median(np.abs(np.diff(a, axis=1)).mean(0))) + 1e-3
    return {"wrap": round(wrap, 1), "normal": round(normal, 1), "ratio": round(wrap / normal, 2)}


def _laplacian_pyramid(a, levels):
    import cv2
    g = [a]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=g[i].shape[1::-1]) for i in range(levels)]
    return lap + [g[levels]]


def blend_wrap_band(img_path, band=0.012, win=0.25, levels=5) -> int:
    """좌우 이음새(x=0)를 띠째로 다시 잇기 (선·점선·톤 단차가 띠 폭 안에 있을 때).
    이음새를 가운데로 옮겨 band×W 폭을 버리고, 양쪽 내용을 각자 띠 경계에서 거울로 이어 붙인 뒤
    다중 대역(라플라시안 피라미드)으로 섞음 → 굵은 결은 넓게, 잔결은 좁게 이어짐. 반환: 다시 이은 띠 폭(px)."""
    import cv2
    im = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = im.shape[:2]
    half = max(2, int(band * W / 2))
    wn = max(4 * half, int(win * W / 2))
    r = np.roll(im, W // 2, axis=1)
    c = W // 2
    S = r[:, c - wn:c + wn].astype(np.float32)
    N, cc = S.shape[1], wn
    j = np.arange(N)
    A, B = S.copy(), S.copy()
    A[:, cc - half:] = S[:, np.clip(2 * (cc - half) - 1 - j[cc - half:], 0, cc - half - 1)]     # 왼쪽 내용 → 오른쪽으로 거울
    B[:, :cc + half] = S[:, np.clip(2 * (cc + half) - 1 - j[:cc + half], cc + half, N - 1)]     # 오른쪽 내용 → 왼쪽으로 거울
    m = np.zeros((H, N), np.float32)
    m[:, cc:] = 1.0
    la, lb = _laplacian_pyramid(A, levels), _laplacian_pyramid(B, levels)
    gm = [m]
    for _ in range(levels):
        gm.append(cv2.pyrDown(gm[-1]))
    out = None
    for i in range(levels, -1, -1):
        mm = cv2.GaussianBlur(gm[i], (0, 0), 1.0)[..., None]
        lv = la[i] * (1 - mm) + lb[i] * mm
        out = lv if out is None else cv2.pyrUp(out, dstsize=lv.shape[1::-1]) + lv
    r = r.copy()
    r[:, c - wn:c + wn] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(np.roll(r, -(W // 2), axis=1)).save(img_path)
    return 2 * half


def bridge_wrap_sky(img_path, half=0.06, sigma=None) -> float:
    """이음새 근처 '무늬 없는 곳(하늘·안개)'의 큰 밝기·색 얼룩(이음새 복원 띠가 주변과 다른 톤으로 그려진 것)을 없앰.
    이음새 ±half×W 창에서 줄마다 저주파(흐린 값)를 창 양끝 값 사이의 매끈한 보간으로 바꿈. 무늬 있는 곳(나무·건물)은 그대로.
    반환: 바뀐 평균 정도(0~255)."""
    import cv2
    from .pole import _flat_weight
    im = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = im.shape[:2]
    N = max(8, int(half * W))
    sigma = float(sigma or W / 200.0)                   # 8K: 41px
    r = np.roll(im, W // 2, axis=1)
    c = W // 2
    pad = int(3 * sigma) + 2
    S = r[:, c - N - pad:c + N + pad].astype(np.float32)
    L = cv2.GaussianBlur(S, (0, 0), sigma)
    L = cv2.GaussianBlur(L, (1, 0), sigmaX=0.1, sigmaY=sigma)
    a, b = L[:, pad:pad + 1], L[:, pad + 2 * N - 1:pad + 2 * N]          # 창 양끝 (H,1,3)
    t = ((np.arange(2 * N, dtype=np.float32) + 0.5) / (2 * N))[None, :, None]
    t = t * t * (3 - 2 * t)
    T = a * (1 - t) + b * t
    w = _flat_weight(S)[:, pad:pad + 2 * N, None] ** 2      # 나뭇가지 사이 작은 하늘 틈은 약하게
    x = np.abs(np.arange(2 * N, dtype=np.float32) - (N - 0.5)) / N           # 0 가운데 → 1 창 끝
    taper = (0.5 * (1 + np.cos(np.pi * np.clip((x - 0.6) / 0.4, 0, 1))))[None, :, None]   # 끝 40% 에서 서서히 0
    corr = (T - L[:, pad:pad + 2 * N]) * w * taper
    out = r.copy()
    out[:, c - N:c + N] = np.clip(S[:, pad:pad + 2 * N] + corr + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(np.roll(out, -(W // 2), axis=1)).save(img_path)
    return float(np.abs(corr).mean())


def continuity_report(img_path, band=0.125) -> dict:
    """이음새(뒤쪽 0°/360°) 연결 점검: 하늘·중간·땅 구간마다 이음새 띠(±band, 생성 때 다시 그린 폭)와 양옆(band~2band)을 비교.
      tone  : 띠 평균색이 양옆 평균과 다른 정도 (0~255)
      haze  : 띠 대비(표준편차)가 양옆보다 낮은 정도 (0 = 같음, 0.5 = 절반으로 뿌옇게)
      texture: 띠 결(경계 세기)이 양옆과 다른 비율 (1 = 같음, >1.6 새 구조물/잡음, <0.6 뭉개짐)
      sides : 왼쪽 옆 vs 오른쪽 옆 재질 차이 (색+결, 이음새 양쪽이 서로 다른 재질인지)
    → warnings 에 사람이 읽을 문구. 장면과 무관한 계산이라 어떤 사진이든 같은 기준."""
    import cv2
    im = Image.open(img_path).convert("RGB")
    if im.width > 4096:
        im = im.resize((4096, 2048), Image.BILINEAR)
    a = np.asarray(im)
    H, W = a.shape[:2]
    r = np.roll(a, W // 2, axis=1)
    lab = cv2.cvtColor(r, cv2.COLOR_RGB2LAB).astype(np.float32)
    g = lab[..., 0]
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    c = W // 2
    b, n = int(band * W), int(2 * band * W)
    zones = {"sky": (0.05, 0.35), "middle": (0.35, 0.6), "ground": (0.6, 0.92)}
    out, warns = {}, []
    names = {"sky": "하늘", "middle": "중간(수평선·구조물)", "ground": "땅"}
    for z, (y0, y1) in zones.items():
        ys = slice(int(y0 * H), int(y1 * H))
        band = (slice(None), slice(c - b, c + b))
        L, R = slice(c - n, c - b), slice(c + b, c + n)
        lb, ll, lr = lab[ys][:, band[1]], lab[ys][:, L], lab[ys][:, R]
        eb, el, er = e[ys][:, band[1]].mean(), e[ys][:, L].mean(), e[ys][:, R].mean()
        side_mean = (ll.reshape(-1, 3).mean(0) + lr.reshape(-1, 3).mean(0)) / 2
        tone = float(np.linalg.norm(lb.reshape(-1, 3).mean(0) - side_mean))
        sd_b = float(lb[..., 0].std())
        sd_s = float((ll[..., 0].std() + lr[..., 0].std()) / 2) + 1e-3
        haze = float(max(0.0, 1 - sd_b / sd_s))
        tex = float(eb / max(1e-3, (el + er) / 2))
        sides = float(np.linalg.norm(ll.reshape(-1, 3).mean(0) - lr.reshape(-1, 3).mean(0))
                      + 20 * abs(np.log((el + 1) / (er + 1))))
        out[z] = {"tone": round(tone, 1), "haze": round(haze, 2), "texture": round(tex, 2), "sides": round(sides, 1)}
        flat = sd_s < 4 and (el + er) / 2 < 6                       # 거의 민무늬(맑은 하늘 등)는 결 비교 의미 없음
        if tone > 15:
            warns.append(f"{names[z]}: 이음새 띠 색이 양옆과 다름 ({tone:.0f})")
        if haze > 0.3 and not flat:
            warns.append(f"{names[z]}: 이음새 띠가 뿌옇게(대비 {haze:.0%} 낮음)")
        if not flat and (tex > 1.8 or tex < 0.55):
            warns.append(f"{names[z]}: 이음새 띠 결이 양옆과 다름 (×{tex:.2f}) — 새 구조물·뭉개짐 의심")
        if sides > 60:
            warns.append(f"{names[z]}: 이음새 양쪽 재질/색이 서로 다름 ({sides:.0f})")
    out["warnings"] = warns
    return out
