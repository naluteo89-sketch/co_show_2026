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


def match_wrap_tone(img_path, width=384, probe=48, smooth=96) -> float:
    """좌우 이음새 양쪽 밝기·색 차이를 줄임: 줄마다(세로로 부드럽게) 양쪽 평균 차이를 구해
    이음새에서 width px 까지 서서히 줄어드는 보정을 반씩 나눠 더함. 반환: 보정 전 평균 차이(0~255)."""
    import cv2                                         # opencv 없는 파이썬에서도 서버는 뜨도록 여기서
    im = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    W = im.shape[1]
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
