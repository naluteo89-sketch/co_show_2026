"""
영상화·사건: 360 위의 영역(평면 화면) 잘라내기 · 격자/영역 미리보기 · 움직임 마스크

yaw(+오른쪽)·pitch(+위) 는 make_control 과 같은 기준. 영역 = 그 방향을 바라본 평면(핀홀) 화면.
static/video.js 의 aspectOf/sizeOf/COLORS 가 아래 값과 같아야 함.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .projection import RAD, erp_dirs, plane_uv, sample_erp, to_erp_xy, to_image, view_dirs

ASPECT_MIN, ASPECT_MAX = 0.5, 2.0
TRAIN_ASPECT = 704 / 512                   # 영상 모델 학습 비율
MOTION_COLORS = [(255, 90, 110), (90, 200, 255), (255, 200, 60), (120, 230, 140), (200, 140, 255)]


def region_aspect(r) -> float:
    """영역 화면 가로:세로. aspect 가 없으면 orientation(가로/세로 704:512)에서."""
    a = r.get("aspect")
    if not a:
        a = TRAIN_ASPECT if r.get("orientation", "landscape") != "portrait" else 1 / TRAIN_ASPECT
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
    return to_image(sample_erp(src, *view_dirs(yaw, pitch, hfov, w, h)))


def _font(size):
    try:
        return ImageFont.load_default(size=size)       # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


def grid_overlay(erp_img: Image.Image, W=2048) -> Image.Image:
    """LLM 이 위치를 숫자로 말할 수 있게 30° 격자와 yaw/pitch 값을 그린 ERP."""
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
    dx, dy, dz = view_dirs(yaw, pitch, hfov, n, m)
    idx = ([(0, j) for j in range(n)] + [(i, n - 1) for i in range(m)] +
           [(m - 1, j) for j in range(n - 1, -1, -1)] + [(i, 0) for i in range(m - 1, -1, -1)])
    rr, cc = zip(*idx)
    xs, ys = to_erp_xy(dx[rr, cc], dy[rr, cc], dz[rr, cc], W, H)
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
    dirs = erp_dirs(W, H)
    for r in regions:
        w, h = region_size(r)
        th = math.tan(r["hfov"] * RAD / 2)
        u, v, front = plane_uv(*dirs, r["yaw"], r["pitch"], th, th * h / w)
        out |= front & (np.abs(u) <= 1) & (np.abs(v) <= 1)
    return Image.fromarray((out * 255).astype(np.uint8), "L")


def motion_overlay(erp_img: Image.Image, regions: list, W=2048) -> Image.Image:
    """움직일 영역만 원래 밝기, 나머지(정지)는 어둡게 + 번호 테두리."""
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
