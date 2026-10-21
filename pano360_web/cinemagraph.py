"""
영상화 합성: 영역별로 만든 짧은 영상을 360 정지 이미지의 같은 자리에 다시 넣어 '일부만 움직이는' 360 영상을 만든다.

  1. 움직일 것만 남기기  — 요소 마스크(SAM 3, 없으면 영역 가운데 둥근 마스크) 안쪽만 영상, 밖은 원본 정지 화면
  2. 첫 프레임 색 맞춤    — 영상 첫 프레임의 밝기·색을 원본 영역 화면에 맞춰 시작할 때 튀지 않게
  3. 반복 이음            — 끝 몇 프레임을 처음과 교차 페이드해서 끊김 없이 반복
  4. 360 에 되돌려 넣기   — 평면 영역 화면 → ERP 좌표로 다시 투영, 마스크 가장자리는 부드럽게
  5. MP4 저장            — ffmpeg (PATH 또는 imageio-ffmpeg)
"""

import io
import math
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageFilter

import erp


def _blur_f(a: np.ndarray, sigma: float) -> np.ndarray:
    """실수 배열 가우시안 흐림 (가로·세로 분리, 가장자리 반사)."""
    r = max(1, int(round(sigma * 3)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    a = np.pad(a.astype(np.float32), ((r, r), (r, r)), mode="reflect")
    a = sum(k[i] * a[:, i:i + a.shape[1] - 2 * r] for i in range(2 * r + 1))
    return sum(k[i] * a[i:i + a.shape[0] - 2 * r, :] for i in range(2 * r + 1))


# ----------------------------------------------------------------------------- 마스크
def edge_falloff(w, h, margin=0.08):
    """영역 화면 테두리에서 0 → 안쪽 1 (화면 가장자리는 절대 안 움직이게)."""
    x = np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / (w * margin)
    y = np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / (h * margin)
    f = np.clip(np.minimum(x[None, :], y[:, None]), 0, 1)
    return f * f * (3 - 2 * f)


def element_mask(size, sam_png: bytes | None = None, feather_px=14, grow_px=6) -> tuple[np.ndarray, str]:
    """움직일 요소 마스크 (0~1, 영역 화면 크기). SAM 결과가 비었거나 없으면 가운데 타원."""
    w, h = size
    src = "sam"
    m = None
    if sam_png:
        try:
            im = Image.open(io.BytesIO(sam_png)).convert("L").resize((w, h), Image.BILINEAR)
            if grow_px:
                im = im.filter(ImageFilter.MaxFilter(grow_px * 2 + 1))
            a = np.asarray(im, np.float32) / 255.0
            if a.mean() > 0.003:                     # 거의 비었으면 실패로 봄
                m = a
        except Exception:
            m = None
    if m is None:
        src = "ellipse"
        yy, xx = np.mgrid[0:h, 0:w]
        d = ((xx - w / 2) / (w * 0.42)) ** 2 + ((yy - h / 2) / (h * 0.42)) ** 2
        m = np.clip((1.15 - d) / 0.3, 0, 1)
    if feather_px:
        m = np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(feather_px)), np.float32) / 255.0
    return np.clip(m * edge_falloff(w, h), 0, 1), src


def motion_energy(frames: np.ndarray, blur_px=2) -> np.ndarray:
    """영상에서 실제로 움직인 정도 (픽셀별, 밝기 단위). 프레임 간 차이의 시간 최댓값을 살짝 흐려서."""
    g = frames.mean(-1)                                    # (T,h,w)
    if blur_px:
        g = np.stack([_blur_f(x, blur_px) for x in g])
    d = np.abs(g - g[:1]).max(0)                           # 첫 프레임(=원본 근처)에서 얼마나 벗어났나
    return d


def moving_mask(size, sam_png: bytes | None, frames: np.ndarray, *, lo=None, hi=None, feather_px=14,
                grow_px=6, min_blob=0.0005) -> tuple[np.ndarray, str, dict]:
    """합성 마스크 = max(SAM 요소, 영상에서 실제로 움직인 곳). 요소 모양에 갇히지 않고 물결·물보라처럼
    그 움직임 때문에 생긴 주변 움직임도 살림. 떨림 수준(문턱 lo 아래)은 버림."""
    w, h = size
    e = motion_energy(frames)
    floor = float(np.median(e))                            # 대부분 정지 → 중앙값 ≈ 잡음
    lo = max(3.0, 2.5 * floor) if lo is None else lo
    hi = lo * 3 if hi is None else hi
    m = np.clip((e - lo) / (hi - lo), 0, 1)
    im = Image.fromarray((m * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))       # 점 잡음 제거
    if grow_px:
        im = im.filter(ImageFilter.MaxFilter(grow_px * 2 + 1))
    mov = np.asarray(im, np.float32) / 255.0
    if mov.mean() < min_blob:
        mov = np.zeros_like(mov)
    sam = None
    if sam_png:
        sam, src = element_mask(size, sam_png, feather_px=0, grow_px=grow_px)
        if src != "sam":
            sam = None
    if sam is None and not mov.any():
        mask, src = element_mask(size, None, feather_px=feather_px)
        return mask, "ellipse", {"moving": 0.0, "lo": round(lo, 2)}
    base = mov if sam is None else np.maximum(sam, mov)
    src = "motion" if sam is None else "sam+motion"
    if feather_px:
        base = np.asarray(Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(feather_px)), np.float32) / 255.0
    out = np.clip(base * edge_falloff(w, h), 0, 1)
    extra = float(((mov > 0.5) & ((sam if sam is not None else np.zeros_like(mov)) < 0.5)).mean())
    return out, src, {"moving": round(float((out > 0.5).mean()), 4), "beyond_element": round(extra, 4),
                      "lo": round(lo, 2)}


SPREAD_GROW = {"none": 0.015, "near": 0.10, "wide": 0.30}    # 요소 마스크를 영역 화면 대각선의 몇 % 만큼 넓힐지


def range_mask(size, sam_png: bytes | None, spread="near", feather_frac=0.035) -> tuple[np.ndarray, str]:
    """영상 모델에 줄 '움직여도 되는 범위' (0~1, 영역 화면 크기). 요소(SAM, 없으면 가운데 타원)를 퍼짐만큼 넓히고
    부드럽게, 화면 테두리는 항상 0. 범위 밖은 생성 중 원본으로 고정됨."""
    import cv2
    w, h = size
    base, src = element_mask(size, sam_png, feather_px=0, grow_px=0)
    if src != "sam":
        yy, xx = np.mgrid[0:h, 0:w]
        base = ((((xx - w / 2) / (w * 0.30)) ** 2 + ((yy - h / 2) / (h * 0.30)) ** 2) <= 1).astype(np.float32)
        src = "ellipse"
    diag = math.hypot(w, h)
    grow = SPREAD_GROW.get(spread, 0.10) * diag
    fea = max(4.0, feather_frac * diag)
    dist = cv2.distanceTransform(((base < 0.5) * 255).astype(np.uint8), cv2.DIST_L2, 5)   # 요소에서의 거리
    m = np.clip(1 - (dist - grow) / fea, 0, 1)
    m = m * m * (3 - 2 * m)
    return np.clip(m * edge_falloff(w, h, 0.06), 0, 1).astype(np.float32), src


def mask_png(mask: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).convert("RGB").save(b, "PNG")
    return b.getvalue()


# ----------------------------------------------------------------------------- 프레임
def load_frames(datas: list, size) -> np.ndarray:
    """PNG 바이트들 → (T,h,w,3) float32. 크기가 다르면 영역 화면 크기로."""
    out = []
    for d in datas:
        im = Image.open(io.BytesIO(d)).convert("RGB")
        if im.size != tuple(size):
            im = im.resize(size, Image.LANCZOS)
        out.append(np.asarray(im, np.float32))
    return np.stack(out)


def stabilize(frames: np.ndarray, still: np.ndarray, exclude: np.ndarray | None = None, grid=8) -> tuple[np.ndarray, dict]:
    """영상 모델이 만든 카메라 흔들림(좌우 밀림·확대) 제거: 프레임마다 원본 영역 화면에 맞춰 정렬.
    물처럼 진짜 움직이는 곳은 RANSAC 에서 걸러지고(exclude 로 미리 빼기도 함), 단단한 배경 기준으로 맞춤."""
    import cv2
    h, w = still.shape[:2]
    g0 = cv2.cvtColor(np.clip(still, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ys, xs = np.mgrid[grid // 2:h:grid, grid // 2:w:grid]
    keep = np.ones(ys.shape, bool)
    if exclude is not None:
        keep &= exclude[ys, xs] < 0.2
    m = 0.06                                              # 화면 가장자리는 빼기
    keep &= (xs > w * m) & (xs < w * (1 - m)) & (ys > h * m) & (ys < h * (1 - m))
    p0 = np.stack([xs[keep], ys[keep]], -1).astype(np.float32)
    out, shifts = [], []
    for f in frames:
        g = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        fl = dis.calc(g0, g, None)
        p1 = p0 + fl[p0[:, 1].astype(int), p0[:, 0].astype(int)]
        M, inl = (cv2.estimateAffinePartial2D(p1, p0, method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                              maxIters=3000, confidence=0.995) if len(p0) >= 20 else (None, None))
        if M is None:
            out.append(f); shifts.append(0.0)
            continue
        out.append(cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
        shifts.append(float(np.hypot(*M[:, 2])))
    return np.stack(out).astype(np.float32), {"max_shift_px": round(max(shifts), 2), "mean_shift_px": round(float(np.mean(shifts)), 2)}


def match_first_frame(frames: np.ndarray, still: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """첫 프레임과 원본의 마스크 안 평균 색 차이를 모든 프레임에서 뺌 (밝기·색 튐 방지)."""
    sel = mask > 0.5
    if sel.sum() < 50:
        sel = mask > 0.05
    if sel.sum() < 10:
        return frames
    off = still[sel].mean(0) - frames[0][sel].mean(0)
    return np.clip(frames + off[None, None, None, :], 0, 255)


def make_loop(frames: np.ndarray, fade=8) -> np.ndarray:
    """끝 fade 프레임을 처음과 교차 페이드 → 길이 T-fade 의 끊김 없는 반복."""
    T = len(frames)
    k = int(min(fade, T // 3))
    if k < 2:
        return frames
    out = frames[: T - k].copy()
    for t in range(k):
        w = (t + 1) / (k + 1)
        w = w * w * (3 - 2 * w)
        out[t] = frames[t] * w + frames[T - k + t] * (1 - w)
    return out


# ----------------------------------------------------------------------------- 360 에 되돌려 넣기
class RegionProjector:
    """ERP(W×H) 픽셀 중 이 영역 화면에 들어가는 곳과 그 화면 좌표를 미리 계산."""

    def __init__(self, region: dict, size, mask: np.ndarray, W: int, H: int):
        w, h = size
        rad = math.pi / 180
        th = math.tan(region["hfov"] * rad / 2)
        tv = th * h / w
        dx, dy, dz = erp._dirs(W, H)
        cy, sy = math.cos(region["yaw"] * rad), math.sin(region["yaw"] * rad)
        cp, sp = math.cos(region["pitch"] * rad), math.sin(region["pitch"] * rad)
        x1 = dx * cy - dz * sy
        z1 = dx * sy + dz * cy
        y2 = dy * cp - z1 * sp
        z2 = dy * sp + z1 * cp
        front = z2 > 1e-6
        zz = np.where(front, z2, 1.0)
        u, v = x1 / (zz * th), y2 / (zz * tv)
        inside = front & (np.abs(u) < 1) & (np.abs(v) < 1)
        self.iy, self.ix = np.nonzero(inside)
        px = ((u[inside] + 1) / 2 * w - 0.5).clip(0, w - 1.001)
        py = ((1 - v[inside]) / 2 * h - 0.5).clip(0, h - 1.001)
        self.x0, self.y0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        self.fx, self.fy = (px - self.x0)[:, None], (py - self.y0)[:, None]
        a = self.sample(mask[..., None])[:, 0]
        keep = a > 0.002
        for k in ("iy", "ix", "x0", "y0", "fx", "fy"):
            setattr(self, k, getattr(self, k)[keep])
        self.alpha = a[keep][:, None]

    def sample(self, img: np.ndarray) -> np.ndarray:
        x0, y0, fx, fy = self.x0, self.y0, self.fx, self.fy
        return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
                (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)

    def paste(self, base: np.ndarray, frame: np.ndarray):
        cur = base[self.iy, self.ix]
        base[self.iy, self.ix] = cur * (1 - self.alpha) + self.sample(frame) * self.alpha


# ----------------------------------------------------------------------------- 인코딩
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg 를 찾지 못했어요. `pip install imageio-ffmpeg` 를 하거나 ffmpeg 를 PATH 에 넣어 주세요.")


def compose_video(still_path: str, regions: list, out_mp4: str, *, fps=24, width=4096, crf=18,
                  preview_png: str | None = None, warp=None) -> dict:
    """regions: [{region(yaw,pitch,hfov), size(w,h), still(crop PNG 경로), frames(np), mask(np)}] → 360 MP4.
    warp: influence.FlowWarp — 있으면 바탕이 '영향받는 것만 살짝 움직이는' 원본, 없으면 완전 정지 원본."""
    W, H = int(width), int(width) // 2
    if warp is not None:
        still = warp.base_frame()
    else:
        still = np.asarray(Image.open(still_path).convert("RGB").resize((W, H), Image.LANCZOS), np.float32)
    T = min([len(r["frames"]) for r in regions] + ([warp.T] if warp is not None else []))
    projs = [RegionProjector(r["region"], r["size"], r["mask"], W, H) for r in regions]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for t in range(T):
            fr = warp.frame(t) if warp is not None else still.copy()
            for r, pj in zip(regions, projs):
                pj.paste(fr, r["frames"][t])
            b = np.clip(np.rint(fr), 0, 255).astype(np.uint8)
            if t == 0 and preview_png:
                Image.fromarray(b).save(preview_png)
            p.stdin.write(b.tobytes())
        p.stdin.close()
        err = p.stderr.read().decode("utf-8", "replace")
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg 실패: {err[-800:]}")
    finally:
        if p.poll() is None:
            p.kill()
    moving = sum(len(pj.iy) for pj in projs) / (W * H) + (float(warp.active.mean()) if warp is not None else 0.0)
    return {"frames": int(T), "fps": fps, "seconds": round(T / fps, 2), "size": [W, H],
            "moving_ratio": round(float(moving), 4)}


def static_leak(frames: np.ndarray, mask: np.ndarray) -> float:
    """마스크 밖에 남은 움직임 (프레임 간 평균 차이, 0 에 가까울수록 좋음). 합성 후엔 밖은 원본이라 참고용."""
    if len(frames) < 2:
        return 0.0
    out = mask < 0.05
    d = np.abs(np.diff(frames, axis=0)).mean(-1)
    return round(float(d[:, out].mean()) if out.any() else 0.0, 3)
