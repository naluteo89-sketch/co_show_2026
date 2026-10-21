"""
사건 영상 (한 번 일어나는 연출): 넓은 평면 화면에서 만든 영상 → 360 에 다시 넣기.

  잔잔 → 물이 넘쳐 화면을 덮음 → 다시 잔잔.  시작·끝이 원본과 같으므로
  '영상이 원본에서 달라진 곳'만 생성 영상을 쓰고, 그대로인 곳은 8K 원본을 쓴다 (사람이 그린 마스크 없음).

  1. 영상 프레임 읽기 (ComfyUI 에서 만든 MP4)
  2. 흔들림 정렬: 각 프레임을 첫 프레임 기준으로 맞춤 (영상 모델의 카메라 밀림 제거)
  3. 색 맞춤: 첫 프레임 색을 원본 화면에 맞춤
  4. 달라진 정도 → 섞는 비율 (부드럽게, 앞뒤 프레임으로 번지게, 화면 테두리는 0)
  5. 360 에 재투영해 원본 위에 섞고 MP4 저장
"""

import subprocess

import numpy as np
from PIL import Image

import cinemagraph


def read_video(path: str, size=None, max_frames=400) -> tuple[np.ndarray, float]:
    """MP4 → (T,h,w,3) uint8, fps.  (메모리 아끼려고 uint8 로 둠)"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("영상을 열지 못했어요. MP4(H.264) 인지 확인해 주세요.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    out = []
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if size and (f.shape[1], f.shape[0]) != tuple(size):
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA if f.shape[1] > size[0] else cv2.INTER_CUBIC)
        out.append(f)
    cap.release()
    if len(out) < 2:
        raise RuntimeError("영상 프레임이 너무 적어요.")
    return np.stack(out), float(fps)


def stabilize_u8(frames: np.ndarray, grid=12) -> tuple[np.ndarray, dict]:
    """첫 프레임 기준 정렬 (영상 모델의 카메라 밀림 제거). 넘치는 물처럼 크게 바뀌는 곳은 RANSAC 이 걸러냄."""
    import cv2
    h, w = frames.shape[1:3]
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ys, xs = np.mgrid[grid // 2:h:grid, grid // 2:w:grid]
    p0 = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32)
    out, shifts, held = [frames[0]], [0.0], 0
    last = np.float32([[1, 0, 0], [0, 1, 0]])
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        fl = dis.calc(g0, g, None)
        p1 = p0 + fl[p0[:, 1].astype(int), p0[:, 0].astype(int)]
        # 물·물보라로 덮여 질감이 없는 곳(밝고 평평함)은 기준점에서 뺌
        tex = cv2.Laplacian(g, cv2.CV_32F)[p0[:, 1].astype(int), p0[:, 0].astype(int)]
        ok = np.abs(tex) > 4
        M, inl = (cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                              maxIters=3000) if ok.sum() >= 30 else (None, None))
        good = M is not None and inl is not None and inl.sum() >= max(30, 0.2 * ok.sum()) and \
            float(np.hypot(*(M[:, 2] - last[:, 2]))) < 0.02 * max(w, h)
        if good:
            last = M.astype(np.float32)
        else:
            held += 1                                         # 대부분 덮였거나 이상하면 직전 정렬 유지
        out.append(cv2.warpAffine(f, last, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
        shifts.append(float(np.hypot(*last[:, 2])))
    return np.stack(out), {"max_shift_px": round(max(shifts), 2), "mean_shift_px": round(float(np.mean(shifts)), 2),
                           "held_frames": held}


def change_alpha(frames: np.ndarray, lo=10.0, hi=32.0, blur=3.0, grow=0.012, feather=0.02, t_spread=2) -> np.ndarray:
    """프레임별 '첫 프레임(=원본)에서 달라진 정도' → 섞는 비율 (T,h,w) float32 0~1. 화면 테두리는 0."""
    import cv2
    T, h, w = frames.shape[:3]
    g0 = cv2.GaussianBlur(cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), blur)
    a = np.empty((T, h, w), np.float32)
    k = max(3, int(round(grow * max(w, h))) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fe = max(2.0, feather * max(w, h))
    for t in range(T):
        g = cv2.GaussianBlur(cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), blur)
        m = np.clip((np.abs(g - g0) - lo) / (hi - lo), 0, 1)
        m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, small), ker)
        a[t] = cv2.GaussianBlur(m, (0, 0), fe)
    if t_spread:                                              # 깜빡임 방지: 앞뒤 프레임 최댓값 → 이동 평균
        p = np.pad(a, ((t_spread, t_spread), (0, 0), (0, 0)), mode="edge")
        mx = p[0:T].copy()
        for i in range(1, 2 * t_spread + 1):
            np.maximum(mx, p[i:i + T], out=mx)
        p = np.pad(mx, ((1, 1), (0, 0), (0, 0)), mode="edge")
        a = (p[:-2] + p[1:-1] + p[2:]) / 3
    return np.clip(a * cinemagraph.edge_falloff(w, h, 0.12)[None], 0, 1).astype(np.float32)


def compose_event(still_hi: np.ndarray, view: dict, frames: np.ndarray, crop: np.ndarray, out_mp4: str, *,
                  fps: float, width=4096, crf=18, preview: str | None = None, alpha_png: str | None = None) -> dict:
    """view: {yaw, pitch, hfov}. frames: (T,h,w,3) uint8 생성 영상. crop: 원본 사건 화면 (아무 크기, RGB)."""
    import cv2
    W, H = int(width), int(width) // 2
    T, h, w = frames.shape[:3]
    base = cv2.resize(still_hi, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    frames, stab = stabilize_u8(frames)
    ref = cv2.resize(np.asarray(crop), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    off = (ref.reshape(-1, 3).mean(0) - frames[0].reshape(-1, 3).astype(np.float32).mean(0)).astype(np.float32)
    alpha = change_alpha(frames)
    cover = alpha.reshape(T, -1).mean(1)
    peak = int(np.argmax(cover))
    pj = cinemagraph.RegionProjector(view, (w, h), np.ones((h, w), np.float32), W, H)
    cmd = [cinemagraph.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for t in range(T):
            fr = base.copy()
            a = pj.sample(alpha[t][..., None])
            if a.max() > 1e-3:
                src = np.clip(frames[t].astype(np.float32) + off, 0, 255)
                fr[pj.iy, pj.ix] = fr[pj.iy, pj.ix] * (1 - a) + pj.sample(src) * a
            b = np.clip(np.rint(fr), 0, 255).astype(np.uint8)
            if preview and t == peak:
                Image.fromarray(b).save(preview, quality=90)
            p.stdin.write(b.tobytes())
        p.stdin.close()
        err = p.stderr.read().decode("utf-8", "replace")
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg 실패: {err[-800:]}")
    finally:
        if p.poll() is None:
            p.kill()
    if alpha_png:
        Image.fromarray((alpha[peak] * 255).astype(np.uint8)).save(alpha_png)
    return {"frames": int(T), "fps": round(fps, 2), "seconds": round(T / fps, 2), "size": [W, H],
            "peak_frame": peak, "peak_cover": round(float(cover[peak]), 3), "start_cover": round(float(cover[0]), 3),
            "end_cover": round(float(cover[-1]), 3), **stab}
