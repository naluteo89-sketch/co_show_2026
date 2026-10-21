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


def change_alpha(frames: np.ndarray, lo=10.0, hi=32.0, blur=3.0, grow=0.012, feather=0.02, t_spread=2,
                 border="all", max_w=1024) -> np.ndarray:
    """프레임별 '첫 프레임(=원본)에서 달라진 정도' → 섞는 비율 (T,h',w') float32 0~1 (메모리 아끼려고 최대 너비 max_w 로
    계산 — 비율은 부드러워서 나중에 키워 써도 됨). 테두리(또는 ERP 면 극점)는 0."""
    import cv2
    T, H0, W0 = frames.shape[:3]
    s = min(1.0, max_w / W0)
    w, h = max(8, int(round(W0 * s))), max(8, int(round(H0 * s)))
    sm = lambda f: cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    bl = max(0.8, blur * s)
    g0 = cv2.GaussianBlur(sm(frames[0]), (0, 0), bl)
    a = np.empty((T, h, w), np.float32)
    k = max(3, int(round(grow * max(w, h))) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fe = max(1.5, feather * max(w, h))
    for t in range(T):
        g = cv2.GaussianBlur(sm(frames[t]), (0, 0), bl)
        m = np.clip((np.abs(g - g0) - lo) / (hi - lo), 0, 1)
        m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, small), ker)
        a[t] = cv2.GaussianBlur(m, (0, 0), fe)
    if t_spread:                                              # 깜빡임 방지: 앞뒤 프레임 최댓값 → 이동 평균
        mx = a.copy()
        for d in range(1, t_spread + 1):
            np.maximum(mx[d:], a[:-d], out=mx[d:])
            np.maximum(mx[:-d], a[d:], out=mx[:-d])
        a[1:-1] = (mx[:-2] + mx[1:-1] + mx[2:]) / 3
        a[0], a[-1] = mx[0], mx[-1]
        del mx
    if border == "poles":                                     # 360(ERP) 전체 영상: 좌우는 이어지므로 위·아래(극점)만 0
        y = np.abs((np.arange(h) + 0.5) / h - 0.5) * 2
        f = np.clip((0.94 - y) / 0.08, 0, 1)
        edge = np.repeat((f * f * (3 - 2 * f))[:, None], w, 1)
    else:
        edge = cinemagraph.edge_falloff(w, h, 0.12)
    a *= edge[None]
    return np.clip(a, 0, 1, out=a)


def compose_event(still_hi: np.ndarray, view: dict, frames: np.ndarray, crop: np.ndarray, out_mp4: str, *,
                  fps: float, width=7680, crf=17, codec="hevc", preview: str | None = None,
                  alpha_png: str | None = None) -> dict:
    """view: {yaw, pitch, hfov}. frames: (T,h,w,3) uint8 생성 영상. crop: 원본 사건 화면 (아무 크기, RGB)."""
    import cv2
    W, H = int(width), int(width) // 2
    T, h, w = frames.shape[:3]
    base = cv2.resize(still_hi, (W, H), interpolation=cv2.INTER_AREA)
    frames, stab = stabilize_u8(frames)
    ref = cv2.resize(np.asarray(crop), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    off = (ref.reshape(-1, 3).mean(0) - frames[0].reshape(-1, 3).astype(np.float32).mean(0)).astype(np.float32)
    alpha = change_alpha(frames)
    cover = alpha.reshape(T, -1).mean(1)
    peak = int(np.argmax(cover))
    pj = cinemagraph.RegionProjector(view, (w, h), np.ones((h, w), np.float32), W, H)
    up = lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    cmd = [cinemagraph.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:g}", "-i", "-", *cinemagraph.encoder_args(codec, crf, fps), "-movflags", "+faststart", out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    broken = False
    try:
        for t in range(T):
            fr = base.copy()                                  # uint8 그대로 (8K 에서 메모리 아낌)
            a = pj.sample(up(alpha[t])[..., None])
            if a.max() > 1e-3:
                src = np.clip(frames[t].astype(np.float32) + off, 0, 255)
                cur = fr[pj.iy, pj.ix].astype(np.float32)
                fr[pj.iy, pj.ix] = np.clip(np.rint(cur * (1 - a) + pj.sample(src) * a), 0, 255).astype(np.uint8)
            if preview and t == peak:
                Image.fromarray(fr).save(preview, quality=90)
            p.stdin.write(fr.tobytes())
        p.stdin.close()
    except BrokenPipeError:
        broken = True
    err = p.stderr.read().decode("utf-8", "replace")
    rc = p.wait()
    if rc != 0 or broken:
        raise RuntimeError(f"ffmpeg 실패 (코드 {rc}; 메모리 부족이면 config motion.event_width 를 5760 으로): {err[-800:]}")
    if alpha_png:
        Image.fromarray((alpha[peak] * 255).astype(np.uint8)).save(alpha_png)
    return {"frames": int(T), "fps": round(fps, 2), "seconds": round(T / fps, 2), "size": [W, H],
            "peak_frame": peak, "peak_cover": round(float(cover[peak]), 3), "start_cover": round(float(cover[0]), 3),
            "end_cover": round(float(cover[-1]), 3), **stab}


def is_erp(frames: np.ndarray) -> bool:
    h, w = frames.shape[1:3]
    return abs(w / h - 2.0) < 0.05


def compose_event_erp(still_hi: np.ndarray, frames: np.ndarray, out_mp4: str, *, fps: float, width=5760, crf=17,
                      codec="hevc", preview: str | None = None, alpha_png: str | None = None) -> dict:
    """360 전체(ERP) 로 만든 영상(예: LTX + 360 LoRA 2K) → 원본에서 달라진 곳만 8K 원본 위에.
    정지한 곳은 8K 원본 그대로라 영상 해상도(2K)와 상관없이 선명함."""
    import cv2
    W, H = int(width), int(width) // 2
    T, h, w = frames.shape[:3]
    base = cv2.resize(still_hi, (W, H), interpolation=cv2.INTER_AREA)
    ref = cv2.resize(still_hi, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    off = (ref.reshape(-1, 3).mean(0) - frames[0].reshape(-1, 3).astype(np.float32).mean(0)).astype(np.float32)
    alpha = change_alpha(frames, border="poles")
    cover = alpha.reshape(T, -1).mean(1)
    peak = int(np.argmax(cover))
    cmd = [cinemagraph.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:g}", "-i", "-", *cinemagraph.encoder_args(codec, crf, fps), "-movflags", "+faststart", out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    broken = False
    try:
        for t in range(T):
            fr = base.copy()
            if alpha[t].max() > 1e-3:
                a = cv2.resize(alpha[t], (W, H), interpolation=cv2.INTER_LINEAR)
                rows = np.nonzero(a.max(1) > 1e-3)[0]
                r0, r1 = rows[0], rows[-1] + 1                # 바뀐 줄만 계산
                src = cv2.resize(np.clip(frames[t].astype(np.float32) + off, 0, 255), (W, H),
                                 interpolation=cv2.INTER_CUBIC)[r0:r1]
                aa = a[r0:r1, :, None]
                fr[r0:r1] = np.clip(np.rint(fr[r0:r1].astype(np.float32) * (1 - aa) + src * aa), 0, 255).astype(np.uint8)
            if preview and t == peak:
                Image.fromarray(fr).save(preview, quality=90)
            p.stdin.write(fr.tobytes())
        p.stdin.close()
    except BrokenPipeError:
        broken = True
    err = p.stderr.read().decode("utf-8", "replace")
    rc = p.wait()
    if rc != 0 or broken:
        raise RuntimeError(f"ffmpeg 실패 (코드 {rc}; 메모리 부족이면 config motion.event_width 를 4096 으로): {err[-800:]}")
    if alpha_png:
        Image.fromarray((alpha[peak] * 255).astype(np.uint8)).save(alpha_png)
    return {"mode": "erp", "frames": int(T), "fps": round(fps, 2), "seconds": round(T / fps, 2), "size": [W, H],
            "gen_size": [int(w), int(h)], "peak_frame": peak, "peak_cover": round(float(cover[peak]), 3),
            "start_cover": round(float(cover[0]), 3), "end_cover": round(float(cover[-1]), 3)}

