"""
효과 레이어 합성 (6단계): 바탕(원본 360, 다시 그리지 않음) + 고른 효과 영상 여러 개 → 4K 360 MP4.

레이어 종류
  erp      360 전체(2:1) I2V 영상 (예: LTX + 360 LoRA). 매트 = 원본(첫 프레임)에서 달라진 곳. 극점만 0.
  view     평면 사건 화면 I2V 영상 (6단계 '사건 만들기' 화면). 매트 = 달라진 곳, 흔들림 정렬, 화면 테두리 0.
  element  효과만 따로 만든 영상 (검은 배경 위 물보라·안개 등, T2V). 매트 = 밝기(검정 = 투명). yaw·pitch·화각으로 배치.
섞는 방식: normal(덮기) / screen(밝게 더하기 — 물보라·안개에 자연스러움). 레이어마다 시작 시각·세기.
"""

import math
import subprocess

import numpy as np
from PIL import Image

import cinemagraph
import event as eventmod

# 효과만 따로 만들 때(T2V) 쓰는 프리셋 — 검은 배경이라 밝기로 떼어냄
PRESETS = [
    {"id": "spray", "name_ko": "물보라 폭발", "blend": "screen", "hfov": 110, "pitch": -5, "opacity": 0.9,
     "prompt": "A massive burst of white water spray and fine mist exploding upward and rushing toward the camera, "
               "isolated on a pure black background, nothing else in frame, locked-off camera, photorealistic, "
               "natural daylight on the water, slow motion."},
    {"id": "mist", "name_ko": "안개 밀려옴", "blend": "screen", "hfov": 120, "pitch": 0, "opacity": 0.7,
     "prompt": "Thick soft white mist and fog rolling slowly toward the camera and drifting across the frame, "
               "isolated on a pure black background, nothing else in frame, locked-off camera, photorealistic."},
    {"id": "splash", "name_ko": "물 튐 · 물방울", "blend": "screen", "hfov": 90, "pitch": -15, "opacity": 0.9,
     "prompt": "White water splashing and thousands of droplets flying toward the camera, isolated on a pure black "
               "background, nothing else in frame, locked-off camera, photorealistic, sunlit droplets, slow motion."},
    {"id": "wave", "name_ko": "물살 밀려옴", "blend": "normal", "hfov": 120, "pitch": -25, "opacity": 1.0,
     "prompt": "A foaming white river surge rushing toward the camera across the bottom of the frame, isolated on a pure "
               "black background, nothing else in frame, locked-off camera, photorealistic."},
    {"id": "leaves", "name_ko": "낙엽 바람", "blend": "normal", "hfov": 110, "pitch": 5, "opacity": 1.0,
     "prompt": "Autumn leaves blown by a gust of wind, swirling and drifting past the camera, isolated on a pure black "
               "background, nothing else in frame, locked-off camera, photorealistic."},
]
PRESET = {p["id"]: p for p in PRESETS}
NEG_ELEMENT = ("scenery, landscape, ground, sky, trees, buildings, people, text, watermark, camera movement, "
               "gray background, colored background, low quality")
COLORS = [(255, 70, 90), (70, 190, 255), (255, 200, 50), (110, 230, 130), (200, 130, 255)]


def probe(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("영상을 열지 못했어요. MP4(H.264/H.265) 인지 확인해 주세요.")
    info = {"fps": round(cap.get(cv2.CAP_PROP_FPS) or 24.0, 3), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]}
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("영상 프레임을 읽지 못했어요.")
    info["thumb"] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return info


def default_layer(info: dict, preset: str | None, ev: dict | None) -> dict:
    w, h = info["size"]
    lay = {"enabled": True, "start": 0.0, "opacity": 1.0, "blend": "normal", "yaw": 0.0, "pitch": 0.0, "hfov": 110.0}
    if abs(w / max(1, h) - 2.0) < 0.05:
        lay.update(kind="erp", key="diff", name="360 전체 영상")
    elif preset and preset in PRESET:
        p = PRESET[preset]
        lay.update(kind="element", key="luma", name=p["name_ko"], blend=p["blend"], opacity=p["opacity"],
                   hfov=p["hfov"], pitch=p["pitch"], preset=preset)
        if ev and ev.get("view"):
            lay.update(yaw=ev["yaw"], pitch=ev["pitch"] + p["pitch"])
    else:
        lay.update(kind="view", key="diff", name="사건 화면 영상")
        if ev and ev.get("view"):
            lay.update(yaw=ev["yaw"], pitch=ev["pitch"], hfov=ev["hfov"])
    return lay


def clamp_layer(l: dict) -> dict:
    l = dict(l)
    l["yaw"] = ((float(l.get("yaw", 0)) + 180) % 360) - 180
    l["pitch"] = max(-85.0, min(85.0, float(l.get("pitch", 0))))
    l["hfov"] = max(20.0, min(170.0, float(l.get("hfov", 110))))
    l["start"] = max(0.0, min(60.0, float(l.get("start", 0))))
    l["opacity"] = max(0.0, min(1.0, float(l.get("opacity", 1))))
    l["blend"] = l.get("blend") if l.get("blend") in ("normal", "screen") else "normal"
    l["key"] = l.get("key") if l.get("key") in ("diff", "luma") else ("luma" if l.get("kind") == "element" else "diff")
    l["enabled"] = bool(l.get("enabled", True))
    return l


# ----------------------------------------------------------------------------- 매트
def luma_alpha(frames: np.ndarray, lo=14.0, hi=110.0, max_w=1024) -> np.ndarray:
    """검은 배경 효과 영상 → 밝은 곳만 (T,h',w')."""
    import cv2
    T, H0, W0 = frames.shape[:3]
    s = min(1.0, max_w / W0)
    w, h = max(8, int(round(W0 * s))), max(8, int(round(H0 * s)))
    a = np.empty((T, h, w), np.float32)
    for t in range(T):
        g = cv2.resize(frames[t], (w, h), interpolation=cv2.INTER_AREA).max(-1).astype(np.float32)
        m = np.clip((g - lo) / (hi - lo), 0, 1)
        a[t] = cv2.GaussianBlur(m * m * (3 - 2 * m), (0, 0), 1.0)
    return a * cinemagraph.edge_falloff(w, h, 0.05)[None]


class Loaded:
    """한 레이어의 프레임·매트를 메모리에 올림 (미리보기·합성 공용)."""

    def __init__(self, lay: dict, path: str, still_hi: np.ndarray, plate_view=None):
        import cv2
        self.l = lay
        frames, self.fps = eventmod.read_video(path)
        self.T = len(frames)
        self.stab = {}
        if lay["kind"] == "view":
            frames, self.stab = eventmod.stabilize_u8(frames)
        h, w = frames.shape[1:3]
        self.size = (w, h)
        self.off = np.zeros(3, np.float32)
        if lay["key"] == "luma":
            self.alpha = luma_alpha(frames)
        else:
            self.alpha = eventmod.change_alpha(frames, border="poles" if lay["kind"] == "erp" else "all")
            if lay["kind"] == "erp":                          # 첫 프레임 색 → 원본
                ref = cv2.resize(still_hi, (w, h), interpolation=cv2.INTER_AREA)
            else:
                ref = plate_view if plate_view is not None else frames[0]
                ref = cv2.resize(np.asarray(ref), (w, h), interpolation=cv2.INTER_AREA)
            self.off = (ref.reshape(-1, 3).astype(np.float32).mean(0) -
                        frames[0].reshape(-1, 3).astype(np.float32).mean(0)).astype(np.float32)
        self.frames = frames
        self.alpha *= float(lay["opacity"])
        self._pj = {}
        self.cover = self.alpha.reshape(self.T, -1).mean(1)

    def index(self, tsec: float) -> int | None:
        i = int(round((tsec - self.l["start"]) * self.fps))
        return i if 0 <= i < self.T else None

    @property
    def end(self) -> float:
        return self.l["start"] + self.T / self.fps

    def projector(self, W, H):
        if W not in self._pj:
            v = {"yaw": self.l["yaw"], "pitch": self.l["pitch"], "hfov": self.l["hfov"]}
            self._pj[W] = cinemagraph.RegionProjector(v, self.size, np.ones(self.size[::-1], np.float32), W, H)
        return self._pj[W]

    def apply(self, fr: np.ndarray, i: int, W: int, H: int, matte: np.ndarray | None = None):
        """fr (H,W,3 uint8) 위에 i 번째 프레임을 섞음. matte 가 있으면 이 레이어의 섞임 정도를 거기에 기록."""
        import cv2
        a_small = self.alpha[i]
        if a_small.max() < 1e-3:
            return
        src_u8 = self.frames[i]
        if self.l["kind"] == "erp":
            a = cv2.resize(a_small, (W, H), interpolation=cv2.INTER_LINEAR)
            rows = np.nonzero(a.max(1) > 1e-3)[0]
            r0, r1 = rows[0], rows[-1] + 1
            src = cv2.resize(src_u8, (W, H), interpolation=cv2.INTER_CUBIC)[r0:r1].astype(np.float32) + self.off
            aa = a[r0:r1, :, None]
            cur = fr[r0:r1].astype(np.float32)
            fr[r0:r1] = _blend(cur, np.clip(src, 0, 255), aa, self.l["blend"])
            if matte is not None:
                np.maximum(matte[r0:r1], a[r0:r1], out=matte[r0:r1])
            return
        pj = self.projector(W, H)
        w, h = self.size
        a = pj.sample(cv2.resize(a_small, (w, h), interpolation=cv2.INTER_LINEAR)[..., None])
        src = np.clip(pj.sample(src_u8.astype(np.float32)) + self.off, 0, 255)
        cur = fr[pj.iy, pj.ix].astype(np.float32)
        fr[pj.iy, pj.ix] = _blend(cur, src, a, self.l["blend"])
        if matte is not None:
            matte[pj.iy, pj.ix] = np.maximum(matte[pj.iy, pj.ix], a[:, 0])


def _blend(cur, src, a, mode):
    if mode == "screen":                                     # 밝게 더하기: 물보라·안개
        s = src * a
        out = cur + s - cur * s / 255.0
    else:
        out = cur * (1 - a) + src * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def timeline(loaded: list, fps=None) -> tuple[float, int]:
    fps = fps or (loaded[0].fps if loaded else 24.0)
    end = max([x.end for x in loaded] + [0.5])
    return float(fps), int(math.ceil(end * fps))


def render(base: np.ndarray, loaded: list, tsec: float, matte=False):
    H, W = base.shape[:2]
    fr = base.copy()
    m = np.zeros((H, W), np.float32) if matte else None
    per = []
    for x in loaded:
        i = x.index(tsec)
        if i is None:
            per.append(None)
            continue
        mm = np.zeros((H, W), np.float32) if matte else None
        x.apply(fr, i, W, H, mm)
        per.append(mm)
        if matte:
            np.maximum(m, mm, out=m)
    return fr, per


def matte_view(base: np.ndarray, per: list) -> np.ndarray:
    """매트 확인용: 바탕은 흐린 흑백, 레이어마다 색으로 섞인 정도 표시."""
    g = base.mean(-1, keepdims=True) * 0.35
    out = np.repeat(g, 3, -1)
    for k, m in enumerate(per):
        if m is None:
            continue
        c = np.array(COLORS[k % len(COLORS)], np.float32)
        out = out * (1 - m[..., None] * 0.85) + c * (m[..., None] * 0.85)
    return np.clip(out, 0, 255).astype(np.uint8)


def compose(base_hi: np.ndarray, loaded: list, out_mp4: str, *, width=4096, crf=17, codec="hevc",
            preview: str | None = None) -> dict:
    import cv2
    W, H = int(width), int(width) // 2
    base = cv2.resize(base_hi, (W, H), interpolation=cv2.INTER_AREA)
    fps, T = timeline(loaded)
    cmd = [cinemagraph.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:g}", "-i", "-", *cinemagraph.encoder_args(codec, crf, fps), "-movflags", "+faststart", out_mp4]
    covers = [float(sum(x.cover[x.index(t / fps)] for x in loaded if x.index(t / fps) is not None)) for t in range(T)]
    peak = int(np.argmax(covers)) if covers else 0
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    broken = False
    try:
        for t in range(T):
            fr, _ = render(base, loaded, t / fps)
            if preview and t == peak:
                Image.fromarray(fr).save(preview, quality=90)
            p.stdin.write(fr.tobytes())
        p.stdin.close()
    except BrokenPipeError:
        broken = True
    err = p.stderr.read().decode("utf-8", "replace")
    rc = p.wait()
    if rc != 0 or broken:
        raise RuntimeError(f"ffmpeg 실패 (코드 {rc}): {err[-800:]}")
    return {"mode": "layers", "frames": T, "fps": round(fps, 2), "seconds": round(T / fps, 2), "size": [W, H],
            "peak_frame": peak, "layers": len(loaded)}
