"""
영향 범위 움직임: 원인(영역)에서 시작된 움직임이 '영향받는 것'에만, 원인에서 멀수록 약하게 퍼지게 한다.

  1. 360 전체를 저해상도로 영상화 (ComfyUI, 원인 영역이 가운데 오도록 돌려서 → 좌우 이음새는 뒤쪽)
  2. 그 영상에서 '프레임 t 의 각 픽셀이 첫 프레임 어디에서 왔는지'(흐름)만 뽑음  — 픽셀은 버림
  3. 흐름 × 영향 가중치(영향 대상 마스크 × 원인에서의 거리 감쇠 × 세기) → 그 외는 0 (원본 그대로)
  4. 반복: 흐름(변위)을 처음·끝 교차 페이드 → 픽셀을 섞지 않으므로 겹쳐 보이지 않음
  5. 매 프레임 '원본'을 그 변위만큼 당겨 샘플 → 흐림이 쌓이지 않음, 가중치 0 인 곳은 원본 픽셀 그대로
"""

import math

import numpy as np
from PIL import Image, ImageFilter

def _blur_f(a: np.ndarray, sigma: float) -> np.ndarray:
    """실수 배열 가우시안 흐림 (가로·세로 분리, 가장자리 반사)."""
    r = max(1, int(round(sigma * 3)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    a = np.pad(a.astype(np.float32), ((r, r), (r, r)), mode="reflect")
    a = sum(k[i] * a[:, i:i + a.shape[1] - 2 * r] for i in range(2 * r + 1))
    return sum(k[i] * a[i:i + a.shape[0] - 2 * r, :] for i in range(2 * r + 1))


RANGE_DEG = {"none": 0.0, "near": 35.0, "wide": 100.0}       # 원인 중심에서 이 각도까지 퍼짐 (넘으면 0)


def _cv2():
    try:
        import cv2
        return cv2
    except Exception:
        raise RuntimeError("opencv 가 필요해요. `pip install opencv-python-headless` 후 다시 실행해 주세요.")


# ----------------------------------------------------------------------------- 가중치
def angle_map(W, H, yaw, pitch) -> np.ndarray:
    """ERP 각 픽셀과 (yaw, pitch) 방향 사이 각도(°)."""
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 2 * math.pi
    lat = (0.5 - (np.arange(H) + 0.5) / H) * math.pi
    y0, p0 = math.radians(yaw), math.radians(pitch)
    c = (np.sin(lat)[:, None] * math.sin(p0) +
         np.cos(lat)[:, None] * math.cos(p0) * np.cos(lon[None, :] - y0))
    return np.degrees(np.arccos(np.clip(c, -1, 1))).astype(np.float32)


def load_mask(png: bytes | None, W, H, grow_px=2) -> np.ndarray | None:
    if not png:
        return None
    import io
    im = Image.open(io.BytesIO(png)).convert("L").resize((W, H), Image.BILINEAR)
    if grow_px:
        im = im.filter(ImageFilter.MaxFilter(grow_px * 2 + 1))
    a = np.asarray(im, np.float32) / 255.0
    return a if a.mean() > 0.0005 else None


def influence_weight(W, H, sources: list, feather_px=6) -> np.ndarray:
    """sources: [{yaw, pitch, targets:[{mask(H×W 0~1), range, strength}]}] → 가중치 H×W (0~1)."""
    w = np.zeros((H, W), np.float32)
    for s in sources:
        ang = angle_map(W, H, s["yaw"], s["pitch"])
        for t in s.get("targets", []):
            R = RANGE_DEG.get(t.get("range"), 0.0)
            if R <= 0 or t.get("mask") is None:
                continue
            f = np.clip(1 - ang / R, 0, 1)
            f = f * f * (3 - 2 * f)                         # 가까울수록 1, R 에서 0
            w = np.maximum(w, t["mask"] * f * float(np.clip(t.get("strength", 0.6), 0, 1)))
    if feather_px and w.any():
        w = np.asarray(Image.fromarray((w * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather_px)),
                       np.float32) / 255.0
    # 극점 근처는 늘어짐이 커서 막음
    lat = np.abs(0.5 - (np.arange(H) + 0.5) / H) * 180
    w *= np.clip((80 - lat) / 15, 0, 1)[:, None]
    return w


def auto_weight(flows: np.ndarray, sources: list, w_px: int, lo_deg=0.06, hi_deg=0.3, feather_px=4) -> np.ndarray:
    """대상을 정하지 않아도: 전체 영상에서 실제로 움직인 곳(흐름 크기) × 원인 거리 감쇠 × 세기.
    lo_deg 아래 흔들림(영상 모델 잡떨림)은 0. sources: [{yaw, pitch, range, strength}]."""
    T, H, W = flows.shape[:3]
    mag = np.linalg.norm(flows, axis=-1).max(0) * (360.0 / w_px)     # 도 단위
    mag = _blur_f(mag, 2)
    m = np.clip((mag - lo_deg) / (hi_deg - lo_deg), 0, 1)
    out = np.zeros((H, W), np.float32)
    for s in sources:
        R = RANGE_DEG.get(s.get("range"), 0.0)
        if R <= 0:
            continue
        f = np.clip(1 - angle_map(W, H, s["yaw"], s["pitch"]) / R, 0, 1)
        out = np.maximum(out, m * f * f * (3 - 2 * f) * float(np.clip(s.get("strength", 0.7), 0, 1)))
    if feather_px and out.any():
        out = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather_px)),
                         np.float32) / 255.0
    lat = np.abs(0.5 - (np.arange(H) + 0.5) / H) * 180
    return out * np.clip((80 - lat) / 15, 0, 1)[:, None]


# ----------------------------------------------------------------------------- 흐름
def backward_flows(frames: np.ndarray) -> np.ndarray:
    """(T,h,w,3) uint8 → (T,h,w,2): 프레임 t 의 각 픽셀이 첫 프레임에서 있던 곳까지의 변위."""
    cv2 = _cv2()
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = [np.zeros(g[0].shape + (2,), np.float32)]
    for t in range(1, len(g)):
        out.append(dis.calc(g[t], g[0], None))
    return np.stack(out)


def loop_displacements(d: np.ndarray, fade=6) -> np.ndarray:
    """변위를 처음·끝 교차 페이드 → 길이 T-fade, 끝이 처음으로 매끄럽게 이어짐 (픽셀이 아니라 변위를 섞음)."""
    T = len(d)
    k = int(min(fade, T // 3))
    if k < 2:
        return d
    out = d[: T - k].copy()
    for t in range(k):
        a = (t + 1) / (k + 1)
        a = a * a * (3 - 2 * a)
        out[t] = d[t] * a + d[T - k + t] * (1 - a)
    return out


def _circ_span(cols: np.ndarray, n: int):
    """켜진 열(bool, 길이 n) → 가장 짧게 감싸는 (시작, 길이) — 좌우 이음새를 넘어도 됨."""
    on = np.nonzero(cols)[0]
    if len(on) == 0:
        return 0, 0
    gaps = np.diff(np.concatenate([on, [on[0] + n]]))       # 각 켜진 열 뒤의 빈칸
    g = int(np.argmax(gaps))
    start = int(on[(g + 1) % len(on)])
    return start, int(n - (gaps[g] - 1))


class FlowWarp:
    """저해상도 360 영상의 흐름을 영향 가중치만큼 원본 고해상도(8K) 정지 화면에 적용.

    화질: 영향받는 부분만 잘라 8K 에서 당겨 샘플한 뒤 출력 크기로 '면적 평균' 축소 → 보간 흐림이 거의 안 남음.
    출력 정지 화면도 같은 8K 의 면적 평균 축소본이어야 경계가 안 보임 (base_frame() 사용).
    """

    def __init__(self, frames_small: np.ndarray, weight_small: np.ndarray, roll_px: int, still_hi: np.ndarray,
                 W: int, H: int, fade=6, max_shift_deg=4.0, smooth_px=3, flows=None):
        cv2 = _cv2()
        h, w = frames_small.shape[1:3]
        d = backward_flows(frames_small) if flows is None else flows     # flows 는 이미 되돌린 좌표
        if roll_px and flows is None:                                        # 원인을 가운데로 돌렸던 것을 되돌림
            d = np.roll(d, -roll_px, axis=2)
        d = loop_displacements(d, fade)
        if smooth_px:                                      # 잔떨림 제거
            d = np.stack([cv2.GaussianBlur(x, (0, 0), smooth_px) for x in d])
        lim = max_shift_deg / 360 * w                       # 너무 크게 끌면 늘어짐 → 상한
        mag = np.linalg.norm(d, axis=-1, keepdims=True)
        d = d * np.minimum(1, lim / np.maximum(mag, 1e-6))
        self.d = (d * weight_small[None, :, :, None]).astype(np.float32)
        self.T = len(self.d)
        self.W, self.H, self.sx, self.sy = W, H, W / w, H / h
        Hh, Wh = still_hi.shape[:2]
        self.k = k = Wh // W if (Wh % W == 0 and Hh == H * (Wh // W)) else 1
        self.still_hi = still_hi if k > 1 else None
        self.base = cv2.resize(still_hi, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) if k > 1 \
            else np.asarray(still_hi, np.float32)
        self.weight = cv2.resize(weight_small, (W, H), interpolation=cv2.INTER_LINEAR)
        self.active = self.weight > 1e-3
        self.max_shift = float(np.abs(self.d).max() * max(self.sx, self.sy)) if self.T else 0.0
        # 영향받는 곳을 감싸는 사각형 (출력 좌표, 열은 이음새를 넘어 감쌀 수 있음)
        m = int(math.ceil(self.max_shift)) + 8
        rows = np.nonzero(self.active.any(1))[0]
        if len(rows):
            r0, r1 = max(0, rows[0] - m), min(H, rows[-1] + 1 + m)
            c0, cl = _circ_span(self.active.any(0), W)
            cl = min(W, cl + 2 * m)
            self.ro = np.arange(r0, r1)
            self.co = (c0 - m + np.arange(cl)) % W
        else:
            self.ro = self.co = np.arange(0)
        if len(self.ro) and k > 1:
            rh = np.arange(self.ro[0] * k, (self.ro[-1] + 1) * k)
            ch = (np.repeat(self.co * k, k) + np.tile(np.arange(k), len(self.co))) % Wh
            self.crop_hi = still_hi[np.ix_(rh, ch)].astype(np.float32)
        else:
            self.crop_hi = self.base[np.ix_(self.ro, self.co)] if len(self.ro) else None
        self.act_crop = self.active[np.ix_(self.ro, self.co)] if len(self.ro) else None

    def base_frame(self) -> np.ndarray:
        return self.base

    def frame(self, t: int) -> np.ndarray:
        """프레임 t (H,W,3 float32). 가중치 0 인 곳은 원본 픽셀 그대로."""
        cv2 = _cv2()
        out = self.base.copy()
        if self.crop_hi is None or not self.act_crop.any():
            return out
        k = self.k
        dd = cv2.resize(self.d[t % self.T], (self.W, self.H), interpolation=cv2.INTER_CUBIC)[np.ix_(self.ro, self.co)]
        hh, ww = self.crop_hi.shape[:2]
        if k > 1:
            dd = cv2.resize(dd, (ww, hh), interpolation=cv2.INTER_LINEAR)
        gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
        mx = gx + dd[..., 0] * np.float32(self.sx * k)
        my = gy + dd[..., 1] * np.float32(self.sy * k)
        warped = cv2.remap(self.crop_hi, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if k > 1:
            warped = cv2.resize(warped, (len(self.co), len(self.ro)), interpolation=cv2.INTER_AREA)
        tile = out[np.ix_(self.ro, self.co)]
        tile[self.act_crop] = warped[self.act_crop]
        out[np.ix_(self.ro, self.co)] = tile
        return out


def roll_for(yaw: float, w: int) -> int:
    """원인 yaw 가 ERP 가운데로 오게 돌리는 픽셀 수 (좌우 이음새를 원인 반대편으로)."""
    return int(round(-yaw / 360 * w))


def sharpness(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """라플라시안 분산 (선명도 비교용)."""
    cv2 = _cv2()
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(lap[mask].var() if mask is not None else lap.var())
