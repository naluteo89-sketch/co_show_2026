"""일반 ComfyUI와 공식 Comfy Cloud의 통신·진행률·모델 선택."""

import asyncio
import json
import uuid

import aiohttp

COMFY_PATHS = {            # 일반 ComfyUI 서버 (로컬, 192.168.1.91:8000 등)
    "ping": "/system_stats",
    "upload": "/upload/image",
    "prompt": "/prompt",
    "history": "/history/{pid}",
    "view": "/view",
    "ws": "/ws?clientId={cid}",
    "object_info": "/object_info",
}
CLOUD_PATHS = {            # 공식 Comfy Cloud (v1 API)
    "upload": "/api/upload/image",
    "prompt": "/api/prompt",
    "status": "/api/job/{pid}/status",
    "history": "/api/history/{pid}",
    "view": "/api/view",
    "ws": "/ws?clientId={cid}&token={key}",
    "object_info": "/api/object_info",
}


class _Base:
    label = ""

    def __init__(self, session: aiohttp.ClientSession, base: str):
        self.s = session
        self.base = base.rstrip("/")
        self.client_id = "pano360-" + uuid.uuid4().hex[:8]
        self.jobs = {}          # pid -> {"node", "value", "max", "started", "outputs": {node: output}, "error"}
        self._ws_task = None
        self._options = None    # 서버의 노드별 선택지 (모델 파일 목록 등) 캐시
        self.fallbacks = {}     # {노드 클래스: [대체 파일명, ...]} — 원래 파일이 없을 때 순서대로 시도

    # ---- websocket: 진행률 + (클라우드) 결과 수신
    def ensure_ws(self):
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop())

    async def close(self):
        if self._ws_task is not None:
            self._ws_task.cancel()
            await asyncio.gather(self._ws_task, return_exceptions=True)
            self._ws_task = None

    def _ws_url(self):
        raise NotImplementedError

    async def _ws_loop(self):
        while True:
            try:
                async with self.s.ws_connect(self._ws_url(), heartbeat=30, max_msg_size=0) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._on_msg(json.loads(msg.data))
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[{self.label}] websocket 재연결 대기: {e}")
                await asyncio.sleep(3)

    def _on_msg(self, m):
        d = m.get("data") or {}
        job = self.jobs.get(d.get("prompt_id"))
        if job is None:
            return
        t = m.get("type")
        if t in ("execution_start",):
            job["started"] = True
        elif t == "executing" and d.get("node"):
            job.update(started=True, node=str(d["node"]), value=0, max=0)
        elif t == "progress":
            job.update(started=True, node=str(d.get("node")), value=d.get("value", 0), max=d.get("max", 0))
        elif t == "executed" and d.get("output"):
            job["outputs"][str(d.get("node"))] = d["output"]
        elif t == "execution_error":
            job["error"] = f"{d.get('node_type')}: {d.get('exception_message')}"

    def _track(self, pid):
        self.jobs[pid] = {"started": False, "node": None, "value": 0, "max": 0, "outputs": {}, "error": None}

    def _running(self, pid):
        job = self.jobs.get(pid)
        if job is None:                      # 서버 재시작 등으로 추적 정보가 없을 때
            return {"state": "queued"}
        if not job["started"]:
            return {"state": "queued"}
        return {"state": "running", "node": job["node"], "value": job["value"], "max": job["max"]}

    @staticmethod
    def _collect(outputs, node_ids):
        files = []
        for nid in node_ids:
            for im in (outputs.get(str(nid)) or {}).get("images", []):
                files.append({**im, "node": str(nid)})
        return files

    # ---- 모델 이름 자동 맞춤
    # 워크플로의 파일명(예: flux2-vae.safetensors)이 서버에 하위폴더로 있으면(예: FLUX2\\flux2-vae.safetensors)
    # 서버 목록에서 같은 파일명을 찾아 바꿔 넣는다.
    def _headers(self):
        return {}

    async def _load_options(self, refresh=False):
        if self._options is not None and not refresh:
            return self._options
        async with self.s.get(self.base + self._paths["object_info"], headers=self._headers()) as r:
            r.raise_for_status()
            info = await r.json(content_type=None)
        opts = {}
        for cls, d in info.items():
            for sect in ("required", "optional"):
                for name, spec in ((d.get("input") or {}).get(sect) or {}).items():
                    if not isinstance(spec, list) or not spec:
                        continue
                    if isinstance(spec[0], list):
                        values = spec[0]
                    elif spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                        values = spec[1].get("options") or []
                    else:
                        continue
                    if values and all(isinstance(v, str) for v in values):
                        opts[(cls, name)] = values
        self._options = opts
        return opts

    # 정밀도·변형 표시 — 이것만 다르면 같은 모델로 본다
    _VARIANT = {"fp8", "fp16", "bf16", "fp32", "e4m3fn", "e5m2", "scaled", "mixed", "base", "distilled",
                "safetensors", "ckpt", "pth", "pt", "gguf", "q8", "q6", "q5", "q4", "model", "comfy", "comfyui"}

    @classmethod
    def _tokens(cls, name):
        import re
        b = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
        return [t for t in re.split(r"[^a-z0-9]+", b) if t]

    @classmethod
    def _family_match(cls, val, choices):
        """val 의 핵심 토큰(정밀도 표시 제외)이 모두 들어 있는 파일 중 가장 비슷한 것 하나."""
        core = [t for t in cls._tokens(val) if t not in cls._VARIANT]
        if len(core) < 2:
            return []
        want = set(cls._tokens(val))
        best, best_score = None, -1
        for c in choices:
            nc = cls._norm(c)
            if not all(t in nc for t in core):
                continue
            ct = set(cls._tokens(c))
            score = len(want & ct) * 10 - len(ct - want)        # 겹치는 토큰 많고 군더더기 적을수록
            if score > best_score:
                best, best_score = c, score
        if best:
            print(f"[api] ⚠ '{val}' 이 서버에 없어 같은 계열 '{best}' 로 대체해요. 다르면 config.json model_fallbacks 에 정확한 이름을 넣어 주세요.")
        return [best] if best else []

    @staticmethod
    def _norm(name):
        b = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
        b = b.rsplit(".", 1)[0] if "." in b else b
        return "".join(ch for ch in b if ch.isalnum())

    @staticmethod
    def _base(name):
        return name.replace("\\", "/").rsplit("/", 1)[-1].lower()

    async def resolve_models(self, wf: dict) -> list:
        """wf 를 제자리에서 고치고, 바꾼 내역 [(노드, 입력, 원래값, 바뀐값)] 반환."""
        try:
            opts = await self._load_options()
        except Exception as e:
            print(f"[api] object_info 조회 실패 — 모델 이름 자동 맞춤 생략: {e}")
            return []
        changed, missing = [], []
        for nid, node in wf.items():
            if node["class_type"].startswith("LoadImage"):    # 방금 올린 이미지는 목록 캐시에 없음
                continue
            for name, val in node["inputs"].items():
                choices = opts.get((node["class_type"], name))
                if not choices or not isinstance(val, str) or val in choices:
                    continue
                hits = [c for c in choices if self._base(c) == self._base(val)]
                if not hits:                        # 구분자·대소문자만 다른 경우 (flux-2-klein ↔ flux2_klein)
                    hits = [c for c in choices if self._norm(c) == self._norm(val)]
                if not hits:                        # 같은 모델 계열의 다른 파일명 (fp8/bf16/base 등만 다른 경우)
                    hits = self._family_match(val, choices)
                if not hits:                        # 대체 후보 (config.json "model_fallbacks")
                    for alt in self.fallbacks.get(node["class_type"], []):
                        hits = [c for c in choices if self._base(c) == self._base(alt)]
                        if hits:
                            break
                if hits:
                    node["inputs"][name] = hits[0]
                    changed.append((nid, name, val, hits[0]))
                elif "." in val:    # 파일처럼 보이는데 서버에 없음
                    keys = [t for t in self._tokens(val) if t not in self._VARIANT and len(t) > 2] or ["klein", "flux"]
                    near = [c for c in choices if any(t in self._norm(c) for t in keys)][:10]
                    missing.append(f"노드 {nid} {node['class_type']}.{name}: '{val}'"
                                   + (f"\n    서버에 있는 비슷한 파일: {near}" if near else f"\n    서버 목록 일부: {choices[:8]}"))
        for c in changed:
            print(f"[api] 모델 이름 맞춤 · 노드 {c[0]} {c[1]}: {c[2]} → {c[3]}")
        if missing and not getattr(self, "_retried", False):   # 서버에 파일을 새로 넣었을 수 있으니 한 번 새로 읽기
            self._options, self._retried = None, True
            try:
                return await self.resolve_models(wf)
            finally:
                self._retried = False
        if missing:
            raise RuntimeError("서버에 없는 파일이 있어요:\n" + "\n".join(missing))
        self._apply_distilled_profile(wf, changed)
        self._soften_ultrasharp(wf)
        return changed

    def _soften_ultrasharp(self, wf):
        """서버에 4x-UltraSharp 만 있으면 선이 딱딱해지고 색면이 뭉개져 그림처럼 보임 → Lanczos 비율을 높임."""
        b = getattr(self, "ultrasharp_blend", None)
        if not b:
            return
        if not any(n["class_type"] == "UpscaleModelLoader" and "ultrasharp" in self._norm(str(n["inputs"].get("model_name", "")))
                   for n in wf.values()):
            return
        for nid, n in wf.items():
            if n["class_type"] == "ImageBlend" and isinstance(n["inputs"].get("blend_factor"), (int, float)):
                n["inputs"]["blend_factor"] = max(float(n["inputs"]["blend_factor"]), float(b))
        print(f"[api] UltraSharp 사용 → AI/Lanczos 블렌드를 {b} 로 높임 (과한 선·색면 뭉개짐 완화)")

    def _apply_distilled_profile(self, wf, changed):
        """base(비증류) 모델을 요청했는데 서버엔 증류(distilled) 모델만 있어서 대체된 경우,
        증류 모델에 맞게 샘플러를 바꾼다. 증류 모델을 CFG 4·25스텝으로 돌리면 색이 과하게 밝아지고
        그림(일러스트)처럼 뭉개진다."""
        prof = getattr(self, "distilled_profile", None)
        if not prof:
            return
        swapped = [c for c in changed if wf[c[0]]["class_type"] == "UNETLoader"
                   and "base" in self._tokens(c[2]) and "base" not in self._tokens(c[3])]
        if not swapped:
            return
        for nid, node in wf.items():
            if node["class_type"] == "KSampler":
                node["inputs"]["cfg"] = float(prof.get("cfg", 1.0))
                partial = float(node["inputs"].get("denoise", 1)) < 0.99   # 이음새·리파인처럼 일부만 다시 그리는 샘플러
                node["inputs"]["steps"] = int(prof.get("partial_steps", 8) if partial else prof.get("steps", 4))
        print(f"[api] ⚠ 증류 모델 '{swapped[0][3]}' 사용 → 샘플러를 CFG {prof.get('cfg', 1.0)}, "
              f"{prof.get('steps', 6)}스텝으로 맞춤. 가장 좋은 결과는 서버에 base 모델을 넣는 거예요.")
        self.last_profile = "distilled"


class ComfyUI(_Base):
    """일반 ComfyUI 서버 (키 없음)"""
    label = "comfyui"
    _paths = COMFY_PATHS

    def _ws_url(self):
        return self.base.replace("http", "ws", 1) + COMFY_PATHS["ws"].format(cid=self.client_id)

    async def ping(self):
        async with self.s.get(self.base + COMFY_PATHS["ping"], timeout=aiohttp.ClientTimeout(total=3)) as r:
            return r.status == 200

    async def upload(self, data: bytes, name: str) -> str:
        fd = aiohttp.FormData()
        fd.add_field("image", data, filename=name, content_type="image/png")
        fd.add_field("type", "input")
        fd.add_field("subfolder", "pano360")
        fd.add_field("overwrite", "true")
        async with self.s.post(self.base + COMFY_PATHS["upload"], data=fd) as r:
            if r.status != 200:
                raise RuntimeError(f"로컬 업로드 실패: {await r.text()}")
            res = await r.json()
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    async def submit(self, wf: dict) -> str:
        await self.resolve_models(wf)
        self.ensure_ws()
        async with self.s.post(self.base + COMFY_PATHS["prompt"], json={"prompt": wf, "client_id": self.client_id}) as r:
            res = await r.json()
            if r.status != 200:
                raise RuntimeError(json.dumps(res, ensure_ascii=False)[:3000])
        self._track(res["prompt_id"])
        return res["prompt_id"]

    async def status(self, pid, node_ids):
        async with self.s.get(self.base + COMFY_PATHS["history"].format(pid=pid)) as r:
            h = (await r.json()).get(pid)
        if h:
            st = h.get("status", {})
            if st.get("status_str") == "error":
                msg = "실행 실패"
                for name, d in st.get("messages", []):
                    if name == "execution_error":
                        msg = f"{d.get('node_type')}: {d.get('exception_message')}"
                return {"state": "error", "error": msg}
            files = self._collect(h.get("outputs", {}), node_ids)
            return {"state": "done", "files": files} if files else {"state": "error", "error": "결과 이미지가 없어요."}
        return self._running(pid)

    async def fetch(self, f) -> bytes:
        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
        async with self.s.get(self.base + COMFY_PATHS["view"], params=params) as r:
            r.raise_for_status()
            return await r.read()


class ComfyCloud(_Base):
    """공식 Comfy Cloud (X-API-Key)"""
    label = "comfy_cloud"
    _paths = CLOUD_PATHS
    TERMINAL_FAIL = {"error", "non_retryable_error", "lost", "cancelled"}

    def __init__(self, session, base, api_key):
        super().__init__(session, base)
        self.key = api_key
        self.h = {"X-API-Key": api_key}

    def _headers(self):
        return self.h

    def _ws_url(self):
        return self.base.replace("http", "ws", 1) + CLOUD_PATHS["ws"].format(cid=self.client_id, key=self.key)

    async def ping(self):
        return bool(self.key)

    async def upload(self, data: bytes, name: str) -> str:
        fd = aiohttp.FormData()
        fd.add_field("image", data, filename=name, content_type="image/png")
        fd.add_field("type", "input")
        fd.add_field("overwrite", "true")
        async with self.s.post(self.base + CLOUD_PATHS["upload"], data=fd, headers=self.h) as r:
            if r.status != 200:
                raise RuntimeError(f"클라우드 업로드 실패 ({r.status}): {await r.text()}")
            res = await r.json()
        return res["name"]                       # Cloud 는 subfolder 무시(평면 저장)

    async def submit(self, wf: dict) -> str:
        await self.resolve_models(wf)
        self.ensure_ws()
        await asyncio.sleep(0.5)                 # ws 연결이 먼저 붙도록 (결과가 ws 로만 옴)
        async with self.s.post(self.base + CLOUD_PATHS["prompt"], json={"prompt": wf, "client_id": self.client_id},
                               headers=self.h) as r:
            res = await r.json(content_type=None)
            if r.status != 200 or res.get("error"):
                raise RuntimeError(json.dumps(res, ensure_ascii=False)[:3000])
        self._track(res["prompt_id"])
        return res["prompt_id"]

    async def status(self, pid, node_ids):
        async with self.s.get(self.base + CLOUD_PATHS["status"].format(pid=pid), headers=self.h) as r:
            st = (await r.json(content_type=None)).get("status", "")
        job = self.jobs.get(pid, {})
        if st == "success":
            files = self._collect(job.get("outputs", {}), node_ids)
            if not files:                        # ws 로 결과를 못 받은 경우 history 시도
                try:
                    async with self.s.get(self.base + CLOUD_PATHS["history"].format(pid=pid), headers=self.h) as r:
                        h = (await r.json(content_type=None)).get(pid) or {}
                    files = self._collect(h.get("outputs", {}), node_ids)
                except Exception:
                    pass
            return {"state": "done", "files": files} if files else \
                {"state": "error", "error": "완료됐지만 결과 정보를 받지 못했어요 (서버 재시작 등). Comfy Cloud 기록에서 확인해 주세요."}
        if st in self.TERMINAL_FAIL:
            return {"state": "error", "error": job.get("error") or f"클라우드 작업 상태: {st}"}
        out = self._running(pid)
        if out["state"] == "queued" and st in ("running", "in_progress", "executing"):
            out["state"] = "running"
        return out

    async def fetch(self, f) -> bytes:
        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
        async with self.s.get(self.base + CLOUD_PATHS["view"], params=params, headers=self.h, allow_redirects=False) as r:
            if r.status in (301, 302, 303, 307, 308):
                url = r.headers["Location"]
            elif r.status == 200:
                return await r.read()
            else:
                raise RuntimeError(f"클라우드 다운로드 실패 ({r.status})")
        async with self.s.get(url) as r2:        # 서명 URL — API 키를 보내지 않음
            r2.raise_for_status()
            return await r2.read()


def make_backends(session, backends_cfg: dict, keys: dict, fallbacks: dict | None = None,
                  distilled_profile: dict | None = None, ultrasharp_blend: float | None = None) -> dict:
    """config.json "backends" → {이름: ComfyUI | ComfyCloud}"""
    out = {}
    for name, b in backends_cfg.items():
        if b.get("type", "comfyui") == "comfy_cloud":
            if not keys.get("comfy_cloud"):
                print(f"[api] '{name}' 건너뜀: Comfy Cloud 키 없음")
                continue
            out[name] = ComfyCloud(session, b["url"], keys["comfy_cloud"])
        else:
            out[name] = ComfyUI(session, b["url"])
        out[name].label = name
        out[name].fallbacks = fallbacks or {}
        out[name].distilled_profile = distilled_profile
        out[name].ultrasharp_blend = ultrasharp_blend
    return out


NODE_LABELS = {"48": "UNET(메인 모델) 로더", "63": "LoRA 로더", "44": "CLIP(텍스트 인코더) 로더", "43": "VAE 로더",
               "71": "컨트롤 이미지", "115": "업스케일 모델", "8": "업스케일 모델", "201": "돌린 ERP", "202": "극점 마스크"}


def explain_comfy_error(msg: str) -> str:
    """ComfyUI 검증 오류(JSON 문자열)를 사람이 읽기 쉽게 정리. 해석 못 하면 원문 그대로."""
    try:
        data = json.loads(msg)
    except Exception:
        return msg
    lines = []
    err = data.get("error") or {}
    if isinstance(err, dict) and err.get("message"):
        lines.append(f"ComfyUI: {err.get('message')} {err.get('details') or ''}".strip())
    for nid, ne in (data.get("node_errors") or {}).items():
        label = NODE_LABELS.get(str(nid), ne.get("class_type", ""))
        for e in ne.get("errors", []):
            lines.append(f"- 노드 {nid} ({label}): {e.get('message')} — {e.get('details', '')[:300]}")
    return "\n".join(lines) or msg


# 이전 모듈을 사용하는 스크립트의 클래스 이름 유지. 구현은 위 클래스 하나로 통합.
LocalComfy = ComfyUI
CloudComfy = ComfyCloud
