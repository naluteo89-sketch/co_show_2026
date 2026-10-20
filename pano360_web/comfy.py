"""
ComfyUI 백엔드 2종 — 같은 인터페이스
  LocalComfy : 내 PC ComfyUI (http://127.0.0.1:8188)
  CloudComfy : Comfy Cloud (https://cloud.comfy.org, X-API-Key)

upload(bytes, name) -> 워크플로에 넣을 이미지 이름
submit(api_workflow) -> prompt_id
status(prompt_id, output_node_ids) -> {"state": queued|running|done|error, ...}
fetch(file_ref) -> bytes
"""

import asyncio
import json
import uuid

import aiohttp


class _Base:
    label = ""

    def __init__(self, session: aiohttp.ClientSession, base: str):
        self.s = session
        self.base = base.rstrip("/")
        self.client_id = "pano360-" + uuid.uuid4().hex[:8]
        self.jobs = {}          # pid -> {"node", "value", "max", "started", "outputs": {node: output}, "error"}
        self._ws_task = None

    # ---- websocket: 진행률 + (클라우드) 결과 수신
    def ensure_ws(self):
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop())

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


class LocalComfy(_Base):
    label = "local"

    def _ws_url(self):
        return self.base.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"

    async def ping(self):
        async with self.s.get(f"{self.base}/system_stats", timeout=aiohttp.ClientTimeout(total=3)) as r:
            return r.status == 200

    async def upload(self, data: bytes, name: str) -> str:
        fd = aiohttp.FormData()
        fd.add_field("image", data, filename=name, content_type="image/png")
        fd.add_field("type", "input")
        fd.add_field("subfolder", "pano360")
        fd.add_field("overwrite", "true")
        async with self.s.post(f"{self.base}/upload/image", data=fd) as r:
            if r.status != 200:
                raise RuntimeError(f"로컬 업로드 실패: {await r.text()}")
            res = await r.json()
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    async def submit(self, wf: dict) -> str:
        self.ensure_ws()
        async with self.s.post(f"{self.base}/prompt", json={"prompt": wf, "client_id": self.client_id}) as r:
            res = await r.json()
            if r.status != 200:
                raise RuntimeError(json.dumps(res, ensure_ascii=False)[:3000])
        self._track(res["prompt_id"])
        return res["prompt_id"]

    async def status(self, pid, node_ids):
        async with self.s.get(f"{self.base}/history/{pid}") as r:
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
        async with self.s.get(f"{self.base}/view", params=params) as r:
            r.raise_for_status()
            return await r.read()


class CloudComfy(_Base):
    label = "cloud"
    TERMINAL_FAIL = {"error", "non_retryable_error", "lost", "cancelled"}

    def __init__(self, session, base, api_key):
        super().__init__(session, base)
        self.key = api_key
        self.h = {"X-API-Key": api_key}

    def _ws_url(self):
        return self.base.replace("https", "wss", 1) + f"/ws?clientId={self.client_id}&token={self.key}"

    async def ping(self):
        return bool(self.key)

    async def upload(self, data: bytes, name: str) -> str:
        fd = aiohttp.FormData()
        fd.add_field("image", data, filename=name, content_type="image/png")
        fd.add_field("type", "input")
        fd.add_field("overwrite", "true")
        async with self.s.post(f"{self.base}/api/upload/image", data=fd, headers=self.h) as r:
            if r.status != 200:
                raise RuntimeError(f"클라우드 업로드 실패 ({r.status}): {await r.text()}")
            res = await r.json()
        return res["name"]                       # Cloud 는 subfolder 무시(평면 저장)

    async def submit(self, wf: dict) -> str:
        self.ensure_ws()
        await asyncio.sleep(0.5)                 # ws 연결이 먼저 붙도록 (결과가 ws 로만 옴)
        async with self.s.post(f"{self.base}/api/prompt", json={"prompt": wf, "client_id": self.client_id},
                               headers=self.h) as r:
            res = await r.json(content_type=None)
            if r.status != 200 or res.get("error"):
                raise RuntimeError(json.dumps(res, ensure_ascii=False)[:3000])
        self._track(res["prompt_id"])
        return res["prompt_id"]

    async def status(self, pid, node_ids):
        async with self.s.get(f"{self.base}/api/job/{pid}/status", headers=self.h) as r:
            st = (await r.json(content_type=None)).get("status", "")
        job = self.jobs.get(pid, {})
        if st == "success":
            files = self._collect(job.get("outputs", {}), node_ids)
            if not files:                        # ws 로 결과를 못 받은 경우 history 시도
                try:
                    async with self.s.get(f"{self.base}/api/history/{pid}", headers=self.h) as r:
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
        async with self.s.get(f"{self.base}/api/view", params=params, headers=self.h, allow_redirects=False) as r:
            if r.status in (301, 302, 303, 307, 308):
                url = r.headers["Location"]
            elif r.status == 200:
                return await r.read()
            else:
                raise RuntimeError(f"클라우드 다운로드 실패 ({r.status})")
        async with self.s.get(url) as r2:        # 서명 URL — API 키를 보내지 않음
            r2.raise_for_status()
            return await r2.read()
