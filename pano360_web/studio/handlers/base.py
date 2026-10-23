"""
모든 핸들러가 함께 쓰는 것: 응답 · 작업 잠금 · 실행 위치 고르기 · 파일 경로 · 시도(attempt) 기록 갱신
"""

import asyncio
from functools import wraps
import io
import json
import os
import time

import aiohttp
from aiohttp import web
from PIL import Image

from .. import ROOT, erp
from ..clients import comfy
from ..clients.openai_client import OpenAIClient
from ..storage import RunStore

Image.MAX_IMAGE_PIXELS = None


def jresp(obj, status=200):
    return web.json_response(obj, status=status, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def adir(n: int, *sub) -> str:
    """시도 폴더 기준 상대 경로: adir(3, "event") → "a03/event"."""
    return "/".join([f"a{n:02d}", *sub])


def resolve(path: str) -> str:
    """config 의 상대 경로는 pano360_web 폴더 기준."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ROOT, path))


def write_json(path, obj, indent=1):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=indent)


def placement_args(pl) -> tuple:
    """배치(placement) → erp 함수 인자 (yaw, pitch, hfov, vfov)."""
    return float(pl["yaw"]), float(pl["pitch"]), float(pl["hfov"]), float(pl["vfov"]) if pl.get("vfov") else None


def route_ids(req) -> tuple[str, int]:
    return req.match_info["run_id"], int(req.match_info["n"])


def serialized_run(handler):
    """같은 작업의 분석·제출·결과 저장이 이전 meta.json을 덮어쓰지 않게 한다."""
    @wraps(handler)
    async def wrapped(self, req):
        run_id = req.match_info.get("run_id")
        if run_id is None:
            run_id = (await req.json())["run_id"]
        self.store.path(run_id)
        async with self.lock(run_id):
            return await handler(self, req)
    return wrapped


class StudioBase:
    def __init__(self, cfg, keys):
        self.cfg = cfg
        self.keys = keys
        self.openai = None
        self.store = RunStore(cfg["runs_dir"])
        self.s = None
        self.backends = {}
        self.locks = {}
        self.tasks = set()

    async def startup(self, app):
        self.s = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=15))
        self.openai = OpenAIClient(self.s, self.keys["openai"], self.cfg["openai_model"])
        self.backends = comfy.make_backends(self.s, self.cfg["backends"], self.keys, self.cfg.get("model_fallbacks"),
                                            self.cfg.get("distilled_profile"),
                                            self.cfg.get("ultrasharp_blend"), self.cfg.get("model_sources"))

    async def cleanup(self, app):
        await asyncio.gather(*(b.close() for b in self.backends.values()))
        if self.s is not None:
            await self.s.close()

    def lock(self, run_id) -> asyncio.Lock:
        return self.locks.setdefault(run_id, asyncio.Lock())

    # ------------------------------------------------------------------ 설정
    def _default_backend(self):
        """기본 실행 위치: config "default_backend" (지금은 Comfy Cloud) → 없으면 첫 번째."""
        d = self.cfg.get("default_backend")
        return d if d in self.backends else next(iter(self.backends), "")

    def _pick_backend(self, name=None):
        """→ (이름, 백엔드 또는 None, 없을 때 돌려줄 400 응답)."""
        bk = name or self._default_backend()
        backend = self.backends.get(bk)
        err = None if backend else jresp({"error": f"'{bk}' 실행 위치를 사용할 수 없어요. config.json 의 backends 를 확인해 주세요."}, 400)
        return bk, backend, err

    def _pick_model(self, p):
        """→ (모델, 허용 안 될 때 돌려줄 400 응답)."""
        model = p.get("model") or self.cfg["openai_model"]
        if model not in self.cfg.get("openai_models", [model]):
            return model, jresp({"error": f"허용되지 않은 모델: {model}"}, 400)
        return model, None

    @property
    def upscale_model(self) -> str:
        return self.cfg["model_fallbacks"]["UpscaleModelLoader"][0]

    @property
    def jpg_quality(self) -> int:
        return int(self.cfg.get("jpg_quality", 95))

    # ------------------------------------------------------------------ 파일
    def source(self, run_id):
        return erp.load_upright(self.store.path(run_id, "source" + self.store.read(run_id)["source_ext"]))

    def _source_png(self, run_id) -> bytes:
        b = io.BytesIO()
        self.source(run_id).convert("RGB").save(b, "PNG")
        return b.getvalue()

    def _local(self, run_id, url) -> str:
        """화면 주소(/runs/<run_id>/...?v=) → 작업 폴더 안 실제 경로."""
        return self.store.path(run_id, *url.split("?")[0].split("/")[3:])

    def _final_path(self, run_id, att):
        f = next((x for x in att.get("files", []) if x.get("final")), None) or (att.get("files") or [None])[-1]
        if not f:
            raise RuntimeError("이 시도에는 360 결과 이미지가 없어요.")
        return self._local(run_id, f["url"])

    def _attempt_dir(self, run_id, n, *sub) -> str:
        """시도 폴더(하위 폴더 포함)를 만들고 상대 경로 반환."""
        d = adir(n, *sub)
        os.makedirs(self.store.path(run_id, d), exist_ok=True)
        return d

    # ------------------------------------------------------------------ ComfyUI
    async def _upload_file(self, backend, path, name):
        with open(path, "rb") as fp:
            return await backend.upload(fp.read(), name)

    async def _save_output(self, backend, ref, path):
        with open(path, "wb") as fp:
            fp.write(await backend.fetch(ref))

    async def _submit_workflow(self, backend, graph, path):
        try:
            return await backend.submit(graph)
        finally:
            # 모델 이름 자동 조정 후 실제 제출한 그래프를 실패한 경우에도 보관한다.
            write_json(path, graph)

    # ------------------------------------------------------------------ meta.json
    def _edit_attempt(self, run_id, n, fn):
        """meta.json 을 다시 읽어 n 번째 시도에 fn 적용 후 저장 (호출하는 쪽이 잠금을 잡고 있을 때)."""
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        fn(att)
        self.store.write(run_id, meta)
        return att

    async def _edit_attempt_locked(self, run_id, n, fn):
        async with self.lock(run_id):
            return self._edit_attempt(run_id, n, fn)
