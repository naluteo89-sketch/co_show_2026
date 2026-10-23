"""
화면 시작 정보 · 모델 점검 · 작업 목록 · 첫 페이지
"""

import asyncio
import os

from aiohttp import web

from .. import ROOT, llm, workflow
from .base import jresp


class InfoMixin:
    async def index(self, req):
        return web.FileResponse(os.path.join(ROOT, "static", "index.html"))

    async def config(self, req):
        async def ping(b):
            try:
                return await b.ping()
            except Exception:
                return False
        names = list(self.backends)
        oks = await asyncio.gather(*(ping(self.backends[n]) for n in names))
        backends = [{"name": n, "label": self.cfg["backends"][n].get("label", n),
                     "url": self.cfg["backends"][n]["url"], "ok": ok} for n, ok in zip(names, oks)]
        return jresp({
            "backends": backends,
            "default_backend": self._default_backend(),
            "openai_ready": self.openai.ready,
            "openai_model": self.cfg["openai_model"],
            "openai_models": self.cfg.get("openai_models", [self.cfg["openai_model"]]),
            "default_yaw": self.cfg.get("default_yaw", -20),
            "workflow": os.path.basename(self.cfg["workflow"]),
            "gen_defaults": self.cfg.get("gen_defaults", {}),
            "ltx360": {"positive": llm.video_event_prompt({}), "negative": llm.LTX360_NEG},
            "version": self.cfg.get("version", "?"),
        })

    async def models_check(self, req):
        """이 실행 위치에 필요한 모델이 다 있는지 점검: 360 생성 · 8K 화질 개선 · 원본 사진 업스케일 · 360 영상 노드."""
        bk, backend, err = self._pick_backend(req.query.get("backend"))
        if err:
            return err
        graphs = []
        try:
            graphs.append(("360 이미지 생성 · 발밑·하늘 보정", workflow.ui_to_api(workflow.load_template(self.cfg["workflow"]))))
        except Exception as e:
            graphs.append(("360 이미지 생성", {"_err": str(e)}))
        graphs.append(("8K 화질 개선 · 원본 사진 업스케일",
                       workflow.build_photo_upscale("check.png", self.upscale_model, "VR360/check")[0]))
        vt = os.path.basename(self.cfg.get("video_template", ""))
        try:
            dummy = {"mask": [0, 0, 512, 256], "src": [0, 0, 2048, 1024], "gen": [1024, 512], "gen_lo": [512, 256],
                     "out": [0, 0, 1024, 512]}
            ui, _, _ = workflow.fill_video_node(self._video_template(), image="check.png", positive="x", negative="x",
                                                mask_image="check.png", rect=(0, 0, 256, 256), crop=dummy)
            graphs.append((f"360 영상 ({vt})", workflow.video_node_api(ui)[0]))
        except Exception as e:
            graphs.append((f"360 영상 ({vt})", {"_err": str(e)}))
        out = []
        for title, g in graphs:
            if "_err" in g:
                out.append({"title": title, "error": g["_err"]})
                continue
            try:
                out.append({"title": title, "items": await backend.check_models(g)})
            except Exception as e:
                out.append({"title": title, "error": f"서버 모델 목록을 못 읽었어요: {e}"})
        return jresp({"backend": bk, "groups": out})

    async def list_runs(self, req):
        out = []
        for rid in sorted(os.listdir(self.store.root), reverse=True)[:100]:
            try:
                m = self.store.read(rid)
            except Exception:
                continue
            out.append({"run_id": rid, "name": m.get("original_name"), "attempts": len(m["attempts"]),
                        "summary": (m.get("analysis") or {}).get("summary_ko", ""),
                        "thumb": f"/runs/{rid}/source{m['source_ext']}",
                        "last": next((a for a in reversed(m["attempts"]) if a.get("files")), None)})
        return jresp(out)

    async def get_run(self, req):
        return jresp(self.store.read(req.match_info["run_id"]))
