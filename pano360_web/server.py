"""
웹 화면·라우트 (Pano360 Studio)
실행은 main.py 에서 합니다. 외부 API 호출은 모두 api.py 에 있어요.
"""

import asyncio
import io
import json
import os
import shutil
import time
import uuid

import aiohttp
from aiohttp import web
from PIL import Image

import api
import erp
import llm
import workflow

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))


def jresp(obj, status=200):
    return web.json_response(obj, status=status, dumps=lambda o: json.dumps(o, ensure_ascii=False))


class Studio:
    def __init__(self, cfg, keys):
        self.cfg = cfg
        self.keys = keys
        self.openai = None
        self.runs = cfg["runs_dir"]
        os.makedirs(self.runs, exist_ok=True)
        self.s = None
        self.backends = {}
        self.locks = {}

    async def startup(self, app):
        self.s = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=15))
        self.openai = api.OpenAIClient(self.s, self.keys["openai"], self.cfg["openai_model"])
        self.backends = api.make_backends(self.s, self.cfg["backends"], self.keys, self.cfg.get("model_fallbacks"),
                                         self.cfg.get("distilled_profile"),
                                         self.cfg.get("ultrasharp_blend"))

    async def cleanup(self, app):
        for b in self.backends.values():
            if b._ws_task:
                b._ws_task.cancel()
        await self.s.close()

    # ------------------------------------------------------------------ 파일 헬퍼
    def rdir(self, run_id, *p):
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise web.HTTPBadRequest(text="bad run id")
        d = os.path.join(self.runs, run_id, *p)
        return d

    def read_meta(self, run_id):
        with open(self.rdir(run_id, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def write_meta(self, run_id, meta):
        tmp = self.rdir(run_id, "meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.rdir(run_id, "meta.json"))

    def source(self, run_id):
        return erp.load_upright(self.rdir(run_id, "source" + self.read_meta(run_id)["source_ext"]))

    # ------------------------------------------------------------------ API
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
            "openai_ready": self.openai.ready,
            "openai_model": self.cfg["openai_model"],
            "openai_models": self.cfg.get("openai_models", [self.cfg["openai_model"]]),
            "default_yaw": self.cfg.get("default_yaw", -20),
            "workflow": os.path.basename(self.cfg["workflow"]),
            "gen_defaults": self.cfg.get("gen_defaults", {}),
            "version": self.cfg.get("version", "?"),
        })

    async def upload(self, req):
        form = await req.post()
        f = form.get("image")
        if f is None or not hasattr(f, "file"):
            return jresp({"error": "이미지가 없어요."}, 400)
        data = f.file.read()
        try:
            img = erp.load_upright(io.BytesIO(data))
        except Exception:
            return jresp({"error": "이미지 파일을 읽을 수 없어요."}, 400)
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        os.makedirs(self.rdir(run_id))
        ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
        with open(self.rdir(run_id, "source" + ext), "wb") as out:
            out.write(data)
        meta = {"run_id": run_id, "original_name": f.filename, "source_ext": ext,
                "width": img.width, "height": img.height,
                "exif_hfov": erp.exif_hfov(io.BytesIO(data)), "attempts": [],
                "is_erp": abs(img.width / img.height - 2.0) < 0.02}     # 이미 360(2:1) 이미지인지
        self.write_meta(run_id, meta)
        return jresp(meta)

    async def analyze(self, req):
        p = await req.json()
        run_id = p["run_id"]
        meta = self.read_meta(run_id)
        img = self.source(run_id)
        try:
            model = p.get("model") or self.cfg["openai_model"]
            if model not in self.cfg.get("openai_models", [model]):
                return jresp({"error": f"허용되지 않은 모델: {model}"}, 400)
            a, usage = await self.openai.analyze(img, p.get("note", ""), model)
        except Exception as e:
            return jresp({"error": str(e)}, 502)

        hfov = meta["exif_hfov"] or round(float(a["hfov_estimate"]), 1)
        placement = {
            "yaw": self.cfg.get("default_yaw", -20),
            "pitch": erp.pitch_from_horizon(a["horizon_y"], hfov, meta["width"], meta["height"]),
            "hfov": hfov,
            "vfov": None,
            "hfov_source": "EXIF" if meta["exif_hfov"] else "LLM 추정",
        }
        prompts = llm.assemble(a)
        meta.update(analysis=a, placement=placement, prompts=prompts, note=p.get("note", ""),
                    usage=usage, model=model)
        self.write_meta(run_id, meta)
        return jresp({"analysis": a, "placement": placement, "prompts": prompts, "model": model, "usage": usage})

    def _control(self, run_id, pl, name="control.png"):
        W, H = self.cfg.get("erp_size", [2048, 1024])
        img = erp.make_control(self.source(run_id), float(pl["yaw"]), float(pl["pitch"]), float(pl["hfov"]),
                               float(pl["vfov"]) if pl.get("vfov") else None, W, H)
        path = self.rdir(run_id, name)
        img.save(path, optimize=False)
        return path

    async def control(self, req):
        p = await req.json()
        await asyncio.to_thread(self._control, p["run_id"], p["placement"])
        return jresp({"url": f"/runs/{p['run_id']}/control.png?t={time.time():.0f}"})

    async def generate(self, req):
        p = await req.json()
        run_id, bk = p["run_id"], p.get("backend") or next(iter(self.backends), "")
        backend = self.backends.get(bk)
        if backend is None:
            return jresp({"error": f"'{bk}' 백엔드를 사용할 수 없어요. config.json 의 backends 를 확인해 주세요."}, 400)
        meta = self.read_meta(run_id)
        n = len(meta["attempts"]) + 1
        adir = self.rdir(run_id, f"a{n:02d}")
        os.makedirs(adir, exist_ok=True)

        mode = p.get("mode", "generate")
        extra = {}
        out_nodes = {"final": str(self.cfg["node_ids"]["save_final"])}     # 결과는 360 한 장만
        try:
            if mode == "pole":                        # 발밑·하늘 보정 (1단계: 돌린 ERP 인페인트)
                wf_api, seeds, out_nodes, extra = await self._start_pole(run_id, n, meta, p, backend)
            elif mode == "enhance":                   # 이미 360 → 화질 개선만
                with open(self.rdir(run_id, "source" + meta["source_ext"]), "rb") as f:
                    src_bytes = f.read()
                name = await backend.upload(src_bytes, f"{run_id}_a{n:02d}_erp_source{meta['source_ext']}")
                wf_api = workflow.build_enhance(name, meta["width"], meta["height"], prefix=f"VR360/{run_id}_a{n:02d}",
                                                **self._enhance_kwargs(meta))
                seeds, out_nodes = {}, {"final": "15"}
            else:
                ctrl_path = await asyncio.to_thread(self._control, run_id, p["placement"], f"a{n:02d}/control.png")
                with open(ctrl_path, "rb") as f:
                    ctrl_bytes = f.read()
                name = await backend.upload(ctrl_bytes, f"{run_id}_a{n:02d}_erp_control.png")
                tmpl = workflow.load_template(self.cfg["workflow"])
                st = self._settings(p, meta, p["prompts"].get("positive", ""))
                wf_api, seeds = workflow.build(tmpl, self.cfg["node_ids"], control_image=name, prompts=p["prompts"],
                                               seed1=p.get("seed1"), seed2=p.get("seed2"),
                                               prefix=f"VR360/{run_id}_a{n:02d}", settings=st)
                p["settings"] = {k: v for k, v in st.items() if k != "refine"}
            try:
                pid = await backend.submit(wf_api)      # 제출 전에 모델 이름을 서버 목록에 맞춰 고침
            finally:                                    # 성공/실패 모두 실제로 보낸 워크플로 저장
                with open(os.path.join(adir, "workflow_api.json"), "w", encoding="utf-8") as f:
                    json.dump(wf_api, f, ensure_ascii=False, indent=1)
        except Exception as e:
            msg = str(e)
            with open(os.path.join(adir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(msg)
            print(f"[pano360] 생성 요청 실패 ({run_id} a{n:02d}):\n{msg}")
            return jresp({"error": api.explain_comfy_error(msg)}, 502)

        meta["attempts"].append({"n": n, "backend": bk, "prompt_id": pid, "mode": mode, "out_nodes": out_nodes,
                                 **(extra if mode == "pole" else {}),
                                 "placement": p.get("placement") or {}, "prompts": p.get("prompts") or {},
                                 "settings": p.get("settings") or {}, "seeds": seeds, "state": "queued",
                                 "files": [], "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.write_meta(run_id, meta)
        return jresp({"attempt": n, "prompt_id": pid, "seeds": seeds})

    def _settings(self, p, meta, positive=""):
        """화면 설정 + config 기본값 + 리파인(LoRA 없는 디테일 패스) 프롬프트."""
        st = dict(self.cfg.get("gen_defaults", {}))
        st.update({k: v for k, v in (p.get("settings") or {}).items() if v is not None and v != ""})
        rf = dict(self.cfg.get("refine", {}))
        if rf.get("enabled", False):
            rp = llm.refine_prompts(meta.get("analysis") or {}, positive)
            rf.update(positive=rp["positive"], negative=rp["negative"])
            st["refine"] = rf
        return st

    def _enhance_kwargs(self, meta, seed=0):
        """8K 화질 개선 그래프 옵션 (리파인 포함)."""
        q = self.cfg.get("quest", {})
        kw = dict(final_w=q.get("final_width", 8192),
                  upscale_model=self.cfg["model_fallbacks"]["UpscaleModelLoader"][0],
                  blend=q.get("blend", 0.2), noise=q.get("noise", 0.004))
        rf = dict(self.cfg.get("refine", {}))
        if rf.get("enabled", False):
            rp = llm.refine_prompts(meta.get("analysis") or {})
            rf.update(positive=rp["positive"], negative=rp["negative"], seed=seed)
            kw.update(refine=rf, template_ui=workflow.load_template(self.cfg["workflow"]), ids=self.cfg["node_ids"])
        return kw

    # ------------------------------------------------------------------ 극점 보정
    def _pole_cfg(self):
        c = {"nadir_cap": 30, "zenith_cap": 22, "feather": 6, "denoise": 0.85, "lora_strength": 0.6, "work_width": 2048}
        c.update(self.cfg.get("pole", {}))
        return c

    def _pole_base_path(self, run_id, meta, from_n):
        if from_n:
            prev = meta["attempts"][int(from_n) - 1]
            files = prev.get("files", [])
            f = next((x for x in files if x.get("final")), None) or next((x for x in files if x["tag"] == "2k"), None)
            if f:
                return self.rdir(run_id, *f["url"].split("/")[3:])
            raise RuntimeError(f"#{from_n} 시도에 결과 이미지가 없어요.")
        if meta.get("is_erp"):
            return self.rdir(run_id, "source" + meta["source_ext"])
        raise RuntimeError("보정할 360 이미지가 없어요. 생성 결과가 있는 시도를 골라 주세요.")

    async def _start_pole(self, run_id, n, meta, p, backend):
        pc = self._pole_cfg()
        base_path = self._pole_base_path(run_id, meta, p.get("from_attempt"))
        aprefix = f"a{n:02d}"

        def prep():
            base = Image.open(base_path).convert("RGB")
            ww = int(pc["work_width"])
            work = base if base.width == ww else base.resize((ww, ww // 2), Image.LANCZOS)
            rot, mask = erp.pole_prepare(work, pc["nadir_cap"], pc["zenith_cap"], pc["feather"])
            work.save(self.rdir(run_id, aprefix, "pole_base.png"))
            rot.save(self.rdir(run_id, aprefix, "pole_rot.png"))
            mask.save(self.rdir(run_id, aprefix, "pole_mask.png"))
        await asyncio.to_thread(prep)

        def rd(name):
            with open(self.rdir(run_id, aprefix, name), "rb") as f:
                return f.read()
        rot_name = await backend.upload(rd("pole_rot.png"), f"{run_id}_{aprefix}_pole_rot.png")
        mask_name = await backend.upload(rd("pole_mask.png"), f"{run_id}_{aprefix}_pole_mask.png")
        prompts = p.get("pole_prompts") or llm.pole_prompts(meta.get("analysis") or {})
        g = {**self.cfg.get("gen_defaults", {}), **{k: v for k, v in (p.get("settings") or {}).items() if v}}
        tmpl = workflow.load_template(self.cfg["workflow"])
        ids = self.cfg["node_ids"]
        wf_api, seed = workflow.build_pole(
            tmpl, image=rot_name, mask=mask_name, positive=prompts["positive"], negative=prompts["negative"],
            seed=p.get("seed1"), steps=g.get("steps", 25), cfg=g.get("cfg", 4.0), denoise=pc["denoise"],
            lora_strength=pc["lora_strength"], prefix=f"VR360/{run_id}_{aprefix}",
            loader_ids=tuple(str(ids[k]) for k in ("unet", "lora", "clip", "vae")))
        extra = {"stage": 1, "pole": pc, "pole_prompts": prompts, "from_attempt": p.get("from_attempt")}
        return wf_api, {"seed1": seed, "seed2": None}, {"rot": "210"}, extra

    async def _pole_stage2(self, run_id, n, att, backend, st):
        """1단계 끝 → 되돌려 합성 → 2K 저장 → 8K 화질 개선 제출."""
        aprefix = f"a{n:02d}"
        f = next(x for x in st["files"] if x["node"] == "210")
        data = await backend.fetch(f)
        with open(self.rdir(run_id, aprefix, "pole_rot_out.png"), "wb") as out:
            out.write(data)
        pc = att["pole"]

        def merge():
            base = Image.open(self.rdir(run_id, aprefix, "pole_base.png"))
            rep = Image.open(self.rdir(run_id, aprefix, "pole_rot_out.png"))
            merged = erp.pole_merge(base, rep, pc["nadir_cap"], pc["zenith_cap"], pc["feather"])
            merged.save(self.rdir(run_id, aprefix, "result_2k.png"))
            return merged.size
        W, H = await asyncio.to_thread(merge)
        with open(self.rdir(run_id, aprefix, "result_2k.png"), "rb") as fp:
            name = await backend.upload(fp.read(), f"{run_id}_{aprefix}_pole_fixed.png")
        meta = self.read_meta(run_id)
        wf = workflow.build_enhance(name, W, H, prefix=f"VR360/{run_id}_{aprefix}", **self._enhance_kwargs(meta))
        pid = await backend.submit(wf)
        with open(self.rdir(run_id, aprefix, "workflow_api_stage2.json"), "w", encoding="utf-8") as fp:
            json.dump(wf, fp, ensure_ascii=False, indent=1)
        att.update(stage=2, prompt_id=pid, out_nodes={"final": "15"})   # 중간 파일은 폴더에만 두고 결과는 한 장

    async def status(self, req):
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        meta = self.read_meta(run_id)
        att = meta["attempts"][n - 1]
        if att["state"] in ("done", "error"):
            return jresp(att)
        backend = self.backends.get(att["backend"])
        ids = self.cfg["node_ids"]
        out_nodes = att.get("out_nodes") or {"final": str(ids["save_final"])}
        st = await backend.status(att["prompt_id"], list(out_nodes.values()))
        if st["state"] == "done" and att.get("mode") == "pole" and att.get("stage") == 1:
            lock = self.locks.setdefault((run_id, n), asyncio.Lock())
            async with lock:
                meta = self.read_meta(run_id)
                att = meta["attempts"][n - 1]
                if att.get("stage") == 1:
                    try:
                        await self._pole_stage2(run_id, n, att, backend, st)
                    except Exception as e:
                        att.update(state="error", error=f"극점 합성/8K 단계 실패: {e}")
                    self.write_meta(run_id, meta)
            return jresp({**att, "state": att["state"] if att["state"] == "error" else "running",
                          "node_title": "극점 합성 → 8K 화질 개선 시작", "value": 0, "max": 0})
        if st["state"] == "done":
            lock = self.locks.setdefault((run_id, n), asyncio.Lock())
            async with lock:
                meta = self.read_meta(run_id)
                att = meta["attempts"][n - 1]
                if att["state"] != "done":
                    files = []
                    final_tag = self.cfg.get("final_tag", "final")
                    for f in [x for x in st["files"] if x["node"] == out_nodes["final"]][:1]:
                        tag = final_tag
                        local = f"a{n:02d}/result_{tag}.png"
                        data = await backend.fetch(f)
                        with open(self.rdir(run_id, local), "wb") as out:
                            out.write(data)
                        tm = float(self.cfg.get("tone_match", 0))
                        if tm > 0 and att.get("mode", "generate") == "generate" and not meta.get("is_erp"):
                            # 생성 결과의 밝기·대비·색을 원본 사진 쪽으로 (그림 같은 밝고 평평한 톤 완화)
                            raw = self.rdir(run_id, f"a{n:02d}/result_{tag}_raw.png")
                            os.replace(self.rdir(run_id, local), raw)

                            def _tone(raw=raw, dst=self.rdir(run_id, local)):
                                erp.tone_match(Image.open(raw), self.source(run_id), tm).save(dst)
                            await asyncio.to_thread(_tone)
                            att["tone_match"] = tm
                        try:
                            att["seam"] = await asyncio.to_thread(erp.wrap_seam_score, self.rdir(run_id, local))
                            print(f"[pano360] 좌우 이음새 {att['seam']}")
                        except Exception as e:
                            print(f"[pano360] 이음새 측정 실패: {e}")
                        item = {"tag": tag, "url": f"/runs/{run_id}/{local}", "final": tag == final_tag}
                        if tag == final_tag:          # Quest 용 JPG (용량 1/5, 로딩 빠름)
                            jpg = f"a{n:02d}/{run_id}_a{n:02d}_{tag}.jpg"
                            await asyncio.to_thread(erp.save_quest_jpg, self.rdir(run_id, local), self.rdir(run_id, jpg),
                                                    self.cfg.get("jpg_quality", 95))
                            item["jpg"] = f"/runs/{run_id}/{jpg}"
                        files.append(item)
                    att.update(state="done", files=files, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
                    self.write_meta(run_id, meta)
            return jresp(att)
        if st["state"] == "error":
            att.update(state="error", error=st["error"])
            self.write_meta(run_id, meta)
            return jresp(att)
        node = st.get("node")
        return jresp({**att, "state": st["state"], "node": node,
                      "node_title": self.cfg.get("node_titles", {}).get(str(node), node),
                      "value": st.get("value"), "max": st.get("max")})

    async def to_viewer(self, req):
        """최종 JPG 를 Unity 뷰어 StreamingAssets 에 복사하고 photos.json 갱신."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        att = self.read_meta(run_id)["attempts"][n - 1]
        f = next((x for x in att.get("files", []) if x.get("jpg")), None)
        if not f:
            return jresp({"error": "Quest 용 JPG 가 없어요."}, 400)
        vdir = self.cfg.get("viewer_dir", "")
        vdir = vdir if os.path.isabs(vdir) else os.path.normpath(os.path.join(HERE, vdir))
        if not os.path.isdir(vdir):
            return jresp({"error": f"뷰어 폴더가 없어요: {vdir}"}, 400)
        src = self.rdir(run_id, *f["jpg"].split("/")[3:])
        name = os.path.basename(src)
        await asyncio.to_thread(shutil.copyfile, src, os.path.join(vdir, name))
        lst = os.path.join(vdir, "photos.json")
        try:
            with open(lst, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            data = {"files": []}
        if name not in data["files"]:
            data["files"].append(name)
        with open(lst, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)
        return jresp({"ok": True, "path": os.path.join(vdir, name), "count": len(data["files"])})

    async def list_runs(self, req):
        out = []
        for rid in sorted(os.listdir(self.runs), reverse=True)[:100]:
            try:
                m = self.read_meta(rid)
            except Exception:
                continue
            out.append({"run_id": rid, "name": m.get("original_name"), "attempts": len(m["attempts"]),
                        "summary": (m.get("analysis") or {}).get("summary_ko", ""),
                        "thumb": f"/runs/{rid}/source{m['source_ext']}",
                        "last": next((a for a in reversed(m["attempts"]) if a.get("files")), None)})
        return jresp(out)

    async def get_run(self, req):
        return jresp(self.read_meta(req.match_info["run_id"]))

    async def index(self, req):
        return web.FileResponse(os.path.join(HERE, "static", "index.html"))


def create_app(cfg, keys) -> web.Application:
    st = Studio(cfg, keys)
    app = web.Application(client_max_size=200 * 1024 * 1024)
    app.on_startup.append(st.startup)
    app.on_cleanup.append(st.cleanup)
    app.add_routes([
        web.get("/", st.index),
        web.get("/api/config", st.config),
        web.post("/api/upload", st.upload),
        web.post("/api/analyze", st.analyze),
        web.post("/api/control", st.control),
        web.post("/api/generate", st.generate),
        web.get("/api/run/{run_id}/attempt/{n}", st.status),
        web.get("/api/run/{run_id}", st.get_run),
        web.post("/api/run/{run_id}/attempt/{n}/to_viewer", st.to_viewer),
        web.get("/api/runs", st.list_runs),
        web.static("/runs", cfg["runs_dir"]),
        web.static("/static", os.path.join(HERE, "static")),
    ])
    return app
