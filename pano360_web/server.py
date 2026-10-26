"""
웹 화면·라우트 (Pano360 Studio)
실행은 main.py, 외부 통신은 openai_client.py와 comfy.py, 작업 저장은 storage.py에서 관리합니다.
"""

import asyncio
from functools import wraps
import io
import json
import os
import shutil
import time
import uuid

import aiohttp
from aiohttp import web
import numpy as np
from PIL import Image

import comfy
from openai_client import OpenAIClient
from storage import RunStore, InvalidRunPath
import erp
import llm
import workflow

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))


def jresp(obj, status=200):
    return web.json_response(obj, status=status, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def serialized_run(handler):
    """같은 작업의 분석·제출·결과 저장이 이전 meta.json을 덮어쓰지 않게 한다."""
    @wraps(handler)
    async def wrapped(self, req):
        run_id = req.match_info.get("run_id")
        if run_id is None:
            run_id = (await req.json())["run_id"]
        self.store.path(run_id)
        async with self.locks.setdefault(run_id, asyncio.Lock()):
            return await handler(self, req)
    return wrapped


class Studio:
    def __init__(self, cfg, keys):
        self.cfg = cfg
        self.keys = keys
        self.openai = None
        self.store = RunStore(cfg["runs_dir"])
        self.s = None
        self.backends = {}
        self.locks = {}

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


    def _source_png(self, run_id) -> bytes:
        b = io.BytesIO()
        self.source(run_id).convert("RGB").save(b, "PNG")
        return b.getvalue()

    def _paste_photo_file(self, run_id, n, att, dst):
        """dst(최종 360)에 원본 사진을 다시 붙임. AI 업스케일된 원본(photo_up.png)이 있으면 그걸로 → 원본 자리도 선명."""
        kp = self.cfg.get("keep_photo", {})
        pl = att.get("placement") or {}
        up = self.store.path(run_id, f"a{n:02d}", "photo_up.png")
        fin = Image.open(dst)
        src = Image.open(up).convert("RGB") if os.path.isfile(up) else self.source(run_id)
        mode, scale = erp.photo_paste_mode(fin.width, src.width, float(pl["hfov"]), kp.get("paste_mode", "auto"))
        erp.paste_photo(fin, src, float(pl["yaw"]), float(pl["pitch"]), float(pl["hfov"]),
                        float(pl["vfov"]) if pl.get("vfov") else None, kp.get("paste_feather", 0.05), mode=mode,
                        detail=float(kp.get("detail_strength", 0.9))).save(dst)
        return ("upscaled-" if os.path.isfile(up) else "") + mode, scale

    def source(self, run_id):
        return erp.load_upright(self.store.path(run_id, "source" + self.store.read(run_id)["source_ext"]))

    def _default_backend(self):
        """기본 실행 위치: config "default_backend" (지금은 Comfy Cloud) → 없으면 첫 번째."""
        d = self.cfg.get("default_backend")
        return d if d in self.backends else next(iter(self.backends), "")

    async def models_check(self, req):
        """이 실행 위치에 필요한 모델이 다 있는지 점검: 360 생성 · 8K 화질 개선 · 원본 사진 업스케일 · 360 영상 노드."""
        bk = req.query.get("backend") or self._default_backend()
        backend = self.backends.get(bk)
        if backend is None:
            return jresp({"error": f"'{bk}' 실행 위치를 사용할 수 없어요."}, 400)
        graphs = []
        try:
            graphs.append(("360 이미지 생성 · 발밑·하늘 보정", workflow.ui_to_api(workflow.load_template(self.cfg["workflow"]))))
        except Exception as e:
            graphs.append(("360 이미지 생성", {"_err": str(e)}))
        up = self.cfg["model_fallbacks"]["UpscaleModelLoader"][0]
        graphs.append(("8K 화질 개선 · 원본 사진 업스케일", workflow.build_photo_upscale("check.png", up, "VR360/check")[0]))
        vt = os.path.basename(self.cfg.get("video_template", ""))
        try:
            ui, _, _ = workflow.fill_video_node(self._video_template(), image="check.png", positive="x", negative="x",
                                                mask_image="check.png", rect=(0, 0, 256, 256))
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
            "default_backend": self._default_backend(),
            "openai_ready": self.openai.ready,
            "openai_model": self.cfg["openai_model"],
            "openai_models": self.cfg.get("openai_models", [self.cfg["openai_model"]]),
            "default_yaw": self.cfg.get("default_yaw", -20),
            "workflow": os.path.basename(self.cfg["workflow"]),
            "gen_defaults": self.cfg.get("gen_defaults", {}),
            "ltx360": {"positive": self._ltx360_prompt({}, self._video_is_crop()),
                       "negative": workflow.LTX_CROP_NEG if self._video_is_crop() else workflow.LTX360_NEG},
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
        os.makedirs(self.store.path(run_id))
        ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
        with open(self.store.path(run_id, "source" + ext), "wb") as out:
            out.write(data)
        meta = {"run_id": run_id, "original_name": f.filename, "source_ext": ext,
                "width": img.width, "height": img.height,
                "exif_hfov": erp.exif_hfov(io.BytesIO(data)), "attempts": [],
                "is_erp": abs(img.width / img.height - 2.0) < 0.02}     # 이미 360(2:1) 이미지인지
        self.store.write(run_id, meta)
        return jresp(meta)

    @serialized_run
    async def analyze(self, req):
        p = await req.json()
        run_id = p["run_id"]
        meta = self.store.read(run_id)
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
        self.store.write(run_id, meta)
        return jresp({"analysis": a, "placement": placement, "prompts": prompts, "model": model, "usage": usage})

    def _control(self, run_id, pl, name="control.png"):
        W, H = self.cfg.get("erp_size", [2048, 1024])
        img = erp.make_control(self.source(run_id), float(pl["yaw"]), float(pl["pitch"]), float(pl["hfov"]),
                               float(pl["vfov"]) if pl.get("vfov") else None, W, H)
        path = self.store.path(run_id, name)
        img.save(path, optimize=False)
        return path

    def _keep_mask(self, run_id, pl, name, feather):
        W, H = self.cfg.get("erp_size", [2048, 1024])
        src = self.source(run_id)
        m = erp.keep_mask(src.size, float(pl["yaw"]), float(pl["pitch"]), float(pl["hfov"]),
                          float(pl["vfov"]) if pl.get("vfov") else None, W, H, feather)
        path = self.store.path(run_id, name)
        m.save(path)
        with open(path, "rb") as f:
            return f.read()

    async def control(self, req):
        p = await req.json()
        await asyncio.to_thread(self._control, p["run_id"], p["placement"])
        return jresp({"url": f"/runs/{p['run_id']}/control.png?t={time.time():.0f}"})

    @serialized_run
    async def import_result(self, req):
        """생성 없이: 올린 360(2:1) 이미지를 그대로 결과로 등록 → 5·6단계(마스크·프롬프트·영상) 바로 사용."""
        run_id = req.match_info["run_id"]
        meta = self.store.read(run_id)
        ratio = meta["width"] / max(1, meta["height"])
        if abs(ratio - 2.0) > 0.05:
            return jresp({"error": f"360(가로:세로 2:1) 이미지만 바로 쓸 수 있어요. 지금 {meta['width']}×{meta['height']}"}, 400)
        n = len(meta["attempts"]) + 1
        d = f"a{n:02d}"
        os.makedirs(self.store.path(run_id, d), exist_ok=True)
        name = f"{d}/{run_id}_{d}_360.jpg"

        def save():
            im = self.source(run_id).convert("RGB")
            if im.width != 2 * im.height:
                im = im.resize((im.width, im.width // 2), Image.LANCZOS)
            im.save(self.store.path(run_id, *name.split("/")), "JPEG", quality=int(self.cfg.get("jpg_quality", 95)),
                    subsampling=0)                      # optimize 는 큰 이미지에서 libjpeg 버퍼 오류가 날 수 있어 끔
            return im.size
        size = await asyncio.to_thread(save)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        att = {"n": n, "backend": "-", "mode": "import", "state": "done", "prompt_id": None, "out_nodes": {},
               "placement": {}, "prompts": {}, "settings": {}, "seeds": {},
               "files": [{"tag": "import", "url": f"/runs/{run_id}/{name}", "jpg": f"/runs/{run_id}/{name}", "final": True}],
               "size": list(size), "created": now, "finished": now}
        meta["attempts"].append(att)
        self.store.write(run_id, meta)
        return jresp(att)


    @serialized_run
    async def generate(self, req):
        p = await req.json()
        run_id, bk = p["run_id"], p.get("backend") or self._default_backend()
        backend = self.backends.get(bk)
        if backend is None:
            return jresp({"error": f"'{bk}' 백엔드를 사용할 수 없어요. config.json 의 backends 를 확인해 주세요."}, 400)
        meta = self.store.read(run_id)
        n = len(meta["attempts"]) + 1
        adir = self.store.path(run_id, f"a{n:02d}")
        os.makedirs(adir, exist_ok=True)

        mode = p.get("mode", "generate")
        extra = {}
        out_nodes = {"final": str(self.cfg["node_ids"]["save_final"])}     # 결과는 360 한 장만
        try:
            if mode == "pole":                        # 발밑·하늘 보정 (1단계: 돌린 ERP 인페인트)
                wf_api, seeds, out_nodes, extra = await self._start_pole(run_id, n, meta, p, backend)
            elif mode == "enhance":                   # 이미 360 → 화질 개선만
                with open(self.store.path(run_id, "source" + meta["source_ext"]), "rb") as f:
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
                kp = self.cfg.get("keep_photo", {})
                mask_name = None
                if kp.get("enabled", True):          # 사진 영역은 그대로 두고 초록 부분만 생성
                    mask_bytes = await asyncio.to_thread(self._keep_mask, run_id, p["placement"], f"a{n:02d}/keep_mask.png",
                                                         kp.get("mask_feather", 0.04))
                    mask_name = await backend.upload(mask_bytes, f"{run_id}_a{n:02d}_keep_mask.png")
                tmpl = workflow.load_template(self.cfg["workflow"])
                st = self._settings(p, meta, p["prompts"].get("positive", ""))
                wf_api, seeds = workflow.build(tmpl, self.cfg["node_ids"], control_image=name, prompts=p["prompts"],
                                               seed1=p.get("seed1"), seed2=p.get("seed2"),
                                               prefix=f"VR360/{run_id}_a{n:02d}", settings=st, keep_mask=mask_name)
                if kp.get("enabled", True) and kp.get("upscale_photo", True):
                    # 원본 사진 자리 화질: 사진을 원래 해상도로 AI 업스케일 → 마지막에 이걸 붙임
                    ph_name = await backend.upload(await asyncio.to_thread(self._source_png, run_id), f"{run_id}_a{n:02d}_photo.png")
                    out_nodes["photo_up"] = workflow.add_photo_upscale(wf_api, ph_name, f"VR360/{run_id}_a{n:02d}",
                                                                       self.cfg["model_fallbacks"]["UpscaleModelLoader"][0])
                p["settings"] = {k: v for k, v in st.items() if k != "refine"}
            pid = await self._submit_workflow(backend, wf_api, os.path.join(adir, "workflow_api.json"))
        except Exception as e:
            msg = str(e)
            with open(os.path.join(adir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(msg)
            print(f"[pano360] 생성 요청 실패 ({run_id} a{n:02d}):\n{msg}")
            return jresp({"error": comfy.explain_comfy_error(msg)}, 502)

        meta["attempts"].append({"n": n, "backend": bk, "prompt_id": pid, "mode": mode, "out_nodes": out_nodes,
                                 **(extra if mode == "pole" else {}),
                                 "placement": p.get("placement") or {}, "prompts": p.get("prompts") or {},
                                 "settings": p.get("settings") or {}, "seeds": seeds, "state": "queued",
                                 "files": [], "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.store.write(run_id, meta)
        return jresp({"attempt": n, "prompt_id": pid, "seeds": seeds})

    async def _submit_workflow(self, backend, graph, path):
        try:
            return await backend.submit(graph)
        finally:
            # 모델 이름 자동 조정 후 실제 제출한 그래프를 실패한 경우에도 보관한다.
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(graph, fp, ensure_ascii=False, indent=1)

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
                return self.store.path(run_id, *f["url"].split("/")[3:])
            raise RuntimeError(f"#{from_n} 시도에 결과 이미지가 없어요.")
        if meta.get("is_erp"):
            return self.store.path(run_id, "source" + meta["source_ext"])
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
            work.save(self.store.path(run_id, aprefix, "pole_base.png"))
            rot.save(self.store.path(run_id, aprefix, "pole_rot.png"))
            mask.save(self.store.path(run_id, aprefix, "pole_mask.png"))
        await asyncio.to_thread(prep)

        def rd(name):
            with open(self.store.path(run_id, aprefix, name), "rb") as f:
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
        with open(self.store.path(run_id, aprefix, "pole_rot_out.png"), "wb") as out:
            out.write(data)
        pc = att["pole"]

        def merge():
            base = Image.open(self.store.path(run_id, aprefix, "pole_base.png"))
            rep = Image.open(self.store.path(run_id, aprefix, "pole_rot_out.png"))
            merged = erp.pole_merge(base, rep, pc["nadir_cap"], pc["zenith_cap"], pc["feather"])
            merged.save(self.store.path(run_id, aprefix, "result_2k.png"))
            return merged.size
        W, H = await asyncio.to_thread(merge)
        with open(self.store.path(run_id, aprefix, "result_2k.png"), "rb") as fp:
            name = await backend.upload(fp.read(), f"{run_id}_{aprefix}_pole_fixed.png")
        meta = self.store.read(run_id)
        wf = workflow.build_enhance(name, W, H, prefix=f"VR360/{run_id}_{aprefix}", **self._enhance_kwargs(meta))
        pid = await self._submit_workflow(backend, wf, self.store.path(run_id, aprefix, "workflow_api_stage2.json"))
        att.update(stage=2, prompt_id=pid, out_nodes={"final": "15"})   # 중간 파일은 폴더에만 두고 결과는 한 장

    @serialized_run
    async def status(self, req):
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        if att["state"] in ("done", "error"):
            return jresp(att)
        backend = self.backends.get(att["backend"])
        ids = self.cfg["node_ids"]
        out_nodes = att.get("out_nodes") or {"final": str(ids["save_final"])}
        st = await backend.status(att["prompt_id"], list(out_nodes.values()))
        if st["state"] == "done" and att.get("mode") == "pole" and att.get("stage") == 1:
            meta = self.store.read(run_id)
            att = meta["attempts"][n - 1]
            if att.get("stage") == 1:
                try:
                    await self._pole_stage2(run_id, n, att, backend, st)
                except Exception as e:
                    att.update(state="error", error=f"극점 합성/8K 단계 실패: {e}")
                self.store.write(run_id, meta)
            return jresp({**att, "state": att["state"] if att["state"] == "error" else "running",
                          "node_title": "극점 합성 → 8K 화질 개선 시작", "value": 0, "max": 0})
        if st["state"] == "done":
            meta = self.store.read(run_id)
            att = meta["attempts"][n - 1]
            if att["state"] != "done":
                files = []
                final_tag = self.cfg.get("final_tag", "final")
                for f in [x for x in st["files"] if x["node"] == out_nodes["final"]][:1]:
                    tag = final_tag
                    local = f"a{n:02d}/result_{tag}.png"
                    data = await backend.fetch(f)
                    with open(self.store.path(run_id, local), "wb") as out:
                        out.write(data)
                    tm = float(self.cfg.get("tone_match", 0))
                    if tm > 0 and att.get("mode", "generate") == "generate" and not meta.get("is_erp"):
                        # 생성 결과의 밝기·대비·색을 원본 사진 쪽으로 (그림 같은 밝고 평평한 톤 완화)
                        raw = self.store.path(run_id, f"a{n:02d}/result_{tag}_raw.png")
                        os.replace(self.store.path(run_id, local), raw)

                        def _tone(raw=raw, dst=self.store.path(run_id, local)):
                            erp.tone_match(Image.open(raw), self.source(run_id), tm).save(dst)
                        await asyncio.to_thread(_tone)
                        att["tone_match"] = tm
                    kp = self.cfg.get("keep_photo", {})
                    pl = att.get("placement") or {}
                    if kp.get("enabled", True) and att.get("mode", "generate") == "generate" and pl.get("hfov"):
                        # 원본 사진을 원래 해상도로 같은 자리에 다시 붙임 → 사진에 있던 건 절대 안 사라짐
                        pf = next((x for x in st["files"] if x["node"] == out_nodes.get("photo_up")), None)
                        if pf:                       # 원래 해상도로 AI 업스케일한 원본 사진
                            with open(self.store.path(run_id, f"a{n:02d}", "photo_up.png"), "wb") as fp:
                                fp.write(await backend.fetch(pf))
                        mode, scale = await asyncio.to_thread(self._paste_photo_file, run_id, n, att,
                                                              self.store.path(run_id, local))
                        att["photo_pasted"] = mode
                        print(f"[pano360] 원본 사진 다시 붙임: {mode} (사진이 최종보다 {scale:.2f}배 흐림)")
                    try:
                        att["seam"] = await asyncio.to_thread(erp.wrap_seam_score, self.store.path(run_id, local))
                        print(f"[pano360] 좌우 이음새 {att['seam']}")
                    except Exception as e:
                        print(f"[pano360] 이음새 측정 실패: {e}")
                    item = {"tag": tag, "url": f"/runs/{run_id}/{local}", "final": tag == final_tag}
                    if tag == final_tag:          # Quest 용 JPG (용량 1/5, 로딩 빠름)
                        jpg = f"a{n:02d}/{run_id}_a{n:02d}_{tag}.jpg"
                        await asyncio.to_thread(erp.save_quest_jpg, self.store.path(run_id, local), self.store.path(run_id, jpg),
                                                self.cfg.get("jpg_quality", 95))
                        item["jpg"] = f"/runs/{run_id}/{jpg}"
                    files.append(item)
                att.update(state="done", files=files, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
                self.store.write(run_id, meta)
            return jresp(att)
        if st["state"] == "error":
            att.update(state="error", error=st["error"])
            self.store.write(run_id, meta)
            return jresp(att)
        node = st.get("node")
        return jresp({**att, "state": st["state"], "node": node,
                      "node_title": self.cfg.get("node_titles", {}).get(str(node), node),
                      "value": st.get("value"), "max": st.get("max")})

    async def to_viewer(self, req):
        """최종 JPG 를 Unity 뷰어 StreamingAssets 에 복사하고 photos.json 갱신."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        att = self.store.read(run_id)["attempts"][n - 1]
        f = next((x for x in att.get("files", []) if x.get("jpg")), None)
        if not f:
            return jresp({"error": "Quest 용 JPG 가 없어요."}, 400)
        vdir = self.cfg.get("viewer_dir", "")
        vdir = vdir if os.path.isabs(vdir) else os.path.normpath(os.path.join(HERE, vdir))
        if not os.path.isdir(vdir):
            return jresp({"error": f"뷰어 폴더가 없어요: {vdir}"}, 400)
        src = self.store.path(run_id, *f["jpg"].split("/")[3:])
        name = os.path.basename(src)
        names = [name]
        drop = None
        vid = (att.get("event") or {}).get("video") or {}
        ev_dir = self.store.path(run_id, f"a{n:02d}", "event")
        ev_mp4 = next((os.path.join(ev_dir, x) for x in (os.path.basename((vid.get(k) or "").split("?")[0])
                                                           for k in ("audio_url", "silent_url")) if x), None)
        if vid.get("state") == "done" and "front_shift" in vid and ev_mp4 and os.path.isfile(ev_mp4):
            # 사건 영상이 있으면 사진·영상·마스크를 한 세트로 (영상과 같은 만큼 돌려서 사건이 정면에 오게)
            import event_compose
            base = os.path.splitext(name)[0]
            sh = int(vid["front_shift"])
            await asyncio.to_thread(event_compose.roll_image, src, os.path.join(vdir, name), sh, quality=95)
            live = None
            amb = (att.get("event") or {}).get("ambient") or {}
            if amb.get("state") == "done" and amb.get("crop") == vid.get("crop") and amb.get("front_shift") == sh:
                try:                                     # 평소 움직임이 있으면: 평소 반복 → 사건 → 평소 (끊김 없는 한 파일)
                    live = await asyncio.to_thread(self._build_live, run_id, n, att, vid, amb)
                except Exception as ex:
                    return jresp({"error": f"평소 움직임 + 사건 합치기 실패: {ex}"[:800]}, 500)
            if live:
                await asyncio.to_thread(shutil.copyfile, live, os.path.join(vdir, base + "_live.mp4"))
                names += [base + "_live.mp4"]
                drop = base + "_event.mp4"                  # 예전에 넣은 사건만 영상은 뺌 (뷰어가 live 로 대신)
                if os.path.isfile(os.path.join(vdir, drop)):
                    os.remove(os.path.join(vdir, drop))
            else:
                await asyncio.to_thread(shutil.copyfile, ev_mp4, os.path.join(vdir, base + "_event.mp4"))
                names += [base + "_event.mp4"]
                drop = base + "_live.mp4"
                if os.path.isfile(os.path.join(vdir, drop)):
                    os.remove(os.path.join(vdir, drop))
            await asyncio.to_thread(event_compose.roll_image, os.path.join(ev_dir, "mask.png"),
                                    os.path.join(vdir, base + "_event_mask.png"), sh)
            names += [base + "_event_mask.png"]
        else:
            await asyncio.to_thread(shutil.copyfile, src, os.path.join(vdir, name))
        lst = os.path.join(vdir, "photos.json")
        try:
            with open(lst, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            data = {"files": []}
        data["files"] = [x for x in data["files"] if not (drop and x == drop)]
        for nm in names:
            if nm not in data["files"]:
                data["files"].append(nm)
        with open(lst, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)
        return jresp({"ok": True, "path": os.path.join(vdir, name), "files": names, "count": len(data["files"])})

    def _build_live(self, run_id, n, att, vid, amb) -> str:
        """평소 움직임(amb_crop) + 사건(crop) → 크롭 공간에서 이어 붙여(event_compose.build_live) → 4K 360 (live_4k.mp4).
        둘 다 같은 범위·같은 정면 돌림이어야 함. 이미 만든 게 더 새 것이면 그대로 씀."""
        import event_compose
        d = f"a{n:02d}/event"

        def pick(v, pre):
            ss = sorted(v.get("saves") or [], key=lambda s: not s.get("audio"))      # 소리 있는 것 먼저
            for s in ss:
                f = self.store.path(run_id, d, f"{pre}crop_{s['node']}.mp4")
                if os.path.isfile(f):
                    return f
            raise RuntimeError(f"{'평소 움직임' if pre else '사건'} 크롭 영상이 없어요")
        a_mp4, e_mp4 = pick(amb, "amb_"), pick(vid, "")
        out = self.store.path(run_id, d, "live_4k.mp4")
        if os.path.isfile(out) and os.path.getmtime(out) > max(os.path.getmtime(a_mp4), os.path.getmtime(e_mp4)):
            return out
        lc = self.store.path(run_id, d, "live_crop.mp4")
        fps = int(vid.get("fps") or 24)
        cfg = self.cfg.get("live") or {}
        plan = event_compose.build_live(a_mp4, e_mp4, lc, fps=fps, calm_seconds=float(cfg.get("calm_seconds", 8)),
                                        fade=int(cfg.get("fade_frames", 12)))
        sh = int(vid.get("front_shift") or 0)
        mask = self.store.path(run_id, d, "mask_front.png" if sh else "mask.png")
        event_compose.compose_crop_video(lc, self._final_path(run_id, att), mask, vid["crop"], out, fps=fps,
                                         audio=True, shift=sh, end_fade=0)
        return out

    # ------------------------------------------------------------------ 영상화: 움직일 곳 (일부만) + 프롬프트
    MOTION_MODEL = {"base": "LTX-2.3-22B", "lora": "ltx-2.3-22b-lora-cinemagraph-0.9", "trigger": llm.MOTION_TRIGGER,
                    "train_size": [704, 512], "frames": 25,
                    "note": "화면 전체가 아니라 영역만 움직임. 결과 영상은 이 영역 자리에만 다시 합성하고 나머지는 8K 정지 이미지."}

    def _final_path(self, run_id, att):
        f = next((x for x in att.get("files", []) if x.get("final")), None) or (att.get("files") or [None])[-1]
        if not f:
            raise RuntimeError("이 시도에는 360 결과 이미지가 없어요.")
        return self.store.path(run_id, *f["url"].split("/")[3:])

    def _motion_context(self, meta, att):
        a = meta.get("analysis") or {}
        pr = att.get("prompts") or meta.get("prompts") or {}
        parts = [f"scene: {a.get('scene', '')}", f"lighting: {a.get('lighting', '')}",
                 f"panorama positive prompt: {pr.get('positive', '')}"]
        return "\n".join(p for p in parts if p.split(": ", 1)[1])

    def _motion_assets(self, run_id, n, plan):
        """영역별 평면 크롭(영상 모델 입력) + 정지/움직임 미리보기 + ERP 마스크 저장, plan 에 주소 기록."""
        d = f"a{n:02d}/motion"
        os.makedirs(self.store.path(run_id, d), exist_ok=True)
        att = self.store.read(run_id)["attempts"][n - 1]
        img = Image.open(self._final_path(run_id, att)).convert("RGB")
        t = int(time.time())
        for i, r in enumerate(plan["regions"], 1):
            w, h = erp.region_size(r)
            erp.perspective(img, r["yaw"], r["pitch"], r["hfov"], w, h).save(self.store.path(run_id, d, f"region_{i}.png"))
            r.update(crop=f"/runs/{run_id}/{d}/region_{i}.png?t={t}", crop_size=[w, h])
        small = img.resize((2048, 1024), Image.LANCZOS)
        erp.motion_overlay(small, plan["regions"]).save(self.store.path(run_id, d, "overlay.jpg"), quality=88)
        erp.region_mask(plan["regions"]).save(self.store.path(run_id, d, "mask.png"))
        plan.update(overlay=f"/runs/{run_id}/{d}/overlay.jpg?t={t}", mask=f"/runs/{run_id}/{d}/mask.png?t={t}",
                    video_model=self.MOTION_MODEL)
        with open(self.store.path(run_id, d, "plan.json"), "w", encoding="utf-8") as fp:
            json.dump(plan, fp, ensure_ascii=False, indent=1)
        return plan

    def _motion_finish(self, plan):
        plan["regions"] = [llm.clamp_region(r) for r in plan.get("regions", [])][:6]
        for r in plan["regions"]:
            if r.pop("regen", True) or not r.get("positive") or r.get("prompt_ver", 1) < llm.PROMPT_VER:
                r.update(llm.motion_prompts(r, plan.get("lighting", "")), prompt_ver=llm.PROMPT_VER)
        plan["ltx"] = llm.global_motion_prompt(plan["regions"], plan.get("lighting", ""))   # 360 전체용 (LTX 도구에 바로)
        return plan

    @serialized_run
    async def motion_plan(self, req):
        """결과 360 + 생성 프롬프트를 LLM 에 주고, 일부만 움직일 영역과 영상 프롬프트를 받음."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json()
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        model = p.get("model") or self.cfg["openai_model"]
        if model not in self.cfg.get("openai_models", [model]):
            return jresp({"error": f"허용되지 않은 모델: {model}"}, 400)
        try:
            path = self._final_path(run_id, att)
            grid = await asyncio.to_thread(lambda: erp.grid_overlay(Image.open(path)))
            photo = None if meta.get("is_erp") else self.source(run_id)
            plan, usage = await self.openai.plan_motion(grid, photo, self._motion_context(meta, att),
                                                        p.get("note", ""), model)
        except Exception as e:
            return jresp({"error": str(e)}, 502)
        plan = self._motion_finish(plan)
        plan.update(model=model, usage=usage, note=p.get("note", ""), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        plan = await asyncio.to_thread(self._motion_assets, run_id, n, plan)
        meta = self.store.read(run_id)
        meta["attempts"][n - 1]["motion"] = plan
        self.store.write(run_id, meta)
        return jresp(plan)

    @serialized_run
    async def motion_update(self, req):
        """화면에서 고친 영역(위치·움직임·프롬프트) 저장 → 크롭·미리보기 다시."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json()
        meta = self.store.read(run_id)
        plan = dict(meta["attempts"][n - 1].get("motion") or {})
        plan["regions"] = p.get("regions") or []
        if p.get("lighting"):
            plan["lighting"] = p["lighting"]
        plan = self._motion_finish(plan)
        plan["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            plan = await asyncio.to_thread(self._motion_assets, run_id, n, plan)
        except Exception as e:
            return jresp({"error": str(e)}, 400)
        meta = self.store.read(run_id)
        meta["attempts"][n - 1]["motion"] = plan
        self.store.write(run_id, meta)
        return jresp(plan)

    # ================================================================== 사건 연출 (프롬프트 · 움직일 범위)
    EVENT_VIEW = (2560, 1440)          # 사건 화면 (16:9)

    def _event_finish(self, ev, lighting=""):
        ev = llm.clamp_event(ev)
        ev["seconds"] = max(2.0, min(8.0, float(ev.get("seconds", 5) or 5)))
        if not ev.pop("prompt_edited", False) or not ev.get("positive"):
            ev.update(llm.event_prompts(ev, lighting))
        return ev

    def _event_assets(self, run_id, n, ev):
        d = f"a{n:02d}/event"
        os.makedirs(self.store.path(run_id, d), exist_ok=True)
        att = self.store.read(run_id)["attempts"][n - 1]
        img = Image.open(self._final_path(run_id, att)).convert("RGB")
        w, h = self.EVENT_VIEW
        t = int(time.time())
        erp.perspective(img, ev["yaw"], ev["pitch"], ev["hfov"], w, h).save(self.store.path(run_id, d, "event_view.png"))
        small = img.resize((2048, 1024), Image.LANCZOS)
        erp.motion_overlay(small, [{"yaw": ev["yaw"], "pitch": ev["pitch"], "hfov": ev["hfov"], "aspect": w / h}]) \
            .save(self.store.path(run_id, d, "overlay.jpg"), quality=88)
        ev.update(view=f"/runs/{run_id}/{d}/event_view.png?t={t}", view_name=f"{run_id}_event_view.png",
                  overlay=f"/runs/{run_id}/{d}/overlay.jpg?t={t}", view_size=[w, h],
                  ltx360={"positive": self._ltx360_prompt(ev, self._video_is_crop()),
                          "negative": workflow.LTX_CROP_NEG if self._video_is_crop() else workflow.LTX360_NEG})
        ev.pop("workflows", None)
        return ev

    @serialized_run
    async def event_plan(self, req):
        """LLM: 결과 360 을 보고 한 번 일어나는 연출(어디서·무엇이·어떻게)과 영상 프롬프트."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json()
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        model = p.get("model") or self.cfg["openai_model"]
        if model not in self.cfg.get("openai_models", [model]):
            return jresp({"error": f"허용되지 않은 모델: {model}"}, 400)
        try:
            path = self._final_path(run_id, att)
            grid = await asyncio.to_thread(lambda: erp.grid_overlay(Image.open(path)))
            photo = None if meta.get("is_erp") else self.source(run_id)
            ev, usage = await self.openai.plan_event(grid, photo, self._motion_context(meta, att), p.get("note", ""), model)
        except Exception as e:
            return jresp({"error": str(e)}, 502)
        a = meta.get("analysis") or {}
        ev["seconds"] = p.get("seconds", 5)
        ev = self._event_finish(ev, a.get("lighting", ""))
        ev.update(model=model, usage=usage, note=p.get("note", ""), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        ev = await asyncio.to_thread(self._event_assets, run_id, n, ev)
        meta = self.store.read(run_id)
        prev = meta["attempts"][n - 1].get("event") or {}
        if prev.get("video"):
            ev["video"] = prev["video"]                       # 사건을 다시 짜도 만든 영상 기록은 유지
        meta["attempts"][n - 1]["event"] = ev
        self.store.write(run_id, meta)
        return jresp(ev)

    @serialized_run
    async def event_update(self, req):
        """화면에서 고친 사건(위치·내용·프롬프트) 저장 → 사건 화면·워크플로 다시."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json()
        meta = self.store.read(run_id)
        ev = dict(meta["attempts"][n - 1].get("event") or {})
        for k in ("name_ko", "summary_ko", "source", "yaw", "pitch", "hfov", "calm_start", "event", "peak", "calm_end",
                  "keep", "seconds", "positive", "negative", "prompt_edited"):
            if k in p:
                ev[k] = p[k]
        try:
            ev = self._event_finish(ev, (meta.get("analysis") or {}).get("lighting", ""))
        except Exception as e:
            return jresp({"error": str(e)}, 400)
        ev["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        ev = await asyncio.to_thread(self._event_assets, run_id, n, ev)
        meta = self.store.read(run_id)
        meta["attempts"][n - 1]["event"] = ev
        self.store.write(run_id, meta)
        return jresp(ev)

    @staticmethod
    def _ltx360_prompt(ev: dict, crop: bool = False) -> str:
        if ev.get("event"):
            event = ev["event"].strip().rstrip(".") + "." + (f" At the peak, {ev['peak'].strip().rstrip('.')}." if ev.get("peak") else "")
        else:
            event = workflow.LTX360_EVENT
        keep = (ev.get("keep") or "The bridge, rocks, cliffs and buildings").strip().rstrip(".")
        tmpl = workflow.LTX_CROP_POS if crop else workflow.LTX360_POS      # 크롭 생성 노드는 360 이 아닌 일반 화면 문구
        return tmpl.format(event=event, keep=keep[:1].upper() + keep[1:])

    def _video_is_crop(self) -> bool:
        try:
            return workflow.video_has_crop(self._video_template())
        except Exception:
            return False

    async def event_base(self, req):
        """범위 칠하기용: 결과 360 작은 그림(2048) + 저장된 범위."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        att = self.store.read(run_id)["attempts"][n - 1]
        d = f"a{n:02d}/event"
        os.makedirs(self.store.path(run_id, d), exist_ok=True)
        base = self.store.path(run_id, d, "base_2k.jpg")
        if not os.path.isfile(base):
            def mk():
                im = Image.open(self._final_path(run_id, att)).convert("RGB")
                im.resize((2048, 1024), Image.LANCZOS).save(base, quality=88)
            await asyncio.to_thread(mk)
        mp = self.store.path(run_id, d, "mask.png")
        v = int(os.path.getmtime(mp)) if os.path.isfile(mp) else 0
        return jresp({"base": f"/runs/{run_id}/{d}/base_2k.jpg", "mask": f"/runs/{run_id}/{d}/mask.png?v={v}" if v else None})

    async def event_mask(self, req):
        """웹에서 칠한 움직임 범위 저장 (흰색 = 영상, 검정 = 원본). 빈 값이면 지움."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json()
        d = f"a{n:02d}/event"
        os.makedirs(self.store.path(run_id, d), exist_ok=True)
        mp = self.store.path(run_id, d, "mask.png")
        data = p.get("png") or ""
        if not data:
            if os.path.isfile(mp):
                os.remove(mp)
            return jresp({"mask": None, "cover": 0})
        import base64
        try:
            raw = base64.b64decode(data.split(",", 1)[-1])
            im = Image.open(io.BytesIO(raw)).convert("L").resize((2048, 1024), Image.BILINEAR)
        except Exception:
            return jresp({"error": "범위 그림을 읽지 못했어요."}, 400)
        im.save(mp)
        cover = float(np.asarray(im, np.float32).mean() / 255.0)
        return jresp({"mask": f"/runs/{run_id}/{d}/mask.png?v={int(time.time())}", "cover": round(cover, 4)})

    # ------------------------------------------------------------------ 6단계: 마스크 인페인팅 노드로 360 영상
    def _video_template(self):
        path = self.cfg.get("video_template", "video_nodes/quest360_flood_mask_inpaint_4k.json")
        path = path if os.path.isabs(path) else os.path.join(HERE, path)
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)

    @staticmethod
    def _mask_rect(mask_path, canvas, pad):
        """칠한 범위(흰색)를 감싸는 사각형 → 노드 캔버스 좌표 (x, y, w, h). 경계 흐림만큼 넓힘."""
        m = np.asarray(Image.open(mask_path).convert("L").resize(canvas, Image.BILINEAR)) > 127
        ys, xs = np.nonzero(m)
        if not len(xs):
            return None
        W, H = canvas
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    async def event_video_start(self, req):
        """화질 올린 360 + 칠한 범위 + 프롬프트 → 사용자 노드 파일(video_template)에 값만 넣어 실행.
        노드 파일이 사각 마스크(SolidMask+MaskComposite)면 칠한 범위의 좌표를, 마스크 그림(LoadImage)이면 그림을 넣음."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json()
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        bk = p.get("backend") or self._default_backend()
        backend = self.backends.get(bk)
        if backend is None:
            return jresp({"error": f"'{bk}' 실행 위치를 사용할 수 없어요."}, 400)
        d = f"a{n:02d}/event"
        mp = self.store.path(run_id, d, "mask.png")
        if not os.path.isfile(mp):
            return jresp({"error": "먼저 움직일 범위를 칠하고 '범위 저장'을 눌러 주세요."}, 400)
        try:
            src = self._final_path(run_id, att)
            tmpl = self._video_template()
        except Exception as e:
            return jresp({"error": str(e)}, 400)
        crop_mode = workflow.video_has_crop(tmpl)
        kind = "ambient" if p.get("kind") == "ambient" else "video"     # ambient = 평소 움직임 (뷰어에서 사건 앞뒤로 반복)
        if kind == "ambient" and not crop_mode:
            return jresp({"error": "평소 움직임은 크롭 생성 노드(event_cropgen)에서만 만들 수 있어요."}, 400)
        ev0 = att.get("event") or {}
        if kind == "ambient":
            ev0 = {**ev0, "event": workflow.AMBIENT_EVENT, "peak": ""}
        pos = (p.get("positive") or "").strip() or self._ltx360_prompt(ev0, crop_mode)
        neg = (p.get("negative") or "").strip() or (workflow.LTX_CROP_NEG if crop_mode else workflow.LTX360_NEG)
        secs = max(2, min(8, int(p.get("seconds") or 4)))
        if crop_mode and p.get("timeline") is True:      # 초 단위 시간표 문장은 기본으로 붙이지 않음 (timeline:true 일 때만)
            pos = pos.rstrip() + " " + (workflow.ambient_timeline(secs) if kind == "ambient" else workflow.event_timeline(secs))
        mode = workflow.video_mask_mode(tmpl)
        rect = None
        if mode == "rect":
            f = workflow._ui_node(tmpl, workflow.VIDEO_IDS["rect_feather"])
            pad = max((f or {}).get("widgets_values_named", {"left": 0}).values(), default=0) if f else 0
            rect = await asyncio.to_thread(self._mask_rect, mp, workflow.video_rect_canvas(tmpl), int(pad))
            if not rect:
                return jresp({"error": "칠한 범위가 비어 있어요."}, 400)

        # 모든 사건은 정면 기준: 칠한 범위의 중심이 정면(360 가운데)에 오도록 사진·마스크를 먼저 돌려서 생성한다.
        # (영상·뷰어용 사진·마스크 모두 같은 값으로 돌아가 있음 → Unity 에서 정면을 보면 사건이 보임)
        import event_compose
        shift = await asyncio.to_thread(event_compose.front_shift, mp)
        mpf = mp
        if shift:
            mpf = self.store.path(run_id, d, "mask_front.png")
            await asyncio.to_thread(event_compose.roll_image, mp, mpf, shift)
        if mode == "rect" and shift:
            rect = await asyncio.to_thread(self._mask_rect, mpf, workflow.video_rect_canvas(tmpl), int(pad))

        def pano_jpg():
            # 360 전체 생성 노드는 2K~4K 로 쓰므로 4K JPG. 크롭 생성 노드는 칠한 곳을 원본(8K)에서 잘라 쓰므로 원본 크기 그대로.
            im = Image.open(src).convert("RGB")
            if shift:
                im = Image.fromarray(np.roll(np.asarray(im), int(round(shift * im.width / 2048)), axis=1))
            if im.width > 4096 and not crop_mode:
                im = im.resize((4096, 2048), Image.LANCZOS)
            b = io.BytesIO()
            im.save(b, "JPEG", quality=95 if crop_mode else 96)
            return b.getvalue(), im.size
        crop = None
        try:
            data, src_size = await asyncio.to_thread(pano_jpg)
            if crop_mode:
                crop = await asyncio.to_thread(lambda: workflow.crop_plan(Image.open(mpf), src_size))
                if not crop:
                    return jresp({"error": "칠한 범위가 비어 있어요."}, 400)
            tag = "pano8k" if crop_mode and src_size[0] > 4096 else "pano4k"
            name = await backend.upload(data, f"{run_id}_a{n:02d}_{tag}.jpg")
            mname = None
            if mode == "image":
                with open(mpf, "rb") as fp:
                    mname = await backend.upload(fp.read(), f"{run_id}_a{n:02d}_motion_mask.png")
            ui, seed, info = workflow.fill_video_node(tmpl, image=name, positive=pos, negative=neg, seconds=secs,
                                                      lora_strength=p.get("lora_strength"), seed=p.get("seed"),
                                                      prefix=f"VR360/{run_id}_a{n:02d}_{'amb' if kind == 'ambient' else 'flood'}",
                                                      mask_image=mname, rect=rect,
                                                      crop=crop)
            pre = "amb_" if kind == "ambient" else ""
            with open(self.store.path(run_id, d, f"{pre}video_node.json"), "w", encoding="utf-8") as fp:
                json.dump(ui, fp, ensure_ascii=False, indent=1)     # 값이 들어간 노드 파일 (ComfyUI 에서 그대로 열림)
            g, saves = workflow.video_node_api(ui)
            pid = await self._submit_workflow(backend, g, self.store.path(run_id, d, f"{pre}workflow_video.json"))
        except Exception as e:
            return jresp({"error": comfy.explain_comfy_error(str(e))}, 502)
        vid = {"state": "running", "backend": bk, "prompt_id": pid, "saves": saves, "seed": seed, "seconds": secs,
               "fps": info["fps"], "mask_mode": mode, "rect": rect, "crop": crop,
               "front_shift": shift, "template": os.path.basename(self.cfg.get("video_template", "")),
               "kind": kind, "node_url": f"/runs/{run_id}/{d}/{pre}video_node.json?t={int(time.time())}",
               "positive": pos, "negative": neg, "lora_strength": p.get("lora_strength"),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        async with self.locks.setdefault(run_id, asyncio.Lock()):
            meta = self.store.read(run_id)
            meta["attempts"][n - 1].setdefault("event", {})[kind] = vid
            self.store.write(run_id, meta)
        return jresp(vid)

    async def event_video_status(self, req):
        """진행 확인. 끝나면 노드의 영상 출력(SaveVideo)을 받아 작업 폴더에 저장."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        kind = "ambient" if req.query.get("kind") == "ambient" else "video"
        pre = "amb_" if kind == "ambient" else ""
        vid = (self.store.read(run_id)["attempts"][n - 1].get("event") or {}).get(kind)
        if not vid:
            return jresp({"state": "none"})
        if vid["state"] in ("done", "error", "composing"):
            return jresp(vid)
        backend = self.backends.get(vid["backend"])
        if backend is None:
            return jresp({**vid, "state": "error", "error": "실행 위치가 없어졌어요."})
        saves = vid.get("saves") or []
        st = await backend.status(vid["prompt_id"], [s["node"] for s in saves])
        if st["state"] == "error":
            vid.update(state="error", error=comfy.explain_comfy_error(str(st.get("error", "")))[:800])
        elif st["state"] != "done":
            return jresp({**vid, "progress": st})
        else:
            try:
                d = f"a{n:02d}/event"
                t = int(time.time())
                for s in saves:
                    f = next((x for x in st["files"] if x["node"] == s["node"]), None)
                    if not f:
                        continue
                    fn = f"{pre or 'flood_'}{s['node']}{'_audio' if s['audio'] else ''}.mp4"
                    raw = f"{pre}crop_{s['node']}.mp4" if vid.get("crop") else fn    # 크롭 노드: 받은 건 칠한 범위 영상
                    with open(self.store.path(run_id, d, raw), "wb") as fp:
                        fp.write(await backend.fetch(f))
                    key = "audio_url" if s["audio"] and not vid.get("audio_url") else "silent_url" if not vid.get("silent_url") else None
                    if key:
                        vid[key] = f"/runs/{run_id}/{d}/{fn}?t={t}"
                if not vid.get("silent_url") and not vid.get("audio_url"):
                    raise RuntimeError("결과 영상이 없어요")
                if vid.get("crop"):
                    vid.update(state="composing", progress={"state": "running", "node_title": "4K 360 에 붙이는 중"})
                    asyncio.create_task(self._compose_crop_video(run_id, n, saves, kind))
                else:
                    vid.update(state="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            except Exception as ex:
                vid.update(state="error", error=f"결과 받기 실패: {ex}")
        async with self.locks.setdefault(run_id, asyncio.Lock()):
            meta = self.store.read(run_id)
            meta["attempts"][n - 1].setdefault("event", {})[kind] = vid
            self.store.write(run_id, meta)
        return jresp(vid)

    async def _compose_crop_video(self, run_id, n, saves, kind="video"):
        """크롭 노드 결과(칠한 범위 영상) → 8K 사진(4K 로 줄임) 위 칠한 곳에 붙여 4K 360 MP4 (ffmpeg, 한 프레임씩)."""
        import event_compose
        d = f"a{n:02d}/event"
        try:
            meta = self.store.read(run_id)
            att = meta["attempts"][n - 1]
            vid = att["event"][kind]
            pre = "amb_" if kind == "ambient" else ""
            still = self._final_path(run_id, att)
            shift = int(vid.get("front_shift") or 0)          # 생성 때 돌린 값 (사건 = 정면)
            mask = self.store.path(run_id, d, "mask_front.png" if shift else "mask.png")
            info = None
            for s in saves:
                raw = self.store.path(run_id, d, f"{pre}crop_{s['node']}.mp4")
                if not os.path.isfile(raw):
                    continue
                out = self.store.path(run_id, d, f"{pre or 'flood_'}{s['node']}{'_audio' if s['audio'] else ''}.mp4")
                info = await asyncio.to_thread(event_compose.compose_crop_video, raw, still, mask, vid["crop"], out,
                                               fps=int(vid.get("fps") or 24), audio=bool(s["audio"]), shift=shift,
                                               end_fade=0 if kind == "ambient" else 0.75)
            upd = {"state": "done", "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "composed": info}
        except Exception as ex:
            upd = {"state": "error", "error": f"4K 360 합성 실패: {ex}"[:800]}
        async with self.locks.setdefault(run_id, asyncio.Lock()):
            meta = self.store.read(run_id)
            v = meta["attempts"][n - 1]["event"][kind]
            v.pop("progress", None)
            v.update(upd)
            self.store.write(run_id, meta)

    async def repaste_photo(self, req):
        """이미 만든 결과: 원본 사진을 원래 해상도로 AI 업스케일(ComfyUI)해서 같은 자리에 다시 붙임 → JPG 도 다시."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json() if req.can_read_body else {}
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        if att.get("mode", "generate") != "generate" or not (att.get("placement") or {}).get("hfov"):
            return jresp({"error": "사진을 놓아서 만든 결과에만 쓸 수 있어요."}, 400)
        f = next((x for x in att.get("files", []) if x.get("final")), None)
        if not f:
            return jresp({"error": "결과 이미지가 없어요."}, 400)
        bk = p.get("backend") or att.get("backend")
        backend = self.backends.get(bk)
        if backend is None:
            return jresp({"error": f"'{bk}' 백엔드를 사용할 수 없어요."}, 400)
        up = self.store.path(run_id, f"a{n:02d}", "photo_up.png")
        try:
            if not os.path.isfile(up):
                name = await backend.upload(await asyncio.to_thread(self._source_png, run_id), f"{run_id}_a{n:02d}_photo.png")
                g, out = workflow.build_photo_upscale(name, self.cfg["model_fallbacks"]["UpscaleModelLoader"][0],
                                                      f"VR360/{run_id}_a{n:02d}")
                pid = await self._submit_workflow(backend, g, self.store.path(run_id, f"a{n:02d}", "workflow_photo_up.json"))
                t0 = time.time()
                while True:
                    st = await backend.status(pid, [out])
                    if st["state"] == "done":
                        break
                    if st["state"] == "error" or time.time() - t0 > 600:
                        raise RuntimeError(st.get("error") or "업스케일이 너무 오래 걸려요")
                    await asyncio.sleep(2)
                pf = next(x for x in st["files"] if x["node"] == out)
                with open(up, "wb") as fp:
                    fp.write(await backend.fetch(pf))
            local = self.store.path(run_id, *f["url"].split("/")[3:])
            mode, scale = await asyncio.to_thread(self._paste_photo_file, run_id, n, att, local)
            if f.get("jpg"):
                await asyncio.to_thread(erp.save_quest_jpg, local, self.store.path(run_id, *f["jpg"].split("/")[3:]),
                                        self.cfg.get("jpg_quality", 95))
        except Exception as e:
            return jresp({"error": comfy.explain_comfy_error(str(e))}, 502)
        async with self.locks.setdefault(run_id, asyncio.Lock()):
            meta = self.store.read(run_id)
            a = meta["attempts"][n - 1]
            a["photo_pasted"] = mode
            t = int(time.time())
            for x in a.get("files", []):
                x["v"] = t                                     # 화면 캐시 무효화용 (경로는 그대로)
            self.store.write(run_id, meta)
        return jresp({"ok": True, "mode": mode, "files": a["files"]})

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

    async def index(self, req):
        return web.FileResponse(os.path.join(HERE, "static", "index.html"))


@web.middleware
async def path_errors(request, handler):
    try:
        return await handler(request)
    except InvalidRunPath as exc:
        return jresp({"error": str(exc)}, 400)


def create_app(cfg, keys) -> web.Application:
    st = Studio(cfg, keys)
    app = web.Application(client_max_size=200 * 1024 * 1024, middlewares=[path_errors])
    app.on_startup.append(st.startup)
    app.on_cleanup.append(st.cleanup)
    app.add_routes([
        web.get("/", st.index),
        web.get("/api/config", st.config),
        web.get("/api/models/check", st.models_check),
        web.post("/api/upload", st.upload),
        web.post("/api/analyze", st.analyze),
        web.post("/api/control", st.control),
        web.post("/api/generate", st.generate),
        web.post("/api/run/{run_id}/import", st.import_result),
        web.get("/api/run/{run_id}/attempt/{n}", st.status),
        web.get("/api/run/{run_id}", st.get_run),
        web.post("/api/run/{run_id}/attempt/{n}/to_viewer", st.to_viewer),
        web.post("/api/run/{run_id}/attempt/{n}/motion", st.motion_plan),
        web.post("/api/run/{run_id}/attempt/{n}/event", st.event_plan),
        web.post("/api/run/{run_id}/attempt/{n}/event/update", st.event_update),
        web.post("/api/run/{run_id}/attempt/{n}/repaste", st.repaste_photo),
        web.get("/api/run/{run_id}/attempt/{n}/event/base", st.event_base),
        web.post("/api/run/{run_id}/attempt/{n}/event/mask", st.event_mask),
        web.post("/api/run/{run_id}/attempt/{n}/event/video", st.event_video_start),
        web.get("/api/run/{run_id}/attempt/{n}/event/video", st.event_video_status),
        web.post("/api/run/{run_id}/attempt/{n}/motion/update", st.motion_update),
        web.get("/api/runs", st.list_runs),
        web.static("/runs", cfg["runs_dir"]),
        web.static("/static", os.path.join(HERE, "static")),
    ])
    return app
