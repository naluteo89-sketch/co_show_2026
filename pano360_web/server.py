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
import cinemagraph
import event as eventmod
import workflow_ui
import influence
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
                                         self.cfg.get("ultrasharp_blend"))

    async def cleanup(self, app):
        await asyncio.gather(*(b.close() for b in self.backends.values()))
        if self.s is not None:
            await self.s.close()

    def source(self, run_id):
        return erp.load_upright(self.store.path(run_id, "source" + self.store.read(run_id)["source_ext"]))

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
            "motion": {k: self.cfg.get("motion", {}).get(k) for k in ("backend", "duration", "fps", "show_render")},
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
    async def generate(self, req):
        p = await req.json()
        run_id, bk = p["run_id"], p.get("backend") or next(iter(self.backends), "")
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
                        def _paste(dst=self.store.path(run_id, local)):
                            erp.paste_photo(Image.open(dst), self.source(run_id), float(pl["yaw"]), float(pl["pitch"]),
                                            float(pl["hfov"]), float(pl["vfov"]) if pl.get("vfov") else None,
                                            kp.get("paste_feather", 0.05)).save(dst)
                        await asyncio.to_thread(_paste)
                        att["photo_pasted"] = True
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

    # ------------------------------------------------------------------ 영상화: 영역 영상 생성 → 360 에 합성
    def _mcfg(self):
        m = {"duration": 1, "fps": 24, "video_width": 4096, "crf": 18, "loop_fade": 6, "sam": True,
             "sam_ckpt": "sam3.1_multiplex_fp16.safetensors", "sam_threshold": 0.4, "mask_feather": 14,
             "ckpt": "ltx-2.3-22b-dev-fp8.safetensors", "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
             "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
             "cinemagraph_lora": "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors",
             "upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "img_compression": 18,
             "influence": True, "global_width": 1536, "global_lora_strength": 1.0, "influence_max_shift_deg": 4.0}
        m.update(self.cfg.get("motion", {}))
        return m

    @serialized_run
    async def motion_render(self, req):
        """영역마다 (SAM 마스크 + LTX 영상) 작업 제출."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        p = await req.json()
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        plan = att.get("motion") or {}
        if not plan.get("regions"):
            return jresp({"error": "먼저 '움직일 곳 찾기'로 영역을 만들어 주세요."}, 400)
        bk = p.get("backend") or self.cfg.get("motion", {}).get("backend") or next(iter(self.backends), "")
        backend = self.backends.get(bk)
        if backend is None:
            return jresp({"error": f"'{bk}' 백엔드를 사용할 수 없어요."}, 400)
        mc = self._mcfg()
        if any(r.get("prompt_ver", 1) < llm.PROMPT_VER for r in plan["regions"]):   # 예전 계획 → 새 기본 프롬프트로
            plan = self._motion_finish(plan)
            meta["attempts"][n - 1]["motion"] = plan
            self.store.write(run_id, meta)
        dur = float(p.get("duration") or mc["duration"])
        fps = int(p.get("fps") or mc["fps"])
        d = f"a{n:02d}/motion"
        jobs = []
        try:
            for i, r in enumerate(plan["regions"], 1):
                with open(self.store.path(run_id, d, f"region_{i}.png"), "rb") as fp:
                    name = await backend.upload(fp.read(), f"{run_id}_a{n:02d}_region_{i}.png")
                job = {"i": i, "image": name}
                if mc.get("sam"):
                    try:
                        g = workflow.build_sam_mask(name, r["element"], ckpt=mc["sam_ckpt"], threshold=mc["sam_threshold"],
                                                    prefix=f"VR360/{run_id}_a{n:02d}_r{i}")
                        job["mask_pid"] = await self._submit_workflow(backend, g, self.store.path(run_id, d, f"workflow_mask_{i}.json"))
                    except Exception as e:                  # SAM 모델이 없으면 둥근 마스크로
                        job["mask_error"] = str(e)[:300]
                if mc.get("range_mode", True) and job.get("mask_pid"):
                    pass                                    # 범위 고정 생성: SAM 이 끝나면 범위 마스크로 영상 제출 (상태 확인 때)
                else:
                    await self._submit_region_video(backend, run_id, n, r, job, mc, dur, fps, p.get("seed"), None)
                jobs.append(job)
            glob = None
            if mc.get("influence") and llm.needs_global(plan["regions"]):
                glob = await self._submit_global(backend, run_id, n, plan, mc, dur, fps, p.get("seed"))
        except Exception as e:
            print(f"[pano360] 영상 작업 제출 실패: {e}")
            return jresp({"error": comfy.explain_comfy_error(str(e))}, 502)
        render = {"state": "running", "backend": bk, "duration": dur, "fps": fps, "jobs": jobs, "seed_req": p.get("seed"),
                  "range_mode": bool(mc.get("range_mode", True)), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        if glob:
            render["global"] = glob
        meta = self.store.read(run_id)
        meta["attempts"][n - 1].setdefault("motion", {})["render"] = render
        self.store.write(run_id, meta)
        return jresp(render)

    async def _submit_region_video(self, backend, run_id, n, r, job, mc, dur, fps, seed, sam_png):
        """영역 영상 제출. 범위 고정 생성이면 SAM 요소를 퍼짐만큼 넓힌 범위 마스크를 같이 보냄."""
        d = f"a{n:02d}/motion"
        i = job["i"]
        w, h = r.get("crop_size") or erp.region_size(r)
        rng_name = None
        if mc.get("range_mode", True):
            def mk():
                m, src = cinemagraph.range_mask((w, h), sam_png, r.get("spread", "near"))
                png = cinemagraph.mask_png(m)
                with open(self.store.path(run_id, d, f"range_{i}.png"), "wb") as fp:
                    fp.write(png)
                return png, src
            png, src = await asyncio.to_thread(mk)
            rng_name = await backend.upload(png, f"{run_id}_a{n:02d}_range_{i}.png")
            job["range_src"] = src
        g, s = workflow.build_cinemagraph(job["image"], positive=r["positive"], negative=r["negative"], width=w, height=h,
                                          duration=dur, fps=fps, seed=seed, m=mc, lora_strength=r.get("strength", 1.2),
                                          prefix=f"VR360/{run_id}_a{n:02d}_r{i}", range_mask=rng_name)
        job.update(seed=s, video_pid=await self._submit_workflow(backend, g, self.store.path(run_id, d, f"workflow_video_{i}.json")))

    async def _submit_global(self, backend, run_id, n, plan, mc, dur, fps, seed):
        """영향 퍼짐용: 360 전체 저해상도 영상 + 영향 대상 SAM 마스크 (원인 영역이 가운데 오게 돌려서)."""
        d = f"a{n:02d}/motion"
        gw = int(mc["global_width"]) // 64 * 64
        gh = gw // 2
        src = max(plan["regions"], key=lambda r: llm.SPREAD_DEG.get(r.get("spread"), 0) +
                  sum(x.get("strength", 0) for x in r.get("influence") or []))
        roll = influence.roll_for(src["yaw"], gw)
        att = self.store.read(run_id)["attempts"][n - 1]

        def prep():
            im = Image.open(self._final_path(run_id, att)).convert("RGB").resize((gw, gh), Image.LANCZOS)
            im = Image.fromarray(np.roll(np.asarray(im), roll, axis=1))
            out = self.store.path(run_id, d, "global_in.png")
            im.save(out)
            with open(out, "rb") as fp:
                return fp.read()
        name = await backend.upload(await asyncio.to_thread(prep), f"{run_id}_a{n:02d}_global.png")
        rng_name = None
        if mc.get("range_mode", True):                  # 원인 주변 퍼짐 범위만 움직이고 나머지 360 은 고정
            def mk():
                src_list = [{"yaw": r["yaw"], "pitch": r["pitch"], "range": r.get("spread", "near"), "strength": 1.0}
                            for r in plan["regions"] if r.get("spread", "near") in ("near", "wide")]
                for r in plan["regions"]:
                    for x in r.get("influence") or []:
                        if x.get("range") in ("near", "wide"):
                            src_list.append({"yaw": r["yaw"], "pitch": r["pitch"], "range": x["range"], "strength": 1.0})
                m = np.zeros((gh, gw), np.float32)
                for sdef in src_list:
                    R = influence.RANGE_DEG[sdef["range"]] * 1.15
                    f = np.clip(1 - influence.angle_map(gw, gh, sdef["yaw"], sdef["pitch"]) / R, 0, 1)
                    m = np.maximum(m, np.clip(f * 3, 0, 1))
                m = np.roll(m, roll, axis=1)
                png = cinemagraph.mask_png(m)
                with open(self.store.path(run_id, d, "global_range.png"), "wb") as fp:
                    fp.write(png)
                return png
            rng_name = await backend.upload(await asyncio.to_thread(mk), f"{run_id}_a{n:02d}_global_range.png")
        pr = llm.global_motion_prompt(plan["regions"], plan.get("lighting", ""))
        g, gseed = workflow.build_cinemagraph(name, positive=pr["positive"], negative=pr["negative"], width=gw, height=gh,
                                              duration=dur, fps=fps, seed=seed, m=mc,
                                              lora_strength=mc.get("global_lora_strength", 1.0),
                                              prefix=f"VR360/{run_id}_a{n:02d}_g", range_mask=rng_name)
        glob = {"image": name, "roll": roll, "size": [gw, gh], "seed": gseed, "positive": pr["positive"],
                "video_pid": await self._submit_workflow(backend, g, self.store.path(run_id, d, "workflow_global.json")),
                "targets": []}
        for i, r in enumerate(plan["regions"]):
            for j, x in enumerate(r.get("influence") or []):
                if x.get("range") not in ("near", "wide") or x.get("strength", 0) <= 0:
                    continue
                t = {"ri": i, "xi": j, "target": x["target"]}
                try:
                    gm = workflow.build_sam_mask(name, x["target"], ckpt=mc["sam_ckpt"], threshold=mc["sam_threshold"],
                                                 prefix=f"VR360/{run_id}_a{n:02d}_g{i + 1}_{j + 1}")
                    t["pid"] = await self._submit_workflow(backend, gm, self.store.path(run_id, d, f"workflow_inf_{i + 1}_{j + 1}.json"))
                except Exception as e:
                    t["error"] = str(e)[:300]
                glob["targets"].append(t)
        return glob

    @serialized_run
    async def motion_render_status(self, req):
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        meta = self.store.read(run_id)
        plan = meta["attempts"][n - 1].get("motion") or {}
        rd = plan.get("render")
        if not rd:
            return jresp({"state": "none"})
        if rd["state"] in ("done", "error", "clips"):
            return jresp(rd)
        backend = self.backends.get(rd["backend"])
        mc = self._mcfg()
        mc["range_mode"] = rd.get("range_mode", mc.get("range_mode", True))
        pending = None
        for j in rd["jobs"]:                                  # SAM 끝난 영역 → 범위 마스크 만들어 영상 제출
            if j.get("video_pid"):
                continue
            ms = await backend.status(j["mask_pid"], ["6"]) if j.get("mask_pid") else {"state": "done", "files": []}
            if ms["state"] not in ("done", "error"):
                pending = pending or j["i"]
                continue
            mf = next((f for f in ms.get("files", []) if f["node"] == "6"), None)
            sam = await backend.fetch(mf) if mf else None
            if ms["state"] == "error":
                j["mask_error"] = str(ms.get("error", ""))[:300]
            try:
                await self._submit_region_video(backend, run_id, n, plan["regions"][j["i"] - 1], j, mc,
                                                rd["duration"], rd["fps"], rd.get("seed_req"), sam)
            except Exception as e:
                rd.update(state="error", error=f"영역 {j['i']} 영상 제출 실패: {comfy.explain_comfy_error(str(e))}")
                self.store.write(run_id, meta)
                return jresp(rd)
            self.store.write(run_id, meta)
        if pending:
            return jresp({**rd, "progress": f"영역 {pending}/{len(rd['jobs'])} 움직임 범위(SAM) 찾는 중"})
        k = len(rd["jobs"])
        done, prog, results = 0, None, []
        for j in rd["jobs"]:
            vs = await backend.status(j["video_pid"], ["33"])
            ms = await backend.status(j["mask_pid"], ["6"]) if j.get("mask_pid") else {"state": "done", "files": []}
            if vs["state"] == "error":
                rd.update(state="error", error=f"영역 {j['i']} 영상 생성 실패: {vs.get('error')}")
                self.store.write(run_id, meta)
                return jresp(rd)
            if vs["state"] == "done" and ms["state"] in ("done", "error"):
                done += 1
            elif prog is None:
                prog = (j["i"], vs)
            results.append((j, vs, ms))
        glob = rd.get("global")
        if glob and not glob.get("error"):
            gs = await backend.status(glob["video_pid"], ["33"])
            if gs["state"] == "error":
                glob["error"] = f"전체 영상 실패: {gs.get('error')}"     # 영향 퍼짐 없이 계속
            else:
                tdone = True
                for t in glob["targets"]:
                    if t.get("pid"):
                        ts = await backend.status(t["pid"], ["6"])
                        tdone = tdone and ts["state"] in ("done", "error")
                k += 1
                if gs["state"] == "done" and tdone:
                    done += 1
                elif prog is None:
                    prog = ("전체", gs)
        if done < k:
            i, vs = prog
            step = {"19": "1단계 생성", "23": "2배 업스케일", "30": "2단계 생성", "32": "디코드"}.get(str(vs.get("node")), "준비")
            who = "360 전체(영향 퍼짐)" if i == "전체" else f"영역 {i}/{len(rd['jobs'])}"
            return jresp({**rd, "progress": f"{who} 영상 만드는 중 · {step}" if vs.get("state") == "running"
                          else f"{who} 대기 중",
                          "value": vs.get("value"), "max": vs.get("max")})
        rd.update(state="clips", clips_finished=time.strftime("%Y-%m-%d %H:%M:%S"))   # 합성은 '360 합성' 버튼으로 따로
        self.store.write(run_id, meta)
        return jresp(rd)

    @serialized_run
    async def motion_compose(self, req):
        """이미 만든 영역 영상 + 마스크를 받아 360 에 합성 (영상은 다시 만들지 않음, 몇 번이든 가능)."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        meta = self.store.read(run_id)
        plan = meta["attempts"][n - 1].get("motion") or {}
        rd = plan.get("render")
        if not rd or not (rd["state"] in ("clips", "done") or (rd["state"] == "error" and rd.get("compose_failed"))):
            return jresp({"error": "먼저 '① 영역 영상 만들기'를 끝내 주세요."}, 400)
        backend = self.backends.get(rd["backend"])
        if backend is None:
            return jresp({"error": f"'{rd['backend']}' 백엔드를 사용할 수 없어요."}, 400)
        results = []
        try:
            for j in rd["jobs"]:
                vs = await backend.status(j["video_pid"], ["33"])
                ms = await backend.status(j["mask_pid"], ["6"]) if j.get("mask_pid") else {"state": "done", "files": []}
                if vs["state"] != "done":
                    raise RuntimeError(f"영역 {j['i']} 영상을 서버에서 찾지 못했어요 ({vs['state']}). 영역 영상을 다시 만들어 주세요.")
                results.append((j, vs, ms))
            info = await self._compose_motion(run_id, n, plan, rd, results, backend)
            for k in ("error", "compose_failed", "compose_error"):
                rd.pop(k, None)
            rd.update(state="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"), **info)
        except Exception as e:
            print(f"[pano360] 영상 합성 실패: {e}")
            for k in ("error", "compose_failed"):
                rd.pop(k, None)
            rd.update(state="clips", compose_error=str(e)[:500])
            self.store.write(run_id, meta)
            return jresp({"error": f"합성 실패: {e}"}, 500)
        self.store.write(run_id, meta)
        return jresp(rd)

    async def _compose_motion(self, run_id, n, plan, rd, results, backend):
        d = f"a{n:02d}/motion"
        mc = self._mcfg()
        regs, notes = [], []
        for (j, vs, ms), r in zip(results, plan["regions"]):
            files = sorted((f for f in vs["files"] if f["node"] == "33"), key=lambda f: f["filename"])
            datas = [await backend.fetch(f) for f in files]
            mfile = next((f for f in ms.get("files", []) if f["node"] == "6"), None)
            sam = await backend.fetch(mfile) if mfile else None
            if sam:
                with open(self.store.path(run_id, d, f"mask_{j['i']}.png"), "wb") as fp:
                    fp.write(sam)

            def prep(r=r, j=j, datas=datas, sam=sam):
                crop = np.asarray(Image.open(self.store.path(run_id, d, f"region_{j['i']}.png")).convert("RGB"), np.float32)
                size = (crop.shape[1], crop.shape[0])
                fr = cinemagraph.load_frames(datas, size)
                stab = {}
                rpath = self.store.path(run_id, d, f"range_{j['i']}.png")
                rng = (np.asarray(Image.open(rpath).convert("L").resize(size, Image.BILINEAR), np.float32) / 255.0
                       if j.get("range_src") and os.path.isfile(rpath) else None)
                if mc.get("stabilize", True):                    # 영상 모델의 카메라 흔들림 제거 (원본 기준 정렬)
                    if rng is not None:
                        excl = rng
                    else:
                        excl, _ = cinemagraph.element_mask(size, sam, feather_px=0, grow_px=12) if sam else (None, "")
                    fr, stab = cinemagraph.stabilize(fr, crop, excl)
                if rng is not None:                                 # 범위 고정 생성: 범위 그대로 (밖은 이미 원본)
                    mask, src, mst = rng, "range:" + j["range_src"], {}
                elif r.get("spread", "near") != "none" and mc.get("mask_mode", "motion") == "motion":
                    # 요소 + 영상에서 실제로 움직인 곳 (요소 때문에 생긴 주변 물결·물보라 포함)
                    mask, src, mst = cinemagraph.moving_mask(size, sam, fr, feather_px=mc["mask_feather"])
                else:                                               # 요소 모양 안만
                    (mask, src), mst = cinemagraph.element_mask(size, sam, feather_px=mc["mask_feather"]), {}
                leak = cinemagraph.static_leak(fr, mask)
                fr = cinemagraph.make_loop(cinemagraph.match_first_frame(fr, crop, mask), mc["loop_fade"])
                Image.fromarray((mask * 255).astype(np.uint8)).save(self.store.path(run_id, d, f"mask_used_{j['i']}.png"))
                return {"region": r, "size": size, "frames": fr, "mask": mask}, src, leak, {**mst, **stab}
            item, src, leak, mst = await asyncio.to_thread(prep)
            regs.append(item)
            notes.append({"i": j["i"], "frames": len(datas), "mask": src, "leak": leak, **mst,
                          "mask_error": j.get("mask_error", "")})
        att = self.store.read(run_id)["attempts"][n - 1]
        warp, ginfo = await self._global_warp(run_id, n, plan, rd, backend, mc, att)
        out = self.store.path(run_id, d, "motion_360.mp4")
        info = await asyncio.to_thread(cinemagraph.compose_video, self._final_path(run_id, att), regs, out,
                                       fps=rd["fps"], width=mc["video_width"], crf=mc["crf"],
                                       preview_png=self.store.path(run_id, d, "motion_first.jpg"), warp=warp)
        info["influence"] = ginfo
        t = int(time.time())
        return {**info, "regions": notes, "video": f"/runs/{run_id}/{d}/motion_360.mp4?t={t}",
                "preview": f"/runs/{run_id}/{d}/motion_first.jpg?t={t}"}

    async def _global_warp(self, run_id, n, plan, rd, backend, mc, att):
        """전체 영상의 흐름 × 영향 가중치 → FlowWarp (없거나 실패하면 None, 사유)."""
        glob = rd.get("global")
        if not glob:
            return None, {"used": False, "note": "영향 퍼짐 없음"}
        if glob.get("error"):
            return None, {"used": False, "note": glob["error"]}
        d = f"a{n:02d}/motion"
        gw, gh = glob["size"]
        gs = await backend.status(glob["video_pid"], ["33"])
        files = sorted((f for f in gs.get("files", []) if f["node"] == "33"), key=lambda f: f["filename"])
        if gs["state"] != "done" or not files:
            return None, {"used": False, "note": "전체 영상을 찾지 못했어요"}
        datas = [await backend.fetch(f) for f in files]
        masks, notes = {}, []
        for t in glob["targets"]:
            png = None
            if t.get("pid"):
                ts = await backend.status(t["pid"], ["6"])
                mf = next((f for f in ts.get("files", []) if f["node"] == "6"), None)
                png = await backend.fetch(mf) if mf else None
            masks[(t["ri"], t["xi"])] = png
            notes.append({"target": t["target"], "mask": bool(png), "error": t.get("error", "")})

        def prep():
            fr = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((gw, gh), Image.LANCZOS))
                           for b in datas])
            sources = []
            for i, r in enumerate(plan["regions"]):
                tg = []
                for j, x in enumerate(r.get("influence") or []):
                    m = influence.load_mask(masks.get((i, j)), gw, gh)
                    if m is not None:
                        m = np.roll(m, -glob["roll"], axis=1)          # 돌렸던 것을 되돌림
                        Image.fromarray((m * 255).astype(np.uint8)).save(self.store.path(run_id, d, f"inf_mask_{i + 1}_{j + 1}.png"))
                        tg.append({"mask": m, "range": x["range"], "strength": x["strength"]})
                if tg:
                    sources.append({"yaw": r["yaw"], "pitch": r["pitch"], "targets": tg})
            # 대상을 안 적어도: 전체 영상에서 실제로 움직인 곳 × 원인에서의 거리 감쇠 (영역의 '퍼짐' 범위)
            flows = influence.backward_flows(fr)
            if glob["roll"]:
                flows = np.roll(flows, -glob["roll"], axis=2)
            auto = []
            for r in plan["regions"]:
                if r.get("spread", "none") in ("near", "wide"):
                    auto.append({"yaw": r["yaw"], "pitch": r["pitch"], "range": r["spread"],
                                 "strength": r.get("spread_strength", 0.7)})
            wgt = influence.influence_weight(gw, gh, sources) if sources else np.zeros((gh, gw), np.float32)
            if auto:
                wgt = np.maximum(wgt, influence.auto_weight(flows, auto, w_px=gw))
            if not wgt.any():
                return None, 0.0
            Image.fromarray((wgt * 255).astype(np.uint8)).save(self.store.path(run_id, d, "influence_weight.png"))
            hi = np.asarray(Image.open(self._final_path(run_id, att)).convert("RGB"))
            W = int(mc["video_width"])
            fw = influence.FlowWarp(fr, wgt, 0, hi, W, W // 2, fade=mc["loop_fade"],
                                    max_shift_deg=mc["influence_max_shift_deg"], flows=flows)
            return fw, float(fw.active.mean())
        fw, ratio = await asyncio.to_thread(prep)
        if fw is None:
            return None, {"used": False, "note": "전체 영상에서 원인 주변에 움직인 곳이 없어요", "targets": notes}
        t = int(time.time())
        return fw, {"used": True, "ratio": round(ratio, 4), "max_shift_px": round(fw.max_shift, 1), "targets": notes,
                    "weight": f"/runs/{run_id}/{d}/influence_weight.png?t={t}"}

    # ================================================================== 사건 영상 (한 번 일어나는 연출)
    EVENT_VIEW = (2560, 1440)          # 사건 화면 (16:9). 영상 모델은 1280×720 으로 줄여 씀

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
        wfs = {}
        for v in ("wan14", "wan5"):
            g = workflow_ui.wan_event_workflow(ev["positive"], ev["negative"], image=f"{run_id}_event_view.png", variant=v,
                                               seconds=ev["seconds"])
            with open(self.store.path(run_id, d, f"comfy_event_{v}.json"), "w", encoding="utf-8") as fp:
                json.dump(g, fp, ensure_ascii=False, indent=1)
            wfs[v] = f"/runs/{run_id}/{d}/comfy_event_{v}.json?t={t}"
        ev.update(view=f"/runs/{run_id}/{d}/event_view.png?t={t}", view_name=f"{run_id}_event_view.png",
                  overlay=f"/runs/{run_id}/{d}/overlay.jpg?t={t}", workflows=wfs, view_size=[w, h])
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

    async def event_compose(self, req):
        """ComfyUI 에서 만든 사건 영상(MP4)을 올리면 → 달라진 곳만 360 원본 위에 섞어 MP4."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        self.store.path(run_id)
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        ev = att.get("event")
        if not ev:
            return jresp({"error": "먼저 '사건 만들기'로 사건 화면을 만들어 주세요."}, 400)
        form = await req.post()
        f = form.get("video")
        if f is None or not hasattr(f, "file"):
            return jresp({"error": "영상 파일이 없어요."}, 400)
        d = f"a{n:02d}/event"
        gen = self.store.path(run_id, d, "event_gen.mp4")
        with open(gen, "wb") as fp:
            shutil.copyfileobj(f.file, fp)
        mc = self._mcfg()

        def work():
            frames, fps = eventmod.read_video(gen)
            hi = np.asarray(Image.open(self._final_path(run_id, att)).convert("RGB"))
            crop = np.asarray(Image.open(self.store.path(run_id, d, "event_view.png")).convert("RGB"))
            return eventmod.compose_event(hi, ev, frames, crop, self.store.path(run_id, d, "event_360.mp4"), fps=fps,
                                          width=mc["video_width"], crf=mc["crf"],
                                          preview=self.store.path(run_id, d, "event_peak.jpg"),
                                          alpha_png=self.store.path(run_id, d, "event_alpha.png"))
        try:
            info = await asyncio.to_thread(work)
        except Exception as e:
            print(f"[pano360] 사건 합성 실패: {e}")
            return jresp({"error": f"합성 실패: {e}"}, 500)
        t = int(time.time())
        info.update(video=f"/runs/{run_id}/{d}/event_360.mp4?t={t}", preview=f"/runs/{run_id}/{d}/event_peak.jpg?t={t}",
                    alpha=f"/runs/{run_id}/{d}/event_alpha.png?t={t}", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        async with self.locks.setdefault(run_id, asyncio.Lock()):
            meta = self.store.read(run_id)
            meta["attempts"][n - 1].setdefault("event", {})["result"] = info
            self.store.write(run_id, meta)
        return jresp(info)

    async def _to_viewer(self, src, name):
        if not os.path.isfile(src):
            return jresp({"error": "아직 만든 영상이 없어요."}, 400)
        vdir = self.cfg.get("viewer_dir", "")
        vdir = vdir if os.path.isabs(vdir) else os.path.normpath(os.path.join(HERE, vdir))
        if not os.path.isdir(vdir):
            return jresp({"error": f"뷰어 폴더가 없어요: {vdir}"}, 400)
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

    async def event_to_viewer(self, req):
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        return await self._to_viewer(self.store.path(run_id, f"a{n:02d}", "event", "event_360.mp4"),
                                     f"{run_id}_a{n:02d}_event.mp4")

    async def motion_to_viewer(self, req):
        """360 영상을 Unity 뷰어 StreamingAssets 에 복사 (빌드 때 목록에 자동 포함)."""
        run_id, n = req.match_info["run_id"], int(req.match_info["n"])
        src = self.store.path(run_id, f"a{n:02d}", "motion", "motion_360.mp4")
        if not os.path.isfile(src):
            return jresp({"error": "아직 만든 영상이 없어요."}, 400)
        vdir = self.cfg.get("viewer_dir", "")
        vdir = vdir if os.path.isabs(vdir) else os.path.normpath(os.path.join(HERE, vdir))
        if not os.path.isdir(vdir):
            return jresp({"error": f"뷰어 폴더가 없어요: {vdir}"}, 400)
        name = f"{run_id}_a{n:02d}_motion.mp4"
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
        web.post("/api/upload", st.upload),
        web.post("/api/analyze", st.analyze),
        web.post("/api/control", st.control),
        web.post("/api/generate", st.generate),
        web.get("/api/run/{run_id}/attempt/{n}", st.status),
        web.get("/api/run/{run_id}", st.get_run),
        web.post("/api/run/{run_id}/attempt/{n}/to_viewer", st.to_viewer),
        web.post("/api/run/{run_id}/attempt/{n}/motion", st.motion_plan),
        web.post("/api/run/{run_id}/attempt/{n}/event", st.event_plan),
        web.post("/api/run/{run_id}/attempt/{n}/event/update", st.event_update),
        web.post("/api/run/{run_id}/attempt/{n}/event/compose", st.event_compose),
        web.post("/api/run/{run_id}/attempt/{n}/event/to_viewer", st.event_to_viewer),
        web.post("/api/run/{run_id}/attempt/{n}/motion/update", st.motion_update),
        web.post("/api/run/{run_id}/attempt/{n}/motion/render", st.motion_render),
        web.get("/api/run/{run_id}/attempt/{n}/motion/render", st.motion_render_status),
        web.post("/api/run/{run_id}/attempt/{n}/motion/compose", st.motion_compose),
        web.post("/api/run/{run_id}/attempt/{n}/motion/to_viewer", st.motion_to_viewer),
        web.get("/api/runs", st.list_runs),
        web.static("/runs", cfg["runs_dir"]),
        web.static("/static", os.path.join(HERE, "static")),
    ])
    return app
