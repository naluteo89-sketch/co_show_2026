"""
1~4단계: 사진 업로드 → LLM 분석 → 컨트롤 이미지 → 360 생성 제출 → 진행 확인·결과 후처리
"""

import asyncio
import io
import os
import time
import uuid

from PIL import Image

from .. import erp, llm, workflow
from ..clients import comfy
from .base import adir, jresp, now, placement_args, route_ids, serialized_run


def version_b_placement(pl: dict, viewer_scale: float, max_hfov: float = 140.0, max_extra_tilt: float | None = None) -> dict:
    """버전 B: 보는 사람을 작게(viewer_scale < 1) = 사진 세계를 크게 → 사진 화각을 1/viewer_scale 배로 키워 배치.
    수평선 위치는 그대로(눈높이 = 수평선) 되도록 pitch 를 같은 비율로 맞춤. 생성(LoRA)이 채울 곳이 줄어듦."""
    import math
    k = 1.0 / max(0.3, min(1.0, float(viewer_scale)))
    h0 = float(pl["hfov"])
    h1 = min(max_hfov, h0 * k)
    p0 = abs(float(pl.get("pitch") or 0))
    if max_extra_tilt is not None and p0 > 0.5:
        # A+B: 사진이 위·아래로 기운 만큼 키우면 기울기도 커짐(가까이서 올려다본 것처럼 휨) → 기울기가 원래보다
        # max_extra_tilt° 넘게 커지지 않는 데까지만 키움. 수평인 사진은 B 그대로, 많이 기운 사진은 A 에 가깝게.
        rmax = math.tan(math.radians(min(85.0, p0 + max_extra_tilt))) / math.tan(math.radians(p0))
        h1 = min(h1, math.degrees(2 * math.atan(math.tan(math.radians(h0) / 2) * rmax)))
    h1 = max(h0, h1)
    r = math.tan(math.radians(h1) / 2) / math.tan(math.radians(h0) / 2)
    out = {**pl, "hfov": round(h1, 2),
           "pitch": round(math.degrees(math.atan(math.tan(math.radians(float(pl.get("pitch") or 0))) * r)), 2)}
    if pl.get("vfov"):
        out["vfov"] = round(min(170.0, math.degrees(2 * math.atan(math.tan(math.radians(float(pl["vfov"])) / 2) * r))), 2)
    out["version_b"] = {"viewer_scale": float(viewer_scale), "from_hfov": h0, "from_pitch": pl.get("pitch"),
                        "max_extra_tilt": max_extra_tilt}
    return out


def photo_rows_at_edge(ctrl_path, margin=8) -> tuple[int, int]:
    """사진을 걸쳐 놓은 control 그림의 x=0 열에서 사진이 있는 줄 범위 (초록 아닌 곳), 위아래 margin 만큼 안쪽."""
    import numpy as np
    a = np.asarray(Image.open(ctrl_path).convert("RGB"))[:, :2].astype(int)
    green = (a[..., 1] > 200) & (a[..., 0] < 60) & (a[..., 2] < 60)
    rows = np.where(~green.all(axis=1))[0]
    if rows.size == 0:
        return 0, 0
    return int(rows.min()) + margin, int(rows.max()) + 1 - margin


class GenerateMixin:
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
        model, err = self._pick_model(p)
        if err:
            return err
        try:
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
        img = erp.make_control(self.source(run_id), *placement_args(pl), W, H)
        path = self.store.path(run_id, name)
        img.save(path, optimize=False)
        return path

    def _keep_mask(self, run_id, pl, name, feather):
        W, H = self.cfg.get("erp_size", [2048, 1024])
        m = erp.keep_mask(self.source(run_id).size, *placement_args(pl), W, H, feather)
        path = self.store.path(run_id, name)
        m.save(path)
        return path

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
        d = self._attempt_dir(run_id, n)
        name = f"{d}/{run_id}_{d}_360.jpg"

        def save():
            im = erp.to_2to1(self.source(run_id).convert("RGB"))
            im.save(self.store.path(run_id, *name.split("/")), "JPEG", quality=self.jpg_quality,
                    subsampling=0)                      # optimize 는 큰 이미지에서 libjpeg 버퍼 오류가 날 수 있어 끔
            return im.size
        size = await asyncio.to_thread(save)
        t = now()
        att = {"n": n, "backend": "-", "mode": "import", "state": "done", "prompt_id": None, "out_nodes": {},
               "placement": {}, "prompts": {}, "settings": {}, "seeds": {},
               "files": [{"tag": "import", "url": f"/runs/{run_id}/{name}", "jpg": f"/runs/{run_id}/{name}", "final": True}],
               "size": list(size), "created": t, "finished": t}
        meta["attempts"].append(att)
        self.store.write(run_id, meta)
        return jresp(att)

    @serialized_run
    async def generate(self, req):
        p = await req.json()
        run_id = p["run_id"]
        p["user_settings"] = dict(p.get("settings") or {})
        bk, backend, err = self._pick_backend(p.get("backend"))
        if err:
            return err
        meta = self.store.read(run_id)
        n = len(meta["attempts"]) + 1
        d = self._attempt_dir(run_id, n)
        tag = f"{run_id}_{d}"                     # ComfyUI 에 올리는 파일 이름 앞부분
        prefix = f"VR360/{tag}"                   # ComfyUI 저장 이름

        mode = p.get("mode", "generate")
        extra = {}
        out_nodes = {"final": str(self.cfg["node_ids"]["save_final"])}     # 결과는 360 한 장만
        try:
            if mode == "pole":                        # 발밑·하늘 보정 (1단계: 돌린 ERP 인페인트)
                wf_api, seeds, out_nodes, extra = await self._start_pole(run_id, n, meta, p, backend)
            elif mode == "enhance":                   # 이미 360 → 화질 개선만
                name = await self._upload_file(backend, self.store.path(run_id, "source" + meta["source_ext"]),
                                               f"{tag}_erp_source{meta['source_ext']}")
                wf_api = workflow.build_enhance(name, meta["width"], meta["height"], prefix=prefix,
                                                **self._enhance_kwargs(meta))
                seeds, out_nodes = {}, {"final": "15"}
            else:
                st0 = self._settings(p, meta)
                ver = str(st0.get("version", "A")).upper()
                if ver in ("B", "AB"):
                    # B: 보는 사람을 작게 → 사진을 크게 배치 / AB: 같지만 기울기가 크게 늘지 않는 데까지만 키움
                    # (이후 단계·결과 기록 모두 이 배치 기준)
                    p["placement"] = version_b_placement(
                        p["placement"], float(st0.get("viewer_scale", 0.6)),
                        max_extra_tilt=float(st0.get("max_extra_tilt", 8)) if ver == "AB" else None)
                gen_pl, wrap = p["placement"], None
                if self.cfg.get("wrap_in_photo", False):
                    # 사진을 0°/360° 에 걸쳐 놓고 생성 → 지평선 부근 이음새는 사진 안(원본)에, 하늘·바닥만 이음새 복원
                    W = self.cfg.get("erp_size", [2048, 1024])[0]
                    shift, yaw_gen = workflow.wrap_shift_for(float(p["placement"]["yaw"]), W)
                    gen_pl = {**p["placement"], "yaw": yaw_gen}
                ctrl_path = await asyncio.to_thread(self._control, run_id, gen_pl, f"{d}/control.png")
                if gen_pl is not p["placement"]:
                    top, bottom = await asyncio.to_thread(photo_rows_at_edge, ctrl_path)
                    if bottom - top >= 64:
                        wrap = {"shift": shift, "top": top, "bottom": bottom}
                name = await self._upload_file(backend, ctrl_path, f"{tag}_erp_control.png")
                kp = self.cfg.get("keep_photo", {})
                mask_name = None
                if kp.get("enabled", True):          # 사진 영역은 그대로 두고 초록 부분만 생성
                    mask_path = await asyncio.to_thread(self._keep_mask, run_id, gen_pl, f"{d}/keep_mask.png",
                                                        kp.get("mask_feather", 0.04))
                    mask_name = await self._upload_file(backend, mask_path, f"{tag}_keep_mask.png")
                tmpl = workflow.load_template(self.cfg["workflow"])
                st = self._settings(p, meta, p["prompts"].get("positive", ""))
                wf_api, seeds = workflow.build(tmpl, self.cfg["node_ids"], control_image=name, prompts=p["prompts"],
                                               seed1=p.get("seed1"), seed2=p.get("seed2"),
                                               prefix=prefix, settings=st, keep_mask=mask_name, wrap=wrap)
                if kp.get("enabled", True) and kp.get("upscale_photo", True):
                    # 원본 사진 자리 화질: 사진을 원래 해상도로 AI 업스케일 → 마지막에 이걸 붙임
                    ph_name = await backend.upload(await asyncio.to_thread(self._source_png, run_id), f"{tag}_photo.png")
                    out_nodes["photo_up"] = workflow.add_photo_upscale(wf_api, ph_name, prefix, self.upscale_model)
                if "108" in wf_api:                   # 진단용: 이음새 복원 직후 2K
                    wf_api["973"] = workflow.node("SaveImage", {"images": ["108", 0], "filename_prefix": f"{prefix}_dbg_2k_after_seam"},
                                                  "진단 · 이음새 복원 직후 2K")
                    out_nodes["dbg_seam2k"] = "973"
                p["settings"] = {k: v for k, v in st.items() if k != "refine"}
                if wrap:
                    p["settings"]["wrap_in_photo"] = wrap
                rv = self.cfg.get("review", {})
                if rv.get("enabled", False) and not wrap and self.openai and self.openai.ready:
                    # 2K 초안 → LLM 검수 → (다시 생성 | 통과한 초안만 8K) — 상태 확인(status)에서 이어서 진행
                    out_nodes = {"draft": workflow.draft_only(wf_api, self.cfg["node_ids"], prefix)}
                    extra = {"stage": "draft", "gen": {"control": name, "prefix": prefix,
                                                       "user_settings": p.get("user_settings") or {},
                                                       "base_prompts": p["prompts"]},
                             "review": {"max_tries": int(rv.get("max_tries", 3)), "tries": []}}
            pid = await self._submit_workflow(backend, wf_api, self.store.path(run_id, d, "workflow_api.json"))
        except Exception as e:
            msg = str(e)
            with open(self.store.path(run_id, d, "error.txt"), "w", encoding="utf-8") as f:
                f.write(msg)
            print(f"[pano360] 생성 요청 실패 ({run_id} {d}):\n{msg}")
            return jresp({"error": comfy.explain_comfy_error(msg)}, 502)

        meta["attempts"].append({"n": n, "backend": bk, "prompt_id": pid, "mode": mode, "out_nodes": out_nodes,
                                 **extra,
                                 "placement": p.get("placement") or {}, "prompts": p.get("prompts") or {},
                                 "settings": p.get("settings") or {}, "seeds": seeds, "state": "queued",
                                 "files": [], "created": now()})
        self.store.write(run_id, meta)
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
        kw = dict(final_w=q.get("final_width", 8192), upscale_model=self.upscale_model,
                  blend=q.get("blend", 0.2), noise=q.get("noise", 0.004))
        rf = dict(self.cfg.get("refine", {}))
        if rf.get("enabled", False):
            rp = llm.refine_prompts(meta.get("analysis") or {})
            rf.update(positive=rp["positive"], negative=rp["negative"], seed=seed)
            kw.update(refine=rf, template_ui=workflow.load_template(self.cfg["workflow"]), ids=self.cfg["node_ids"])
        return kw

    @serialized_run
    async def status(self, req):
        run_id, n = route_ids(req)
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        if att["state"] in ("done", "error"):
            return jresp(att)
        backend = self.backends.get(att["backend"])
        out_nodes = att.get("out_nodes") or {"final": str(self.cfg["node_ids"]["save_final"])}
        if att.get("stage") == "review":                  # LLM 검수 중 (뒤에서 진행)
            k = len((att.get("review") or {}).get("tries", [])) + 1
            return jresp({**att, "state": "running", "node_title": f"2K 초안 {k} LLM 검수 중", "value": 0, "max": 0})
        st = await backend.status(att["prompt_id"], list(out_nodes.values()))
        if st["state"] == "done" and att.get("stage") == "draft":
            try:
                await self._draft_done(run_id, n, att, backend, st)
            except Exception as e:
                att.update(state="error", error=f"2K 초안 저장 실패: {e}")
            self.store.write(run_id, meta)
            return jresp({**att, "state": att["state"] if att["state"] == "error" else "running",
                          "node_title": "2K 초안 완성 → LLM 검수 시작", "value": 0, "max": 0})
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
                for f in [x for x in st["files"] if x["node"] == out_nodes["final"]][:1]:
                    files.append(await self._finish_result(run_id, n, meta, att, backend, st, out_nodes, f))
                att.update(state="done", files=files, finished=now())
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

    # ------------------------------------------------------------------ 2K 초안 → LLM 검수 → 다시 생성 / 8K
    async def _draft_done(self, run_id, n, att, backend, st):
        """초안 생성 끝 → 2K 저장 → 검수 단계로 (검수는 뒤에서)."""
        k = len(att["review"]["tries"]) + 1
        local = adir(n, f"draft_{k}.png")
        f = next(x for x in st["files"] if x["node"] == att["out_nodes"]["draft"])
        await self._save_output(backend, f, self.store.path(run_id, local))
        att["review"]["current"] = {"draft": local, "seeds": att["seeds"], "prompts": att["prompts"]}
        att["stage"] = "review"
        task = asyncio.create_task(self._review_step(run_id, n))
        self._bg = getattr(self, "_bg", set())
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def _review_views(self, draft_path, yaw):
        """초안 → 앞·오른쪽·뒤·왼쪽 4화면 (2×2, 각 768×512)."""
        from PIL import ImageDraw
        im = Image.open(draft_path).convert("RGB")
        grid = Image.new("RGB", (1536, 1024))
        for i, (lab, dy) in enumerate((("FRONT", 0), ("RIGHT", 90), ("BACK", 180), ("LEFT", -90))):
            v = erp.perspective(im, yaw + dy, 0, 100, 768, 512)
            ImageDraw.Draw(v).text((12, 8), lab, fill=(255, 255, 0))
            grid.paste(v, ((i % 2) * 768, (i // 2) * 512))
        return grid

    async def _review_step(self, run_id, n):
        """LLM 검수 → 통과면 그 초안을 8K 로, 아니면 고칠 점을 넣어 새 시드로 다시 초안 (최대 max_tries)."""
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        cur = att["review"]["current"]
        res = None
        try:
            views = await asyncio.to_thread(self._review_views, self.store.path(run_id, cur["draft"]),
                                            float((att.get("placement") or {}).get("yaw", 0)))
            views.save(self.store.path(run_id, cur["draft"].replace(".png", "_views.jpg")), quality=85)
            res, _ = await self.openai.review(views, self.source(run_id), cur["prompts"].get("positive", ""),
                                              meta.get("model"))
        except Exception as e:
            print(f"[pano360] 초안 검수 실패 → 이 초안으로 진행: {e}")
        async with self.lock(run_id):
            meta = self.store.read(run_id)
            att = meta["attempts"][n - 1]
            rv = att["review"]
            score = float(res["score"]) if res else -1.0
            rv["tries"].append({**rv.pop("current"), "ok": bool(res and res["ok"]), "score": score,
                                "problems": (res or {}).get("problems", []), "fix_hint": (res or {}).get("fix_hint", ""),
                                "summary_ko": (res or {}).get("summary_ko", "검수 실패")})
            print(f"[pano360] 초안 {len(rv['tries'])} 검수: {'통과' if res and res['ok'] else '불합격'} "
                  f"{score:.1f} — {(res or {}).get('summary_ko', '')}")
            backend = self.backends.get(att["backend"])
            try:
                if res and not res["ok"] and len(rv["tries"]) < rv["max_tries"]:
                    await self._submit_draft(run_id, n, att, backend)
                else:
                    await self._submit_final(run_id, n, meta, att, backend)
            except Exception as e:
                att.update(state="error", error=f"검수 다음 단계 제출 실패: {e}")
            self.store.write(run_id, meta)

    def _gen_graph(self, meta, att, prompts, seeds, settings_user):
        tmpl = workflow.load_template(self.cfg["workflow"])
        st = self._settings({"settings": settings_user}, meta, prompts.get("positive", ""))
        return workflow.build(tmpl, self.cfg["node_ids"], control_image=att["gen"]["control"], prompts=prompts,
                              seed1=seeds.get("seed1"), seed2=seeds.get("seed2"), prefix=att["gen"]["prefix"],
                              settings=st)

    async def _submit_draft(self, run_id, n, att, backend):
        """고칠 점을 Positive 에 넣고 새 시드로 다시 2K 초안."""
        meta = self.store.read(run_id)
        prompts = llm.with_fix_hints(att["gen"]["base_prompts"], [t["fix_hint"] for t in att["review"]["tries"]])
        wf, seeds = self._gen_graph(meta, att, prompts, {}, att["gen"]["user_settings"])
        out = workflow.draft_only(wf, self.cfg["node_ids"], att["gen"]["prefix"])
        k = len(att["review"]["tries"]) + 1
        pid = await self._submit_workflow(backend, wf, self.store.path(run_id, adir(n), f"workflow_api_draft{k}.json"))
        att.update(stage="draft", prompt_id=pid, seeds=seeds, prompts=prompts, out_nodes={"draft": out})

    async def _submit_final(self, run_id, n, meta, att, backend):
        """가장 좋은 초안(통과한 것 우선, 없으면 점수 최고) → 같은 8K 단계."""
        tries = att["review"]["tries"]
        best = max(tries, key=lambda t: (t["ok"], t["score"]))
        att["review"]["chosen"] = tries.index(best) + 1
        name = await self._upload_file(backend, self.store.path(run_id, best["draft"]),
                                       f"{run_id}_{adir(n).replace('/', '_')}_draft{att['review']['chosen']}.png")
        wf, seeds = self._gen_graph(meta, att, best["prompts"], best["seeds"], att["gen"]["user_settings"])
        out = workflow.final_from_draft(wf, self.cfg["node_ids"], name)
        pid = await self._submit_workflow(backend, wf, self.store.path(run_id, adir(n), "workflow_api_final.json"))
        att.update(stage="final", prompt_id=pid, seeds=seeds, prompts=best["prompts"], out_nodes={"final": out})

    async def _finish_result(self, run_id, n, meta, att, backend, st, out_nodes, f) -> dict:
        """최종 결과 받기 → 톤 맞춤 → 원본 사진 다시 붙이기 → 이음새 후처리 → Quest JPG. 결과 파일 항목 반환."""
        tag = self.cfg.get("final_tag", "final")
        local = adir(n, f"result_{tag}.png")
        path = self.store.path(run_id, local)
        await self._save_output(backend, f, path)
        generated = att.get("mode", "generate") == "generate"
        tm = float(self.cfg.get("tone_match", 0))
        if tm > 0 and generated and not meta.get("is_erp"):
            # 생성 결과의 밝기·대비·색을 원본 사진 쪽으로 (그림 같은 밝고 평평한 톤 완화)
            raw = self.store.path(run_id, adir(n, f"result_{tag}_raw.png"))
            os.replace(path, raw)
            await asyncio.to_thread(lambda: erp.tone_match(Image.open(raw), self.source(run_id), tm).save(path))
            att["tone_match"] = tm
        if self.cfg.get("keep_photo", {}).get("enabled", True) and generated and (att.get("placement") or {}).get("hfov"):
            # 원본 사진을 원래 해상도로 같은 자리에 다시 붙임 → 사진에 있던 건 절대 안 사라짐
            pf = next((x for x in st["files"] if x["node"] == out_nodes.get("photo_up")), None)
            if pf:                       # 원래 해상도로 AI 업스케일한 원본 사진
                await self._save_output(backend, pf, self.store.path(run_id, adir(n, "photo_up.png")))
            mode, scale = await asyncio.to_thread(self._paste_photo_file, run_id, n, att, path)
            att["photo_pasted"] = mode
            print(f"[pano360] 원본 사진 다시 붙임: {mode} (사진이 최종보다 {scale:.2f}배 흐림)")
        df = next((x for x in st["files"] if x["node"] == out_nodes.get("dbg_seam2k")), None)
        if df:                           # 진단용 2K (이음새 선이 어느 단계에서 생기는지 확인)
            try:
                await self._save_output(backend, df, self.store.path(run_id, adir(n, "debug_2k_after_seam.png")))
            except Exception as e:
                print(f"[pano360] 진단 2K 저장 실패: {e}")
        post = self.cfg.get("post", {})
        try:     # 좌우 이음새에 생긴 가는 세로선 지우기 (ComfyUI 결과에서 x=0 에 1~2px@2K 선이 나오는 경우)
            cols = await asyncio.to_thread(erp.fix_wrap_line, path)
            if cols:
                att["wrap_line_fixed"] = [cols[0], cols[-1]]
                print(f"[pano360] 이음새 세로선 지움: 열 {cols[0]}~{cols[-1]}")
            gap = await asyncio.to_thread(erp.match_wrap_tone, path)
            att["wrap_tone_gap"] = round(gap, 1)     # VR 에서 이음새 양쪽 밝기 차이가 띠처럼 보이지 않게
            if post.get("wrap_blend", True):         # 굵은 선·점선·남은 단차: 이음새 띠째로 다중 대역으로 다시 이음
                att["wrap_blend_px"] = await asyncio.to_thread(erp.blend_wrap_band, path, post.get("wrap_band", 0.012))
            if post.get("wrap_sky", True):          # 이음새 근처 하늘의 큰 얼룩(복원 띠 톤 차이) → 매끈하게
                att["wrap_sky"] = round(await asyncio.to_thread(erp.bridge_wrap_sky, path), 2)
        except Exception as e:
            print(f"[pano360] 이음새 선 처리 실패: {e}")
        try:     # 이음새 연결 점검 (참고용 경고: 띠 색·뿌연 정도·결·양쪽 재질)
            qa = await asyncio.to_thread(erp.continuity_report, path)
            att["continuity"] = qa
            for w in qa.get("warnings", []):
                print(f"[pano360] 이음새 점검 경고: {w}")
        except Exception as e:
            print(f"[pano360] 이음새 점검 실패: {e}")
        try:
            att["seam"] = await asyncio.to_thread(erp.wrap_seam_score, path)
            print(f"[pano360] 좌우 이음새 {att['seam']}")
        except Exception as e:
            print(f"[pano360] 이음새 측정 실패: {e}")
        jpg = adir(n, f"{run_id}_a{n:02d}_{tag}.jpg")          # Quest 용 JPG (용량 1/5, 로딩 빠름)
        await asyncio.to_thread(erp.save_quest_jpg, path, self.store.path(run_id, jpg), self.jpg_quality)
        return {"tag": tag, "url": f"/runs/{run_id}/{local}", "final": True, "jpg": f"/runs/{run_id}/{jpg}"}
