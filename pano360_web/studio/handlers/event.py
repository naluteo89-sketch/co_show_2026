"""
6단계 사건 연출: LLM 사건 계획 · 움직일 범위(마스크) 칠하기 · 노드 파일로 360 영상 만들기 · 4K 360 합성
"""

import asyncio
import base64
import io
import os
import time

import numpy as np
from PIL import Image

from .. import erp, llm, workflow
from ..clients import comfy
from ..video import compose
from .base import adir, jresp, now, resolve, route_ids, serialized_run, write_json


class EventMixin:
    EVENT_VIEW = (2560, 1440)          # 사건 화면 (16:9)

    # ------------------------------------------------------------------ 사건 계획
    def _event_finish(self, ev, lighting=""):
        ev = llm.clamp_event(ev)
        ev["seconds"] = max(2.0, min(8.0, float(ev.get("seconds", 5) or 5)))
        if not ev.pop("prompt_edited", False) or not ev.get("positive"):
            ev.update(llm.event_prompts(ev, lighting))
        return ev

    def _event_assets(self, run_id, n, ev):
        d = self._attempt_dir(run_id, n, "event")
        img = self._final_image(run_id, n)
        w, h = self.EVENT_VIEW
        t = int(time.time())
        erp.perspective(img, ev["yaw"], ev["pitch"], ev["hfov"], w, h).save(self.store.path(run_id, d, "event_view.png"))
        self._save_overlay(run_id, d, img, [{"yaw": ev["yaw"], "pitch": ev["pitch"], "hfov": ev["hfov"], "aspect": w / h}])
        ev.update(view=f"/runs/{run_id}/{d}/event_view.png?t={t}", view_name=f"{run_id}_event_view.png",
                  overlay=f"/runs/{run_id}/{d}/overlay.jpg?t={t}", view_size=[w, h],
                  ltx360={"positive": llm.video_event_prompt(ev), "negative": llm.LTX360_NEG})
        ev.pop("workflows", None)
        return ev

    @serialized_run
    async def event_plan(self, req):
        """LLM: 결과 360 을 보고 한 번 일어나는 연출(어디서·무엇이·어떻게)과 영상 프롬프트."""
        run_id, n = route_ids(req)
        p = await req.json()
        meta = self.store.read(run_id)
        model, err = self._pick_model(p)
        if err:
            return err
        try:
            grid, photo, ctx = await self._llm_inputs(run_id, meta, meta["attempts"][n - 1])
            ev, usage = await self.openai.plan_event(grid, photo, ctx, p.get("note", ""), model)
        except Exception as e:
            return jresp({"error": str(e)}, 502)
        ev["seconds"] = p.get("seconds", 5)
        ev = self._event_finish(ev, (meta.get("analysis") or {}).get("lighting", ""))
        ev.update(model=model, usage=usage, note=p.get("note", ""), created=now())
        ev = await asyncio.to_thread(self._event_assets, run_id, n, ev)

        def put(a):
            if (a.get("event") or {}).get("video"):
                ev["video"] = a["event"]["video"]                # 사건을 다시 짜도 만든 영상 기록은 유지
            a["event"] = ev
        self._edit_attempt(run_id, n, put)
        return jresp(ev)

    @serialized_run
    async def event_update(self, req):
        """화면에서 고친 사건(위치·내용·프롬프트) 저장 → 사건 화면·워크플로 다시."""
        run_id, n = route_ids(req)
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
        ev["updated"] = now()
        ev = await asyncio.to_thread(self._event_assets, run_id, n, ev)
        self._edit_attempt(run_id, n, lambda a: a.__setitem__("event", ev))
        return jresp(ev)

    # ------------------------------------------------------------------ 움직일 범위 (마스크)
    async def event_base(self, req):
        """범위 칠하기용: 결과 360 작은 그림(2048) + 저장된 범위."""
        run_id, n = route_ids(req)
        d = self._attempt_dir(run_id, n, "event")
        base = self.store.path(run_id, d, "base_2k.jpg")
        if not os.path.isfile(base):
            await asyncio.to_thread(lambda: self._final_image(run_id, n).resize((2048, 1024), Image.LANCZOS)
                                    .save(base, quality=88))
        mp = self.store.path(run_id, d, "mask.png")
        v = int(os.path.getmtime(mp)) if os.path.isfile(mp) else 0
        return jresp({"base": f"/runs/{run_id}/{d}/base_2k.jpg", "mask": f"/runs/{run_id}/{d}/mask.png?v={v}" if v else None})

    async def event_mask(self, req):
        """웹에서 칠한 움직임 범위 저장 (흰색 = 영상, 검정 = 원본). 빈 값이면 지움."""
        run_id, n = route_ids(req)
        p = await req.json()
        d = self._attempt_dir(run_id, n, "event")
        mp = self.store.path(run_id, d, "mask.png")
        data = p.get("png") or ""
        if not data:
            if os.path.isfile(mp):
                os.remove(mp)
            return jresp({"mask": None, "cover": 0})
        try:
            raw = base64.b64decode(data.split(",", 1)[-1])
            im = Image.open(io.BytesIO(raw)).convert("L").resize((2048, 1024), Image.BILINEAR)
        except Exception:
            return jresp({"error": "범위 그림을 읽지 못했어요."}, 400)
        im.save(mp)
        cover = float(np.asarray(im, np.float32).mean() / 255.0)
        return jresp({"mask": f"/runs/{run_id}/{d}/mask.png?v={int(time.time())}", "cover": round(cover, 4)})

    # ------------------------------------------------------------------ 노드 파일로 360 영상
    def _video_template(self):
        return workflow.load_template(resolve(self.cfg.get("video_template",
                                                           "../comfy_nodes/video/quest360_flood_mask_inpaint_4k.json")))

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

    def _store_video(self, run_id, n, vid):
        self._edit_attempt(run_id, n, lambda a: a.setdefault("event", {}).__setitem__("video", vid))

    async def event_video_start(self, req):
        """화질 올린 360 + 칠한 범위 + 프롬프트 → 노드 파일(video_template)에 값만 넣어 실행.
        크롭 생성 노드(event_cropgen_*)면: 칠한 범위가 정면에 오게 사진·마스크를 돌리고(front_shift) → 칠한 범위만 잘라
        고해상도로 생성 → 끝나면 서버가 4K 360 에 붙임(video/compose.py). 그 밖의 노드는 예전처럼 사각/그림 마스크만 넣음."""
        run_id, n = route_ids(req)
        p = await req.json()
        att = self.store.read(run_id)["attempts"][n - 1]
        bk, backend, err = self._pick_backend(p.get("backend"))
        if err:
            return err
        d = adir(n, "event")
        tag = f"{run_id}_a{n:02d}"
        mp = self.store.path(run_id, d, "mask.png")
        if not os.path.isfile(mp):
            return jresp({"error": "먼저 움직일 범위를 칠하고 '범위 저장'을 눌러 주세요."}, 400)
        try:
            src = self._final_path(run_id, att)
            tmpl = self._video_template()
        except Exception as e:
            return jresp({"error": str(e)}, 400)
        crop_mode = workflow.video_has_crop(tmpl)
        ev = att.get("event") or {}
        secs = max(2, min(14, int(p.get("seconds") or 4)))
        pos = (p.get("positive") or "").strip() or llm.video_event_prompt(ev, crop=crop_mode)
        if crop_mode:
            pos = pos.rstrip() + " " + llm.event_timeline(secs)          # 길이 안에 시작→절정→끝
        neg = (p.get("negative") or "").strip() or (llm.LTX_CROP_NEG if crop_mode else llm.LTX360_NEG)
        mode = workflow.video_mask_mode(tmpl)
        rect = crop = None
        shift = 0
        mask_send = mp
        if crop_mode:                                  # 모든 사건은 정면 기준: 칠한 범위 가운데가 정면(360 가운데)에 오게 돌림
            shift = await asyncio.to_thread(compose.front_shift, mp)
            mask_send = self.store.path(run_id, d, "mask_front.png")
            await asyncio.to_thread(compose.roll_image, mp, mask_send, shift)
        elif mode == "rect":
            rect = await asyncio.to_thread(self._mask_rect, mp, workflow.video_rect_canvas(tmpl),
                                           workflow.video_rect_feather(tmpl))
            if not rect:
                return jresp({"error": "칠한 범위가 비어 있어요."}, 400)

        def pano_jpg():
            """크롭 생성은 원본(8K)에서 잘라야 선명하므로 원래 크기 그대로(돌려서), 그 밖은 4K 로 줄여 올림."""
            im = Image.open(src).convert("RGB")
            if crop_mode:
                if shift:
                    im = Image.fromarray(erp.roll_array(np.asarray(im), shift))
            elif im.width > 4096:
                im = im.resize((4096, 2048), Image.LANCZOS)
            b = io.BytesIO()
            im.save(b, "JPEG", quality=96)
            return b.getvalue(), im.size
        try:
            data, src_size = await asyncio.to_thread(pano_jpg)
            if crop_mode:
                crop = await asyncio.to_thread(lambda: workflow.crop_plan(Image.open(mask_send), src_size))
                if not crop:
                    return jresp({"error": "칠한 범위가 비어 있어요."}, 400)
            name = await backend.upload(data, f"{tag}_pano{'_front' if crop_mode else '4k'}.jpg")
            mname = None
            if mode == "image":
                mname = await self._upload_file(backend, mask_send, f"{tag}_motion_mask{'_front' if crop_mode else ''}.png")
            ui, seed, info = workflow.fill_video_node(tmpl, image=name, positive=pos, negative=neg, seconds=secs,
                                                      lora_strength=p.get("lora_strength"), seed=p.get("seed"),
                                                      prefix=f"VR360/{tag}_flood", mask_image=mname, rect=rect,
                                                      crop=crop)
            write_json(self.store.path(run_id, d, "video_node.json"), ui)     # 값이 들어간 노드 파일 (ComfyUI 에서 그대로 열림)
            g, saves = workflow.video_node_api(ui)
            pid = await self._submit_workflow(backend, g, self.store.path(run_id, d, "workflow_video.json"))
        except Exception as e:
            return jresp({"error": comfy.explain_comfy_error(str(e))}, 502)
        vid = {"state": "running", "backend": bk, "prompt_id": pid, "saves": saves, "seed": seed, "seconds": secs,
               "fps": info["fps"], "mask_mode": mode, "rect": rect, "crop": crop, "shift": shift, "crop_mode": crop_mode,
               "template": os.path.basename(self.cfg.get("video_template", "")),
               "node_url": f"/runs/{run_id}/{d}/video_node.json?t={int(time.time())}",
               "positive": pos, "negative": neg, "lora_strength": p.get("lora_strength"),
               "created": now()}
        async with self.lock(run_id):
            self._store_video(run_id, n, vid)
        return jresp(vid)

    async def _compose_event(self, run_id, n, crop_mp4, has_audio):
        """크롭 영상 → 4K 360 (바탕 = 8K 사진을 같은 만큼 돌려 줄인 것, 칠한 범위 안만 영상, 끝은 사진으로)."""
        d = adir(n, "event")
        try:
            att = self.store.read(run_id)["attempts"][n - 1]
            vid = att["event"]["video"]
            out = self.store.path(run_id, d, "flood_4k_audio.mp4" if has_audio else "flood_4k.mp4")
            info = await asyncio.to_thread(compose.compose_crop_video, crop_mp4, self._final_path(run_id, att),
                                           self.store.path(run_id, d, "mask_front.png"), vid["crop"], out,
                                           fps=int(vid.get("fps", 24)), audio=has_audio, shift=int(vid.get("shift", 0)))
            url = f"/runs/{run_id}/{d}/{os.path.basename(out)}?t={int(time.time())}"
            upd = {"state": "done", "composed": info, "finished": now(),
                   ("audio_url" if has_audio else "silent_url"): url}
        except Exception as e:
            upd = {"state": "error", "error": f"4K 360 합성 실패: {e}"}
        await self._edit_attempt_locked(run_id, n, lambda a: a.setdefault("event", {}).setdefault("video", {}).update(upd))

    async def event_video_status(self, req):
        """진행 확인. 끝나면 노드의 영상 출력(SaveVideo)을 받아 저장.
        크롭 생성이면 받은 크롭 영상을 4K 360 에 붙이는 동안 'composing' → 끝나면 'done'."""
        run_id, n = route_ids(req)
        vid = (self.store.read(run_id)["attempts"][n - 1].get("event") or {}).get("video")
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
                d = adir(n, "event")
                t = int(time.time())
                got = []
                for s in saves:
                    f = next((x for x in st["files"] if x["node"] == s["node"]), None)
                    if not f:
                        continue
                    fn = f"{'crop' if vid.get('crop_mode') else 'flood'}_{s['node']}{'_audio' if s['audio'] else ''}.mp4"
                    await self._save_output(backend, f, self.store.path(run_id, d, fn))
                    got.append((fn, s["audio"]))
                    if not vid.get("crop_mode"):
                        key = "audio_url" if s["audio"] and not vid.get("audio_url") else "silent_url" if not vid.get("silent_url") else None
                        if key:
                            vid[key] = f"/runs/{run_id}/{d}/{fn}?t={t}"
                if not got:
                    raise RuntimeError("결과 영상이 없어요")
                if vid.get("crop_mode"):
                    fn, has_audio = got[0]                       # 소리 있는 것 먼저 (video_node_api 정렬)
                    vid.update(state="composing", crop_file=fn,
                               progress={"state": "running", "node_title": "4K 360 에 붙이는 중"})
                    async with self.lock(run_id):
                        self._store_video(run_id, n, vid)
                    task = asyncio.create_task(self._compose_event(run_id, n, self.store.path(run_id, d, fn), has_audio))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                    return jresp(vid)
                vid.update(state="done", finished=now())
            except Exception as ex:
                vid.update(state="error", error=f"결과 받기 실패: {ex}")
        async with self.lock(run_id):
            self._store_video(run_id, n, vid)
        return jresp(vid)
