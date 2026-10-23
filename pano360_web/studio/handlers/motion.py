"""
5단계 영상화: 일부만 움직일 영역(cinemagraph) 계획 · 수정
사건 연출(event.py)도 같은 LLM 입력(_llm_inputs)과 영역 화면(_save_view_assets)을 쓴다.
"""

import asyncio
import time

from PIL import Image

from .. import erp, llm
from .base import jresp, now, route_ids, serialized_run, write_json


class MotionMixin:
    MOTION_MODEL = {"base": "LTX-2.3-22B", "lora": "ltx-2.3-22b-lora-cinemagraph-0.9", "trigger": llm.MOTION_TRIGGER,
                    "train_size": [704, 512], "frames": 25,
                    "note": "화면 전체가 아니라 영역만 움직임. 결과 영상은 이 영역 자리에만 다시 합성하고 나머지는 8K 정지 이미지."}

    # ------------------------------------------------------------------ 영상화·사건 공통
    def _motion_context(self, meta, att):
        a = meta.get("analysis") or {}
        pr = att.get("prompts") or meta.get("prompts") or {}
        parts = [f"scene: {a.get('scene', '')}", f"lighting: {a.get('lighting', '')}",
                 f"panorama positive prompt: {pr.get('positive', '')}"]
        return "\n".join(p for p in parts if p.split(": ", 1)[1])

    async def _llm_inputs(self, run_id, meta, att):
        """LLM 에 줄 것: 격자를 그린 결과 360 · 원본 사진(360 을 올렸으면 없음) · 생성 프롬프트 요약."""
        path = self._final_path(run_id, att)
        grid = await asyncio.to_thread(lambda: erp.grid_overlay(Image.open(path)))
        photo = None if meta.get("is_erp") else self.source(run_id)
        return grid, photo, self._motion_context(meta, att)

    def _final_image(self, run_id, n):
        att = self.store.read(run_id)["attempts"][n - 1]
        return Image.open(self._final_path(run_id, att)).convert("RGB")

    def _save_overlay(self, run_id, d, img, regions):
        erp.motion_overlay(img.resize((2048, 1024), Image.LANCZOS), regions) \
            .save(self.store.path(run_id, d, "overlay.jpg"), quality=88)

    # ------------------------------------------------------------------ 움직일 영역
    def _motion_assets(self, run_id, n, plan):
        """영역별 평면 크롭(영상 모델 입력) + 정지/움직임 미리보기 + ERP 마스크 저장, plan 에 주소 기록."""
        d = self._attempt_dir(run_id, n, "motion")
        img = self._final_image(run_id, n)
        t = int(time.time())
        for i, r in enumerate(plan["regions"], 1):
            w, h = erp.region_size(r)
            erp.perspective(img, r["yaw"], r["pitch"], r["hfov"], w, h).save(self.store.path(run_id, d, f"region_{i}.png"))
            r.update(crop=f"/runs/{run_id}/{d}/region_{i}.png?t={t}", crop_size=[w, h])
        self._save_overlay(run_id, d, img, plan["regions"])
        erp.region_mask(plan["regions"]).save(self.store.path(run_id, d, "mask.png"))
        plan.update(overlay=f"/runs/{run_id}/{d}/overlay.jpg?t={t}", mask=f"/runs/{run_id}/{d}/mask.png?t={t}",
                    video_model=self.MOTION_MODEL)
        write_json(self.store.path(run_id, d, "plan.json"), plan)
        return plan

    def _motion_finish(self, plan):
        plan["regions"] = [llm.clamp_region(r) for r in plan.get("regions", [])][:6]
        for r in plan["regions"]:
            if r.pop("regen", True) or not r.get("positive") or r.get("prompt_ver", 1) < llm.PROMPT_VER:
                r.update(llm.motion_prompts(r, plan.get("lighting", "")), prompt_ver=llm.PROMPT_VER)
        plan["ltx"] = llm.global_motion_prompt(plan["regions"], plan.get("lighting", ""))   # 360 전체용 (LTX 도구에 바로)
        return plan

    def _store_motion(self, run_id, n, plan):
        self._edit_attempt(run_id, n, lambda a: a.__setitem__("motion", plan))
        return jresp(plan)

    @serialized_run
    async def motion_plan(self, req):
        """결과 360 + 생성 프롬프트를 LLM 에 주고, 일부만 움직일 영역과 영상 프롬프트를 받음."""
        run_id, n = route_ids(req)
        p = await req.json()
        meta = self.store.read(run_id)
        model, err = self._pick_model(p)
        if err:
            return err
        try:
            grid, photo, ctx = await self._llm_inputs(run_id, meta, meta["attempts"][n - 1])
            plan, usage = await self.openai.plan_motion(grid, photo, ctx, p.get("note", ""), model)
        except Exception as e:
            return jresp({"error": str(e)}, 502)
        plan = self._motion_finish(plan)
        plan.update(model=model, usage=usage, note=p.get("note", ""), created=now())
        plan = await asyncio.to_thread(self._motion_assets, run_id, n, plan)
        return self._store_motion(run_id, n, plan)

    @serialized_run
    async def motion_update(self, req):
        """화면에서 고친 영역(위치·움직임·프롬프트) 저장 → 크롭·미리보기 다시."""
        run_id, n = route_ids(req)
        p = await req.json()
        plan = dict(self.store.read(run_id)["attempts"][n - 1].get("motion") or {})
        plan["regions"] = p.get("regions") or []
        if p.get("lighting"):
            plan["lighting"] = p["lighting"]
        plan = self._motion_finish(plan)
        plan["updated"] = now()
        try:
            plan = await asyncio.to_thread(self._motion_assets, run_id, n, plan)
        except Exception as e:
            return jresp({"error": str(e)}, 400)
        return self._store_motion(run_id, n, plan)
