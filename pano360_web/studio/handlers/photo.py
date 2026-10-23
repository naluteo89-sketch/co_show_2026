"""
원본 사진 다시 붙이기 (생성 직후 자동 · 이미 만든 결과에 '원본 자리 화질 맞추기')
"""

import asyncio
import os
import time

from PIL import Image

from .. import erp, workflow
from ..clients import comfy
from .base import adir, jresp, placement_args, route_ids


class PhotoMixin:
    def _paste_photo_file(self, run_id, n, att, dst):
        """dst(최종 360)에 원본 사진을 다시 붙임. AI 업스케일된 원본(photo_up.png)이 있으면 그걸로 → 원본 자리도 선명."""
        kp = self.cfg.get("keep_photo", {})
        yaw, pitch, hfov, vfov = placement_args(att.get("placement") or {})
        up = self.store.path(run_id, adir(n, "photo_up.png"))
        fin = Image.open(dst)
        src = Image.open(up).convert("RGB") if os.path.isfile(up) else self.source(run_id)
        mode, scale = erp.photo_paste_mode(fin.width, src.width, hfov, kp.get("paste_mode", "auto"))
        erp.paste_photo(fin, src, yaw, pitch, hfov, vfov, kp.get("paste_feather", 0.05), mode=mode,
                        detail=float(kp.get("detail_strength", 0.9))).save(dst)
        return ("upscaled-" if os.path.isfile(up) else "") + mode, scale

    async def _upscale_photo(self, run_id, n, backend, dst):
        """원본 사진만 ComfyUI 로 AI 업스케일해 dst 에 저장 (최대 10분 기다림)."""
        d = adir(n)
        name = await backend.upload(await asyncio.to_thread(self._source_png, run_id), f"{run_id}_{d}_photo.png")
        g, out = workflow.build_photo_upscale(name, self.upscale_model, f"VR360/{run_id}_{d}")
        pid = await self._submit_workflow(backend, g, self.store.path(run_id, d, "workflow_photo_up.json"))
        t0 = time.time()
        while True:
            st = await backend.status(pid, [out])
            if st["state"] == "done":
                break
            if st["state"] == "error" or time.time() - t0 > 600:
                raise RuntimeError(st.get("error") or "업스케일이 너무 오래 걸려요")
            await asyncio.sleep(2)
        await self._save_output(backend, next(x for x in st["files"] if x["node"] == out), dst)

    async def repaste_photo(self, req):
        """이미 만든 결과: 원본 사진을 원래 해상도로 AI 업스케일(ComfyUI)해서 같은 자리에 다시 붙임 → JPG 도 다시."""
        run_id, n = route_ids(req)
        p = await req.json() if req.can_read_body else {}
        att = self.store.read(run_id)["attempts"][n - 1]
        if att.get("mode", "generate") != "generate" or not (att.get("placement") or {}).get("hfov"):
            return jresp({"error": "사진을 놓아서 만든 결과에만 쓸 수 있어요."}, 400)
        f = next((x for x in att.get("files", []) if x.get("final")), None)
        if not f:
            return jresp({"error": "결과 이미지가 없어요."}, 400)
        _, backend, err = self._pick_backend(p.get("backend") or att.get("backend"))
        if err:
            return err
        up = self.store.path(run_id, adir(n, "photo_up.png"))
        try:
            if not os.path.isfile(up):
                await self._upscale_photo(run_id, n, backend, up)
            local = self._local(run_id, f["url"])
            mode, scale = await asyncio.to_thread(self._paste_photo_file, run_id, n, att, local)
            if f.get("jpg"):
                await asyncio.to_thread(erp.save_quest_jpg, local, self._local(run_id, f["jpg"]), self.jpg_quality)
        except Exception as e:
            return jresp({"error": comfy.explain_comfy_error(str(e))}, 502)

        def mark(a):
            a["photo_pasted"] = mode
            t = int(time.time())
            for x in a.get("files", []):
                x["v"] = t                                     # 화면 캐시 무효화용 (경로는 그대로)
        a = await self._edit_attempt_locked(run_id, n, mark)
        return jresp({"ok": True, "mode": mode, "files": a["files"]})
