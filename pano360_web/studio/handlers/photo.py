"""
원본 사진 다시 붙이기 (생성 직후 자동 · 이미 만든 결과에 '원본 자리 화질 맞추기')
"""

import asyncio
import io
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


    async def text_keep(self, req):
        """이미 만든 결과: 분석의 글자 영역(text_regions · 간판·현판)만 원본 사진 픽셀로 되살림 → JPG 도 다시.
        예전 분석엔 text_regions 가 없으니 먼저 분석(/api/analyze)을 다시 해야 함 (생성 결과·프롬프트 기록은 그대로)."""
        run_id, n = route_ids(req)
        meta = self.store.read(run_id)
        att = meta["attempts"][n - 1]
        boxes = [r.get("box") for r in ((meta.get("analysis") or {}).get("text_regions") or []) if isinstance(r, dict)]
        if not boxes:
            return jresp({"error": "분석에 글자 영역이 없어요 (분석을 다시 하거나 글자가 없는 사진)."}, 400)
        f = next((x for x in att.get("files", []) if x.get("final")), None)
        if not f or not (att.get("placement") or {}).get("hfov"):
            return jresp({"error": "결과 이미지가 없어요."}, 400)
        tk = self.cfg.get("text_keep", {})
        local = self._local(run_id, f["url"])

        def work():
            erp.paste_text_regions(Image.open(local), self.source(run_id), boxes, *placement_args(att["placement"]),
                                   pad=float(tk.get("pad", 0.15)), soft=float(tk.get("soft", 0.35))).save(local)
            if f.get("jpg"):
                erp.save_quest_jpg(local, self._local(run_id, f["jpg"]), self.jpg_quality)
        await asyncio.to_thread(work)

        def mark(a):
            a["text_kept"] = len(boxes)
            t = int(time.time())
            for x in a.get("files", []):
                x["v"] = t
        a = await self._edit_attempt_locked(run_id, n, mark)
        return jresp({"ok": True, "boxes": len(boxes), "files": a["files"]})


    CLEAN_PROMPT = ("Remove all the people from this photo. Keep everything else exactly the same: the same place, "
                    "buildings, signs and their letters, ground, water, sky, light and colors.")

    async def clean_people(self, req):
        """원본 사진에서 사람 지우기 (생성 전 · 사진 속 사람은 생성 결과에 그대로 남으므로).
        Comfy 서버의 FLUX.2 Klein 편집(참조 이미지 + 지시문)으로 사람만 지운 사진을 만들어 원본 자리에 둠.
        원래 사진은 source_orig.* 로 보관. 끝나면 분석을 다시 해야 함 (배치·프롬프트는 새 사진 기준)."""
        run_id = req.match_info["run_id"]
        p = await req.json() if req.can_read_body else {}
        meta = self.store.read(run_id)
        _, backend, err = self._pick_backend(p.get("backend") or "cloud")
        if err:
            return err
        ext = meta["source_ext"]
        orig = self.store.path(run_id, "source_orig" + ext)
        src_path = self.store.path(run_id, "source" + ext)
        if not os.path.isfile(orig):
            import shutil
            shutil.copyfile(src_path, orig)
        img = erp.load_upright(orig).convert("RGB")
        b = io.BytesIO()
        img.save(b, "PNG")
        try:
            name = await backend.upload(b.getvalue(), f"{run_id}_clean_src.png")
            seed = int(p.get("seed") or time.time()) % 2_000_000_000
            g = {
                "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors", "weight_dtype": "default"}},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX2\\flux2-vae.safetensors"}},
                "4": {"class_type": "LoadImage", "inputs": {"image": name}},
                "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                                                                         "megapixels": float(p.get("megapixels", 2.0)), "resolution_steps": 16}},
                "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": p.get("prompt") or self.CLEAN_PROMPT, "clip": ["2", 0]}},
                "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]}},
                "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
                "10": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0],
                                                            "latent_image": ["6", 0], "seed": seed, "steps": 4, "cfg": 1.0,
                                                            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
                "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
                "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": f"VR360/{run_id}_clean"}},
            }
            pid = await self._submit_workflow(backend, g, self.store.path(run_id, "workflow_clean.json"))
            t0 = time.time()
            while True:
                st = await backend.status(pid, ["12"])
                if st["state"] == "done":
                    break
                if st["state"] == "error" or time.time() - t0 > 600:
                    raise RuntimeError(st.get("error") or "사람 지우기가 너무 오래 걸려요")
                await asyncio.sleep(2)
            tmp = self.store.path(run_id, "source_clean.png")
            await self._save_output(backend, next(x for x in st["files"] if x["node"] == "12"), tmp)
        except Exception as e:
            return jresp({"error": comfy.explain_comfy_error(str(e))}, 502)

        def put():                                         # 원래 크기로 되돌려 원본 자리에
            c = Image.open(tmp).convert("RGB").resize(img.size, Image.LANCZOS)
            c.save(self.store.path(run_id, "source.png"))
            if ext != ".png" and os.path.isfile(src_path):
                os.remove(src_path)
        await asyncio.to_thread(put)
        meta = self.store.read(run_id)
        meta.update(source_ext=".png", people_cleaned=True)
        self.store.write(run_id, meta)
        return jresp({"ok": True, "size": list(img.size)})
