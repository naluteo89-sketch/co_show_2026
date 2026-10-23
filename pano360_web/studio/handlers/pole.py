"""
발밑·하늘 보정 (2단계): 1) 돌린 ERP 의 극점만 다시 그림 → 2) 되돌려 합성 → 8K 화질 개선 제출
"""

import asyncio

from PIL import Image

from .. import erp, llm, workflow
from .base import adir


class PoleMixin:
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
                return self._local(run_id, f["url"])
            raise RuntimeError(f"#{from_n} 시도에 결과 이미지가 없어요.")
        if meta.get("is_erp"):
            return self.store.path(run_id, "source" + meta["source_ext"])
        raise RuntimeError("보정할 360 이미지가 없어요. 생성 결과가 있는 시도를 골라 주세요.")

    async def _start_pole(self, run_id, n, meta, p, backend):
        pc = self._pole_cfg()
        base_path = self._pole_base_path(run_id, meta, p.get("from_attempt"))
        d = adir(n)
        path = lambda name: self.store.path(run_id, d, name)

        def prep():
            base = Image.open(base_path).convert("RGB")
            ww = int(pc["work_width"])
            work = base if base.width == ww else base.resize((ww, ww // 2), Image.LANCZOS)
            rot, mask = erp.pole_prepare(work, pc["nadir_cap"], pc["zenith_cap"], pc["feather"])
            work.save(path("pole_base.png"))
            rot.save(path("pole_rot.png"))
            mask.save(path("pole_mask.png"))
        await asyncio.to_thread(prep)

        rot_name = await self._upload_file(backend, path("pole_rot.png"), f"{run_id}_{d}_pole_rot.png")
        mask_name = await self._upload_file(backend, path("pole_mask.png"), f"{run_id}_{d}_pole_mask.png")
        prompts = p.get("pole_prompts") or llm.pole_prompts(meta.get("analysis") or {})
        g = {**self.cfg.get("gen_defaults", {}), **{k: v for k, v in (p.get("settings") or {}).items() if v}}
        ids = self.cfg["node_ids"]
        wf_api, seed = workflow.build_pole(
            workflow.load_template(self.cfg["workflow"]), image=rot_name, mask=mask_name,
            positive=prompts["positive"], negative=prompts["negative"],
            seed=p.get("seed1"), steps=g.get("steps", 25), cfg=g.get("cfg", 4.0), denoise=pc["denoise"],
            lora_strength=pc["lora_strength"], prefix=f"VR360/{run_id}_{d}",
            loader_ids=tuple(str(ids[k]) for k in ("unet", "lora", "clip", "vae")))
        extra = {"stage": 1, "pole": pc, "pole_prompts": prompts, "from_attempt": p.get("from_attempt")}
        return wf_api, {"seed1": seed, "seed2": None}, {"rot": "210"}, extra

    async def _pole_stage2(self, run_id, n, att, backend, st):
        """1단계 끝 → 되돌려 합성 → 2K 저장 → 8K 화질 개선 제출."""
        d = adir(n)
        path = lambda name: self.store.path(run_id, d, name)
        await self._save_output(backend, next(x for x in st["files"] if x["node"] == "210"), path("pole_rot_out.png"))
        pc = att["pole"]

        def merge():
            merged = erp.pole_merge(Image.open(path("pole_base.png")), Image.open(path("pole_rot_out.png")),
                                    pc["nadir_cap"], pc["zenith_cap"], pc["feather"])
            merged.save(path("result_2k.png"))
            return merged.size
        W, H = await asyncio.to_thread(merge)
        name = await self._upload_file(backend, path("result_2k.png"), f"{run_id}_{d}_pole_fixed.png")
        meta = self.store.read(run_id)
        wf = workflow.build_enhance(name, W, H, prefix=f"VR360/{run_id}_{d}", **self._enhance_kwargs(meta))
        pid = await self._submit_workflow(backend, wf, path("workflow_api_stage2.json"))
        att.update(stage=2, prompt_id=pid, out_nodes={"final": "15"})   # 중간 파일은 폴더에만 두고 결과는 한 장
