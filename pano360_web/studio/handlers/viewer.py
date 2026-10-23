"""
Unity 뷰어로 보내기: 최종 JPG(+사건 영상 세트)를 StreamingAssets 에 복사하고 photos.json 갱신
"""

import asyncio
import json
import os
import shutil

from ..video import compose
from .base import adir, jresp, resolve, route_ids, write_json


class ViewerMixin:
    async def to_viewer(self, req):
        """최종 JPG 를 Unity 뷰어 StreamingAssets 에 복사하고 photos.json 갱신.
        사건 영상(크롭 생성)이 있으면 세트로: X.jpg(사건이 정면에 오게 돌린 8K) + X_event.mp4 + X_event_mask.png
        → 뷰어가 사진 → (칠한 범위만) 영상 → 사진으로 재생."""
        run_id, n = route_ids(req)
        att = self.store.read(run_id)["attempts"][n - 1]
        f = next((x for x in att.get("files", []) if x.get("jpg")), None)
        if not f:
            return jresp({"error": "Quest 용 JPG 가 없어요."}, 400)
        vdir = resolve(self.cfg.get("viewer_dir", ""))
        if not os.path.isdir(vdir):
            return jresp({"error": f"뷰어 폴더가 없어요: {vdir}"}, 400)
        src = self._local(run_id, f["jpg"])
        vid = (att.get("event") or {}).get("video") or {}
        if vid.get("state") == "done" and vid.get("crop_mode") and (vid.get("audio_url") or vid.get("silent_url")):
            base = f"{run_id}_a{n:02d}"
            mp4 = self._local(run_id, vid.get("audio_url") or vid.get("silent_url"))
            names = [base + ".jpg", base + "_event.mp4", base + "_event_mask.png"]

            def write_set():
                compose.roll_image(src, os.path.join(vdir, names[0]), int(vid.get("shift", 0)),
                                   quality=self.jpg_quality, subsampling=0)
                shutil.copyfile(mp4, os.path.join(vdir, names[1]))
                shutil.copyfile(self.store.path(run_id, adir(n, "event"), "mask_front.png"), os.path.join(vdir, names[2]))
            await asyncio.to_thread(write_set)
        else:
            names = [os.path.basename(src)]
            await asyncio.to_thread(shutil.copyfile, src, os.path.join(vdir, names[0]))
        lst = os.path.join(vdir, "photos.json")
        try:
            with open(lst, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            data = {"files": []}
        for nm in names:
            if nm not in data["files"]:
                data["files"].append(nm)
        write_json(lst, data, indent=4)
        return jresp({"ok": True, "path": os.path.join(vdir, names[0]), "files": names, "count": len(data["files"])})
