"""외부 API 없이 업로드부터 결과 저장까지 검증한다."""

import asyncio
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp.test_utils import TestClient, TestServer
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import erp, llm, server, workflow
from studio.clients import comfy
from studio.storage import InvalidRunPath, RunStore
from studio.video import compose

NODES = ROOT.parent / "comfy_nodes"


def png(size=(64, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, (60, 100, 140)).save(buf, "PNG")
    return buf.getvalue()


class FakeBackend:
    def __init__(self):
        self.submitted = []
        self.fetch_count = 0
        self.closed = False

    async def ping(self):
        return True

    async def upload(self, data, name):
        Image.open(io.BytesIO(data)).verify()
        return name

    async def submit(self, graph):
        await asyncio.sleep(0.01)  # 동시에 들어온 HTTP 요청이 겹칠 수 있게 한다.
        self.submitted.append(copy.deepcopy(graph))
        return str(len(self.submitted))

    async def status(self, pid, node_ids):
        return {"state": "done", "files": [{"node": str(n), "filename": "fake.png"} for n in node_ids]}

    async def fetch(self, ref):
        self.fetch_count += 1
        return png()

    async def close(self):
        self.closed = True


class StoreTests(unittest.TestCase):
    def test_roundtrip_and_path_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            Path(store.path("run")).mkdir()
            store.write("run", {"name": "연못", "attempts": []})
            self.assertEqual(store.read("run")["name"], "연못")
            for rid, parts in [("../outside", ()), ("C:outside", ()), ("run", ("../../outside",)),
                               ("run", (str(Path(tmp).parent / "outside"),))]:
                with self.subTest(rid=rid, parts=parts), self.assertRaises(InvalidRunPath):
                    store.path(rid, *parts)

    def test_failed_serialization_keeps_previous_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            folder = Path(store.path("run"))
            folder.mkdir()
            store.write("run", {"attempts": [1]})
            with self.assertRaises(TypeError):
                store.write("run", {"invalid": object()})
            self.assertEqual(store.read("run"), {"attempts": [1]})
            self.assertEqual([p.name for p in folder.iterdir()], ["meta.json"])


class WorkflowTests(unittest.TestCase):
    def test_build_preserves_template_and_photo_mask(self):
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        template = workflow.load_template(ROOT / cfg["workflow"])
        original = copy.deepcopy(template)
        prompts = dict(positive="scene", negative="bad", seam_positive="seam", seam_negative="bad seam")
        graph, seeds = workflow.build(template, cfg["node_ids"], control_image="control.png",
                                      prompts=prompts, seed1=17, seed2=23, keep_mask="mask.png")
        self.assertEqual(template, original)
        self.assertEqual(seeds, {"seed1": 17, "seed2": 23})
        self.assertNotIn(str(cfg["node_ids"]["save_2k"]), graph)
        main = graph[str(cfg["node_ids"]["sampler_main"])]["inputs"]
        mask = graph[main["latent_image"][0]]
        self.assertEqual(mask["class_type"], "SetLatentNoiseMask")
        self.assertEqual(graph[mask["inputs"]["mask"][0]]["inputs"]["image"], "mask.png")
        for item in graph.values():
            for value in item["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    self.assertIn(value[0], graph)

class StudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.cfg.update(runs_dir=str(self.root / "runs"), viewer_dir=str(self.root / "viewer"),
                        workflow=str((ROOT / self.cfg["workflow"]).resolve()),
                        erp_size=[64, 32], refine={"enabled": False}, tone_match=0,
                        backends={"fake": {"url": "http://unused", "type": "comfyui"}})
        self.cfg["quest"]["final_width"] = 64
        self.cfg["pole"]["work_width"] = 64
        (self.root / "viewer").mkdir()
        self.backend = FakeBackend()
        factory = patch("studio.clients.comfy.make_backends", return_value={"fake": self.backend})
        factory.start()
        self.addCleanup(factory.stop)
        app = server.create_app(self.cfg, {"openai": "", "comfy_cloud": ""})
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def request_json(self, method, url, **kwargs):
        response = await self.client.request(method, url, **kwargs)
        data = await response.json()
        self.assertEqual(response.status, 200, data)
        return data

    async def upload(self, size=(64, 32)):
        form = aiohttp.FormData()
        form.add_field("image", png(size), filename="source.png", content_type="image/png")
        return await self.request_json("POST", "/api/upload", data=form)

    async def test_static_modules_and_config(self):
        static = ROOT / "static"
        files = [p.relative_to(static).as_posix() for p in static.rglob("*")
                 if p.is_file() and p.suffix in (".js", ".css", ".html") and "_old" not in p.parts]
        self.assertIn("js/main.js", files)
        for path in ["/"] + [f"/static/{f}" for f in files]:
            response = await self.client.get(path)
            self.assertEqual(response.status, 200, path)
            self.assertTrue(await response.read())
            if path.endswith(".js"):
                self.assertIn("javascript", response.content_type)
        cfg = await self.request_json("GET", "/api/config")
        self.assertFalse(cfg["openai_ready"])
        self.assertTrue(cfg["backends"][0]["ok"])

    async def test_generate_saves_results_and_viewer_copy(self):
        self.cfg["keep_photo"]["enabled"] = True           # 원본 사진 보존 경로까지 검사 (config 기본값은 꺼짐)
        uploaded = await self.upload((48, 32))
        rid = uploaded["run_id"]
        result = await self.request_json("POST", "/api/generate", json={
            "run_id": rid, "backend": "fake", "seed1": 1, "seed2": 2,
            "placement": {"yaw": 0, "pitch": 0, "hfov": 70, "vfov": None},
            "prompts": dict(positive="scene", negative="bad", seam_positive="seam", seam_negative="bad"),
        })
        status_url = f"/api/run/{rid}/attempt/{result['attempt']}"
        a, b = await asyncio.gather(self.request_json("GET", status_url), self.request_json("GET", status_url))
        self.assertEqual(a, b)
        self.assertEqual(a["state"], "done")
        self.assertTrue(a["photo_pasted"])
        self.assertEqual(self.backend.fetch_count, 3)    # 동시 조회여도 결과·업스케일 원본·진단 2K 를 한 번씩만 받음
        self.assertEqual(len(a["files"]), 1)
        self.assertEqual((await self.client.get(a["files"][0]["jpg"])).status, 200)
        for _ in range(2):
            copied = await self.request_json("POST", status_url + "/to_viewer", json={})
            self.assertEqual(copied["count"], 1)
        self.assertTrue(Path(copied["path"]).is_file())
        self.assertTrue((self.root / "runs" / rid / "a01" / "workflow_api.json").exists())
        self.assertEqual(len(await self.request_json("GET", "/api/runs")), 1)

    async def test_concurrent_attempts_and_two_stage_pole(self):
        rid = (await self.upload())["run_id"]
        payload = {"run_id": rid, "backend": "fake", "mode": "enhance"}
        results = await asyncio.gather(*(self.request_json("POST", "/api/generate", json=payload) for _ in range(2)))
        self.assertEqual(sorted(x["attempt"] for x in results), [1, 2])
        pole = await self.request_json("POST", "/api/generate", json={**payload, "mode": "pole"})
        url = f"/api/run/{rid}/attempt/{pole['attempt']}"
        first, second = await asyncio.gather(self.request_json("GET", url), self.request_json("GET", url))
        self.assertEqual(first["stage"], 2)
        self.assertEqual(second["state"], "done")
        self.assertEqual(len(self.backend.submitted), 4)  # enhance 2회 + pole 두 단계
        self.assertTrue((self.root / "runs" / rid / "a03" / "workflow_api_stage2.json").exists())

    async def test_invalid_path_and_failed_upload(self):
        bad = await self.client.post("/api/control", json={"run_id": "../outside", "placement": {}})
        self.assertEqual(bad.status, 400)
        bad = await self.client.post("/api/generate", json={"run_id": "../outside"})
        self.assertEqual(bad.status, 400)
        bad = await self.client.post("/api/upload", data={})
        self.assertEqual(bad.status, 400)

    async def test_cleanup_closes_backend(self):
        await self.client.close()
        self.assertTrue(self.backend.closed)



class ImageAndVideoTests(unittest.TestCase):
    """영상화: 일부만 움직이는 영역 · 프롬프트."""
    def test_photo_paste_keeps_detail_when_photo_is_small(self):
        self.assertEqual(erp.photo_paste_mode(8192, 1200, 72)[0], "detail")      # 1°당 16.7px < 22.8px
        self.assertEqual(erp.photo_paste_mode(8192, 4000, 72)[0], "original")
        rng = np.random.default_rng(0)
        fin = Image.fromarray((rng.random((256, 512, 3)) * 255).astype(np.uint8))
        photo = Image.new("RGB", (60, 40), (200, 100, 50))
        out = np.asarray(erp.paste_photo(fin, photo, 0, 0, 60, mode="detail"), np.float32)
        f = np.asarray(fin, np.float32)
        self.assertTrue(np.array_equal(out[:, :100], f[:, :100]))            # 사진 밖은 그대로
        c = out[118:138, 246:266]
        self.assertLess(abs(c[..., 0].mean() - 200), 25)                     # 사진 자리 색은 원본 사진
        self.assertGreater(c.std(), 5)                                        # 결과의 디테일은 남음

    def test_event_prompt(self):
        e = llm.clamp_event({"yaw": 400, "pitch": 80, "hfov": 10, "calm_start": "calm", "event": "water surges",
                             "peak": "spray fills view", "calm_end": "settles", "keep": "bridge"})
        self.assertEqual((e["yaw"], e["pitch"], e["hfov"]), (40.0, 40.0, 60.0))
        pr = llm.event_prompts(e)
        self.assertIn("begins calm", pr["positive"]); self.assertIn("water surges", pr["positive"])
        self.assertIn("{event}", llm.LTX360_POS)

    def test_video_node_file_filled_with_web_values(self):
        import json, os
        here = NODES / "video"
        rt = workflow.load_template(os.path.join(here, "retest.json"))
        ui, seed, info = workflow.fill_video_node(rt, image="p.jpg", positive="P", negative="N", seconds=5, rect=(300, 200, 900, 400))
        g, saves = workflow.video_node_api(ui)
        self.assertEqual(info["mode"], "rect")
        self.assertEqual((g["1134"]["inputs"]["width"], g["1134"]["inputs"]["height"]), (900, 400))   # 칠한 범위 좌표
        self.assertEqual((g["1136"]["inputs"]["x"], g["1136"]["inputs"]["y"]), (300, 200))
        self.assertEqual((g["1131"]["inputs"]["image"], g["1128"]["inputs"]["text"]), ("p.jpg", "P"))
        self.assertEqual(g["1108"]["inputs"]["length"], 121)                 # 계산 노드 → 숫자
        self.assertTrue(saves[0]["audio"])
        for nd in g.values():
            for v in nd["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                    self.assertIn(v[0], g)
        mi = workflow.load_template(os.path.join(here, "quest360_flood_mask_inpaint_4k.json"))
        ui, _, info = workflow.fill_video_node(mi, image="p.jpg", positive="P", negative="N", mask_image="m.png")
        self.assertEqual(info["mode"], "image")
        self.assertEqual(workflow.video_node_api(ui)[0]["1162"]["inputs"]["image"], "m.png")

    def test_crop_plan_from_painted_mask(self):
        from PIL import Image, ImageDraw
        m = Image.new("L", (2048, 1024)); ImageDraw.Draw(m).rectangle([250, 380, 1300, 790], fill=255)
        c = workflow.crop_plan(m, (8192, 4096))
        x, y, w, h = c["mask"]
        self.assertTrue(x <= 250 and y <= 380 and x + w >= 1300 and y + h >= 790)       # 칠한 범위 + 여유
        gw, gh = c["gen"]
        self.assertTrue(gw % 64 == 0 and gh % 64 == 0 and gw * gh <= workflow.CROP_MAX_PIXELS)
        self.assertAlmostEqual(w / h, gw / gh, delta=0.02)                              # 생성 크기와 같은 비율
        self.assertEqual(c["src"], [x * 4, y * 4, w * 4, h * 4])                         # 8K 원본 좌표
        self.assertEqual(c["out"], [x * 2, y * 2, w * 2, h * 2])                         # 4K 360 좌표
        self.assertIsNone(workflow.crop_plan(Image.new("L", (2048, 1024)), (8192, 4096)))
        wide = Image.new("L", (2048, 1024)); ImageDraw.Draw(wide).rectangle([0, 500, 2047, 600], fill=255)
        with self.assertRaises(ValueError):
            workflow.crop_plan(wide, (8192, 4096))

    def test_crop_video_node_filled(self):
        import json, os
        path = NODES / "video" / "event_cropgen_v1.json"
        t = workflow.load_template(path)
        self.assertTrue(workflow.video_has_crop(t))
        plan = {"mask": [100, 300, 1200, 510], "src": [400, 1200, 4800, 2040], "gen": [2560, 1088],
                "gen_lo": [1280, 544], "out": [200, 600, 2400, 1020]}
        ui, _, info = workflow.fill_video_node(t, image="p.jpg", positive="P", negative="N", seconds=6,
                                               mask_image="m.png", crop=plan)
        g, saves = workflow.video_node_api(ui)
        self.assertEqual(info["mode"], "image")
        self.assertEqual(g["1162"]["inputs"]["image"], "m.png")
        self.assertEqual([g["1202"]["inputs"][k] for k in ("x", "y", "width", "height")], plan["src"])
        self.assertEqual([g["1208"]["inputs"][k] for k in ("x", "y", "width", "height")], plan["mask"])
        self.assertEqual((g["1203"]["inputs"]["width"], g["1204"]["inputs"]["height"]), (2560, 544))
        self.assertNotIn("1221", g)                                   # 4K 360 붙이기는 서버(ffmpeg)가 함 → 노드는 꺼짐
        self.assertEqual(g["1154"]["inputs"]["images"], ["1217", 0])   # 저장되는 건 칠한 범위 영상
        self.assertEqual([ui_n["widgets_values_named"]["x"] for ui_n in ui["nodes"] if ui_n["id"] == 1221], [200])
        self.assertTrue(saves)
        for nd in g.values():
            for v in nd["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                    self.assertIn(v[0], g)
        with self.assertRaises(ValueError):
            workflow.fill_video_node(t, image="p.jpg", positive="P", negative="N", mask_image="m.png")

    def test_compose_crop_video_into_360(self):
        import shutil, subprocess, tempfile, os
        from PIL import ImageDraw
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg 없음")
        d = tempfile.mkdtemp()
        crop_mp4, out = os.path.join(d, "c.mp4"), os.path.join(d, "o.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=320x192:rate=24:duration=1",
                        "-pix_fmt", "yuv420p", crop_mp4], check=True)
        Image.new("RGB", (2048, 1024), (0, 0, 255)).save(os.path.join(d, "s.jpg"))
        m = Image.new("L", (2048, 1024)); ImageDraw.Draw(m).rectangle([600, 400, 900, 600], fill=255); m.save(os.path.join(d, "m.png"))
        info = compose.compose_crop_video(crop_mp4, os.path.join(d, "s.jpg"), os.path.join(d, "m.png"),
                                                {"mask": [560, 380, 400, 240], "out": [1120, 760, 800, 480]}, out,
                                                out_w=1024, audio=False)
        self.assertEqual((info["frames"], info["size"]), (24, [1024, 512]))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        f = np.frombuffer(raw, np.uint8).reshape(512, 1024, 3)
        self.assertGreater(f[250, 375, 0], 200)        # 칠한 곳 = 영상(빨강)
        self.assertGreater(f[50, 50, 2], 200)          # 밖 = 사진(파랑)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.05", "-i", out, "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        last = np.frombuffer(raw, np.uint8)[-512 * 1024 * 3:].reshape(512, 1024, 3)
        self.assertGreater(last[250, 375, 2], 200)     # 마지막 프레임 = 첫 화면(사진) — 뷰어 전환이 자연스러움

    def test_front_shift_puts_event_in_front(self):
        import tempfile, os
        from PIL import ImageDraw
        d = tempfile.mkdtemp()
        for box, cx in (([300, 400, 500, 600], 400), ([1900, 400, 2047, 600], None)):
            m = Image.new("L", (2048, 1024)); ImageDraw.Draw(m).rectangle(box, fill=255)
            if cx is None:                                   # 좌우 끝에 걸친 범위
                ImageDraw.Draw(m).rectangle([0, 400, 100, 600], fill=255); cx = 2000
            p = os.path.join(d, "m.png"); m.save(p)
            sh = compose.front_shift(p)
            self.assertLessEqual(abs((cx + sh) % 2048 - 1024), 30)          # 돌린 뒤 중심이 정면(1024)
            out = os.path.join(d, "r.png")
            compose.roll_image(p, out, sh)
            col = np.asarray(Image.open(out)).sum(0)
            self.assertGreater(col[1024], 0)

    def test_event_timeline_fits_clip(self):
        t = llm.event_timeline(6)
        self.assertIn("0-0.9s", t); self.assertIn("5.1-6s", t); self.assertIn("finishes inside the clip", t)

    def test_ambient_prompt_and_live_plan(self):
        import workflow, event_compose
        self.assertIn("loops", workflow.ambient_timeline(6))
        self.assertIn("ordinary ongoing motion", workflow.event_timeline(6))     # 사건 전에도 평소 흐름은 보이게
        pl = event_compose.live_plan(145, 145, 24, calm_seconds=8, fade=12)
        self.assertEqual(pl["loop_len"], 133)
        self.assertEqual(pl["total"], pl["event_start"] + 145)
        self.assertGreaterEqual(pl["event_start"], 6 * 24)

    def test_build_live_wraps_seamlessly(self):
        import event_compose, subprocess, tempfile, os
        exe = event_compose.ffmpeg_exe()
        with tempfile.TemporaryDirectory() as d:
            a, e, o = (os.path.join(d, x) for x in ("a.mp4", "e.mp4", "o.mp4"))
            subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x96:r=24:d=4",
                            "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", a], check=True)
            subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=160x96:r=24:d=3",
                            "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", e], check=True)
            pl = event_compose.build_live(a, e, o, calm_seconds=5, audio=False)
            fr = list(event_compose._decode(o, (160, 96), exe))
            self.assertEqual(len(fr), pl["total"])
            step = np.median([np.abs(fr[i].astype(float) - fr[i - 1]).mean() for i in range(1, pl["event_start"])])
            wrap = np.abs(fr[0].astype(float) - fr[-1]).mean()
            self.assertLess(wrap, step * 2.5 + 2)                                # 끝 → 처음이 한 프레임 움직임 정도

    def test_seam_second_pass_and_paste_harmonize(self):
        import json
        t = workflow.load_template(NODES / "image" / "test_quest3.json")
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        pr = {k: "x" for k in ("positive", "negative", "seam_positive", "seam_negative")}
        api_, _ = workflow.build(t, cfg["node_ids"], control_image="c.png", prompts=pr, settings={"steps": 4, "cfg": 1.0})
        self.assertEqual(api_["916"]["inputs"]["denoise"], 0.6)                  # 이음새 2차: 좁은 띠를 약하게
        self.assertEqual(api_["105"]["inputs"]["image"], ["920", 0])             # 되돌리기 전에 2차 결과 사용
        # 이음새 노이즈 마스크: 띠 안은 1, 바깥으로만 부드럽게 (1차 32px · 2차 16px @2K)
        self.assertEqual(api_["82"]["inputs"]["mask"], ["934", 0])
        self.assertEqual(api_["914"]["inputs"]["mask"], ["937", 0])
        self.assertEqual((api_["933"]["inputs"]["left"], api_["934"]["inputs"]["x"]), (32, 768 - 32))
        self.assertEqual(api_["932"]["inputs"]["width"], 512 + 64)
        self.assertEqual((api_["936"]["inputs"]["left"], api_["937"]["inputs"]["x"]), (16, 928 - 16))
        # 2차 프롬프트엔 초록 채우기 문장이 없음 (1차는 그대로)
        self.assertEqual(api_["915"]["inputs"]["conditioning"], ["938", 0])
        self.assertNotIn("green", api_["938"]["inputs"]["text"])
        self.assertTrue(api_["77"]["inputs"]["text"].startswith("Fill the green"))
        off, _ = workflow.build(t, cfg["node_ids"], control_image="c.png", prompts=pr,
                                settings={"steps": 4, "cfg": 1.0, "seam_noise_feather": 0})
        self.assertEqual(off["82"]["inputs"]["mask"], ["102", 0])                # 0 이면 예전 딱딱한 마스크
        # 사진 붙이기: 경계 바깥 생성 부분 톤이 사진 쪽으로 이어짐 (사각형 경계 단차 감소)
        fin = Image.new("RGB", (512, 256), (90, 90, 90))
        ph = Image.new("RGB", (64, 64), (200, 200, 200))
        pasted = erp.photo_composite(fin, ph, 0, 0, 40, None, feather=0.05, color_strength=0.0)
        h = np.asarray(erp.harmonize_paste(fin, pasted, ph.size, 0, 0, 40), np.float32)
        p = np.asarray(pasted, np.float32)
        y, xe = 128, 256 + int(512 * 20 / 360) + 3                               # 사진 오른쪽 경계 바로 바깥
        self.assertGreater(h[y, xe, 0], p[y, xe, 0] + 10)
        self.assertAlmostEqual(float(h[y, 256, 0]), float(p[y, 256, 0]), delta=3)  # 사진 가운데는 원본 그대로

    def test_wrap_line_removed(self):
        import tempfile, os
        from PIL import Image
        rng = np.random.default_rng(0)
        img = (rng.random((256, 1024, 3)) * 60 + 80).astype(np.uint8)
        img[:, 2:5] = [200, 120, 20]                                    # 이음새 오른쪽에 주황 세로선
        p = os.path.join(tempfile.mkdtemp(), "r.png"); Image.fromarray(img).save(p)
        cols = erp.fix_wrap_line(p)
        self.assertTrue(cols and min(cols) <= 2 and max(cols) >= 4)
        out = np.asarray(Image.open(p)).astype(np.float32)
        self.assertLess(abs(out[:, 3, 2].mean() - out[:, 30, 2].mean()), 15)   # 파랑 부족이 사라짐
        self.assertEqual(erp.fix_wrap_line(p), [])                         # 선 없는 그림은 그대로
        clean = os.path.join(os.path.dirname(p), "c.png"); Image.fromarray((rng.random((256, 1024, 3)) * 255).astype(np.uint8)).save(clean)
        self.assertEqual(erp.fix_wrap_line(clean), [])

    def test_wrap_tone_matched(self):
        import tempfile, os
        from PIL import Image
        img = np.full((128, 1024, 3), 100, np.uint8); img[:, 512:] = 140        # 이음새 양쪽 밝기 다름
        p = os.path.join(tempfile.mkdtemp(), "t.png"); Image.fromarray(img).save(p)
        self.assertGreater(erp.match_wrap_tone(p, width=256), 30)
        out = np.asarray(Image.open(p)).astype(np.float32)
        self.assertLess(abs(out[64, 0, 0] - out[64, -1, 0]), 3)                    # 이음새에서 같아짐
        self.assertEqual(int(out[64, 400, 0]), 100)                                # 멀리는 그대로

    def test_photo_upscale_ids_do_not_clash_with_photo_lock(self):
        api = {"31": {"class_type": "KSampler", "inputs": {"latent_image": ["951", 0]}},
               "950": {"class_type": "LoadImageMask", "inputs": {}}, "951": {"class_type": "SetLatentNoiseMask", "inputs": {}}}
        workflow.add_photo_upscale(api, "p.png", "x")
        self.assertEqual(api["951"]["class_type"], "SetLatentNoiseMask")      # 사진 잠금 노드가 덮이지 않음

    def test_region_crop_mask_and_prompt(self):
        W, H = 512, 256
        img = Image.new("RGB", (W, H), (40, 40, 40))
        img.paste((255, 0, 0), (W // 2 - 10, H // 2 - 10, W // 2 + 10, H // 2 + 10))   # yaw 0, pitch 0 에 빨간 점
        w, h = 128, 80
        crop = erp.perspective(img, 0, 0, 40, w, h)
        self.assertEqual(crop.getpixel((w // 2, h // 2)), (255, 0, 0))
        self.assertNotEqual(erp.perspective(img, 180, 0, 40, w, h).getpixel((w // 2, h // 2)), (255, 0, 0))
        r = llm.clamp_region({"name_ko": "물", "element": "water", "yaw": 200, "pitch": 0, "hfov": 60,
                              "orientation": "x", "motion": "ripples", "speed": "?", "strength": 9, "keep_still": "rocks"})
        self.assertEqual((r["yaw"], r["orientation"], r["speed"], r["strength"]), (-160, "landscape", "slow", 3.0))
        mask = erp.region_mask([r], W=256)
        self.assertGreater(sum(mask.getdata()) / 255, 0)                 # 일부만 움직임
        self.assertLess(sum(mask.getdata()) / 255, 256 * 128 * 0.5)
        pr = llm.motion_prompts(r, "overcast daylight")
        self.assertTrue(pr["positive"].startswith(llm.MOTION_TRIGGER))
        self.assertIn("Keep the existing image exactly", pr["positive"])     # 원본 유지 · 360 파노라마 강조
        self.assertIn("360 degree panorama", pr["positive"])
        self.assertIn("camera movement", pr["negative"])
        self.assertIn("Only the water moves", llm.motion_prompts({**r, "spread": "none"})["positive"])



if __name__ == "__main__":
    unittest.main()
