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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api
import comfy
import server
import workflow
from storage import InvalidRunPath, RunStore


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

    def test_compatibility_imports_use_single_implementation(self):
        self.assertIs(api.ComfyUI, comfy.ComfyUI)
        self.assertIs(comfy.LocalComfy, comfy.ComfyUI)
        self.assertIs(comfy.CloudComfy, comfy.ComfyCloud)


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
        factory = patch("server.comfy.make_backends", return_value={"fake": self.backend})
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
        for path in ["/", "/static/style.css", "/static/app.js", "/static/api.js",
                     "/static/poller.js", "/static/viewer.js"]:
            response = await self.client.get(path)
            self.assertEqual(response.status, 200, path)
            self.assertTrue(await response.read())
            if path.endswith(".js"):
                self.assertIn("javascript", response.content_type)
        cfg = await self.request_json("GET", "/api/config")
        self.assertFalse(cfg["openai_ready"])
        self.assertTrue(cfg["backends"][0]["ok"])

    async def test_generate_saves_results_and_viewer_copy(self):
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
        self.assertEqual(self.backend.fetch_count, 1)
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


if __name__ == "__main__":
    unittest.main()
