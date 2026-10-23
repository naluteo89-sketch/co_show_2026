"""
Pano360 Studio 실행 파일

    python main.py                 → 서버 시작 + 브라우저 자동으로 열림
    python main.py --host 0.0.0.0  → 같은 네트워크 다른 기기에서도 접속
    python main.py --no-browser    → 브라우저 안 열기

구성
    main.py      실행 (여기)
    studio/      서버 패키지 (구성은 studio/__init__.py 참고)
    static/      웹 화면
    config.json  설정 (키 경로, 모델, ComfyUI 주소, 노드 파일, 노드 번호)
    ../comfy_nodes/  ComfyUI 노드 파일 (이미지 생성 · 영상)
"""

import argparse
import asyncio
import json
import os
import sys
import webbrowser

from aiohttp import web

from studio import server
from studio.clients.credentials import resolve_keys

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "2026-10-20 keep-v3 (원본 사진 영역 유지 + 초록 이음새 + 이음새 리파인)"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["openai_model"] = os.environ.get("OPENAI_MODEL") or cfg.get("openai_model")
    for key, default in (("workflow", None), ("runs_dir", "runs")):
        v = cfg.get(key, default)
        cfg[key] = v if os.path.isabs(v) else os.path.normpath(os.path.join(HERE, v))
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Pano360 Studio")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    keys = resolve_keys(cfg, HERE)

    if not os.path.isfile(cfg["workflow"]):
        sys.exit(f"[pano360] 워크플로 파일이 없어요: {cfg['workflow']}")

    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    print(f"[pano360] 버전 {VERSION}")
    print(f"[pano360] {url}")
    print(f"[pano360] 모델 {cfg['openai_model']} · 워크플로 {cfg['workflow']}")
    print(f"[pano360] OpenAI 키: {'OK' if keys['openai'] else '없음'} / Comfy Cloud 키: {'OK' if keys['comfy_cloud'] else '없음'}")
    for n, b in cfg["backends"].items():
        print(f"[pano360] ComfyUI '{n}': {b.get('label', n)} → {b['url']}")

    cfg["version"] = VERSION
    app = server.create_app(cfg, keys)

    if not a.no_browser:
        async def open_browser(_app):
            async def later():
                await asyncio.sleep(1.0)
                webbrowser.open(url)
            asyncio.get_running_loop().create_task(later())
        app.on_startup.append(open_browser)

    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
