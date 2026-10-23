"""
웹 앱 · 라우트 표 (Pano360 Studio)
요청 처리는 handlers/, 외부 통신은 clients/, 작업 저장은 storage.py 에서 관리합니다.
"""

import os

from aiohttp import web

from . import ROOT
from .handlers import Studio
from .handlers.base import jresp
from .storage import InvalidRunPath


@web.middleware
async def path_errors(request, handler):
    try:
        return await handler(request)
    except InvalidRunPath as exc:
        return jresp({"error": str(exc)}, 400)


def create_app(cfg, keys) -> web.Application:
    st = Studio(cfg, keys)
    app = web.Application(client_max_size=200 * 1024 * 1024, middlewares=[path_errors])
    app.on_startup.append(st.startup)
    app.on_cleanup.append(st.cleanup)
    run = "/api/run/{run_id}"
    att = run + "/attempt/{n}"
    app.add_routes([
        web.get("/", st.index),
        web.get("/api/config", st.config),
        web.get("/api/models/check", st.models_check),
        web.get("/api/runs", st.list_runs),
        # 1~4단계: 업로드 · 분석 · 생성
        web.post("/api/upload", st.upload),
        web.post("/api/analyze", st.analyze),
        web.post("/api/control", st.control),
        web.post("/api/generate", st.generate),
        web.get(run, st.get_run),
        web.post(run + "/import", st.import_result),
        web.get(att, st.status),
        web.post(att + "/repaste", st.repaste_photo),
        web.post(att + "/to_viewer", st.to_viewer),
        # 5단계: 영상화 (일부만 움직임)
        web.post(att + "/motion", st.motion_plan),
        web.post(att + "/motion/update", st.motion_update),
        # 6단계: 사건 연출 · 360 영상
        web.post(att + "/event", st.event_plan),
        web.post(att + "/event/update", st.event_update),
        web.get(att + "/event/base", st.event_base),
        web.post(att + "/event/mask", st.event_mask),
        web.post(att + "/event/video", st.event_video_start),
        web.get(att + "/event/video", st.event_video_status),
        web.static("/runs", cfg["runs_dir"]),
        web.static("/static", os.path.join(ROOT, "static")),
    ])
    return app
