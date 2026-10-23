"""
Pano360 Studio 서버 (실행은 ../main.py)

    server.py    aiohttp 앱 · 라우트 표
    handlers/    기능별 요청 처리 (생성 · 극점 보정 · 사진 붙이기 · 영상화 · 사건 영상 · 뷰어 전달 · 정보)
    clients/     외부 통신 (키 · OpenAI · ComfyUI/Comfy Cloud)
    erp/         360 이미지 계산
    workflow/    ComfyUI 그래프 만들기
    video/       ffmpeg 영상 합성
    llm.py       LLM 지시문 · 스키마 · 프롬프트 조립
    storage.py   작업 폴더 · meta.json
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # pano360_web 폴더
