# _legacy — 웹에서 뺀 예전 모듈

`studio/` 실행 경로에서 import 하지 않는 파일입니다. 2026-10-21 효과 레이어·사건 영상 합성·cinemagraph 합성을 웹에서 빼면서 남은 코드로,
참고용으로만 보관합니다 (테스트 대상 아님, 그대로 실행되지 않을 수 있음). 필요 없으면 지워도 되고, git 기록에 남아 있습니다.

| 파일 | 원래 역할 |
|---|---|
| `cinemagraph.py` | 영역 영상 → ERP 합성, 루프·안정화 (쓰던 `ffmpeg_exe` 는 `studio/video/ffmpeg.py` 로 옮김) |
| `event.py` | 사건 영상 합성 (평면 → ERP) |
| `layers.py` | 효과 레이어 타임라인 합성 |
| `influence.py` | 영향 범위 흐름(optical flow) 왜곡 |
| `workflow_ui.py` | Wan 2.2 사건 영상 UI 워크플로 생성기 |
| `api.py` | 예전 `import api` 호환 진입점 |
