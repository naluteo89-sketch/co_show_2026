# ComfyUI 노드 파일

ComfyUI 화면에 끌어다 놓으면 열리는 UI 포맷 워크플로입니다. 웹 Studio 는 이 파일을 그대로 읽어 값만 넣어 실행합니다
(경로는 `pano360_web/config.json` 의 `workflow` · `video_template`). 웹에서 쓰려면 최신 ComfyUI 에서 저장해 `widgets_values_named` 가 있어야 합니다.

## image/ — 360 이미지

| 파일 | 웹 사용 | 내용 |
|---|---|---|
| `test_quest3.json` | **기본** (`workflow`) | FLUX.2 Klein 4B + 360 ERP 아웃페인트 LoRA, 2048×1024 생성 → 이음새 복원 → AI 업스케일 → 8192×4096 |
| `test.json` | — | 원래 워크플로: 2K 생성 + 이음새 복원 + Lanczos 4K |

노드 번호를 바꾸면 `config.json` 의 `node_ids` 도 맞춰야 합니다.

## video/ — 사건 360 영상

| 파일 | 웹 사용 | 마스크 방식 |
|---|---|---|
| `event_cropgen_v1.json` | **기본** (`video_template`) | 칠한 범위만 잘라 고해상도 생성 → 서버가 4K 360 에 합성 (ImageCrop 1202 가 있으면 이 방식) |
| `quest360_flood_mask_inpaint_4k.json` | 선택 | 360 전체 + 칠한 마스크 그림(1162 LoadImage) 인페인팅 |
| `retest.json` | 선택 | 360 전체 + 사각 마스크(1134 SolidMask · 1136 MaskComposite) |

웹이 값을 넣는 노드 id 는 `pano360_web/studio/workflow/video.py` 맨 위 설명을 참고하세요.

### video/experiments/ — 실험용 (웹에서 안 씀)

| 파일 | 내용 |
|---|---|
| `pano360_cinemagraph_ltx23.json` | LTX-2.3 cinemagraph LoRA (영역 평면 화면 입력) |
| `pano360_wan22_5b_i2v.json` | Wan 2.2 TI2V 5B 이미지→영상 |
| `stream_wind_360_event_fixB.json` | 계곡·바람 장면 360 사건 영상 실험 |
| `region_1.png` | 위 노드 입력용 영역 화면 예시 |
