# CO-SHOW 2026 · Pano360 Studio

사진 한 장으로 **Meta Quest 3용 360° 이미지와 "사건" 360 영상**을 만드는 프로젝트입니다.
웹(Pano360 Studio)에서 LLM 분석 → ComfyUI 생성 → 검수까지 하고, Unity 앱(Pano360 Viewer)으로 Quest 3에서 봅니다.

> 기준일 2026-10-23 · 1차 종료(`367f038e`) 이후 폴더 정리 반영

## 무엇을 하나요

```text
원본 사진
  → LLM(OpenAI)이 장면·조명·화각·수평선 분석 → 프롬프트 + 배치(yaw/pitch/hFOV)
  → ERP 컨트롤 이미지(빈 곳 #00ff00) → ComfyUI: FLUX.2 Klein 4B + 360 ERP 아웃페인트 LoRA (2048×1024)
  → 좌우 이음새 복원 → AI 업스케일 ×4 + 타일 리파인 → 8192×4096 + Quest용 JPG
  → (선택) 사건 영상: 움직일 범위를 칠함 → 그 범위만 LTX-2.3 로 생성 → 4K 360 영상에 합성
  → Unity 뷰어로 복사 → Quest 3: 사진 → (칠한 범위만) 사건 영상 → 다시 사진
```

| 단계 | 사람이 하는 일 | 코드가 하는 일 |
|---|---|---|
| 분석 | 메모·모델 선택, 결과 확인 | 사진 + JSON 스키마로 OpenAI 호출, 고정 문구(LoRA 트리거·공통 네거티브)와 조립 |
| 360 생성 | 배치·프롬프트 수정, 실행 위치 선택 | 노드 파일에 값 주입, 모델 이름 자동 매칭, 이음새 점수·세로선 제거 |
| 사건 영상 | 360 위에 움직일 범위 칠하기, 길이·LoRA 세기 | 범위를 정면으로 돌림 → 범위만 잘라 고해상도 생성 → 4K 360 합성(소리 포함) |
| Quest | 결과 고르기 → "Quest 뷰어에 추가" | 사진·영상·마스크 세트를 `StreamingAssets`에 복사, `photos.json` 갱신 |

## 폴더 구조

```text
co_show_2026/
├─ README.md
├─ pano360_web/                 웹 Studio (Python aiohttp + HTML/JS) → 자세한 설명: pano360_web/README.md
│  ├─ main.py · run.bat         실행 (http://127.0.0.1:7870)
│  ├─ config.json               모델·실행 위치·노드 파일 경로·화질 설정
│  ├─ studio/                   서버 패키지
│  │  ├─ server.py              aiohttp 앱 · 라우트 표
│  │  ├─ handlers/              기능별 요청 처리: generate · pole · photo · motion · event · viewer · info
│  │  ├─ clients/               credentials · openai_client · comfy (ComfyUI·Comfy Cloud, 모델 자동 매칭)
│  │  ├─ erp/                   projection · control · pole · paste · seam · regions (360 이미지 계산)
│  │  ├─ workflow/              core · generate · enhance · video (노드 파일 → API, 값 주입, 크롭 계획)
│  │  ├─ video/                 ffmpeg · compose (크롭 생성 영상 → 4K 360 합성)
│  │  ├─ llm.py                 LLM 지시문·스키마·프롬프트 조립 (영상 노드 프롬프트 포함)
│  │  └─ storage.py             작업 폴더 · meta.json
│  ├─ static/                   화면(index.html · app.js · event.js · video.js · viewer.js · dom.js …)
│  ├─ tests/                    서버·워크플로·폴링 회귀 검사
│  ├─ _legacy/                  웹에서 뺀 예전 모듈 (실행 경로에서 안 씀)
│  └─ runs/                     작업 기록 (저장소 제외)
├─ comfy_nodes/
│  ├─ image/test_quest3.json    360 이미지 생성 (8192×4096 출력, 웹 기본)
│  ├─ image/test.json           원본 워크플로 (2K 생성 + 이음새 복원 + Lanczos 4K)
│  └─ video/
│     ├─ event_cropgen_v1.json  사건 영상: 칠한 범위만 잘라 생성 (웹 기본)
│     ├─ retest.json            360 전체에 사각 마스크로 생성
│     ├─ quest360_flood_mask_inpaint_4k.json  마스크 그림 + 인페인팅
│     └─ experiments/           실험용 노드
├─ Pano360Viewer/               Unity 6 · Meta Quest 3 뷰어 → Pano360Viewer/README.md
├─ tools/erp_control_maker.html 단독 ERP 컨트롤 이미지 도구 (서버 studio/erp/control.py 와 같은 계산)
└─ docs/
   ├─ 2026 AIM 챌린지 … 기획서.md
   ├─ screenshots/              웹 화면 캡처
   ├─ analysis/                 화질·이음새·마스크·와이프 비교 이미지
   └─ patches/                  Unity 셰이더 변경 기록 (unity_video_filter.diff)
```

## 실행

```powershell
cd C:\co_show_2026\pano360_web
python -m pip install -r requirements.txt   # aiohttp, Pillow, numpy, opencv-python-headless
run.bat                                       # = python main.py, 브라우저가 http://127.0.0.1:7870 으로 열림
```

- API 키는 저장소 밖에서 읽습니다: `C:\keydata\gpt_key`(변수 `open_Aikey`), `C:\keydata\comfy_key`(변수 `comfy_key`). 같은 이름의 환경변수가 우선.
- 실행 위치(`config.json` → `backends`): **Comfy Cloud(기본)**, LAN ComfyUI 서버 `192.168.1.91:8000`, 내 PC `127.0.0.1:8188`.
- 사건 영상 합성에는 ffmpeg 가 필요합니다.
- 코드를 고친 뒤에는 `run.bat`을 다시 실행하고, 화면 오른쪽 위 버전 표시로 확인합니다.

## 주요 설정 (`pano360_web/config.json`)

| 항목 | 값 | 설명 |
|---|---|---|
| `workflow` | `../comfy_nodes/image/test_quest3.json` | 360 이미지 생성 노드 파일 |
| `video_template` | `../comfy_nodes/video/event_cropgen_v1.json` | 사건 영상 노드 파일 (크롭 생성) |
| `default_backend` | `comfycloud` | 기본 실행 위치 |
| `openai_model` | `gpt-6.1-sol` | 분석 모델 (`openai_models` 에서 화면 선택) |
| `keep_photo.enabled` | `false` | 8K 결과에 원본 사진을 다시 붙일지 (10-23 기본 꺼짐) |
| `refine` | denoise 0.3, 타일 1536 | LoRA 없이 8K 디테일 리파인 |
| `viewer_dir` | `../Pano360Viewer/Assets/StreamingAssets/Pano360` | Quest 뷰어로 복사할 곳 |

## 사건 영상 (크롭 생성 방식)

360 전체(4K)를 영상 모델에 넣으면 움직일 곳이 작게 그려지고, ComfyUI 에서 4K 프레임 수백 장을 붙이면 서버 메모리가 부족했습니다. 그래서 다음과 같이 나눴습니다.

1. **정면으로 돌리기**: 칠한 범위의 가로 중심(원형 평균)이 360 가운데에 오도록 사진·마스크를 돌림 (`studio/video/compose.py` `front_shift`). 360 은 좌우가 이어져 있어 화질 손실이 없습니다.
2. **범위만 생성**: 8K 원본에서 칠한 범위(+여유 12%)만 잘라 LTX-2.3 22B(distilled) + Seamless-Equirectangular LoRA 로 생성. 생성 크기 상한 2560×1088 (`studio/workflow/video.py` `crop_plan`).
3. **시간표 고정**: 길이에 맞춰 "0~15% 잔잔 → 50%까지 절정 → 85%까지 가라앉음 → 첫 화면과 같음" 문장을 프롬프트에 붙임 (`studio/llm.py` `event_timeline`).
4. **4K 합성**: 바탕은 8K 사진을 4K 로 줄인 정지 화면, 칠한 범위 안만 영상. 마지막 0.75초는 사진으로 서서히 돌아가서 마지막 프레임 = 첫 화면 (`studio/video/compose.py` `compose_crop_video`). 소리는 생성 영상의 것을 그대로 붙임.
5. **Quest 세트**: `X.jpg`(정면으로 돌린 8K) + `X_event.mp4` + `X_event_mask.png` 를 뷰어에 복사. 뷰어는 사진 → 범위 안만 영상 → 사진으로 재생하고, 끝날 때 범위 가장자리부터 불규칙하게 벗겨지듯 사진으로 돌아갑니다(와이프).

## 검증

```powershell
cd pano360_web
python -m unittest discover -s tests -v
node --test tests/poller.test.mjs
```

- 서버 테스트는 가짜 ComfyUI 백엔드와 임시 폴더를 사용해 실제 키·GPU 없이 돌아갑니다.
- 2026-10-23 결과: 360 생성 6회 성공(Comfy 서버), 좌우 이음새 점수 ×0.89~×1.78 (2 이하 양호). Comfy Cloud 2회는 LoRA 이름 검증 실패로 중단.

## 문서

- [웹 Studio 상세](pano360_web/README.md): 단계별 기능, 설정, 화질·이음새·극점 보정, 사건 영상 노드 값
- [Unity 뷰어](Pano360Viewer/README.md): 설치·빌드·조작, 360 영상·사건 영상 재생
- [기획서](docs/): 목표·실험·일정
- [ERP Outpaint LoRA](https://huggingface.co/nomadoor/flux-2-klein-4B-360-erp-outpaint-lora)
