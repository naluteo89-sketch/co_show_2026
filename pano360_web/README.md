# Pano360 Studio

사진 한 장 → **LLM(OpenAI)이 프롬프트·배치 분석** → ERP 컨트롤 이미지 → **ComfyUI(로컬/Comfy Cloud)** → 360 이미지

프롬프트를 사람이 매번 쓰는 대신, LLM이 사진을 보고 재료(장면·조명·네거티브·이음새 요소·수평선·화각)를 JSON으로 주면
서버가 고정 문구(LoRA 트리거, 조명 일관성, 공통 네거티브)와 합쳐 `test_quest3.json` 워크플로에 넣어 실행합니다.

## 흐름
```
[브라우저 :7870]
  1 사진 업로드 ─▶ EXIF 화각 읽기
  2 LLM 분석   ─▶ OpenAI Responses API (이미지 + JSON 스키마) ─▶ 프롬프트 4종 + yaw/pitch/hFOV
  3 확인·수정   ─▶ 컨트롤 이미지 미리보기 (erp_control_maker.html 과 동일 계산)
  4 생성        ─▶ 컨트롤 PNG 업로드 → test_quest3.json 을 API 포맷으로 변환·주입 → /prompt
  5 결과        ─▶ 최종 360 한 장 (8K PNG + Quest용 JPG) + 브라우저 360 뷰어
```
모든 시도는 `runs/<날짜-시간>/meta.json`에 프롬프트·배치·시드·결과와 함께 남아서 비교하고 다시 불러올 수 있어요.

## 설치·실행 (Windows)
1. `pano360_web` 폴더를 `C:\co_show_2026\` 안에 둡니다 (워크플로는 `../test_quest3.json`).
2. Python 3.10 이상을 사용합니다. `pano360_web` 폴더에서 `python -m pip install -r requirements.txt`로 의존성을 설치합니다.
   ComfyUI 파이썬을 쓰는 경우에도 aiohttp, Pillow, numpy가 설치되어 있는지 확인합니다.
3. 키는 `C:\keydata`에서 자동으로 읽어요 (`config.json`):
   - `openai_key_file`: `C:\keydata\gpt_key` · 변수 `open_Aikey`
   - `comfy_cloud_key_file`: `C:\keydata\comfy_key` · 변수 `comfy_key`
   - 파일 내용이 키 한 줄이어도, `open_Aikey = "..."` 형식이어도 돼요. 확장자 없으면 .txt/.env/.key/.py 도 찾아요.
   - 같은 이름의 환경변수가 있으면 그게 우선이에요.
   그다음 `python main.py` 또는 `run.bat` 더블클릭 (브라우저가 자동으로 열려요). 시작 로그에 `OpenAI 키: OK` 가 떠야 해요.
4. 브라우저에서 `http://127.0.0.1:7870`
   - 다른 기기에서 접속하려면 `python main.py --host 0.0.0.0`

## config.json
| 항목 | 설명 |
|---|---|
| `openai_model` | 분석에 쓸 모델. 이미지 입력 + Structured Outputs 지원 모델이어야 함. 환경변수 `OPENAI_MODEL`로 덮어쓰기 가능 |
| `backends` | 이미지 생성할 ComfyUI 목록. `type: comfyui` = 일반 ComfyUI 서버 (기본: `cloud` → `http://192.168.1.91:8000`, `local` → `http://127.0.0.1:8188`). 공식 Comfy Cloud를 쓰려면 `{"type": "comfy_cloud", "url": "https://cloud.comfy.org"}` 추가 (키: `comfy_cloud_key_file`) |
| `workflow` | 실행할 워크플로 (UI 포맷 그대로, `widgets_values_named` 필요) |
| `node_ids` | 값을 넣을 노드 번호. 워크플로를 고치면 여기도 맞추기 |
| `default_yaw` | 사진을 붙일 가로 방향 (기본 -20°, 기존 워크플로 값) |

## Meta Quest 3 기준 (test_quest3.json)
| 단계 | 설정 | 이유 |
|---|---|---|
| 생성 | 2048×1024 (LoRA 학습 해상도 유지) | 해상도를 올려 생성하면 구도·이음새가 무너짐 |
| 이음새 패딩 | 좌우 128px 이어붙인 뒤 업스케일 → 잘라냄 | 업스케일러가 좌우 끝을 따로 처리해서 뒤쪽에 세로선이 생기는 것 방지 |
| AI 업스케일 x4 | NMKD-Siax → Nomos → Remacri → RealESRGAN → UltraSharp 순으로 서버에 있는 것 자동 선택 | 두 눈이 같은 화면을 크게 보므로 과샤픈·헤일로가 바로 티 남 → 자연스러운 계열 우선 |
| 블렌드 | AI 80% + Lanczos 20% | AI 특유의 질감·테두리 완화 |
| 디더 | 노이즈 0.004 | Quest LCD에서 하늘 계조 끊김(밴딩) 완화 |
| 최종 | **8192×4096**, JPG 95 (4:4:4) | 8K ≈ 22.8 px/° ↔ Quest 3 약 25 PPD. 뷰어가 밉맵 없이 LOD0 샘플링이라 16K는 반짝임(에일리어싱), 4K는 흐림 |

결과 화면의 **Quest 뷰어에 추가**를 누르면 JPG가 `Pano360Viewer/Assets/StreamingAssets/Pano360/`에 복사되고 `photos.json`이 갱신돼요 → Unity에서 Build And Run.
업스케일 순서는 `config.json`의 `model_fallbacks`에서 바꿀 수 있어요.

## 화질 개선 메모 (2026-10-20 점검)
기존 결과(2048×1024)에서 확인된 문제와 대응:
| 문제 | 원인 추정 | 대응 |
|---|---|---|
| 기름그림 같은 과한 질감·테두리 헤일로 | CFG 4.0 과조리, 프롬프트에 억제어 없음 | 네거티브에 `HDR look, oversharpened, halos…` 추가. 화면 '생성 설정'에서 CFG 3.0~3.5 비교 권장 |
| 하늘이 하얗게 날아감 (상단 15% 평균 248/255, 화면의 10%가 거의 흰색) | 안개 낀 장면 + 노출 일관성 문구 | LLM이 하늘 상태를 구체적으로 쓰도록 지시, 네거티브 `overexposed sky, blown-out highlights` |
| 발밑(아래쪽)이 늘어지고 뭉개짐 | 등장방형 극점 + 생성 해상도 한계 | LLM이 발밑 바닥 재질을 구체적으로 쓰도록 지시. '발밑·하늘 보정' 버튼으로 해결 (아래 참고) |
| 이미 360인 이미지를 올리면 사진처럼 다시 붙여서 망가짐 | 입력 판별 없음 | 2:1 이미지는 자동 감지 → **8K 화질 개선만** 실행 |
| x2 업스케일 모델이 선택되면 8K 크롭이 어긋남 | 배율 고정 가정 | 업스케일 뒤 크기를 정확히 다시 맞추는 노드 추가 (x2/x4 모두 대응) |

비교 실험은 시드를 고정하고 설정을 하나씩 바꿔 보세요. 각 시도의 설정이 `meta.json`과 화면 기록에 남아요.

## 업스케일 블러·LoRA 영향 줄이기 (리파인 패스)
- **Lanczos 블렌드 제거**: 8K에 단순 확대본을 20~45% 섞던 단계가 블러의 원인 → 생성 모드·8K 화질 개선에서 제거.
- **디테일 리파인 (LoRA 없음)**: AI 업스케일한 8K를 1536px 타일 18장(겹침 192px)으로 나눠, LoRA 없이 기본 모델로
  약하게(denoise 0.3) 다시 그림 → 업스케일러의 뭉개짐 대신 실제 질감. 프롬프트는 "고해상도 사진" + LLM 장면 묘사,
  네거티브에 painting/illustration/cartoon/flat colors 등. (`config.json` → `refine`, 끄려면 `"enabled": false`)
- **LoRA 강도**: 1차 생성 0.95 → **0.8**, 이음새 복원은 별도 LoRA **0.6** (`gen_defaults.lora_strength`, `seam_lora_strength`).
- 증류 모델이면 부분 재생성(이음새·리파인)은 8스텝, 전체 생성은 4스텝 (`distilled_profile`).
- 리파인은 시간이 걸려요 (타일 18장). 빠르게 확인만 할 땐 `refine.enabled`를 false로.

## 좌우 이음새 (360 뒤쪽 세로선)
- 원인: 이음새 복원 단계가 이음새가 그대로 있는 그림을 참조로 받아, 편집 모델(FLUX Klein)이 이음새까지 그대로 다시 그림
  (측정: 이음새 색 차이 39 vs 일반 이웃 8.7 → 4.5배).
- 해결 1: 이음새 띠(512px)를 **초록(#00ff00)으로 칠해** 아웃페인트 LoRA가 학습한 방식("Fill the green spaces…")으로 채움.
  denoise 1.0 고정, 이음새 LoRA 0.8.
- 해결 2: 8K 리파인 타일이 좌우 끝을 따로 그리지 않도록, 오른쪽 끝에 왼쪽 앞부분을 이어 붙여 이음새를 가로지르는
  타일을 만들고 끝나면 부드럽게 되돌림 (시뮬레이션: 이음새 차이 17 → 1.8).
- 결과 화면에 **좌우 이음새 점수**(×배율, 2 이하 양호) 표시. 3.5를 넘으면 다시 생성 권장.

## 발밑·하늘 보정 (극점 보정 패스)
결과 화면의 **발밑·하늘 보정 + 8K** (또는 360 이미지를 올렸을 때 같은 버튼):
1. 2K 360 이미지를 90° 굴려 발밑은 왼쪽, 하늘 꼭대기는 오른쪽 **적도 위**로 옮김 → 늘어짐 없이 보임
2. 두 원(발밑 반경 30°+4°, 하늘 22°+4°)만 같은 모델로 다시 그림 (denoise 0.85, LoRA 0.6). LLM이 쓴 `ground_below` / `sky_above` 묘사 사용
3. 되돌려서 극점 부분만 6° 페더로 합성 → 가운데(수평선 부근)는 픽셀 하나도 안 바뀜
4. 이어서 자동으로 8K 화질 개선 → Quest용 JPG
- 결과는 최종 360 **한 장**(8K)만 나와요. 보정 전/중간 파일은 `runs/<작업>/aNN/` 폴더에만 남아요(`pole_base.png` = 보정 전 2K).
- 범위·강도는 `config.json` 의 `pole` 에서 조정 (`nadir_cap`, `zenith_cap`, `denoise`, `lora_strength`).
- 작업 파일: `pole_rot.png`(돌린 이미지), `pole_mask.png`(다시 그릴 부분), `pole_rot_out.png`(모델 결과)

## 영상화 프롬프트 (5단계 · 일부만 움직임)
결과가 나오면 아래에 **영상화 · 움직일 곳과 프롬프트** 칸이 생겨요. 화면 전체가 아니라 **일부만** 움직이는 cinemagraph 방식이에요.
1. **움직일 곳 찾기 (LLM)**: 결과 360(yaw/pitch 30° 격자를 그려서) + 원본 사진 + 360 생성 프롬프트·분석을 gpt 에 보내면
   움직일 영역 1~4곳(물결·구름·버드나무 가지 등)을 위치(yaw, pitch, 화각)와 움직임으로 돌려줘요. 건물·바위·땅·카메라는 움직이지 않게 지시.
2. 영역마다 **영상 모델에 들어갈 평면 화면**(8K 에서 잘라낸 1408×1024 / 1024×1408, 학습 비율 704×512)과
   **Cinemagraph 프롬프트**를 만들어요. 고정 문구: 트리거 `CINEMAGRAPH_MOTION`, 카메라 고정, "Only the … moves",
   "Everything else stays perfectly still", 반복(loop) — LLM 은 무엇이 어떻게 움직일지만 정함.
3. 미리보기에서 **마우스로 끌어** 새 영역을 고르거나, 영역의 **위치 다시 고르기**를 누른 뒤 끌어서 위치를 바꿀 수 있어요(yaw·pitch·화각·비율을 끈 모양대로 자동, 비율 0.5~2 · 화면 크기는 약 1.4MP 로 64 배수). 새 영역은 '움직이는 것'·'움직임'을 채우고 **수정 반영**.
   미리보기: 움직일 곳만 밝게, 정지 부분은 어둡게. 위치·움직임을 고치고 **수정 반영**하면 화면·프롬프트를 다시 만들어요
   (Positive 를 직접 고친 영역은 그대로 둠). **프롬프트 JSON 복사**로 ComfyUI 에 옮길 수 있어요.
4. 저장: `runs/<작업>/aNN/motion/` — `plan.json`(영역·프롬프트·모델 정보), `region_N.png`(영상 모델 입력), `overlay.jpg`, `mask.png`(ERP 움직임 마스크)
- 영상 모델: LTX-2.3-22B + `ltx-2.3-22b-lora-cinemagraph-0.9` (LoRA 강도 0.8~2.0). 생성된 영상은 같은 영역 자리에만 다시 합성하고 나머지는 8K 정지 이미지.

## 360 영상 만들기 (영상화 6~7단계)
5단계 칸 아래 버튼 두 개로 나뉘어 있어요.
- **① 영역 영상 만들기**: 실행 위치(기본 Comfy Cloud) · 길이(초) · fps 를 고르고 누르면 영역 영상과 SAM 마스크를 만들어요(클라우드).
- **② 360 합성**: 만들어진 영역 영상을 받아 이 PC 에서 360 에 합성해 MP4 로 저장. 영역 영상을 다시 만들지 않고 몇 번이든 다시 합성할 수 있어요.

| 단계 | 어디서 | 내용 |
|---|---|---|
| 영역 영상 | ComfyUI / Comfy Cloud | `region_N.png` + 영역 프롬프트 → LTX-2.3 + Cinemagraph LoRA (1단계 704×512 → 2배 → 1408×1024, 프레임 8n+1) |
| 요소 마스크 | ComfyUI / Comfy Cloud | SAM 3 에 요소 이름(예: `pond water surface`) → 움직일 것 모양 마스크. 모델이 없으면 영역 가운데 둥근 마스크 |
| 다른 것 고정 | 웹앱 (Python) | 마스크 안만 영상, 밖은 원본 정지 화면. 영역 화면 테두리는 항상 0 → 경계 안 보임 |
| 색·반복 | 웹앱 | 첫 프레임 색을 원본에 맞춤, 끝 6프레임을 처음과 교차 페이드해 끊김 없는 반복 |
| 360 합성 | 웹앱 | 평면 영역 → ERP 로 다시 투영해 8K 정지 이미지(→4K) 위 같은 자리에 프레임마다 합성 |
| 저장 | 웹앱 (ffmpeg) | `motion/motion_360.mp4` 4096×2048 H.264 · **Quest 뷰어에 추가** 로 Unity StreamingAssets 에 복사 |

- 필요: `pip install imageio-ffmpeg` (또는 ffmpeg 를 PATH 에). Comfy Cloud 키(`C:\keydata\comfy_key`).
- Cloud/서버 모델: `ltx-2.3-22b-dev-fp8`, `ltx-2.3-22b-distilled-lora-384`, `gemma_3_12B_it_fp4_mixed`,
  `ltx-2.3-spatial-upscaler-x2-1.1`, `ltx-2.3-22b-lora-cinemagraph-0.9`(없으면 config `motion.cinemagraph_lora` 를 "" 로), `sam3.1_multiplex_fp16`(선택)
- 설정: `config.json` → `motion` (backend, duration, fps, video_width, loop_fade, sam, 모델 파일명)
- 기록: `motion/workflow_video_N.json`, `workflow_mask_N.json`(실제 보낸 그래프), `mask_N.png`(SAM), `mask_used_N.png`(실제 합성 마스크)

### 범위 고정 생성 (기본, `motion.range_mode`)
붙여넣기 대신 **영상을 만들 때부터 범위를 줌**. SAM 으로 요소를 찾고 → 퍼짐(끔/주변/넓게)만큼 넓힌 범위 마스크(`range_N.png`) →
LTX 가 모든 프레임을 원본으로 채운 잠재에서 시작해 **범위 밖은 매 단계 원본으로 되돌림**(SetLatentNoiseMask, 첫 프레임은 전부 고정).
→ 범위 밖·카메라는 원본 그대로(좌우 흔들림 없음), 범위 경계는 모델이 주변과 어울리게. 합성은 같은 범위 마스크로.
- 프롬프트 기본: "360 파노라마 사진에서 본 화면 · 기존 이미지를 그대로 유지(구도·사물·디테일·색)" + 카메라 고정. 전체 영상은 "등장방형 360 파노라마".
- 그래도 남는 흔들림은 합성 전에 원본 기준으로 정렬(`motion.stabilize`).

### 영향 퍼짐 (원인에서 시작된 움직임이 '영향받는 것'에만)
영역마다 **영향 퍼짐** 목록(LLM 이 장면상 자연스러울 때만 채움, 기본은 비어 있음 → 영역만 움직임).
대상(영어, SAM 으로 찾음) · 범위(끔 / 주변 ~35° / 넓게 ~100°) · 세기(0~1).

| 단계 | 내용 |
|---|---|
| 360 전체 영상 | 8K 결과를 1536×768 로 줄이고 원인 영역이 가운데 오게 돌려 LTX 로 짧게 영상화 (`global_in.png`, `workflow_global.json`) |
| 영향 대상 마스크 | 같은 이미지에서 대상마다 SAM 3 (`inf_mask_N_M.png`) |
| 흐름만 사용 | 영상 픽셀은 버리고 '어디서 왔는지'(흐름)만 뽑음 → × 대상 마스크 × 원인에서 멀수록 감쇠 × 세기 (`influence_weight.png`) |
| 원본에 적용 | 영향받는 부분만 8K 원본에서 당겨 샘플 → 면적 평균으로 4K. 영향 밖은 원본 픽셀 그대로, 선명도 유지(시험: 원본 대비 1.01) |
| 반복 | 픽셀이 아니라 변위를 처음·끝 교차 페이드 → 겹쳐 보이지 않음 |

- 필요: `pip install opencv-python-headless`
- 설정: `motion.influence`(끄기), `global_width`, `global_lora_strength`, `influence_max_shift_deg`(최대 이동 각도, 늘어짐 방지)
- 전체 영상이 실패하면 영향 퍼짐 없이 영역만 합성해요(결과 정보에 사유 표시).

## LLM이 정하는 것 / 고정된 것
| LLM (사진마다 다름) | 고정 (llm.py 상단) |
|---|---|
| 장면 묘사(카메라 뒤·바닥·하늘 포함), 조명 표현, 피해야 할 조명, 장면별 실패 요소, 이음새에서 이어질 요소, 수평선 위치, 화각 추정, 한국어 요약·주의점 | LoRA 트리거 문장, 조명 일관성 문장, 공통 네거티브, 이음새 프롬프트 틀 |

- 화각: EXIF 35mm 환산 초점거리가 있으면 그걸 우선 사용, 없으면 LLM 추정
- pitch: LLM이 찾은 수평선 위치와 화각으로 계산
- 품질을 올리려면 `llm.py`의 `SYSTEM` 지시문을 다듬는 게 가장 효과적이에요. 결과가 좋았던 시도의 meta.json 을 예시로 넣는 것도 좋아요.

## 192.168.1.91 ComfyUI 서버 주의
- 그 PC의 ComfyUI가 외부 접속을 허용해야 해요 (ComfyUI Desktop: 설정 > Server-Config > Host `0.0.0.0` / 포터블: `--listen 0.0.0.0 --port 8000`).
- 브라우저에서 `http://192.168.1.91:8000` 이 열리면 연결 가능한 상태예요.
- 그 서버에 FLUX.2 Klein 모델, qwen_3_4b, flux2-vae, 360 LoRA 파일이 있어야 해요.

## (선택) 공식 Comfy Cloud 주의
- Cloud에 `flux-2-klein-base-4b-fp8`, `qwen_3_4b`, `flux2-vae`, **`flux-2-klein-4B-360-erp-outpaint-lora_V1`** 가 있어야 해요 (LoRA 업로드 여부 확인).
- 현재 구현의 API 경로: `/api/upload/image`, `/api/prompt`, `/api/job/{id}/status`, `/api/view`, `wss://.../ws`. 경로 변경은 `comfy.py`의 `CLOUD_PATHS`·`ComfyCloud`에서 관리합니다.
- 웹소켓으로 결과를 수신하고, 누락되면 history 조회를 시도합니다. 서버의 상태 API가 호출될 때 결과를 `runs/`에 저장합니다. 브라우저를 닫거나 다른 작업으로 이동하면 해당 작업의 조회가 중단되므로 다시 열어 완료 여부를 확인하세요.

## 파일
```
main.py                  실행·설정 로딩 (python main.py)
server.py                HTTP 라우트·작업 단계 조율·결과 후처리
storage.py               작업 경로 검사·meta.json 읽기·원자적 저장
credentials.py           환경변수·로컬 키 파일 읽기
openai_client.py         OpenAI Responses 요청·응답 처리
comfy.py                 ComfyUI·Comfy Cloud 통신·모델 선택·웹소켓 수명 관리
api.py                   이전 import api 호출용 호환 진입점 (중복 구현 없음)
llm.py                   LLM 지시문·응답 스키마·프롬프트 조립
erp.py                   ERP 투영·극점 회전·사진 합성·JPG 저장
workflow.py              생성·화질 개선·극점 보정 그래프 구성
static/index.html        화면 구조
static/style.css         화면 스타일
static/app.js            업로드·분석·생성·결과 UI
static/api.js            브라우저의 JSON 요청·오류 처리
static/poller.js         작업 상태 조회·취소·늦은 응답 차단
static/viewer.js         WebGL 360 뷰어
requirements.txt         Python 의존성
tests/                   서버 통합·워크플로·저장·폴링 회귀 검사
runs/                    작업 기록 (자동 생성)
```

## 구조 정리와 검증

- 기존 `api.py`와 `comfy.py`에 있던 ComfyUI 구현을 `comfy.py` 하나로 통합했습니다. 이전 이름 `LocalComfy`, `CloudComfy`는 같은 클래스를 가리키는 호환 별칭입니다.
- 생성·화질 개선·극점 보정의 화면 제출과 진행률 처리를 공유합니다. 상태 조회는 이전 요청이 끝난 뒤 예약하여 느린 서버에 중복 요청이 쌓이지 않게 합니다.
- 작업을 전환하면 상태 조회와 미리보기 예약을 중단하고, 이전 화면의 늦은 응답을 무시합니다.
- 같은 작업의 분석·생성 제출·결과 저장은 서버 프로세스 안에서 순서대로 처리합니다. 여러 서버 프로세스가 같은 `runs/`를 공유하는 구성은 지원하지 않습니다.
- 실행 명령, API 경로, `config.json`, 기존 작업 기록 형식은 유지합니다. HTML은 서버 주소로 열어야 합니다.

`pano360_web` 폴더에서:

```powershell
python -m unittest discover -s tests -v
node --test tests/poller.test.mjs
```

서버 테스트는 임시 작업 폴더와 가짜 ComfyUI 백엔드를 사용합니다. 업로드, 생성 결과 저장, 동시 요청, 극점 보정 두 단계, Unity 폴더 복사 및 정적 모듈 제공을 검사하며 실제 API 키나 GPU 작업은 사용하지 않습니다. 폴링 테스트는 Node.js 22.7 이상을 사용합니다. 실제 ComfyUI 생성 품질·브라우저 WebGL 렌더링·Quest 재생은 별도 확인 대상입니다.

## 6. 사건 영상 (한 번 일어나는 연출)
잔잔하게 시작 → 물이 넘쳐 화면을 덮음 → 다시 잔잔하게 끝. (예: 다리 아래 폭포 물이 불어나 앞 바위 위로 넘쳐 밀려왔다가 빠짐)
1. **사건 만들기 (LLM)** — 결과 360 을 보고 사건 위치(yaw·pitch·화각 100~130°)와 시작·사건·절정·끝 문장, 영상 프롬프트를 정함.
   사건 화면 `event/event_view.png`(2560×1440, 16:9)와 ComfyUI 워크플로 2개를 만듦.
   - `comfy_event_wan14.json`: Wan 2.2 I2V 14B fp8 + 4스텝 LoRA, 1280×720, 16fps (24GB)
   - `comfy_event_wan5.json`: Wan 2.2 TI2V 5B, 1280×704, 24fps (가벼움)
2. ComfyUI 에서 사건 화면 PNG 를 넣고 실행 → MP4.
3. **생성 영상 올리기 → 360 에 합치기** — 영상 흔들림을 첫 프레임 기준으로 정렬하고, **첫 프레임(=원본)에서 달라진 곳만**
   생성 영상을 섞음(사람이 그린 마스크 없음, 앞뒤 프레임으로 부드럽게, 사건 화면 테두리는 0). 나머지는 8K 원본 → `event/event_360.mp4` (4096×2048).
   `event_alpha.png`(절정 때 섞인 범위), `event_peak.jpg`(절정 장면).
- 사건 화면 1280×720 / 화각 120° ≈ 1°당 10.7px 로 4K 360(1°당 11.4px)과 거의 같아 4K 출력에서는 업스케일 없이 맞음.
