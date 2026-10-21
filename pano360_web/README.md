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
- 결과 아래 **영역 마스크 PNG 받기**(360 크기, 흰색 = 움직일 곳). 영상 생성·합성은 웹에서 하지 않고 ComfyUI(LTX) 에서 해요 (2026-10-21 정리).

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

## 생성 없이 바로 쓰기 (이미 만든 360)
360(2:1) 이미지를 올리면 **생성 없이 이 이미지로 바로 다음 단계** 버튼 → ComfyUI 없이 결과로 등록(`aNN/<run>_aNN_360.jpg`, mode `import`)
→ 5·6단계(움직일 곳·프롬프트·범위 칠하기·영상 만들기)를 바로 씀. POST `/api/run/{run_id}/import`.

## 6. 사건 연출 · 프롬프트와 움직일 범위
잔잔하게 시작 → 물이 넘쳐 화면을 덮음 → 다시 잔잔하게 끝.
1. **사건 만들기 (LLM)** — 결과 360 을 보고 사건 위치(yaw·pitch·화각)와 시작·사건·절정·끝 문장, 영상 프롬프트를 정함.
   사건 화면 `event/event_view.png`(2560×1440, 16:9) PNG 받기. 칸을 고치고 **수정 반영**하면 다시 만듦.
2. **움직일 범위 (마스크)** — 360 위에 붓으로 칠하고 **범위 저장** → `event/mask.png`(2048×1024, 흰색 = 움직일 곳) → **마스크 PNG 받기**.
3. **360 전체 영상 프롬프트** — LTX-2.3 + 360 LoRA 에 360 이미지를 그대로 넣을 때 쓸 Positive/Negative (사건 만들기 내용이 들어감) 복사.
4. **영상 만들기 — 노드 파일로** — 사용자 노드 파일(`video_template`, 기본 `video_nodes/retest.json`)에 **값만 넣어** 그대로 실행.
   넣는 값(노드 id): 1131 원본 360(4K JPG 로 올림) · 1128/1112 Positive/Negative · 1102 길이 · 1132 LoRA 세기 · 1100 시드 · SaveVideo 이름.
   마스크: 노드에 1162 LoadImage 가 있으면 칠한 마스크 그림을, 없으면 사각 마스크(1134 SolidMask 크기 · 1136 MaskComposite 위치,
   1133 캔버스 기준)에 **칠한 범위를 감싸는 좌표**(+경계 흐림 폭)를 넣음. 계산 노드(ComfyMathExpression)는 숫자로 바꿔 보냄.
   값 넣은 노드 파일 `event/video_node.json` 을 받을 수 있음(ComfyUI 에서 그대로 열림). 결과 `event/flood_<노드>.mp4`.
   - 모양 그대로 움직이게 하려면 `video_nodes/quest360_flood_mask_inpaint_4k.json`(마스크 그림 + 인페인팅)을 `video_template` 로.
- 서버 모델 목록(object_info)에 없는 파일은 막지 않고 그대로 보냄 (Comfy Cloud 에 직접 가져온 LoRA 는 목록에 안 나올 수 있음).
- 효과 레이어·사건 영상 합성·Quest 뷰어 영상 복사는 2026-10-21 웹에서 뺐어요.

## 원본 사진 자리 화질 (keep_photo.upscale_photo, 기본 켜짐)
생성 그래프는 원본 사진을 **2K 360 안에 줄여 넣었다가** 4배로 키움 → 1200px 사진이 폭 약 410px 로 줄어 원본 디테일이 버려짐.
그래서 생성할 때 **원본 사진을 원래 해상도 그대로 AI 업스케일**(같은 업스케일 모델, `photo_up.png`)하고, 마지막에 그걸 같은 자리에 붙임.
- 이미 만든 결과: 결과 아래 **원본 자리 화질 맞추기** 버튼 → 원본 업스케일만 ComfyUI 로 돌려 다시 붙이고 JPG 도 다시 만듦.

## 원본 사진 다시 붙이기 (keep_photo.paste_mode)
8K 로 키운 뒤 원본 사진을 같은 자리에 다시 붙임(사진에 있던 건 절대 안 사라짐). 원본이 작으면(1°당 픽셀이 8K 의 95% 미만)
그대로 붙일 때 그 부분만 흐려지므로 **detail** 방식: 색·형태(저주파)는 원본 사진, 선명한 디테일(고주파)은 8K 업스케일 결과.
- `paste_mode`: auto(기본) / original / detail, `detail_strength`: 사진 자리의 AI 디테일 세기 0~1 (기본 0.9, 결이 과하면 낮춤)
- 예: 1200px 사진 · 화각 72° → 선명도(라플라시안) 원본 붙이기 1199 → detail 2600, 주변 생성 영역 2354

## Comfy Cloud 만으로 돌리기 (2026-10-21)
- 기본 실행 위치 `default_backend: comfycloud` — 360 생성 · 발밑·하늘 보정 · 8K 화질 개선 · 원본 사진 업스케일이 모두 Comfy Cloud 로.
- **모델 점검** 버튼(실행 위치 아래): 고른 곳에 LoRA·업스케일러가 다 있는지 ✔/⚠/✘ 로 보여 줌.
- Cloud 에 없는 LoRA 는 Cloud 왼쪽 **Models → Import** 에 Hugging Face 링크를 넣어 가져오기 (Creator 이상, safetensors 만).
  가져온 모델은 `올린사람__저장소__파일명` 이름이 되는데 자동으로 찾아 씀. 링크 목록은 `config.json` `model_sources`.
  - 360 아웃페인트 LoRA: https://huggingface.co/nomadoor/flux-2-klein-4B-360-erp-outpaint-lora (폴더 loras)
- 업스케일러 `4x_NMKD-Siax_200k.pth` 가 Cloud 에 없으면 `model_fallbacks` 순서대로 대체 (점검에 → 로 표시).
