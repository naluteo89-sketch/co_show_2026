# Pano360 Viewer (Unity 6 · Meta Quest)

Meta Quest에서 360° 사진(등장방형 equirectangular JPG/PNG)을 보는 뷰어입니다.
OpenXR + Meta Quest Support 기반이고, 컨트롤러와 핸드트래킹을 모두 지원합니다.

## 처음 열 때

1. Unity Hub → **Add → Add project from disk** → 이 폴더(`Pano360Viewer`) 선택
   - 에디터 버전: **6000.3.26f1** 기준 (다른 Unity 6 버전으로 열어도 됩니다)
   - 해당 에디터에 **Android Build Support (OpenJDK, Android SDK & NDK)** 모듈이 설치되어 있어야 합니다.
2. 프로젝트가 열리면 자동으로:
   - Quest용 Player 설정 적용, `Assets/Scenes/Main.unity` 생성 및 빌드 씬 등록
   - **OpenXR**, **XR Hands** 패키지 설치 (Input System 활성화 팝업이 뜨면 **Yes** → 에디터 재시작)
   - 설치 후 OpenXR 로더 + Meta Quest Support + Touch 컨트롤러 프로필 + Hand Tracking 자동 활성화
3. 상단 메뉴 **Pano360 → 3. Android 플랫폼으로 전환**
4. **Pano360 → 4. XR(OpenXR · Meta Quest) 설정 적용** 한 번 더 실행 (확인용)
5. `Edit > Project Settings > XR Plug-in Management > Project Validation`에 경고가 있으면 **Fix All**

자동 설정이 안 됐을 때 수동 설정:
- XR Plug-in Management → Android 탭 → **OpenXR** 체크
- OpenXR → Android 탭 → Feature Groups **Meta Quest** 의 *Meta Quest Support*, *Hand Tracking Subsystem* 체크,
  Interaction Profiles에 *Oculus Touch Controller Profile* 추가

## 빌드 & 실행

1. Quest를 개발자 모드로 설정하고 USB로 PC에 연결 (헤드셋에서 USB 디버깅 허용)
2. `File > Build Profiles` → Android → Run Device에서 Quest 선택 → **Build And Run**
3. 헤드셋의 앱 라이브러리 → **알 수 없는 출처(Unknown Sources)** 에서 *Pano360 Viewer* 실행

## 사진 넣기

| 방법 | 위치 | 비고 |
|---|---|---|
| 앱에 포함 | `Assets/StreamingAssets/Pano360/` | 빌드할 때 목록(photos.json) 자동 생성 |
| 기기 폴더 | 헤드셋 `내부 저장소/Android/data/com.witches.pano360/files/Photos/` | 앱을 한 번 실행하면 폴더가 생깁니다. USB 파일 전송 또는 Meta Quest Developer Hub로 복사 → 앱 재실행 |

adb로 넣기:
```
adb push 내사진.jpg /sdcard/Android/data/com.witches.pano360/files/Photos/
```

- 2:1 비율 등장방형 이미지 권장 (예: 8192×4096, 5760×2880). 8K 이하 권장.
- 파일명에 `_TB`, `_OU`, `_3D` 가 들어가면 **위/아래(Top-Bottom) 스테레오 3D**로 표시합니다 (위쪽 = 왼쪽 눈).
- `00_test_grid.jpg`는 방향 확인용 테스트 이미지입니다. 필요 없으면 지우세요.

## 조작

| 동작 | 컨트롤러 | 핸드트래킹 | 에디터 |
|---|---|---|---|
| 다음 사진 | A / X, 오른쪽 트리거, 스틱 → | 오른손 핀치 | → / Space |
| 이전 사진 | B / Y, 왼쪽 트리거, 스틱 ← | 왼손 핀치 | ← |
| 정면 맞추기 | 스틱 누르기 | 양손 동시 핀치 | R |
| 둘러보기 | 머리 회전 | 머리 회전 | 마우스 오른쪽 드래그 |
| **홈으로(취소)** | 아래쪽 HOME 버튼에 레이 맞추고 트리거 · 왼쪽 ≡ 메뉴 버튼 | HOME 버튼을 바라보며 핀치 | Esc / H |
| 홈에서 시작 | 트리거 / A / X | 핀치 | → / Space |

- 앱은 **홈 화면**(검은 배경 + 사진 수 안내)에서 시작해요. 시작하면 첫 사진 정면이 지금 보는 방향에 맞춰져요.
- 사진을 보는 동안 시선 아래(약 32°)에 **HOME** 버튼이 떠 있고, 고개를 크게 돌리면 따라와요.
- 사진을 불러오는 중에 취소하면 로딩이 끝나는 즉시 홈으로 가요.
- `PanoViewer` 옵션: `startAtHome`(끄면 바로 첫 사진), `showHomeButton`. `PanoHomeButton.dwellToActivate`를 켜면 버튼을 2초 바라보기만 해도 홈으로 가요(전시 중 오작동 방지로 기본 꺼짐).

## 구조

```
Assets/Pano360/
  Runtime/   PanoViewer(본체·홈 화면) · PanoInput(입력) · HeadTracker(HMD 회전) · PhotoSource(사진 목록) · PanoLabel(안내 텍스트) · PanoHomeButton(HOME 취소 버튼)
  Resources/ Pano360Equirect.shader  — 시선 방향으로 UV를 계산하는 360 셰이더 (이음새·극점 왜곡 없음)
  Editor/    Pano360Setup(자동 설정·메뉴) · PhotoListBuilder · QuestManifestPatcher(핸드트래킹 권한)
  Editor/XR/ Pano360XRConfig — OpenXR 설치 후에만 컴파일되는 XR 자동 설정
```

- 패키지명 변경: `Pano360Setup.cs`의 `com.witches.pano360` 수정 (기기 폴더 경로도 같이 바뀜)
- 머리 위치 이동을 반영하려면 Main Camera의 `HeadTracker.trackPosition` 체크 (360 사진은 보통 끔)
- 사진 전환마다 시선 방향을 정면으로 맞추려면 `PanoViewer.recenterOnChange` 체크
