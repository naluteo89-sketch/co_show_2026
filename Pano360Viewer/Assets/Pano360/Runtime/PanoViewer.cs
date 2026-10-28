using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Video;
using UnityEngine.XR;

namespace Pano360
{
    /// <summary>
    /// 360 사진·영상 뷰어 본체. 씬에 이 컴포넌트 하나만 있으면 카메라/구/입력/안내 텍스트를 스스로 구성합니다.
    /// (씬에 없어도 자동 생성됩니다.)
    /// </summary>
    public class PanoViewer : MonoBehaviour
    {
        [Tooltip("사진을 그리는 구의 반지름(m)")]
        public float sphereRadius = 50f;
        [Tooltip("사진 전환 시 페이드 시간(초)")]
        public float fadeDuration = 0.25f;
        [Tooltip("사진 번호/이름 표시 시간(초)")]
        public float labelDuration = 2.5f;
        [Tooltip("새 사진을 열 때마다 사진 정면을 현재 시선 방향에 맞춤")]
        public bool recenterOnChange = false;
        [Tooltip("앱을 홈 화면에서 시작 (끄면 바로 첫 사진)")]
        public bool startAtHome = true;
        [Tooltip("보는 중 아래쪽에 HOME(취소) 버튼 표시")]
        public bool showHomeButton = true;
        [Tooltip("홈 화면 아래쪽에 EXIT(앱 끄기) 버튼 표시. 사진을 보는 중에는 숨김 (관람 중 실수로 꺼지지 않게)")]
        public bool showExitButton = true;
        [Tooltip("눈 렌더 해상도 배율. Quest 3 기본(1.0)=1680x1760/눈 으로 패널(2064x2208)의 약 80%. 1.25 ≈ 패널과 1:1")]
        [Range(1f, 1.5f)] public float eyeResolutionScale = 1.25f;
        [Tooltip("360 영상 반복 재생")]
        public bool loopVideo = true;
        [Tooltip("360 영상 소리 재생")]
        public bool videoAudio = true;

        [Header("사건 영상 (사진 → 영상 → 사진)")]
        [Tooltip("사진을 띄운 뒤 사건 영상을 시작하기까지 (초)")]
        public float eventDelay = 4f;
        [Tooltip("사진 ↔ 영상 전환 시간 (초)")]
        public float eventCrossfade = 0.6f;
        [Tooltip("사건이 끝난 뒤 같은 사진에서 다시 재생 (끄면 한 번만)")]
        public bool eventRepeat = true;
        [Tooltip("다시 재생하기까지 사진만 보여 주는 시간 (초)")]
        public float eventInterval = 8f;
        [Tooltip("켜면 사진 → 영상으로 바뀐 뒤 영상 마지막 장면에서 멈춤 (사진으로 돌아가지 않음, 반복 없음)")]
        public bool eventEndOnVideo = false;
        [Tooltip("끝날 때 범위 마스크를 가장자리부터 안쪽으로 지우며 사진으로 돌아옴 (마스크가 있을 때). 끄면 전체가 한 번에 흐려지며 전환")]
        public bool eventWipeOut = true;
        [Tooltip("지우며 돌아오는 시간 (초). 영상이 끝나기 이만큼 전에 시작")]
        public float eventWipeDuration = 2.5f;
        [Tooltip("지워지는 경계 불규칙함 (0 = 매끈, 0.5 = 물결처럼)")]
        [Range(0f, 0.6f)] public float eventWipeIrregularity = 0.25f;
        [Tooltip("범위 마스크(이름_event_mask.png) 경계 부드럽기. 밉맵 단계 0=딱딱 3≈1.4° 5≈5.6°")]
        [Range(0f, 6f)] public float eventMaskSoftness = 3f;

        Camera cam;
        Transform head;
        Transform sphere;
        Material mat;
        PanoLabel label;
        PanoInput input;
        PanoHomeButton homeButton;
        PanoHomeButton exitButton;

        enum Mode { Home, Viewing }
        Mode mode = Mode.Home;
        bool homeRequested;

        readonly List<PhotoEntry> photos = new List<PhotoEntry>();
        int index = -1;
        Texture2D current;
        // 360 영상: VideoPlayer → RenderTexture → 같은 Equirect 셰이더
        VideoPlayer videoPlayer;
        RenderTexture videoRT;
        bool HasContent => current != null || videoRT != null;
        bool busy;
        float fade = 1f;
        Coroutine eventCo;               // 사진 위 사건 영상 반복
        bool eventPlaying;
        bool eventHeld;                  // 영상 마지막 장면에서 멈춘 상태 (헤드셋을 다시 써도 재생 안 함)
        Texture2D eventMask;             // 사건 영상 범위 (흰색만 영상, 나머지는 고화질 사진)
        Texture2D eventWipeField;        // 범위 가장자리부터의 거리 (끝날 때 벗겨지는 순서)

        static readonly int MainTexId = Shader.PropertyToID("_MainTex");
        static readonly int StereoId = Shader.PropertyToID("_Stereo");
        static readonly int FadeId = Shader.PropertyToID("_Fade");
        static readonly int BlendTexId = Shader.PropertyToID("_BlendTex");
        static readonly int BlendId = Shader.PropertyToID("_Blend");
        static readonly int BlendMaskId = Shader.PropertyToID("_BlendMask");
        static readonly int UseBlendMaskId = Shader.PropertyToID("_UseBlendMask");
        static readonly int MaskLodId = Shader.PropertyToID("_MaskLod");
        static readonly int WipeId = Shader.PropertyToID("_Wipe");
        static readonly int WipeNoiseId = Shader.PropertyToID("_WipeNoise");
        static readonly int WipeFieldId = Shader.PropertyToID("_WipeField");

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            if (FindAnyObjectByType<PanoViewer>() == null)
                new GameObject("Pano360 Viewer").AddComponent<PanoViewer>();
        }

        void Awake()
        {
            // 렌더 해상도: 기본 1.0 은 패널보다 낮아 사진이 약간 흐리고 고해상도 사진에서 반짝임(에일리어싱)이 생김
            if (XRSettings.enabled) XRSettings.eyeTextureResolutionScale = eyeResolutionScale;

            // 카메라
            cam = Camera.main;
            if (cam == null)
            {
                var go = new GameObject("Main Camera") { tag = "MainCamera" };
                cam = go.AddComponent<Camera>();
                go.AddComponent<AudioListener>();
            }
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = sphereRadius * 4f;
            head = cam.transform;
            if (cam.GetComponent<HeadTracker>() == null) cam.gameObject.AddComponent<HeadTracker>();

            // 사진 구
            var shader = Shader.Find("Pano360/Equirect");
            if (shader == null) Debug.LogError("[Pano360] 셰이더 'Pano360/Equirect'를 찾을 수 없습니다.");
            mat = new Material(shader);
            mat.SetFloat(FadeId, fade);

            var sphereGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphereGo.name = "Pano360 Sphere";
            Destroy(sphereGo.GetComponent<Collider>());
            sphere = sphereGo.transform;
            sphere.localScale = Vector3.one * sphereRadius * 2f;
            var r = sphereGo.GetComponent<MeshRenderer>();
            r.sharedMaterial = mat;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;

            // 입력 / 안내
            input = GetComponent<PanoInput>();
            if (input == null) input = gameObject.AddComponent<PanoInput>();
            label = PanoLabel.Create(head);

            // HOME(취소) 버튼
            homeButton = PanoHomeButton.Create(head);
            homeButton.onActivate = () => RequestHome();

            // EXIT(앱 끄기) 버튼 — 홈 화면에서만
            exitButton = PanoHomeButton.Create(head, "EXIT");
            exitButton.onActivate = () => QuitApp();

            // 360 영상 재생기 (사진을 볼 때는 멈춰 있음)
            videoPlayer = gameObject.AddComponent<VideoPlayer>();
            videoPlayer.playOnAwake = false;
            videoPlayer.source = VideoSource.Url;
            videoPlayer.renderMode = VideoRenderMode.RenderTexture;
            videoPlayer.skipOnDrop = true;                 // 디코딩이 밀리면 프레임을 건너뛰어 시간을 맞춤
            videoPlayer.isLooping = loopVideo;
            videoPlayer.audioOutputMode = videoAudio ? VideoAudioOutputMode.Direct : VideoAudioOutputMode.None;
            videoPlayer.errorReceived += (vp, msg) => Debug.LogError($"[Pano360] 영상 오류 {vp.url}: {msg}");
            input.pointerOverHome = node => homeButton.IsHoveredBy(node);
            input.gazeOverHome = () => homeButton.GazeHover;
            input.pointerOverQuit = node => exitButton.IsHoveredBy(node);
            input.gazeOverQuit = () => exitButton.GazeHover;
        }

        IEnumerator Start()
        {
            label.Show("Loading...", float.PositiveInfinity);
            yield return PhotoSource.Collect(photos);

            if (photos.Count == 0)
            {
                label.Show(
                    "No 360 photos or videos found.\n\n" +
                    "Put equirectangular JPG/PNG/MP4 files in\n" +
                    "Assets/StreamingAssets/Pano360 (bundled)\nor on the headset:\n" +
                    PhotoSource.DeviceFolder,
                    float.PositiveInfinity);
                yield break;
            }

            if (startAtHome) ShowHome();
            else yield return StartViewing();
        }

        // ------------------------------------------------------------------ 홈 화면
        void ShowHome()
        {
            mode = Mode.Home;
            homeRequested = false;
            homeButton.SetVisible(false);
            exitButton.SetVisible(showExitButton);
            label.follow = true;
            label.Show(
                "PANO360\n\n" +
                $"{photos.Count} items ({photos.FindAll(x => x.video).Count} videos)\n\n" +
                "Trigger / A : Start" +
                (showExitButton ? "\nEXIT (below) : Quit app" : ""),
                float.PositiveInfinity);
        }

        /// 앱 끄기 (Quest: 앱 종료, 에디터: 플레이 모드 끝)
        public void QuitApp()
        {
            StopEvent();
            StopVideo();
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        /// 보는 중 취소 → 홈. 사진을 불러오는 중이면 끝난 뒤 이동
        public void RequestHome()
        {
            if (mode == Mode.Home) return;
            if (busy) { homeRequested = true; return; }
            StartCoroutine(GoHome());
        }

        IEnumerator GoHome()
        {
            busy = true;
            StopEvent();
            homeButton.SetVisible(false);
            yield return FadeTo(1f);
            mat.SetTexture(MainTexId, null);
            if (current != null) { Destroy(current); current = null; }
            StopVideo();
            index = -1;
            busy = false;
            ShowHome();
        }

        IEnumerator StartViewing()
        {
            mode = Mode.Viewing;
            exitButton.SetVisible(false);
            label.follow = false;
            label.Hide();
            Recenter();                     // 첫 사진 정면을 지금 보는 방향으로
            yield return ShowPhoto(0);
        }

        void Update()
        {
            var action = input.Consume();

            if (action == PanoAction.Quit)
            {
                if (mode == Mode.Home) QuitApp();   // 보는 중에는 EXIT 버튼이 없음 (키보드 Q 도 홈 화면에서만)
                return;
            }

            if (mode == Mode.Home)
            {
                if (busy || photos.Count == 0) return;
                if (action == PanoAction.Next || action == PanoAction.Prev || action == PanoAction.Recenter)
                    StartCoroutine(StartViewing());
                return;
            }

            if (action == PanoAction.Home)
            {
                RequestHome();
                return;
            }
            if (action == PanoAction.Recenter)
            {
                Recenter();
                if (index >= 0) label.Show("Recentered", 1f);
                return;
            }
            if (busy || photos.Count == 0) return;

            if (action == PanoAction.Next) StartCoroutine(ShowPhoto((index + 1) % photos.Count));
            else if (action == PanoAction.Prev) StartCoroutine(ShowPhoto((index - 1 + photos.Count) % photos.Count));
        }

        void LateUpdate()
        {
            // 구는 항상 눈 위치를 중심으로 (위치 추적을 켜도 사진이 따라오도록)
            sphere.position = head.position;
        }

        /// 사진의 정면(가운데)을 현재 시선 방향(Yaw)에 맞춤
        public void Recenter()
        {
            var yaw = head.eulerAngles.y;
            sphere.rotation = Quaternion.Euler(0f, yaw, 0f);
        }

        IEnumerator ShowPhoto(int i)
        {
            busy = true;
            StopEvent();
            yield return FadeTo(1f);

            var p = photos[i];
            if (p.video)
            {
                yield return PlayVideo(i, p);
            }
            else
            using (var req = UnityWebRequestTexture.GetTexture(p.url, true))
            {
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    var tex = DownloadHandlerTexture.GetContent(req);
                    tex.wrapModeU = TextureWrapMode.Repeat;
                    tex.wrapModeV = TextureWrapMode.Clamp;
                    tex.filterMode = FilterMode.Bilinear;

                    StopVideo();
                    mat.SetTexture(MainTexId, tex);
                    mat.SetFloat(StereoId, p.stereo ? 1f : 0f);
                    if (current != null) Destroy(current);
                    current = tex;

                    if (recenterOnChange) Recenter();
                    // 2:1(스테레오는 1:1)이 아니면 위아래로 늘어나 보임 → 경고
                    float want = p.stereo ? 1f : 2f, ratio = (float)tex.width / tex.height;
                    string warn = Mathf.Abs(ratio - want) > 0.03f
                        ? $"\n\nWARNING: {tex.width}x{tex.height} is not {(p.stereo ? "1:1" : "2:1")} - image will look stretched"
                        : "";
                    label.Show($"{i + 1} / {photos.Count}{(p.HasEvent ? (p.eventLoops ? "  [LIVE]" : string.IsNullOrEmpty(p.eventMaskUrl) ? "  [EVENT]" : "  [EVENT+MASK]") : "")}\n{p.name}{(p.stereo ? "  [3D]" : "")}\n{tex.width}x{tex.height}{warn}",
                               warn.Length > 0 ? labelDuration * 2f : labelDuration);
                }
                else
                {
                    Debug.LogError($"[Pano360] 로드 실패 {p.url}: {req.error}");
                    label.Show($"Failed to load\n{p.name}\n{req.error}", labelDuration * 2f);
                }
            }

            index = i;
            if (homeRequested)              // 불러오는 동안 취소를 눌렀으면 바로 홈으로
            {
                busy = false;
                yield return GoHome();
                yield break;
            }
            if (HasContent) yield return FadeTo(0f);
            if (showHomeButton && mode == Mode.Viewing) homeButton.SetVisible(true);
            busy = false;
            if (!p.video && p.HasEvent && current != null) eventCo = StartCoroutine(EventCycle(p));
        }

        // ------------------------------------------------------------------ 사건 영상: 원본 사진 → 영상 → 원본 사진
        // 평소엔 고화질(8K) 사진. 사건 영상(4K)은 처음·끝이 원본과 같으므로 부드럽게 바꿨다가 끝나면 다시 사진으로.
        IEnumerator EventCycle(PhotoEntry p)
        {
            yield return LoadEventMask(p);
            float wait = eventDelay;
            while (true)
            {
                // 기다리는 동안 미리 준비 (시작할 때 끊김 없도록). live 는 평소 움직임부터 시작하므로 바로
                if (p.eventLoops) wait = Mathf.Min(wait, 0.5f);
                videoPlayer.url = p.eventUrl;
                videoPlayer.isLooping = p.eventLoops;
                videoPlayer.Prepare();
                float t0 = Time.realtimeSinceStartup;
                yield return new WaitForSeconds(wait);
                while (!videoPlayer.isPrepared && Time.realtimeSinceStartup - t0 < 20f) yield return null;
                if (!videoPlayer.isPrepared)
                {
                    Debug.LogError($"[Pano360] 사건 영상 준비 실패 {p.eventUrl}");
                    label.Show("Failed to load event video", labelDuration);
                    yield break;
                }

                int w = (int)videoPlayer.width, h = (int)videoPlayer.height;
                if (videoRT == null || videoRT.width != w || videoRT.height != h)
                {
                    ReleaseVideoRT();
                    videoRT = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32)
                    {
                        name = "Pano360 Event", useMipMap = false, autoGenerateMips = false,
                        wrapModeU = TextureWrapMode.Repeat, wrapModeV = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear
                    };
                    videoRT.Create();
                }
                videoPlayer.targetTexture = videoRT;
                mat.SetTexture(BlendTexId, videoRT);
                eventPlaying = true;
                videoPlayer.Play();
                // 첫 프레임이 그려질 때까지 기다린 뒤 사진 → 영상
                float tw = Time.realtimeSinceStartup;
                while (videoPlayer.frame < 1 && Time.realtimeSinceStartup - tw < 3f) yield return null;
                yield return BlendTo(1f);

                if (p.eventLoops)
                {
                    // XXX_live.mp4: 평소 움직임 → 사건 → 평소 가 끝↔처음까지 이어진 영상 → 사진으로 돌아가지 않고 계속 반복
                    videoPlayer.isLooping = true;
                    eventCo = null;
                    yield break;
                }

                double len = videoPlayer.length;
                if (eventEndOnVideo)
                {
                    // 영상으로 끝냄: 마지막 장면 직전에 멈춰 그대로 둠 (끝까지 가면 화면이 비는 기기가 있어 2프레임 앞에서 멈춤)
                    double stopAt = len - 2.0 / Mathf.Max(1f, (float)videoPlayer.frameRate);
                    while (videoPlayer.isPlaying && videoPlayer.time < stopAt) yield return null;
                    videoPlayer.Pause();
                    eventHeld = true;
                    eventCo = null;
                    yield break;
                }
                if (eventWipeOut && eventMask != null)
                {
                    // 범위를 가장자리부터 지우며 사진으로 (물이 빠지듯) — 영상이 아직 움직이는 동안 진행
                    float dur = Mathf.Clamp(eventWipeDuration, 0.2f, (float)len * 0.8f);
                    while (videoPlayer.isPlaying && videoPlayer.time < len - dur - 0.05) yield return null;
                    mat.SetFloat(WipeNoiseId, eventWipeIrregularity);
                    float wipe = 0f, wipeStart = Time.time;
                    while (wipe < 1f)
                    {
                        wipe = Mathf.Clamp01((Time.time - wipeStart) / dur);
                        mat.SetFloat(WipeId, Mathf.SmoothStep(0f, 1f, wipe));
                        yield return null;
                    }
                    mat.SetFloat(BlendId, 0f);
                    mat.SetFloat(WipeId, 0f);
                }
                else
                {
                    // 끝나기 직전에 영상 → 사진
                    while (videoPlayer.isPlaying && videoPlayer.time < len - eventCrossfade) yield return null;
                    yield return BlendTo(0f);
                }
                videoPlayer.Stop();
                eventPlaying = false;

                if (!eventRepeat) yield break;
                wait = eventInterval;
            }
        }

        /// 범위 마스크: 있으면 흰색 부분만 영상으로 바뀌고 나머지는 8K 사진 그대로 (영상 해상도가 낮아도 배경은 선명)
        IEnumerator LoadEventMask(PhotoEntry p)
        {
            ReleaseEventMask();
            if (string.IsNullOrEmpty(p.eventMaskUrl)) yield break;
            using (var req = UnityWebRequest.Get(p.eventMaskUrl))
            {
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"[Pano360] 사건 범위 마스크 로드 실패 {p.eventMaskUrl}: {req.error} — 화면 전체에 영상");
                    yield break;
                }
                // 그림 형식(흑백/RGBA/알파만)과 상관없이 밝기만 꺼내 R8 + 밉맵으로 → 낮은 밉 단계로 샘플하면 경계가 부드러움
                var src = new Texture2D(2, 2);
                if (!src.LoadImage(req.downloadHandler.data, false))
                {
                    Destroy(src);
                    Debug.LogError($"[Pano360] 사건 범위 마스크를 읽지 못함 {p.eventMaskUrl}");
                    yield break;
                }
                bool alphaOnly = src.format == TextureFormat.Alpha8;
                var px = src.GetPixels32();
                var bytes = new byte[px.Length];
                for (int k = 0; k < px.Length; k++) bytes[k] = alphaOnly ? px[k].a : px[k].r;
                var tex = new Texture2D(src.width, src.height, TextureFormat.R8, true, true);
                eventWipeField = BuildWipeField(bytes, src.width, src.height);
                Destroy(src);
                tex.SetPixelData(bytes, 0);
                tex.wrapModeU = TextureWrapMode.Repeat;    // 360 좌우가 이어짐
                tex.wrapModeV = TextureWrapMode.Clamp;
                tex.filterMode = FilterMode.Trilinear;
                tex.Apply(true, true);
                eventMask = tex;
            }
            mat.SetTexture(BlendMaskId, eventMask);
            mat.SetTexture(WipeFieldId, eventWipeField);
            mat.SetFloat(MaskLodId, eventMaskSoftness);
            mat.SetFloat(UseBlendMaskId, 1f);
        }

        void ReleaseEventMask()
        {
            if (mat != null) { mat.SetFloat(UseBlendMaskId, 0f); mat.SetTexture(BlendMaskId, null); mat.SetTexture(WipeFieldId, null); }
            if (eventMask != null) { Destroy(eventMask); eventMask = null; }
            if (eventWipeField != null) { Destroy(eventWipeField); eventWipeField = null; }
        }

        /// 마스크 → 256x128 에서 가장자리까지 거리(체임퍼) → 0~1. 문턱을 올리면 끝부분부터 안쪽으로 벗겨짐
        static Texture2D BuildWipeField(byte[] mask, int w, int h)
        {
            const int W = 256, H = 128;
            var d = new float[W * H];
            const float INF = 1e9f;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    // 칸 안의 마스크 평균 (상자 축소)
                    int x0 = x * w / W, x1 = Mathf.Max(x0 + 1, (x + 1) * w / W), y0 = y * h / H, y1 = Mathf.Max(y0 + 1, (y + 1) * h / H);
                    int sum = 0, cnt = 0;
                    for (int yy = y0; yy < y1; yy += 2)
                        for (int xx = x0; xx < x1; xx += 2) { sum += mask[yy * w + xx]; cnt++; }
                    d[y * W + x] = sum > cnt * 127 ? INF : 0f;
                }
            float D(int x, int y) => (x < 0 || y < 0 || x >= W || y >= H) ? 0f : d[y * W + x];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int i = y * W + x; if (d[i] == 0f) continue;
                    d[i] = Mathf.Min(d[i], Mathf.Min(Mathf.Min(D(x - 1, y - 1) + 1.414f, D(x, y - 1) + 1f), Mathf.Min(D(x + 1, y - 1) + 1.414f, D(x - 1, y) + 1f)));
                }
            for (int y = H - 1; y >= 0; y--)
                for (int x = W - 1; x >= 0; x--)
                {
                    int i = y * W + x; if (d[i] == 0f) continue;
                    d[i] = Mathf.Min(d[i], Mathf.Min(Mathf.Min(D(x + 1, y + 1) + 1.414f, D(x, y + 1) + 1f), Mathf.Min(D(x - 1, y + 1) + 1.414f, D(x + 1, y) + 1f)));
                }
            float max = 1e-3f;
            for (int i = 0; i < d.Length; i++) if (d[i] < INF) max = Mathf.Max(max, d[i]);
            var outBytes = new byte[W * H];
            for (int i = 0; i < d.Length; i++) outBytes[i] = (byte)Mathf.RoundToInt(Mathf.Clamp01(d[i] / max) * 255f);
            var tex = new Texture2D(W, H, TextureFormat.R8, false, true)
            { wrapModeU = TextureWrapMode.Repeat, wrapModeV = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
            tex.SetPixelData(outBytes, 0);
            tex.Apply(false, true);
            return tex;
        }

        IEnumerator BlendTo(float target)
        {
            float b = mat.GetFloat(BlendId);
            while (!Mathf.Approximately(b, target))
            {
                b = Mathf.MoveTowards(b, target, Time.deltaTime / Mathf.Max(0.01f, eventCrossfade));
                mat.SetFloat(BlendId, b);
                yield return null;
            }
        }

        void StopEvent()
        {
            if (eventCo != null) { StopCoroutine(eventCo); eventCo = null; }
            eventHeld = false;
            if (mat != null) { mat.SetFloat(BlendId, 0f); mat.SetFloat(WipeId, 0f); mat.SetTexture(BlendTexId, null); }
            ReleaseEventMask();
            // 사진 위 사건 영상이면 멈춤 (일반 360 영상을 보는 중이면 그대로 둠 — 다음 항목에서 처리)
            if (videoRT == null || mat == null || mat.GetTexture(MainTexId) != videoRT)
            {
                eventPlaying = false;
                StopVideo();
            }
        }

        IEnumerator FadeTo(float target)
        {
            if (fadeDuration <= 0f || !HasContent)
            {
                fade = target;
                mat.SetFloat(FadeId, fade);
                yield break;
            }
            while (!Mathf.Approximately(fade, target))
            {
                fade = Mathf.MoveTowards(fade, target, Time.deltaTime / fadeDuration);
                mat.SetFloat(FadeId, fade);
                yield return null;
            }
        }

        // ------------------------------------------------------------------ 360 영상
        IEnumerator PlayVideo(int i, PhotoEntry p)
        {
            StopVideo();
            videoPlayer.url = p.url;
            videoPlayer.isLooping = loopVideo;
            videoPlayer.Prepare();
            float t0 = Time.realtimeSinceStartup;
            while (!videoPlayer.isPrepared && Time.realtimeSinceStartup - t0 < 20f) yield return null;
            if (!videoPlayer.isPrepared)
            {
                Debug.LogError($"[Pano360] 영상 준비 실패 {p.url}");
                label.Show($"Failed to load video\n{p.name}\n(codec / resolution not supported?)", labelDuration * 2f);
                videoPlayer.Stop();
                yield break;
            }

            int w = (int)videoPlayer.width, h = (int)videoPlayer.height;
            // 영상 크기 그대로의 RenderTexture (밉맵 없음 — 셰이더가 LOD0 으로 샘플링)
            videoRT = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32)
            {
                name = "Pano360 Video", useMipMap = false, autoGenerateMips = false,
                wrapModeU = TextureWrapMode.Repeat, wrapModeV = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear
            };
            videoRT.Create();
            videoPlayer.targetTexture = videoRT;
            mat.SetTexture(MainTexId, videoRT);
            mat.SetFloat(StereoId, p.stereo ? 1f : 0f);
            if (current != null) { Destroy(current); current = null; }
            if (recenterOnChange) Recenter();

            videoPlayer.Play();
            float want = p.stereo ? 1f : 2f, ratio = (float)w / h;
            string warn = Mathf.Abs(ratio - want) > 0.03f
                ? $"\n\nWARNING: {w}x{h} is not {(p.stereo ? "1:1" : "2:1")} - video will look stretched"
                : "";
            label.Show($"{i + 1} / {photos.Count}  [VIDEO]\n{p.name}{(p.stereo ? "  [3D]" : "")}\n" +
                       $"{w}x{h} / {videoPlayer.frameRate:0.#}fps / {videoPlayer.length:0.0}s{warn}",
                       warn.Length > 0 ? labelDuration * 2f : labelDuration);
        }

        void StopVideo()
        {
            if (videoPlayer != null)
            {
                videoPlayer.Stop();
                videoPlayer.targetTexture = null;
            }
            ReleaseVideoRT();
        }

        void ReleaseVideoRT()
        {
            if (videoRT != null)
            {
                if (mat != null && mat.GetTexture(MainTexId) == videoRT) mat.SetTexture(MainTexId, null);
                if (mat != null && mat.GetTexture(BlendTexId) == videoRT) { mat.SetTexture(BlendTexId, null); mat.SetFloat(BlendId, 0f); }
                videoRT.Release();
                Destroy(videoRT);
                videoRT = null;
            }
        }

        void OnApplicationPause(bool paused)
        {
            // 헤드셋을 벗으면 영상도 멈췄다가 다시 쓰면 이어서 재생
            if (videoPlayer == null || videoRT == null || eventHeld || (!eventPlaying && mat.GetTexture(MainTexId) != videoRT)) return;
            if (paused) videoPlayer.Pause(); else videoPlayer.Play();
        }

        void OnDestroy()
        {
            StopVideo();
            ReleaseEventMask();
            if (current != null) Destroy(current);
            if (mat != null) Destroy(mat);
        }
    }
}
