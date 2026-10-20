using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.XR;

namespace Pano360
{
    /// <summary>
    /// 360 사진 뷰어 본체. 씬에 이 컴포넌트 하나만 있으면 카메라/구/입력/안내 텍스트를 스스로 구성합니다.
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
        [Tooltip("눈 렌더 해상도 배율. Quest 3 기본(1.0)=1680x1760/눈 으로 패널(2064x2208)의 약 80%. 1.25 ≈ 패널과 1:1")]
        [Range(1f, 1.5f)] public float eyeResolutionScale = 1.25f;

        Camera cam;
        Transform head;
        Transform sphere;
        Material mat;
        PanoLabel label;
        PanoInput input;
        PanoHomeButton homeButton;

        enum Mode { Home, Viewing }
        Mode mode = Mode.Home;
        bool homeRequested;

        readonly List<PhotoEntry> photos = new List<PhotoEntry>();
        int index = -1;
        Texture2D current;
        bool busy;
        float fade = 1f;

        static readonly int MainTexId = Shader.PropertyToID("_MainTex");
        static readonly int StereoId = Shader.PropertyToID("_Stereo");
        static readonly int FadeId = Shader.PropertyToID("_Fade");

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
            input.pointerOverHome = node => homeButton.IsHoveredBy(node);
            input.gazeOverHome = () => homeButton.GazeHover;
        }

        IEnumerator Start()
        {
            label.Show("Loading...", float.PositiveInfinity);
            yield return PhotoSource.Collect(photos);

            if (photos.Count == 0)
            {
                label.Show(
                    "No 360 photos found.\n\n" +
                    "Put equirectangular JPG/PNG files in\n" +
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
            label.follow = true;
            label.Show(
                "PANO360\n\n" +
                $"{photos.Count} photos\n\n" +
                "Trigger / A : Start",
                float.PositiveInfinity);
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
            homeButton.SetVisible(false);
            yield return FadeTo(1f);
            mat.SetTexture(MainTexId, null);
            if (current != null) { Destroy(current); current = null; }
            index = -1;
            busy = false;
            ShowHome();
        }

        IEnumerator StartViewing()
        {
            mode = Mode.Viewing;
            label.follow = false;
            label.Hide();
            Recenter();                     // 첫 사진 정면을 지금 보는 방향으로
            yield return ShowPhoto(0);
        }

        void Update()
        {
            var action = input.Consume();

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
            yield return FadeTo(1f);

            var p = photos[i];
            using (var req = UnityWebRequestTexture.GetTexture(p.url, true))
            {
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    var tex = DownloadHandlerTexture.GetContent(req);
                    tex.wrapModeU = TextureWrapMode.Repeat;
                    tex.wrapModeV = TextureWrapMode.Clamp;
                    tex.filterMode = FilterMode.Bilinear;

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
                    label.Show($"{i + 1} / {photos.Count}\n{p.name}{(p.stereo ? "  [3D]" : "")}\n{tex.width}x{tex.height}{warn}",
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
            if (current != null) yield return FadeTo(0f);
            if (showHomeButton && mode == Mode.Viewing) homeButton.SetVisible(true);
            busy = false;
        }

        IEnumerator FadeTo(float target)
        {
            if (fadeDuration <= 0f || current == null)
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

        void OnDestroy()
        {
            if (current != null) Destroy(current);
            if (mat != null) Destroy(mat);
        }
    }
}
