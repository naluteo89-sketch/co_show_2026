using UnityEngine;
using UnityEngine.XR;
using XRUsages = UnityEngine.XR.CommonUsages;

namespace Pano360
{
    /// <summary>
    /// 사진을 보는 동안 시선 아래쪽에 떠 있는 "HOME" 취소 버튼.
    ///  - 컨트롤러: 레이를 버튼에 맞추고 트리거
    ///  - 핸드트래킹: 버튼을 바라본 채 핀치
    ///  - (선택) dwellToActivate 를 켜면 버튼을 gazeDwell 초 바라보기만 해도 실행
    ///    전시 중 관람객이 발밑을 보다가 실수로 나가지 않도록 기본은 꺼 둠
    /// 버튼은 머리 Yaw 를 느슨하게 따라오고, 아래(pitch -32°)에 있어 사진 감상을 가리지 않습니다.
    /// </summary>
    public class PanoHomeButton : MonoBehaviour
    {
        public float distance = 1.5f;
        public float pitchDown = 32f;
        public Vector2 size = new Vector2(0.46f, 0.15f);
        public float followAngle = 50f;
        public bool dwellToActivate = false;
        public float gazeDwell = 2f;

        public System.Action onActivate;

        Transform head;
        Transform bg, fill, cursor;
        TextMesh text;
        Material bgMat, fillMat, cursorMat;
        float yaw;
        float dwell;
        bool visible;
        bool hoverLeft, hoverRight, hoverGaze;

        static readonly Color Idle = new Color(0.08f, 0.08f, 0.1f, 0.72f);
        static readonly Color Hot = new Color(0.75f, 0.18f, 0.18f, 0.9f);
        static readonly InputFeatureUsage<Quaternion> PointerRotation = new InputFeatureUsage<Quaternion>("PointerRotation");
        static readonly InputFeatureUsage<Vector3> PointerPosition = new InputFeatureUsage<Vector3>("PointerPosition");

        public static PanoHomeButton Create(Transform head)
        {
            var go = new GameObject("Pano360 Home Button");
            var b = go.AddComponent<PanoHomeButton>();
            b.head = head;
            b.Build();
            b.SetVisible(false);
            return b;
        }

        public bool IsHoveredBy(XRNode node) =>
            visible && (node == XRNode.LeftHand ? hoverLeft : node == XRNode.RightHand ? hoverRight : false);

        public bool GazeHover => visible && hoverGaze;

        public void SetVisible(bool v)
        {
            visible = v;
            dwell = 0f;
            gameObject.SetActive(v);
            if (v) SnapToHead();
        }

        void Build()
        {
            var shader = Shader.Find("Sprites/Default"); // 항상 빌드에 포함되는 기본 셰이더
            bgMat = new Material(shader) { color = Idle, renderQueue = 3000 };
            fillMat = new Material(shader) { color = new Color(1f, 1f, 1f, 0.35f), renderQueue = 3001 };
            cursorMat = new Material(shader) { color = Color.white, renderQueue = 3003 };

            bg = MakeQuad("Background", bgMat, transform);
            bg.localScale = new Vector3(size.x, size.y, 1f);

            fill = MakeQuad("Gaze Fill", fillMat, transform);
            fill.localPosition = new Vector3(0f, 0f, -0.001f);
            fill.localScale = new Vector3(0f, size.y, 1f);

            cursor = MakeQuad("Cursor", cursorMat, transform);
            cursor.localScale = Vector3.one * 0.018f;
            cursor.gameObject.SetActive(false);

            var tgo = new GameObject("Text");
            tgo.transform.SetParent(transform, false);
            tgo.transform.localPosition = new Vector3(0f, 0f, -0.002f);
            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text = tgo.AddComponent<TextMesh>();
            text.font = font;
            text.text = "HOME";
            text.fontSize = 80;
            text.characterSize = 0.0075f;
            text.anchor = TextAnchor.MiddleCenter;
            text.alignment = TextAlignment.Center;
            text.color = Color.white;
            var tr = tgo.GetComponent<MeshRenderer>();
            tr.sharedMaterial = font.material;
            tr.sharedMaterial.renderQueue = 3002;
        }

        static Transform MakeQuad(string name, Material m, Transform parent)
        {
            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            q.name = name;
            Destroy(q.GetComponent<Collider>());
            q.transform.SetParent(parent, false);
            var r = q.GetComponent<MeshRenderer>();
            r.sharedMaterial = m;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
            return q.transform;
        }

        float HeadYaw()
        {
            var f = head.forward;
            f.y = 0f;
            if (f.sqrMagnitude < 1e-4f) f = head.up * -1f;
            return Mathf.Atan2(f.x, f.z) * Mathf.Rad2Deg;
        }

        void SnapToHead()
        {
            if (head == null) return;
            yaw = HeadYaw();
            Place();
        }

        void Place()
        {
            var dir = Quaternion.Euler(pitchDown, yaw, 0f) * Vector3.forward;
            transform.position = head.position + dir * distance;
            transform.rotation = Quaternion.LookRotation(transform.position - head.position);
        }

        void LateUpdate()
        {
            if (!visible || head == null) return;

            // 머리를 크게 돌렸을 때만 따라옴
            float hy = HeadYaw();
            float d = Mathf.DeltaAngle(yaw, hy);
            if (Mathf.Abs(d) > followAngle) yaw = Mathf.MoveTowardsAngle(yaw, hy, Mathf.Max(90f, Mathf.Abs(d) * 3f) * Time.deltaTime);
            Place();

            // 시선
            hoverGaze = Hit(head.position, head.forward, out _);

            // 컨트롤러 레이
            Vector3 cursorPos = Vector3.zero;
            hoverLeft = ControllerHit(XRNode.LeftHand, ref cursorPos);
            hoverRight = ControllerHit(XRNode.RightHand, ref cursorPos);
            cursor.gameObject.SetActive(hoverLeft || hoverRight);
            if (hoverLeft || hoverRight) cursor.position = cursorPos - transform.forward * 0.003f;

            bool hot = hoverGaze || hoverLeft || hoverRight;
            bgMat.color = hot ? Hot : Idle;

            // 시선 유지 → 실행
            dwell = (dwellToActivate && hoverGaze) ? dwell + Time.deltaTime : 0f;
            float k = Mathf.Clamp01(dwell / gazeDwell);
            fill.localScale = new Vector3(size.x * k, size.y, 1f);
            fill.localPosition = new Vector3(-size.x * 0.5f * (1f - k), 0f, -0.001f);
            if (dwell >= gazeDwell)
            {
                dwell = 0f;
                onActivate?.Invoke();
            }
        }

        bool ControllerHit(XRNode node, ref Vector3 hitPoint)
        {
            var c = InputDevices.GetDeviceAtXRNode(node);
            if (!c.isValid || (c.characteristics & InputDeviceCharacteristics.Controller) == 0) return false;

            if (!c.TryGetFeatureValue(PointerRotation, out Quaternion rot) &&
                !c.TryGetFeatureValue(XRUsages.deviceRotation, out rot)) return false;
            if (!c.TryGetFeatureValue(PointerPosition, out Vector3 pos) &&
                !c.TryGetFeatureValue(XRUsages.devicePosition, out pos)) return false;

            // 트래킹 공간 → 월드: 카메라는 회전만 따라가므로 머리 위치 차이만큼 옮김
            var hmd = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            Vector3 headTracked = Vector3.zero;
            if (hmd.isValid && !hmd.TryGetFeatureValue(XRUsages.centerEyePosition, out headTracked))
                hmd.TryGetFeatureValue(XRUsages.devicePosition, out headTracked);
            var origin = head.position + (pos - headTracked);

            if (Hit(origin, rot * Vector3.forward, out var p)) { hitPoint = p; return true; }
            return false;
        }

        bool Hit(Vector3 origin, Vector3 dir, out Vector3 point)
        {
            point = default;
            var n = transform.forward;
            float denom = Vector3.Dot(dir, n);
            if (denom < 1e-4f) return false;
            float t = Vector3.Dot(transform.position - origin, n) / denom;
            if (t <= 0f) return false;
            point = origin + dir * t;
            var local = transform.InverseTransformPoint(point);
            const float pad = 1.15f; // 살짝 넉넉하게
            return Mathf.Abs(local.x) <= size.x * 0.5f * pad && Mathf.Abs(local.y) <= size.y * 0.5f * pad;
        }

        void OnDestroy()
        {
            if (bgMat) Destroy(bgMat);
            if (fillMat) Destroy(fillMat);
            if (cursorMat) Destroy(cursorMat);
        }
    }
}
