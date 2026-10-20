using UnityEngine;

namespace Pano360
{
    /// <summary>시선 앞에 잠깐 떠오르는 안내 텍스트 (사진 번호/파일명/오류 등)</summary>
    public class PanoLabel : MonoBehaviour
    {
        public float distance = 2f;
        public float heightOffset = -0.35f;
        /// 켜면 고개를 돌려도 계속 정면에 머무름 (홈 화면 안내용)
        public bool follow;

        TextMesh text;
        MeshRenderer rend;
        Transform head;
        float hideAt;

        public static PanoLabel Create(Transform head)
        {
            var go = new GameObject("Pano360 Label");
            var label = go.AddComponent<PanoLabel>();
            label.head = head;

            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            label.text = go.AddComponent<TextMesh>();
            label.text.font = font;
            label.text.fontSize = 80;
            label.text.characterSize = 0.01f;
            label.text.anchor = TextAnchor.MiddleCenter;
            label.text.alignment = TextAlignment.Center;
            label.text.color = Color.white;

            label.rend = go.GetComponent<MeshRenderer>();
            label.rend.sharedMaterial = font.material;
            label.rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            label.rend.receiveShadows = false;
            label.rend.enabled = false;
            return label;
        }

        public void Show(string message, float duration)
        {
            text.text = message;
            hideAt = float.IsInfinity(duration) ? float.PositiveInfinity : Time.time + duration;
            Place();
            rend.enabled = true;
        }

        public void Hide() => rend.enabled = false;

        void Place()
        {
            if (head == null) return;
            var fwd = head.forward;
            fwd.y = 0;
            if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward;
            fwd.Normalize();
            transform.position = head.position + fwd * distance + Vector3.up * heightOffset;
            transform.rotation = Quaternion.LookRotation(transform.position - head.position);
        }

        void Update()
        {
            if (rend.enabled && Time.time > hideAt) rend.enabled = false;
            if (rend.enabled && follow) Place();
        }
    }
}
