using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using XRUsages = UnityEngine.XR.CommonUsages;
#if PANO_INPUTSYSTEM && ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace Pano360
{
    /// <summary>
    /// HMD 회전을 카메라에 적용합니다 (360 사진은 위치 이동이 없어야 자연스러우므로 기본은 회전만).
    /// 헤드셋이 없을 때(에디터)는 마우스 오른쪽 버튼 드래그로 둘러볼 수 있습니다.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    public class HeadTracker : MonoBehaviour
    {
        [Tooltip("켜면 머리 위치 이동도 반영 (360 사진에서는 보통 끄는 것이 좋음)")]
        public bool trackPosition = false;
        public float mouseSensitivity = 3f;

        float editorYaw, editorPitch;
        bool hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply;
        void OnDisable() => Application.onBeforeRender -= Apply;

        void Update()
        {
            Apply();
            if (!hmdActive) EditorLook();
        }

        void Apply()
        {
            var head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            hmdActive = head.isValid;
            if (!hmdActive) return;

            if (head.TryGetFeatureValue(XRUsages.centerEyeRotation, out Quaternion rot) ||
                head.TryGetFeatureValue(XRUsages.deviceRotation, out rot))
                transform.localRotation = rot;

            if (trackPosition &&
                (head.TryGetFeatureValue(XRUsages.centerEyePosition, out Vector3 pos) ||
                 head.TryGetFeatureValue(XRUsages.devicePosition, out pos)))
                transform.localPosition = pos;
        }

        void EditorLook()
        {
            Vector2 delta = Vector2.zero;
            bool held = false;
#if PANO_INPUTSYSTEM && ENABLE_INPUT_SYSTEM
            if (Mouse.current != null)
            {
                held = Mouse.current.rightButton.isPressed;
                delta = Mouse.current.delta.ReadValue() * 0.1f;
            }
#elif ENABLE_LEGACY_INPUT_MANAGER
            held = Input.GetMouseButton(1);
            delta = new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
#endif
            if (!held) return;
            editorYaw += delta.x * mouseSensitivity;
            editorPitch = Mathf.Clamp(editorPitch - delta.y * mouseSensitivity, -89f, 89f);
            transform.localRotation = Quaternion.Euler(editorPitch, editorYaw, 0f);
        }
    }
}
