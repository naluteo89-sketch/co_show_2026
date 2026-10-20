using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using XRUsages = UnityEngine.XR.CommonUsages;
#if PANO_XR_HANDS
using UnityEngine.XR.Hands;
#endif
#if PANO_INPUTSYSTEM && ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace Pano360
{
    public enum PanoAction { None, Next, Prev, Recenter }

    /// <summary>
    /// 조작 매핑
    ///  컨트롤러: A/X 또는 오른쪽 트리거 = 다음, B/Y 또는 왼쪽 트리거 = 이전,
    ///            스틱 좌/우 = 이전/다음, 스틱 누르기 = 정면 맞추기
    ///  핸드트래킹: 오른손 핀치 = 다음, 왼손 핀치 = 이전, 양손 동시 핀치 = 정면 맞추기
    ///  에디터 키보드: ←/→ = 이전/다음, R = 정면 맞추기
    /// </summary>
    [DefaultExecutionOrder(-50)]
    public class PanoInput : MonoBehaviour
    {
        [Header("Thumbstick")]
        public float stickPress = 0.7f;
        public float stickRelease = 0.3f;

        [Header("Hand pinch (m)")]
        public float pinchOn = 0.015f;
        public float pinchOff = 0.035f;

        struct ControllerState { public bool primary, secondary, trigger, click, stickLatched; }

        ControllerState left, right;
        PanoAction pending;

        public PanoAction Consume()
        {
            var a = pending;
            pending = PanoAction.None;
            return a;
        }

        void Fire(PanoAction a)
        {
            if (pending == PanoAction.None || a == PanoAction.Recenter) pending = a;
        }

        void Update()
        {
            PollController(XRNode.LeftHand, ref left, true);
            PollController(XRNode.RightHand, ref right, false);
#if PANO_XR_HANDS
            PollHands();
#endif
            PollKeyboard();
        }

        void PollController(XRNode node, ref ControllerState s, bool isLeft)
        {
            var d = InputDevices.GetDeviceAtXRNode(node);
            if (!d.isValid || (d.characteristics & InputDeviceCharacteristics.Controller) == 0)
            {
                s = default;
                return;
            }

            d.TryGetFeatureValue(XRUsages.primaryButton, out bool primary);     // A / X
            d.TryGetFeatureValue(XRUsages.secondaryButton, out bool secondary); // B / Y
            d.TryGetFeatureValue(XRUsages.triggerButton, out bool trigger);
            d.TryGetFeatureValue(XRUsages.primary2DAxisClick, out bool click);
            d.TryGetFeatureValue(XRUsages.primary2DAxis, out Vector2 stick);

            if (primary && !s.primary) Fire(PanoAction.Next);
            if (secondary && !s.secondary) Fire(PanoAction.Prev);
            if (trigger && !s.trigger) Fire(isLeft ? PanoAction.Prev : PanoAction.Next);
            if (click && !s.click) Fire(PanoAction.Recenter);

            if (!s.stickLatched && Mathf.Abs(stick.x) > stickPress)
            {
                s.stickLatched = true;
                Fire(stick.x > 0 ? PanoAction.Next : PanoAction.Prev);
            }
            else if (s.stickLatched && Mathf.Abs(stick.x) < stickRelease)
            {
                s.stickLatched = false;
            }

            s.primary = primary;
            s.secondary = secondary;
            s.trigger = trigger;
            s.click = click;
        }

#if PANO_XR_HANDS
        static readonly List<XRHandSubsystem> handSubsystems = new List<XRHandSubsystem>();
        bool leftPinch, rightPinch;
        // 한 손 핀치는 짧게 대기했다가 실행 → 그 사이 다른 손도 핀치하면 '정면 맞추기'
        int waitingHand; // 0 없음, 1 왼손, 2 오른손
        float waitingSince;
        const float BothHandsWindow = 0.2f;

        void PollHands()
        {
            handSubsystems.Clear();
            SubsystemManager.GetSubsystems(handSubsystems);
            XRHandSubsystem hs = null;
            foreach (var s in handSubsystems)
                if (s.running) { hs = s; break; }
            if (hs == null) return;

            bool l = UpdatePinch(hs.leftHand, ref leftPinch);
            bool r = UpdatePinch(hs.rightHand, ref rightPinch);

            if ((l && r) || (l && waitingHand == 2) || (r && waitingHand == 1))
            {
                waitingHand = 0;
                Fire(PanoAction.Recenter);
            }
            else if (l || r)
            {
                waitingHand = l ? 1 : 2;
                waitingSince = Time.time;
            }

            if (waitingHand != 0 && Time.time - waitingSince > BothHandsWindow)
            {
                Fire(waitingHand == 2 ? PanoAction.Next : PanoAction.Prev);
                waitingHand = 0;
            }
        }

        /// 핀치가 새로 시작된 프레임에 true
        bool UpdatePinch(XRHand hand, ref bool pinching)
        {
            if (!hand.isTracked) { pinching = false; return false; }
            if (!hand.GetJoint(XRHandJointID.IndexTip).TryGetPose(out Pose index) ||
                !hand.GetJoint(XRHandJointID.ThumbTip).TryGetPose(out Pose thumb))
                return false;

            float dist = Vector3.Distance(index.position, thumb.position);
            if (!pinching && dist < pinchOn) { pinching = true; return true; }
            if (pinching && dist > pinchOff) pinching = false;
            return false;
        }
#endif

        void PollKeyboard()
        {
#if PANO_INPUTSYSTEM && ENABLE_INPUT_SYSTEM
            var k = Keyboard.current;
            if (k == null) return;
            if (k.rightArrowKey.wasPressedThisFrame || k.spaceKey.wasPressedThisFrame) Fire(PanoAction.Next);
            if (k.leftArrowKey.wasPressedThisFrame) Fire(PanoAction.Prev);
            if (k.rKey.wasPressedThisFrame) Fire(PanoAction.Recenter);
#elif ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.Space)) Fire(PanoAction.Next);
            if (Input.GetKeyDown(KeyCode.LeftArrow)) Fire(PanoAction.Prev);
            if (Input.GetKeyDown(KeyCode.R)) Fire(PanoAction.Recenter);
#endif
        }
    }
}
