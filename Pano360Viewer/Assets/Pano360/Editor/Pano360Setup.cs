using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Pano360.EditorTools
{
    /// <summary>
    /// 프로젝트를 처음 열면 자동으로:
    ///  - Quest용 Player 설정 적용 + Main 씬 생성/등록
    ///  - OpenXR, XR Hands 패키지 설치 (에디터 버전에 맞는 최신 호환 버전)
    /// 설치 후 Pano360XRConfig(별도 어셈블리)가 OpenXR · Meta Quest 기능을 자동으로 켭니다.
    /// 모든 단계는 상단 메뉴 [Pano360]에서 수동 실행도 가능합니다.
    /// </summary>
    [InitializeOnLoad]
    public static class Pano360Setup
    {
        public const string ScenePath = "Assets/Scenes/Main.unity";
        const string BaseMarker = "ProjectSettings/Pano360_BaseSetup.done";
        static readonly string[] RequiredPackages = { "com.unity.xr.openxr", "com.unity.xr.hands" };

        static AddAndRemoveRequest request;

        static Pano360Setup()
        {
            EditorApplication.delayCall += AutoRun;
        }

        static void AutoRun()
        {
            if (Application.isBatchMode) return;

            if (!File.Exists(BaseMarker))
            {
                ApplyProjectSettings();
                File.WriteAllText(BaseMarker, DateTime.Now.ToString("o"));
            }

            if (!SessionState.GetBool("Pano360.PackagesChecked", false))
            {
                SessionState.SetBool("Pano360.PackagesChecked", true);
                InstallPackages();
            }
        }

        [MenuItem("Pano360/1. 필수 패키지 설치 (OpenXR, XR Hands)", priority = 1)]
        public static void InstallPackages()
        {
            var missing = RequiredPackages
                .Where(p => UnityEditor.PackageManager.PackageInfo.FindForPackageName(p) == null)
                .ToArray();
            if (missing.Length == 0)
            {
                Debug.Log("[Pano360] 필수 패키지가 모두 설치되어 있습니다.");
                return;
            }

            Debug.Log("[Pano360] 패키지 설치 중: " + string.Join(", ", missing));
            request = Client.AddAndRemove(missing, null);
            EditorApplication.update += WaitForRequest;
        }

        static void WaitForRequest()
        {
            if (request == null || !request.IsCompleted) return;
            EditorApplication.update -= WaitForRequest;
            if (request.Status == StatusCode.Success)
                Debug.Log("[Pano360] 패키지 설치 완료. 컴파일 후 XR 설정이 자동 적용됩니다.");
            else
                Debug.LogError("[Pano360] 패키지 설치 실패: " + request.Error?.message);
            request = null;
        }

        [MenuItem("Pano360/2. Quest용 프로젝트 설정 적용", priority = 2)]
        public static void ApplyProjectSettings()
        {
            PlayerSettings.companyName = "Witches";
            PlayerSettings.productName = "Pano360 Viewer";

            var android = NamedBuildTarget.Android;
            PlayerSettings.SetApplicationIdentifier(android, "com.witches.pano360");
            PlayerSettings.SetScriptingBackend(android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { UnityEngine.Rendering.GraphicsDeviceType.Vulkan });
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
            EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ASTC;

            EnsureScene();
            AssetDatabase.SaveAssets();
            Debug.Log("[Pano360] Quest용 Player 설정 및 Main 씬 등록 완료.");
        }

        [MenuItem("Pano360/3. Android 플랫폼으로 전환", priority = 3)]
        public static void SwitchToAndroid()
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
        }

        [MenuItem("Pano360/4. XR(OpenXR · Meta Quest) 설정 적용", priority = 4)]
        public static void ConfigureXR()
        {
            var t = Type.GetType("Pano360.EditorTools.XR.Pano360XRConfig, Pano360.Editor.XR");
            if (t == null)
            {
                EditorUtility.DisplayDialog("Pano360",
                    "OpenXR 패키지가 아직 설치되지 않았습니다.\n[Pano360 > 1. 필수 패키지 설치]를 먼저 실행하세요.", "확인");
                return;
            }
            var ok = (bool)t.GetMethod("Apply", BindingFlags.Public | BindingFlags.Static).Invoke(null, null);
            EditorUtility.DisplayDialog("Pano360",
                ok ? "OpenXR · Meta Quest · Hand Tracking 설정을 적용했습니다.\nProject Settings > XR Plug-in Management에서 확인할 수 있습니다."
                   : "XR 설정 중 오류가 발생했습니다. Console을 확인하세요.", "확인");
        }

        [MenuItem("Pano360/포함할 사진 폴더 열기 (StreamingAssets)", priority = 20)]
        public static void OpenPhotoFolder()
        {
            var dir = Path.Combine(Application.dataPath, "StreamingAssets", "Pano360");
            Directory.CreateDirectory(dir);
            EditorUtility.RevealInFinder(dir + Path.DirectorySeparatorChar);
        }

        static void EnsureScene()
        {
            if (!File.Exists(ScenePath))
            {
                if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
                Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
                var cam = camGo.AddComponent<Camera>();
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = Color.black;
                camGo.AddComponent<AudioListener>();
                camGo.AddComponent<HeadTracker>();

                var viewer = new GameObject("Pano360 Viewer");
                viewer.AddComponent<PanoInput>();
                viewer.AddComponent<PanoViewer>();

                EditorSceneManager.SaveScene(scene, ScenePath);
            }
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        }
    }
}
