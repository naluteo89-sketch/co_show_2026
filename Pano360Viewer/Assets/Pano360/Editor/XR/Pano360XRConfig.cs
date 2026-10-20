using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEditor.XR.OpenXR.Features;
using UnityEngine;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;

namespace Pano360.EditorTools.XR
{
    /// <summary>
    /// OpenXR 패키지가 설치되면 컴파일되는 어셈블리.
    /// XR Plug-in Management에 OpenXR 로더를 등록하고 Meta Quest 관련 기능을 켭니다.
    ///  - Android: Meta Quest Support, Touch 컨트롤러 프로필, Hand Tracking
    ///  - Windows(Standalone): Quest Link로 에디터 Play 테스트용
    /// </summary>
    [InitializeOnLoad]
    public static class Pano360XRConfig
    {
        const string Marker = "ProjectSettings/Pano360_XRSetup.done";

        // 클래스 이름으로 켭니다 (패키지 버전에 따라 일부는 없을 수 있음)
        static readonly string[] AndroidFeatures =
        {
            "MetaQuestFeature",
            "OculusTouchControllerProfile",
            "MetaQuestTouchPlusControllerProfile",
            "MetaQuestTouchProControllerProfile",
            "HandTracking",
        };

        static readonly string[] StandaloneFeatures =
        {
            "OculusTouchControllerProfile",
            "MetaQuestTouchPlusControllerProfile",
            "MetaQuestTouchProControllerProfile",
            "HandTracking",
        };

        static Pano360XRConfig()
        {
            EditorApplication.delayCall += () =>
            {
                if (Application.isBatchMode || File.Exists(Marker)) return;
                if (Apply()) File.WriteAllText(Marker, DateTime.Now.ToString("o"));
            };
        }

        public static bool Apply()
        {
            try
            {
                var perTarget = GetOrCreatePerBuildTarget();
                Configure(perTarget, BuildTargetGroup.Android, AndroidFeatures);
                Configure(perTarget, BuildTargetGroup.Standalone, StandaloneFeatures);
                AssetDatabase.SaveAssets();
                Debug.Log("[Pano360] OpenXR · Meta Quest · Hand Tracking 설정 완료.");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError("[Pano360] XR 자동 설정 실패 — Project Settings > XR Plug-in Management에서 수동으로 설정하세요.\n" + e);
                return false;
            }
        }

        static XRGeneralSettingsPerBuildTarget GetOrCreatePerBuildTarget()
        {
            EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey, out XRGeneralSettingsPerBuildTarget settings);
            if (settings != null) return settings;

            const string dir = "Assets/XR";
            if (!AssetDatabase.IsValidFolder(dir)) AssetDatabase.CreateFolder("Assets", "XR");
            settings = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
            AssetDatabase.CreateAsset(settings, dir + "/XRGeneralSettingsPerBuildTarget.asset");
            EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey, settings, true);
            return settings;
        }

        static void Configure(XRGeneralSettingsPerBuildTarget perTarget, BuildTargetGroup group, string[] featureNames)
        {
            if (!perTarget.HasSettingsForBuildTarget(group))
                perTarget.CreateDefaultSettingsForBuildTarget(group);
            if (!perTarget.HasManagerSettingsForBuildTarget(group))
                perTarget.CreateDefaultManagerSettingsForBuildTarget(group);

            var general = perTarget.SettingsForBuildTarget(group);
            general.InitManagerOnStart = true;
            XRPackageMetadataStore.AssignLoader(general.Manager, typeof(OpenXRLoader).FullName, group);
            EditorUtility.SetDirty(general);

            FeatureHelpers.RefreshFeatures(group);
            var oxr = OpenXRSettings.GetSettingsForBuildTargetGroup(group);
            if (oxr == null)
            {
                Debug.LogWarning($"[Pano360] {group}용 OpenXR 설정을 찾지 못했습니다.");
                return;
            }

            foreach (var f in oxr.GetFeatures())
            {
                if (featureNames.Contains(f.GetType().Name))
                {
                    f.enabled = true;
                    EditorUtility.SetDirty(f);
                }
            }
            oxr.renderMode = OpenXRSettings.RenderMode.SinglePassInstanced;
            EditorUtility.SetDirty(oxr);
        }
    }
}
