#if UNITY_ANDROID
using System.IO;
using System.Xml;
using UnityEditor.Android;
using UnityEngine;

namespace Pano360.EditorTools
{
    /// <summary>
    /// 빌드되는 AndroidManifest에 Quest 핸드트래킹 권한/기능이 없으면 추가합니다.
    /// </summary>
    public class QuestManifestPatcher : IPostGenerateGradleAndroidProject
    {
        const string AndroidNs = "http://schemas.android.com/apk/res/android";
        public int callbackOrder => 999;

        public void OnPostGenerateGradleAndroidProject(string path)
        {
            var manifestPath = Path.Combine(path, "src", "main", "AndroidManifest.xml");
            if (!File.Exists(manifestPath)) return;

            var doc = new XmlDocument();
            doc.Load(manifestPath);
            var root = doc.DocumentElement;
            if (root == null) return;

            bool changed = false;
            changed |= AddIfMissing(doc, root, "uses-permission", "com.oculus.permission.HAND_TRACKING", null);
            changed |= AddIfMissing(doc, root, "uses-feature", "oculus.software.handtracking", "false");

            if (changed)
            {
                doc.Save(manifestPath);
                Debug.Log("[Pano360] AndroidManifest에 핸드트래킹 항목 추가");
            }
        }

        static bool AddIfMissing(XmlDocument doc, XmlElement root, string tag, string name, string required)
        {
            foreach (XmlNode n in root.SelectNodes(tag))
                if (n is XmlElement e && e.GetAttribute("name", AndroidNs) == name)
                    return false;

            var el = doc.CreateElement(tag);
            el.SetAttribute("name", AndroidNs, name);
            if (required != null) el.SetAttribute("required", AndroidNs, required);
            root.AppendChild(el);
            return true;
        }
    }
}
#endif
