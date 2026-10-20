using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Pano360.EditorTools
{
    /// <summary>
    /// 빌드 직전에 StreamingAssets/Pano360 폴더의 사진 목록(photos.json)을 생성합니다.
    /// (Android에서는 APK 안의 폴더를 런타임에 탐색할 수 없기 때문)
    /// </summary>
    public class PhotoListBuilder : IPreprocessBuildWithReport
    {
        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report) => Generate();

        [MenuItem("Pano360/사진 목록(photos.json) 갱신", priority = 21)]
        public static void Generate()
        {
            var dir = Path.Combine(Application.dataPath, "StreamingAssets", PhotoSource.StreamingFolder);
            Directory.CreateDirectory(dir);

            var files = Directory.GetFiles(dir)
                .Where(PhotoSource.IsImage)
                .Select(p => Path.GetFileName(p))
                .OrderBy(n => n, System.StringComparer.OrdinalIgnoreCase)
                .ToArray();

            var json = JsonUtility.ToJson(new PhotoList { files = files }, true);
            File.WriteAllText(Path.Combine(dir, PhotoSource.ListFileName), json);
            Debug.Log($"[Pano360] photos.json 생성: {files.Length}장");
        }
    }
}
