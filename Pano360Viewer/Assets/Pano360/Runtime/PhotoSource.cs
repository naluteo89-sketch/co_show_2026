using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace Pano360
{
    public struct PhotoEntry
    {
        public string name;
        public string url;
        public bool stereo; // Top-Bottom 스테레오 여부
    }

    [Serializable]
    public class PhotoList
    {
        public string[] files = Array.Empty<string>();
    }

    /// <summary>
    /// 360 사진 목록을 모읍니다.
    ///  1) 앱에 포함된 사진: Assets/StreamingAssets/Pano360/  (빌드 시 photos.json 목록 자동 생성)
    ///  2) 기기 폴더 사진:  Android/data/&lt;패키지명&gt;/files/Photos/  (USB로 복사)
    /// </summary>
    public static class PhotoSource
    {
        public const string StreamingFolder = "Pano360";
        public const string ListFileName = "photos.json";

        static readonly string[] Extensions = { ".jpg", ".jpeg", ".png" };

        public static string DeviceFolder => Path.Combine(Application.persistentDataPath, "Photos");

        public static bool IsImage(string path)
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();
            return Array.IndexOf(Extensions, ext) >= 0;
        }

        /// 파일명에 _TB, _OU, _3D 가 들어가면 위/아래 스테레오로 처리
        public static bool IsStereoName(string fileName)
        {
            var n = Path.GetFileNameWithoutExtension(fileName).ToUpperInvariant();
            return n.Contains("_TB") || n.Contains("_OU") || n.Contains("_3D");
        }

        public static IEnumerator Collect(List<PhotoEntry> result)
        {
            result.Clear();

            // 1) 앱에 포함된 사진
            var bundled = new List<PhotoEntry>();
            var baseDir = Application.streamingAssetsPath + "/" + StreamingFolder;
            if (Application.platform == RuntimePlatform.Android)
            {
                // Android에서는 StreamingAssets가 APK 내부라 폴더 탐색이 불가 → 빌드 때 만든 목록 사용
                using (var req = UnityWebRequest.Get(baseDir + "/" + ListFileName))
                {
                    yield return req.SendWebRequest();
                    if (req.result == UnityWebRequest.Result.Success)
                    {
                        var list = JsonUtility.FromJson<PhotoList>(req.downloadHandler.text);
                        if (list?.files != null)
                            foreach (var f in list.files)
                                bundled.Add(new PhotoEntry
                                {
                                    name = f,
                                    url = baseDir + "/" + Uri.EscapeDataString(f),
                                    stereo = IsStereoName(f)
                                });
                    }
                    else
                    {
                        Debug.Log($"[Pano360] 포함된 사진 목록 없음: {req.error}");
                    }
                }
            }
            else
            {
                AddFromDirectory(baseDir, bundled);
            }
            bundled.Sort((a, b) => string.Compare(a.name, b.name, StringComparison.OrdinalIgnoreCase));
            result.AddRange(bundled);

            // 2) 기기 폴더 사진
            var local = new List<PhotoEntry>();
            try
            {
                Directory.CreateDirectory(DeviceFolder);
                AddFromDirectory(DeviceFolder, local);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Pano360] 기기 폴더 읽기 실패: {e.Message}");
            }
            local.Sort((a, b) => string.Compare(a.name, b.name, StringComparison.OrdinalIgnoreCase));
            result.AddRange(local);

            Debug.Log($"[Pano360] 사진 {result.Count}장 (포함 {bundled.Count}, 기기 {local.Count}) / 기기 폴더: {DeviceFolder}");
        }

        static void AddFromDirectory(string dir, List<PhotoEntry> into)
        {
            if (!Directory.Exists(dir)) return;
            foreach (var path in Directory.GetFiles(dir))
            {
                if (!IsImage(path)) continue;
                var name = Path.GetFileName(path);
                into.Add(new PhotoEntry
                {
                    name = name,
                    url = new Uri(path).AbsoluteUri, // file:///... (공백/한글 이스케이프)
                    stereo = IsStereoName(name)
                });
            }
        }
    }
}
