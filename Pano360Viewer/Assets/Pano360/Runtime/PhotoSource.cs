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
        public bool video;  // 360 영상 (mp4 등) — VideoPlayer 로 재생
        public string eventUrl; // 사건 영상: 이 사진을 보다가 재생 → 끝나면 다시 사진 (이름_event.mp4)
        public string eventMaskUrl; // 사건 영상 범위: 흰색만 영상, 나머지는 사진 그대로 (이름_event_mask.png)
        public bool eventLoops;     // 이름_live.mp4: 평소 움직임 → 사건 → 평소 가 이어진 끊김 없는 반복 영상 (사진 위에 계속 재생)
        public bool HasEvent => !string.IsNullOrEmpty(eventUrl);
    }

    [Serializable]
    public class PhotoList
    {
        public string[] files = Array.Empty<string>();
    }

    /// <summary>
    /// 360 사진·영상 목록을 모읍니다.
    ///  1) 앱에 포함된 사진: Assets/StreamingAssets/Pano360/  (빌드 시 photos.json 목록 자동 생성)
    ///  2) 기기 폴더 사진:  Android/data/&lt;패키지명&gt;/files/Photos/  (USB로 복사)
    /// </summary>
    public static class PhotoSource
    {
        public const string StreamingFolder = "Pano360";
        public const string ListFileName = "photos.json";

        static readonly string[] Extensions = { ".jpg", ".jpeg", ".png" };
        static readonly string[] VideoExtensions = { ".mp4", ".m4v", ".mov", ".webm" };

        public static string DeviceFolder => Path.Combine(Application.persistentDataPath, "Photos");

        public static bool IsImage(string path)
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();
            return Array.IndexOf(Extensions, ext) >= 0;
        }

        public static bool IsVideo(string path)
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();
            return Array.IndexOf(VideoExtensions, ext) >= 0;
        }

        public static bool IsMedia(string path) => IsImage(path) || IsVideo(path);

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
                                    // VideoPlayer 는 StreamingAssets 경로를 그대로 받음 (이스케이프하면 못 찾음)
                                    url = IsVideo(f) ? baseDir + "/" + f : baseDir + "/" + Uri.EscapeDataString(f),
                                    stereo = IsStereoName(f),
                                    video = IsVideo(f)
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

            PairEvents(result);
            Debug.Log($"[Pano360] 사진·영상 {result.Count}개 (포함 {bundled.Count}, 기기 {local.Count}) / 기기 폴더: {DeviceFolder}");
        }

        public const string EventSuffix = "_event";
        public const string LiveSuffix = "_live";
        public const string EventMaskSuffix = "_event_mask";

        /// "XXX_event.mp4" 를 이름이 XXX 로 시작하는 사진에 붙임 (사진 → 사건 영상 → 사진). 짝이 없으면 일반 영상으로 둠.
        /// "XXX_event_mask.png" 가 있으면 그 흰색 범위에서만 영상이 보이고 나머지는 고화질 사진 그대로.
        public static void PairEvents(List<PhotoEntry> list)
        {
            // 1) 범위 마스크 그림 (XXX_event_mask.png) 은 사진 목록에서 빼 두었다가 짝지은 사진에 붙임
            var masks = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (int i = list.Count - 1; i >= 0; i--)
            {
                var m = list[i];
                if (m.video) continue;
                var stem = Path.GetFileNameWithoutExtension(m.name);
                if (!stem.EndsWith(EventMaskSuffix, StringComparison.OrdinalIgnoreCase)) continue;
                masks[stem.Substring(0, stem.Length - EventMaskSuffix.Length)] = m.url;
                list.RemoveAt(i);
            }
            // 2) XXX_live.mp4 (끊김 없는 반복) 먼저, 그다음 XXX_event.mp4. 같은 사진에 둘 다 있으면 live 만 씀
            foreach (var (suffix, loops) in new[] { (LiveSuffix, true), (EventSuffix, false) })
            {
                for (int i = list.Count - 1; i >= 0; i--)
                {
                    var v = list[i];
                    if (!v.video) continue;
                    var stem = Path.GetFileNameWithoutExtension(v.name);
                    if (!stem.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)) continue;
                    var baseName = stem.Substring(0, stem.Length - suffix.Length);
                    bool Match(PhotoEntry p) => !p.video &&
                        Path.GetFileNameWithoutExtension(p.name).StartsWith(baseName, StringComparison.OrdinalIgnoreCase);
                    int img = list.FindIndex(p => Match(p) && !p.HasEvent);
                    if (img < 0)
                    {
                        if (list.FindIndex(p => Match(p) && p.eventLoops) >= 0) list.RemoveAt(i);   // live 가 이미 있음 → 사건만 영상은 뺌
                        continue;
                    }
                    var e = list[img];
                    e.eventUrl = v.url;
                    e.eventLoops = loops;
                    if (masks.TryGetValue(baseName, out var mu)) e.eventMaskUrl = mu;
                    list[img] = e;
                    list.RemoveAt(i);
                }
            }
        }

        static void AddFromDirectory(string dir, List<PhotoEntry> into)
        {
            if (!Directory.Exists(dir)) return;
            foreach (var path in Directory.GetFiles(dir))
            {
                if (!IsMedia(path)) continue;
                var name = Path.GetFileName(path);
                bool vid = IsVideo(path);
                into.Add(new PhotoEntry
                {
                    name = name,
                    url = vid ? path : new Uri(path).AbsoluteUri, // 영상은 파일 경로, 사진은 file:///... (공백/한글 이스케이프)
                    stereo = IsStereoName(name),
                    video = vid
                });
            }
        }
    }
}
