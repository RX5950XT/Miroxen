#pragma once
// 參考影片／音訊讀取（呼叫 ffmpeg／ffprobe，需在 PATH 中）。
#include <cstdint>
#include <string>
#include <vector>

namespace mx {

struct MediaInfo {
  int width = 0, height = 0;  // 沒有影像軌時為 0
  bool has_video = false, has_audio = false;
  double duration = 0;  // 秒
};
MediaInfo probe_media(const std::string& path);

// 影像軌 → 重新取樣到 fps（ffmpeg fps 濾鏡：整格丟棄／重複）、Lanczos 縮放到 w×h，最多 max_frames 幀。
// 回傳 RGB8 [frames, h, w, 3]。
std::vector<uint8_t> decode_video_rgb(const std::string& path, int w, int h, double fps, int max_frames, int& frames);
// 音軌 → 交錯立體聲 float PCM（L R L R ...），取樣率 rate，最多 max_seconds 秒（單聲道會複製成兩聲道）。
std::vector<float> decode_audio(const std::string& path, int rate, double max_seconds);

}  // namespace mx
