#include "server/media_io.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/common.h"

namespace mx {

namespace {

#ifdef _WIN32
FILE* open_pipe(const std::string& cmd) { return _popen(cmd.c_str(), "rb"); }
int close_pipe(FILE* f) { return _pclose(f); }
#else
FILE* open_pipe(const std::string& cmd) { return popen(cmd.c_str(), "r"); }
int close_pipe(FILE* f) { return pclose(f); }
#endif

// 讀完整個子程序輸出；失敗丟例外。
std::vector<uint8_t> run(const std::string& cmd) {
  FILE* f = open_pipe(cmd);
  MX_CHECK(f, "無法啟動 ffmpeg／ffprobe（請確認已安裝並在 PATH 中）");
  std::vector<uint8_t> out;
  uint8_t buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  int rc = close_pipe(f);
  MX_CHECK(rc == 0, "ffmpeg／ffprobe 失敗（結束代碼 %d）", rc);
  return out;
}

std::string quoted(const std::string& path) {
  MX_CHECK(std::filesystem::exists(path), "找不到媒體檔 %s", path.c_str());
  MX_CHECK(path.find('"') == std::string::npos, "檔名不可含雙引號");
  return "\"" + path + "\"";
}

}  // namespace

MediaInfo probe_media(const std::string& path) {
  auto raw = run("ffprobe -v error -show_entries stream=codec_type,width,height:format=duration -of csv=p=0 " +
                 quoted(path));
  std::string s(raw.begin(), raw.end());
  MediaInfo m;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t e = s.find('\n', pos);
    std::string line = s.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
    pos = e == std::string::npos ? s.size() : e + 1;
    while (!line.empty() && (line.back() == '\r' || line.back() == ',')) line.pop_back();
    if (line.rfind("video", 0) == 0) {
      int w = 0, h = 0;
      if (std::sscanf(line.c_str(), "video,%d,%d", &w, &h) == 2 && !m.has_video) m.width = w, m.height = h;
      m.has_video = true;
    } else if (line.rfind("audio", 0) == 0) {
      m.has_audio = true;
    } else if (!line.empty()) {
      double d = std::atof(line.c_str());
      if (d > 0) m.duration = d;
    }
  }
  return m;
}

std::vector<uint8_t> decode_video_rgb(const std::string& path, int w, int h, double fps, int max_frames, int& frames) {
  MX_CHECK(w > 0 && h > 0 && fps > 0 && max_frames > 0, "影片解碼參數無效");
  auto raw = run(strf("ffmpeg -v error -i %s -an -vf fps=%g,scale=%d:%d:flags=lanczos -frames:v %d "
                      "-f rawvideo -pix_fmt rgb24 -",
                      quoted(path).c_str(), fps, w, h, max_frames));
  const size_t frame = size_t(w) * h * 3;
  frames = int(raw.size() / frame);
  MX_CHECK(frames > 0, "%s 沒有可讀的影格", path.c_str());
  raw.resize(frame * frames);
  return raw;
}

std::vector<float> decode_audio(const std::string& path, int rate, double max_seconds) {
  MX_CHECK(rate > 0 && max_seconds > 0, "音訊解碼參數無效");
  auto raw = run(strf("ffmpeg -v error -i %s -vn -t %.6f -ac 2 -ar %d -f f32le -", quoted(path).c_str(), max_seconds,
                      rate));
  MX_CHECK(raw.size() >= 8, "%s 沒有可讀的音軌", path.c_str());
  std::vector<float> pcm(raw.size() / 4 / 2 * 2);
  std::memcpy(pcm.data(), raw.data(), pcm.size() * 4);
  return pcm;
}

}  // namespace mx
