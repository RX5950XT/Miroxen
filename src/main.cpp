// Miroxen 命令列：
//   miroxen image  -p "prompt" [-o out.png] [-W 1024 -H 1024] [--steps 40] [--seed N] [--lora file:scale]
//                  [--ref 參考圖.png ...]（影像編輯，最多 10 張）[--sigmas 1,0.9375,0.875,0.75,0.5,0.25]
//                  [--shift-terminal 0]（步數蒸餾 LoRA 用，例如 Turbo8：--steps 8 --shift-terminal 0）
//   miroxen video  -p "prompt" [-W 864 -H 480] [--frames 56] [--steps 30] [--first 首幀.png] [--last 尾幀.png]
//   miroxen serve  [--port 8188] [--host 127.0.0.1] [--api-key KEY] [--fake]
// 模型路徑預設在 models/ 底下，可用 --qi-dit / --qi-te / --qi-vae / --qi-tok 覆寫。
// --compute bf16|fp8|fp4：DiT 運算精度（fp8/fp4 用 Blackwell 區塊縮放 tensor core）。
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "server/image_io.h"
#include "app/miroxen_engine.h"
#include "core/common.h"

using namespace mx;

namespace {

struct Args {
  std::string cmd;
  std::map<std::string, std::string> kv;
  std::vector<std::string> loras, refs;
  bool has(const std::string& k) const { return kv.count(k) > 0; }
  std::string get(const std::string& k, const std::string& d = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  int geti(const std::string& k, int d) const { return has(k) ? std::atoi(get(k).c_str()) : d; }
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  static const char* kFlags[] = {"--fake", "--help"};
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    bool flag = false;
    for (auto f : kFlags) flag |= k == f;
    if (flag) {
      a.kv[k] = "1";
      continue;
    }
    MX_CHECK(i + 1 < argc, "參數 %s 缺少值", k.c_str());
    if (k == "--lora") a.loras.push_back(argv[++i]);
    else if (k == "--ref") a.refs.push_back(argv[++i]);
    else a.kv[k] = argv[++i];
  }
  return a;
}

EngineConfig engine_config(const Args& a) {
  EngineConfig c;
  if (a.has("--models")) c.models_dir = a.get("--models");
  c.qi_dit = a.get("--qi-dit", c.qi_dit);
  c.qi_te = a.get("--qi-te", c.qi_te);
  c.qi_vae = a.get("--qi-vae", c.qi_vae);
  c.qi_tok = a.get("--qi-tok", c.qi_tok);
  c.h3_dit = a.get("--h3-dit", c.h3_dit);
  c.h3_te = a.get("--h3-te", c.h3_te);
  c.out_dir = a.get("--out", c.out_dir);
  c.compute = a.get("--compute", c.compute);
  return c;
}

int cmd_image(const Args& a) {
  ImageRequest r;
  r.prompt = a.get("-p", a.get("--prompt"));
  MX_CHECK(!r.prompt.empty(), "請用 -p 指定 prompt");
  r.negative_prompt = a.get("-n", "");
  r.ref_images = a.refs;
  r.width = a.geti("-W", a.refs.empty() ? 1024 : 0);  // 編輯時預設依參考圖比例
  r.height = a.geti("-H", a.refs.empty() ? 1024 : 0);
  r.steps = a.geti("--steps", 40);
  for (std::string s = a.get("--sigmas", ""); !s.empty();) {
    size_t c = s.find(',');
    r.sigmas.push_back(float(std::atof(s.substr(0, c).c_str())));
    s = c == std::string::npos ? "" : s.substr(c + 1);
  }
  r.cfg_scale = float(std::atof(a.get("--cfg", "1").c_str()));
  r.seed = a.has("--seed") ? std::atoll(a.get("--seed").c_str()) : -1;
  r.shift_terminal = a.get("--shift-terminal", "1") != "0";
  for (auto& l : a.loras) {
    size_t c = l.rfind(':');
    LoraSpec s;
    s.path = c != std::string::npos && c > 1 ? l.substr(0, c) : l;
    s.scale = c != std::string::npos && c > 1 ? float(std::atof(l.substr(c + 1).c_str())) : 1.f;
    r.loras.push_back(s);
  }
  auto engine = make_engine(engine_config(a));
  auto res = engine->generate_image(r, [](const Progress& p) {
    if (p.stage == "denoise") std::printf("\r取樣 %d/%d", p.step + 1, p.total), std::fflush(stdout);
    else std::printf("\n[%s]\n", p.stage.c_str());
  }, nullptr);
  std::string out = a.get("-o", "out.png");
  write_file(out, res.png.data(), res.png.size());
  MX_INFO("完成：%s（%dx%d、seed %lld、%.1f 秒）", out.c_str(), res.width, res.height,
          (long long)res.seed, res.seconds);
  return 0;
}

int cmd_video(const Args& a) {
  VideoRequest r;
  r.prompt = a.get("-p", a.get("--prompt"));
  MX_CHECK(!r.prompt.empty(), "請用 -p 指定 prompt");
  r.first_frame = a.get("--first", "");
  r.last_frame = a.get("--last", "");
  const bool keyed = !r.first_frame.empty() || !r.last_frame.empty();
  r.width = a.geti("-W", keyed ? 0 : 864);  // 有關鍵幀時預設依第一張比例用官方畫布（短邊 768）
  r.height = a.geti("-H", keyed ? 0 : 480);
  r.num_frames = a.geti("--frames", 56);
  r.video_shift = float(std::atof(a.get("--shift", "12").c_str()));
  r.steps = a.geti("--steps", 30);
  r.seed = a.has("--seed") ? std::atoll(a.get("--seed").c_str()) : -1;
  for (auto& l : a.loras) {
    size_t c = l.rfind(':');
    LoraSpec s;
    s.path = c != std::string::npos && c > 1 ? l.substr(0, c) : l;
    s.scale = c != std::string::npos && c > 1 ? float(std::atof(l.substr(c + 1).c_str())) : 1.f;
    r.loras.push_back(s);
  }
  auto engine = make_engine(engine_config(a));
  auto res = engine->generate_video(r, [](const Progress& p) {
    if (p.stage == "denoise" || p.stage == "decode_video")
      std::printf("\r%s %d/%d   ", p.stage.c_str(), p.step + 1, p.total), std::fflush(stdout);
    else std::printf("\n[%s]\n", p.stage.c_str());
  }, nullptr);
  MX_INFO("\n完成：%s（seed %lld、%.1f 秒）", res.mp4_path.c_str(), (long long)res.seed, res.seconds);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    if (a.cmd == "image") return cmd_image(a);
    if (a.cmd == "video") return cmd_video(a);
    if (a.cmd == "serve") return cmd_serve(a.kv, engine_config(a));
    std::printf("用法：miroxen image -p \"prompt\" [-o out.png] [-W 1024 -H 1024] [--steps 40] "
                "[--seed N] [--cfg 1] [-n 負面] [--lora 檔案:強度]\n"
                "      miroxen video -p \"prompt\" [-W 864 -H 480] [--frames 56] [--steps 30] [--seed N]\n"
                "      miroxen serve [--port 8188] [--host 127.0.0.1] [--api-key KEY] [--fake]\n");
    return a.cmd.empty() || a.has("--help") ? 0 : 2;
  } catch (const std::exception& e) {
    MX_ERROR("%s", e.what());
    return 1;
  }
}
