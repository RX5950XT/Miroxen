// Qwen-Image-2.1 端到端冒煙測試：用 UC DiT 產生圖片並檢查輸出不是全黑／全灰。
// 參數（環境變數）：MX_E2E_STEPS（預設 20）、MX_E2E_SIZE（預設 768）、MX_E2E_PROMPT。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "models/qwen_image/pipeline.h"
#include "server/image_io.h"

using namespace mx;

int main() {
  const std::string d = "models/image/qwen-image-2.1/";
  QwenImagePaths p{d + "qwen-image-2.1-UC-Q8_0.gguf", d + "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                   d + "vae/qwen_image_2.1_vae_bf16.safetensors", d + "hf/processor/tokenizer.json"};
  for (auto* f : {&p.dit, &p.text_encoder, &p.vae, &p.tokenizer})
    if (!std::filesystem::exists(*f)) {
      std::printf("SKIP：缺少 %s\n", f->c_str());
      return 0;
    }
  try {
    auto env = [](const char* k, const char* def) { const char* v = std::getenv(k); return std::string(v ? v : def); };
    QwenImagePipeline pipe(p, QwenImageDevices{0, -1, 0});
    std::printf("%s\n", pipe.describe().c_str());
    ImageRequest r;
    r.prompt = env("MX_E2E_PROMPT",
                   "A neon shop sign that reads \"MIROXEN\", rainy night, reflections on wet pavement, cinematic");
    r.width = r.height = std::atoi(env("MX_E2E_SIZE", "768").c_str());
    r.steps = std::atoi(env("MX_E2E_STEPS", "20").c_str());
    r.seed = 42;
    auto res = pipe.generate(r, [](const Progress& g) {
      if (g.stage == "denoise") std::printf("\r取樣 %d/%d", g.step + 1, g.total), std::fflush(stdout);
    }, nullptr);
    std::filesystem::create_directories("out/images");
    write_file("out/images/qwen_e2e.png", res.png.data(), res.png.size());
    auto img = decode_image_bytes(res.png.data(), res.png.size());
    double mean = 0, var = 0;
    size_t n = img.pixels.size();
    for (auto v : img.pixels) mean += v;
    mean /= n;
    for (auto v : img.pixels) var += (v - mean) * (v - mean);
    double sd = std::sqrt(var / n);
    std::printf("\n%dx%d、%.1f 秒、像素平均 %.1f、標準差 %.1f → out/images/qwen_e2e.png\n", res.width, res.height,
                res.seconds, mean, sd);
    return sd > 10 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("FAIL：%s\n", e.what());
    return 1;
  }
}
