#include "app/engine.h"
#include "models/qwen_image/pipeline.h"

#include <cstdio>

int main() {
  const auto& t = mx::qwen_turbo_sigmas();
  if (t.size() != 8 || t.front() != 1.f) return 1;
  for (size_t i = 1; i < t.size(); ++i)
    if (t[i] >= t[i - 1]) return 2;
  // Turbo 排程不經位移：shift_terminal 關閉、raw=true 時原值輸出並補上 0。
  auto s = mx::qwen_image_sigmas(0, 4096, false, t, /*raw=*/true);
  if (s.size() != 9 || s[7] != t[7] || s[8] != 0.f) return 3;
  auto shifted = mx::qwen_image_sigmas(0, 4096, false, t);
  if (shifted[6] == t[6]) return 4;  // 一般模式仍套動態位移
  std::puts("Turbo sigmas OK");
}
