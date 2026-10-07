#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "text/qwen3_text.h"

using namespace mx;

static void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

int main() {
  try {
    Qwen3TextEncoderCache cache;
    const std::vector<int32_t> ids{3, 8, 13};
    std::vector<bf16> expected{__float2bfloat16(1.f), __float2bfloat16(2.f)};
    TextVisionInput vis;
    vis.rows = {1};
    vis.embeds = {__float2bfloat16(3.f)};
    vis.deepstack = {{__float2bfloat16(4.f)}};
    vis.pos = {0, 1, 2};
    cache.put(ids, &vis, expected);

    const std::vector<bf16>* got = nullptr;
    got = cache.get(ids, &vis);
    check(got && std::memcmp(got->data(), expected.data(), 4) == 0,
          "同一個圖文輸入沒有命中");
    auto changed = vis;
    changed.embeds[0] = __float2bfloat16(3.5f);
    check(!cache.get(ids, &changed), "視覺嵌入不同仍命中");
    changed = vis;
    changed.deepstack[0][0] = __float2bfloat16(4.5f);
    check(!cache.get(ids, &changed), "DeepStack 不同仍命中");
    changed = vis;
    changed.pos[0]++;
    check(!cache.get(ids, &changed), "位置不同仍命中");
    changed = vis;
    changed.rows[0]++;
    check(!cache.get(ids, &changed), "視覺列不同仍命中");
    check(!cache.get(ids, nullptr), "圖文與純文字輸入混淆");
    check(!cache.get({3, 8, 14}, &vis), "token ids 不同仍命中");

    cache.put({20}, nullptr, expected);
    cache.put({30}, nullptr, expected);
    check(cache.stats().bytes <= Qwen3TextEncoderCache::kMaxBytes, "cache 超過記憶體上限");
    check(!cache.get(ids, &vis), "LRU 沒有淘汰最舊項目");
    check(cache.get({30}, nullptr), "LRU 沒有保留最新項目");
    check(!cache.eligible({}, nullptr, Qwen3TextEncoderCache::kMaxBytes / sizeof(bf16) + 1),
          "超過上限的輸出仍可快取");
    std::puts("PASS：圖文快取精確比對、LRU 容量與記憶體上限");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
