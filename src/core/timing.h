#pragma once
#include <chrono>
#include <map>
#include <string>

namespace mx {
// 階段邊界由呼叫端同步 GPU；只量整段牆鐘時間，不把重疊搬運重複計算。
class PhaseTimer {
 public:
  using Clock = std::chrono::steady_clock;
  void mark(const char* name) {
    const auto now = Clock::now();
    seconds[name] += std::chrono::duration<double>(now - last_).count();
    last_ = now;
  }
  std::map<std::string, double> seconds;
 private:
  Clock::time_point last_ = Clock::now();
};
}  // namespace mx
