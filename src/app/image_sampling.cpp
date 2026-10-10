#include "app/engine.h"

namespace mx {

const std::vector<float>& qwen_turbo_sigmas() {
  static const std::vector<float> s = {1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f};
  return s;
}

}  // namespace mx
