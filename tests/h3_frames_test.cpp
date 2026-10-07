#include "models/minimax_h3/pipeline.h"
#include <iostream>

int main() {
  try {
    for (int requested = 1; requested <= 361; ++requested) {
      const int aligned = mx::h3_align_frames(requested);
      MX_CHECK(aligned >= 22 && aligned >= requested && aligned % 17 == 5, "frame alignment");
      MX_CHECK(mx::h3_latent_frames(aligned) >= 3, "decoder minimum");
    }
    MX_CHECK(mx::h3_align_frames(56) == 56 && mx::h3_align_frames(240) == 243, "existing lengths changed");
    bool rejected = false;
    try { mx::h3_align_frames(0); } catch (const std::exception&) { rejected = true; }
    MX_CHECK(rejected, "zero frames must fail");
    std::cout << "h3_frames_test: 361 lengths and invalid input passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
