// Optional host check against pinned upstream CCCL headers. No NVIDIA execution.
// Build with THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP and include paths to
// CCCL's thrust/, cub/ and libcudacxx/include/. See the validation record.
// Include host headers first to retain Apple libc++ ABI attributes before CCCL
// introduces its interoperability forward declarations. Upstream is unmodified.
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "cuda_column_reference.h"

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <cuda/std/random>

int main() {
  try {
    std::size_t cases = 0, indices = 0, draws = 0;
    for (std::uint32_t seed : {0u, 1u, 7u, 2147483647u, 2147483648u, 0xffffffffu}) {
      thrust::default_random_engine rng(seed);
      thrust::uniform_int_distribution<std::uint32_t> key_distribution;
      auto keys = cuda_reference::ShuffleKeys(seed);
      for (auto key : keys)
        if (key != key_distribution(rng)) throw std::runtime_error("Thrust round key differs");
      for (std::uint32_t n : {0u, 1u, 2u, 3u, 255u, 256u, 257u, 2047u, 2048u, 2049u, 65537u}) {
        std::vector<std::uint32_t> upstream(n);
        std::iota(upstream.begin(), upstream.end(), 0);
        rng.seed(seed);
        thrust::shuffle(thrust::host, upstream.begin(), upstream.end(), rng);
        if (upstream != cuda_reference::Shuffle(n, seed))
          throw std::runtime_error("Thrust permutation differs");
        ++cases;
        indices += n;
      }
      cuda::std::philox4x64 engine(seed);
      cuda::std::uniform_real_distribution<float> distribution;
      for (std::uint64_t i = 0; i < 10000; ++i) {
        float actual = distribution(engine), expected = cuda_reference::Weight(seed, i);
        if (std::memcmp(&actual, &expected, sizeof(float)))
          throw std::runtime_error("CCCL Philox uniform differs");
        ++draws;
      }
      std::printf("Checked seed %u / %zu permutations / %zu indices / %zu draws\n", seed,
                  cases, indices, draws);
      std::fflush(stdout);
    }
    std::printf("PASS pinned upstream Thrust/CCCL host reference: %zu permutations / %zu indices / "
                "%zu exact Philox uniforms and 144 shuffle keys\n", cases, indices, draws);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
