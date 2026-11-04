// Optional host check against unmodified pinned CCCL. No NVIDIA device run.
// Host includes precede CCCL for Apple libc++ interoperability declarations.
#include <cstdio>
#include <cuda/std/random>
#include <random>
#include <stdexcept>

#include "cuda_ranking_pairs.h"
int main() {
  try {
    static_assert(sizeof(std::size_t) == 8);
    std::size_t draws = 0;
    for (std::uint32_t seed : {0u, 1u, 7u, 2147483647u, 2147483648u, 0xffffffffu}) {
      for (std::uint32_t bit = 1; bit < 32; ++bit) {
        for (std::uint32_t range : {(1u << bit) - 1, 1u << bit, (1u << bit) + 1, 0xffffffffu}) {
          for (std::uint64_t start : {0ULL, 1ULL, 3ULL, 0xffffffffULL, 0x100000000ULL,
                                      0x10000000003ULL, 0x7fffffffffffffefULL}) {
            for (std::uint64_t i = 0; i < 16; ++i) {
              cuda::std::philox4x64 engine(seed);
              engine.discard(start + i);
              cuda::std::uniform_int_distribution<std::size_t> distribution(0, range - 1);
              auto position = start + i;
              auto expected = cuda_reference::RankingUniform(seed, &position, range);
              if (distribution(engine) != expected ||
                  engine() != cuda_reference::Philox(seed, position))
                throw std::runtime_error(
                    "CCCL ranking distribution or consumed engine position differs");
              ++draws;
            }
          }
        }
      }
      std::printf("Checked seed %u / %zu exact upstream draws and engine positions\n", seed, draws);
      std::fflush(stdout);
    }
    std::printf("PASS %zu exact pinned CCCL ranking draws and consumed positions\n", draws);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
