// Host-only link/loader check. Taking an address does not invoke nvCOMP/CUDA.
#include <nvcomp/snappy.h>

int main() {
  auto* volatile decompress = &nvcompBatchedSnappyDecompressAsync;
  return decompress == nullptr;
}
