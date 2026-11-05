// Link-only package dependency check. CI never executes this binary.
#include <nvcomp/snappy.h>

int main() {
  auto* volatile decompress = &nvcompBatchedSnappyDecompressAsync;
  return decompress == nullptr;
}
