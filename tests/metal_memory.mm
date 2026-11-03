// Validate the production borrowing helper, including offset/guard/VM boundaries.
#include "../xgboost/src/common/metal_memory.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  @autoreleasepool {
    try {
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No GPU");
      if (!device.hasUnifiedMemory) {
        uint32_t value = 0;
        Check(xgboost::common::metal::TryBorrowMemory(device, &value, 4, 4).buffer == nil,
              "Non-unified GPU must refuse host borrowing");
        printf("PASS non-unified GPU staging requirement; shared-memory cases unavailable\n");
        return 0;
      }
      NSError* error = nil;
      auto lib = [device
          newLibraryWithSource:
              @"#include <metal_stdlib>\nusing namespace metal;\nkernel void copy_values(device "
              @"const uint* input [[buffer(0)]],device uint* output [[buffer(1)]],uint i "
              @"[[thread_position_in_grid]]){output[i]=input[i]^0x5a5a5a5au;}"
                       options:nil
                         error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline =
          [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy_values"]
                                                error:&error];
      Check(pipeline != nil, "Missing pipeline");
      auto queue = [device newCommandQueue];
      size_t cases = 0, values = 0;
      for (size_t n : {1, 31, 256, 257, 4095, 4096, 4097, 65537, 1000003})
        for (size_t offset : {0, 1, 3, 4095}) {
          // Separate logical spans can share VM pages; every surrounding word is guarded.
          std::vector<uint32_t> storage(2 * n + 2 * offset + 64, 0xdeadbeefu);
          auto input = storage.data() + offset + 16, output = input + n + 16;
          for (size_t i = 0; i < n; ++i) input[i] = uint32_t(i * 1777 + 357);
          auto expected = storage;
          for (size_t i = 0; i < n; ++i)
            expected[output - storage.data() + i] = input[i] ^ 0x5a5a5a5au;
          @autoreleasepool {
            auto in = xgboost::common::metal::TryBorrowMemory(device, input, n * 4, 4);
            auto out = xgboost::common::metal::TryBorrowMemory(device, output, n * 4, 4);
            Check(in.buffer && out.buffer, "Valid writable memory was not borrowed");
            auto command = [queue commandBuffer];
            auto e = [command computeCommandEncoder];
            [e setComputePipelineState:pipeline];
            [e setBuffer:in.buffer offset:in.offset atIndex:0];
            [e setBuffer:out.buffer offset:out.offset atIndex:1];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [command commit];
            [command waitUntilCompleted];
            Check(command.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
          }
          Check(storage == expected, "Borrowed output or guard differs from CPU");
          ++cases;
          values += n;
        }
      auto page = size_t(getpagesize());
      auto mapping = static_cast<uint8_t*>(
          mmap(nullptr, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
      Check(mapping != MAP_FAILED, "mmap failed");
      memset(mapping, 0x11, 3 * page);
      Check(mprotect(mapping + page, page, PROT_READ) == 0, "mprotect failed");
      using xgboost::common::metal::TryBorrowMemory;
      Check(TryBorrowMemory(device, mapping + page, 4, 4).buffer == nil,
            "Read-only region must use copied staging");
      Check(TryBorrowMemory(device, mapping, 2 * page, 4).buffer == nil,
            "Multiple VM regions must use copied staging");
      Check(TryBorrowMemory(device, mapping + 1, 4, 4).buffer == nil,
            "Misaligned argument must use copied staging");
      Check(TryBorrowMemory(device, mapping, 0, 4).buffer == nil,
            "Empty memory should not be borrowed");
      Check(TryBorrowMemory(device, nullptr, 16, 4).buffer == nil,
            "Null memory should not be borrowed");
      Check(TryBorrowMemory(device, mapping, SIZE_MAX, 4).buffer == nil,
            "Overflowing region should not be borrowed");
      Check(munmap(mapping, 3 * page) == 0, "munmap failed");
      printf("PASS %zu shared-memory cases / %zu values and 6 fallback checks on %s\n", cases,
             values, device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
