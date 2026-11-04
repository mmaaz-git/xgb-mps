// Independent byte expectations, real private-buffer consumers and ownership.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../xgboost/include/xgboost/c_api.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_output.h"

namespace metal = xgboost::common::metal;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const *message) {
  ++fields;
  if (!ok)
    throw std::runtime_error(message);
}
void API(int code) { Check(code == 0, XGBGetLastError()); }
void Complete(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted,
        "Consumer GPU command failed");
}
std::vector<std::uint32_t> Pattern(std::size_t size) {
  // Exact bytes include signed zero, infinities, NaN payloads and subnormals.
  std::uint32_t edges[] = {0,           0x80000000u, 1,           0x80000001u,
                           0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc54321u,
                           0x3f800000u, 0x00800000u};
  std::vector<std::uint32_t> out(size);
  for (std::size_t i = 0; i < size; ++i)
    out[i] = edges[i % 10];
  return out;
}
void Same(void const *actual, std::vector<std::uint32_t> const &expected) {
  auto bytes = static_cast<unsigned char const *>(actual);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    std::uint32_t value;
    std::memcpy(&value, bytes + i * 4, 4);
    Check(value == expected[i], "Output changed float32 bits");
  }
}
void Consume(id<MTLDevice> device, id<MTLCommandQueue> queue,
             DLTensor const &tensor, std::vector<bst_ulong> const &shape,
             std::vector<std::uint32_t> const &expected) {
  Check(tensor.device.device_type == kDLMetal && tensor.device.device_id == 0,
        "Device ABI");
  Check(tensor.dtype.code == kDLFloat && tensor.dtype.bits == 32 &&
            tensor.dtype.lanes == 1,
        "Datatype ABI");
  Check(tensor.ndim == shape.size() && tensor.byte_offset == 0,
        "Rank/offset ABI");
  std::size_t stride = 1;
  for (std::size_t i = shape.size(); i; --i) {
    Check(tensor.shape[i - 1] == shape[i - 1], "Output shape");
    Check(tensor.strides[i - 1] == stride, "Contiguous element stride");
    stride *= shape[i - 1];
  }
  auto buffer = (__bridge id<MTLBuffer>)tensor.data;
  Check(buffer.storageMode == MTLStorageModePrivate,
        "Output must own private GPU storage");
  if (expected.empty())
    return;
  auto host = [device newBufferWithLength:expected.size() * 4
                                  options:MTLResourceStorageModeShared];
  auto command = [queue commandBuffer];
  auto blit = [command blitCommandEncoder];
  [blit copyFromBuffer:buffer
           sourceOffset:tensor.byte_offset
               toBuffer:host
      destinationOffset:0
                   size:expected.size() * 4];
  [blit endEncoding];
  Complete(command);
  Same(host.contents, expected);
}
void Shapes(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  for (auto shape : std::vector<std::vector<bst_ulong>>{{},
                                                        {1},
                                                        {257},
                                                        {3, 17},
                                                        {2, 3, 5},
                                                        {2, 3, 5, 7},
                                                        {0},
                                                        {0, 7},
                                                        {3, 0, 5}}) {
    @autoreleasepool {
      auto count = std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                                   std::multiplies<std::size_t>{});
      auto expected = Pattern(count);
      std::vector<float> source(count);
      if (count)
        std::memcpy(source.data(), expected.data(), count * 4);
      auto before = metal::GetStatistics();
      void *handle = nullptr;
      API(XGMetalOutputCreate(source.data(), shape.data(), shape.size(),
                              &handle));
      std::fill(source.begin(), source.end(), -99.f);
      source.clear();
      source.shrink_to_fit();
      void *legacy_raw = nullptr, *versioned_raw = nullptr,
           *clone_raw = nullptr;
      char const *name;
      API(XGMetalOutputExportDLPack(handle, 0, 0, &legacy_raw, &name));
      Check(std::strcmp(name, "dltensor") == 0, "Legacy capsule name");
      API(XGMetalOutputExportDLPack(handle, 1, 0, &versioned_raw, &name));
      Check(std::strcmp(name, "dltensor_versioned") == 0,
            "Versioned capsule name");
      API(XGMetalOutputExportDLPack(handle, 1, 1, &clone_raw, nullptr));
      auto legacy = static_cast<DLManagedTensor *>(legacy_raw);
      auto versioned = static_cast<DLManagedTensorVersioned *>(versioned_raw);
      auto clone = static_cast<DLManagedTensorVersioned *>(clone_raw);
      Check(versioned->version.major == 1 && versioned->version.minor == 0,
            "DLPack version");
      Check(versioned->flags == 0 &&
                clone->flags == DLPACK_FLAG_BITMASK_IS_COPIED,
            "Copy flags");
      Check(legacy->dl_tensor.data == versioned->dl_tensor.data,
            "Shared exports must alias");
      Check(clone->dl_tensor.data != legacy->dl_tensor.data,
            "Copy export must be independent");
      std::vector<float> output(count + 2, -971.f);
      API(XGMetalOutputRead(handle, output.data() + 1, count));
      Same(output.data() + 1, expected);
      Check(output.front() == -971.f && output.back() == -971.f,
            "Readback guards");
      API(XGMetalOutputFree(handle));
      Consume(device, queue, legacy->dl_tensor, shape, expected);
      Consume(device, queue, versioned->dl_tensor, shape, expected);
      // Consumer writes may alias other exports, but never a requested copy.
      if (count) {
        auto command = [queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        [blit fillBuffer:(__bridge id<MTLBuffer>)legacy->dl_tensor.data
                   range:NSMakeRange(0, count * 4)
                   value:0];
        [blit endEncoding];
        Complete(command);
      }
      legacy->deleter(legacy);
      Consume(device, queue, versioned->dl_tensor, shape,
              std::vector<std::uint32_t>(count));
      Consume(device, queue, clone->dl_tensor, shape, expected);
      versioned->deleter(versioned);
      clone->deleter(clone);
      auto after = metal::GetStatistics();
      Check(after.output_live_buffers == before.output_live_buffers,
            "Buffer lifetime leak");
      Check(after.output_live_exports == before.output_live_exports,
            "Export lifetime leak");
      Check(after.output_dlpack_exports == before.output_dlpack_exports + 3,
            "Export count");
      Check(after.output_snapshot_dispatches ==
                before.output_snapshot_dispatches + 2 * bool(count),
            "GPU snapshot and copy commands");
      Check(after.output_host_readback_bytes ==
                before.output_host_readback_bytes + count * 4,
            "Only explicit readback counted");
      ++cases;
    }
  }
}
void Fallback() {
  auto length = std::size_t(getpagesize());
  void *mapping = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                       MAP_ANON | MAP_PRIVATE, -1, 0);
  Check(mapping != MAP_FAILED, "mmap source");
  auto expected = Pattern(67);
  std::memcpy(static_cast<char *>(mapping) + 12, expected.data(),
              expected.size() * 4);
  for (bool readonly : {false, true}) {
    Check(mprotect(mapping, length,
                   readonly ? PROT_READ : PROT_READ | PROT_WRITE) == 0,
          "mprotect");
    auto before = metal::GetStatistics();
    std::vector<bst_ulong> shape{67};
    auto output = metal::MakeOutput(
        reinterpret_cast<float *>(static_cast<char *>(mapping) + 12), shape);
    auto after = metal::GetStatistics();
    Check(after.output_snapshot_shared_input_bytes -
                  before.output_snapshot_shared_input_bytes ==
              (readonly ? 0 : 268),
          "Writable source shares pages and byte offset");
    Check(after.output_snapshot_host_copy_bytes -
                  before.output_snapshot_host_copy_bytes ==
              (readonly ? 268 : 0),
          "Read-only source uses counted fallback");
    // An unaligned destination forces safe memcpy fallback, with guards.
    std::vector<unsigned char> target(270, 0xa5);
    output->Readback({reinterpret_cast<float *>(target.data() + 1), 67});
    Same(target.data() + 1, expected);
    Check(target.front() == 0xa5 && target.back() == 0xa5,
          "Unaligned readback guards");
    ++cases;
  }
  Check(munmap(mapping, length) == 0, "munmap");
}
void Invalid() {
  auto reject = [](int code) {
    Check(code != 0, "Invalid output call accepted");
    ++rejected;
  };
  float value = 1;
  bst_ulong shape[] = {1};
  for (auto invalid : std::vector<std::vector<bst_ulong>>{
           {std::numeric_limits<bst_ulong>::max()},
           {1ull << 62, 4},
           {1ull << 50}}) {
    void *output = reinterpret_cast<void *>(1);
    reject(
        XGMetalOutputCreate(&value, invalid.data(), invalid.size(), &output));
    Check(output == nullptr, "Failure must leave handle unset");
  }
  void *output = nullptr;
  reject(XGMetalOutputCreate(nullptr, shape, 1, &output));
  reject(XGMetalOutputCreate(&value, nullptr, 1, &output));
  reject(XGMetalOutputCreate(&value, shape, 1ull << 32, &output));
  reject(XGMetalOutputCreate(&value, shape, 1, nullptr));
  API(XGMetalOutputCreate(&value, shape, 1, &output));
  reject(XGMetalOutputRead(output, &value, 0));
  reject(XGMetalOutputRead(output, nullptr, 1));
  reject(XGMetalOutputRead(nullptr, &value, 1));
  for (auto flags : {std::pair{2, 0}, std::pair{0, -1}}) {
    void *managed = reinterpret_cast<void *>(1);
    char const *name = "sentinel";
    reject(XGMetalOutputExportDLPack(output, flags.first, flags.second,
                                     &managed, &name));
    Check(managed == nullptr && name == nullptr,
          "Invalid export must leave outputs unset");
  }
  reject(XGMetalOutputExportDLPack(output, 0, 0, nullptr, nullptr));
  API(XGMetalOutputFree(output));
  API(XGMetalOutputFree(nullptr));
  ++cases;
}
int main() {
  try {
    @autoreleasepool {
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "A real Metal device is required");
      auto queue = [device newCommandQueue];
      Shapes(device, queue);
      Fallback();
      Invalid();
      auto stats = metal::GetStatistics();
      Check(stats.output_live_buffers == 0 && stats.output_live_exports == 0,
            "Final output leaks");
      std::printf("PASS %zu native output cases / %zu field checks / %zu "
                  "rejected calls\n",
                  cases, fields, rejected);
    }
  } catch (std::exception const &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
