// Native Metal buffers exercise private/shared storage, bounds and exact decoding.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_tensor.h"

namespace metal = xgboost::common::metal;
using xgboost::StringView;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++fields;
  if (!ok) throw std::runtime_error(message);
}
std::string Descriptor(id<MTLBuffer> buffer, long rows, long cols, long offset,
                       long rs, long cs, int code, int bits) {
  std::ostringstream out;
  out << "{\"device_type\":8,\"device_id\":0,\"buffer\":"
      << reinterpret_cast<std::uintptr_t>((__bridge void*)buffer)
      << ",\"byte_offset\":" << offset << ",\"shape\":[" << rows << ',' << cols
      << "],\"strides\":[" << rs << ',' << cs << "],\"dtype_code\":" << code
      << ",\"dtype_bits\":" << bits << ",\"dtype_lanes\":1}";
  return out.str();
}
void Complete(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted, "Test producer GPU work");
}
id<MTLBuffer> Source(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     void const* data, std::size_t bytes, bool private_buffer) {
  auto shared = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
  if (!private_buffer) return shared;
  auto result = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  auto command = [queue commandBuffer];
  auto blit = [command blitCommandEncoder];
  [blit copyFromBuffer:shared sourceOffset:0 toBuffer:result destinationOffset:0 size:bytes];
  [blit endEncoding];
  Complete(command);
  return result;
}
void Same(float actual, float expected) {
  Check((std::isnan(actual) && std::isnan(expected)) || !std::memcmp(&actual, &expected, 4),
        "Native tensor GPU decode differs from independent CPU conversion");
}
template <class T>
void Typed(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<T> source(1024);
  for (unsigned i = 0; i < source.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) source[i] = T(int(i) - 512) / T(7);
    else source[i] = T(std::uint64_t(i) * 0x12345678912345ULL);
  }
  int code = std::is_floating_point_v<T> ? 2 : std::is_signed_v<T> ? 0 : 1;
  for (auto strides : std::vector<std::pair<long, long>>{{17, 2}, {-17, 2}, {17, -2}, {0, 2}, {17, 0}}) {
    @autoreleasepool {
      auto buffer = Source(device, queue, source.data(), source.size() * sizeof(T), private_buffer);
      long start = 400, rows = 19, cols = 7;
      auto json = Descriptor(buffer, rows, cols, start * sizeof(T), strides.first, strides.second,
                             code, sizeof(T) * 8);
      auto before = metal::GetStatistics();
      auto input = metal::ImportTensor(StringView{json});
      auto after = metal::GetStatistics();
      Check(after.tensor_copy_dispatches == before.tensor_copy_dispatches + 1, "Real GPU snapshot");
      auto span = ((rows - 1) * std::abs(strides.first) + (cols - 1) * std::abs(strides.second) + 1) * sizeof(T);
      Check(after.tensor_copied_bytes - before.tensor_copied_bytes == span, "Only addressed span copied");
      // Mutate AND release the producer allocation. The snapshot must be independent.
      auto command = [queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      [blit fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:0xa5];
      [blit endEncoding];
      Complete(command);
      buffer = nil;
      auto prepared = metal::PrepareInput(*input, NAN, 0, rows, cols, {});
      std::vector<float> result(rows * cols);
      prepared->Readback(result);
      for (long r = 0; r < rows; ++r)
        for (long c = 0; c < cols; ++c)
          Same(result[r * cols + c], float(source[start + r * strides.first + c * strides.second]));
      ++cases;
    }
  }
}
void Half(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<std::uint16_t> bits(65536);
  for (unsigned i = 0; i < bits.size(); ++i) bits[i] = i;
  auto buffer = Source(device, queue, bits.data(), bits.size() * 2, private_buffer);
  auto json = Descriptor(buffer, 256, 256, 0, 256, 1, 2, 16);
  auto input = metal::ImportTensor(StringView{json});
  auto prepared = metal::PrepareInput(*input, NAN, 0, 256, 256, {});
  std::vector<float> actual(65536);
  prepared->Readback(actual);
  for (unsigned i = 0; i < bits.size(); ++i) {
    __fp16 value;
    std::memcpy(&value, &bits[i], 2);
    Same(actual[i], float(value));
  }
  ++cases;
}
void Windows(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  float value = 3.5f;
  auto buffer = Source(device, queue, &value, 4, true);
  // A broadcast tensor crosses the actual 2^26-slot window boundary without
  // requiring a huge allocation. First/last partial rows remain observable.
  auto json = Descriptor(buffer, 1 << 20, 129, 0, 0, 0, 2, 32);
  auto input = metal::ImportTensor(StringView{json});
  for (std::uint64_t begin : {0ULL, (1ULL << 26) - 3, (1ULL << 20) * 129 - 137}) {
    auto view = metal::PrepareAdapter(*input, NAN, begin, 137);
    Check(view->NonMissing() == 137, "Flattened window nnz");
    Check(view->BaseRow() == begin / 129, "Flattened window global base");
    xgboost::SparsePage page;
    view->Readback(&page);
    auto const& values = page.data.ConstHostVector();
    Check(values.size() == 137, "Flattened window sparse entry count");
    for (unsigned i = 0; i < values.size(); ++i) {
      Check(values[i].index == (begin + i) % 129, "Partial row feature IDs");
      Same(values[i].fvalue, value);
    }
    ++cases;
  }
}
void Invalid(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  std::vector<float> values(32, 1);
  auto buffer = Source(device, queue, values.data(), values.size() * 4, false);
  auto reject = [&](std::string json) {
    bool failed = false;
    try { metal::ImportTensor(StringView{json}); } catch (std::exception const&) { failed = true; }
    Check(failed, "Invalid native tensor descriptor accepted");
    ++rejected;
  };
  reject(Descriptor(buffer, 3, 3, 0, -3, 1, 2, 32));
  reject(Descriptor(buffer, 100, 3, 0, 3, 1, 2, 32));
  reject(Descriptor(buffer, 3, 3, 1, 3, 1, 2, 32));
  reject(Descriptor(buffer, -1, 3, 0, 3, 1, 2, 32));
  reject(Descriptor(buffer, 3, 0, 0, 3, 1, 2, 32));
  reject(Descriptor(buffer, 3, 3, 0, 3, 1, 2, 8));
  reject(Descriptor(buffer, 3, 3, 0, 3, 1, 4, 16));
  reject(Descriptor(buffer, 3, 3, 0, 3, 1, 2, 128));
  reject(Descriptor(nil, 3, 3, 0, 3, 1, 2, 32));
  auto empty = Descriptor(nil, 0, 7, 0, 7, 1, 2, 16);
  auto before = metal::GetStatistics();
  auto tensor = metal::ImportTensor(StringView{empty});
  Check(tensor->NumRows() == 0 && tensor->NumColumns() == 7, "Empty native shape");
  Check(metal::GetStatistics().tensor_copy_dispatches == before.tensor_copy_dispatches,
        "Empty input performs no GPU copy");
}
int main() {
  @autoreleasepool {
    try {
      auto device = MTLCreateSystemDefaultDevice();
      auto queue = [device newCommandQueue];
      for (bool private_buffer : {false, true}) {
        Half(device, queue, private_buffer);
        Typed<float>(device, queue, private_buffer);
        Typed<double>(device, queue, private_buffer);
        Typed<std::int8_t>(device, queue, private_buffer);
        Typed<std::int16_t>(device, queue, private_buffer);
        Typed<std::int32_t>(device, queue, private_buffer);
        Typed<std::int64_t>(device, queue, private_buffer);
        Typed<std::uint8_t>(device, queue, private_buffer);
        Typed<std::uint16_t>(device, queue, private_buffer);
        Typed<std::uint32_t>(device, queue, private_buffer);
        Typed<std::uint64_t>(device, queue, private_buffer);
        std::printf("Storage mode %d passed: %zu cases / %zu exact fields\n", private_buffer, cases, fields);
        std::fflush(stdout);
      }
      Windows(device, queue);
      Invalid(device, queue);
      Check(metal::GetStatistics().input_staging_bytes == 0, "Native payload never staged on CPU");
      std::printf("PASS native tensors: %zu cases / %zu exact fields / %zu invalid descriptors\n", cases, fields, rejected);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
