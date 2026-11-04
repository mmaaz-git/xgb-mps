// Independent scalar oracles for native metadata conversion and CUDA group rules.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_metadata.h"

namespace metal = xgboost::common::metal;
using xgboost::MetaInfo;
using xgboost::StringView;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++fields;
  if (!ok) throw std::runtime_error(message);
}
void Same(float a, float b) {
  Check((std::isnan(a) && std::isnan(b)) || !std::memcmp(&a, &b, 4), "Typed metadata conversion bits");
}
void Complete(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted, "Native test producer failed");
}
id<MTLBuffer> Source(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     void const* data, std::size_t bytes, bool private_buffer) {
  if (!bytes) return nil;
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
std::string Descriptor(id<MTLBuffer> buffer, long rows, long cols, long offset,
                       long rs, long cs, int code, int bits, int dimensions = 2) {
  std::ostringstream out;
  out << "{\"device_type\":8,\"device_id\":0,\"buffer\":"
      << reinterpret_cast<std::uintptr_t>((__bridge void*)buffer)
      << ",\"byte_offset\":" << offset << ",\"shape\":[";
  if (dimensions) out << rows;
  if (dimensions == 2) out << ',' << cols;
  out << "],\"strides\":[";
  if (dimensions) out << rs;
  if (dimensions == 2) out << ',' << cs;
  out << "],\"dtype_code\":" << code << ",\"dtype_bits\":" << bits << ",\"dtype_lanes\":1}";
  return out.str();
}
template <class Fn>
void Reject(Fn&& fn) {
  bool failed = false;
  try { fn(); } catch (std::exception const&) { failed = true; }
  Check(failed, "Invalid metadata accepted");
  ++rejected;
}
template <class T>
void Typed(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<T> values(1024);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) values[i] = T(int(i) - 512) / T(7);
    else values[i] = T(std::uint64_t(i) * 0x9e3779b97f4a7c15ULL);
  }
  int code = std::is_floating_point_v<T> ? 2 : std::is_signed_v<T> ? 0 : 1;
  for (auto strides : std::vector<std::pair<long, long>>{{17, 2}, {-17, -2}, {0, 0}}) {
    for (auto key : {"label", "base_margin"}) {
      @autoreleasepool {
        auto buffer = Source(device, queue, values.data(), values.size() * sizeof(T), private_buffer);
        auto json = Descriptor(buffer, 19, 7, 400 * sizeof(T), strides.first, strides.second,
                               code, sizeof(T) * 8);
        MetaInfo info;
        info.num_row_ = 13;  // Native CUDA metadata shape is not inferred from sample count.
        auto before = metal::GetStatistics();
        metal::SetMetadata(&info, StringView{key}, StringView{json});
        auto after = metal::GetStatistics();
        Check(after.metadata_values == before.metadata_values + 19 * 7, "Metadata value accounting");
        Check(after.input_staging_bytes == before.input_staging_bytes, "No CPU input staging");
        Check(after.metadata_shared_output_bytes - before.metadata_shared_output_bytes +
              after.metadata_host_copy_bytes - before.metadata_host_copy_bytes == 19 * 7 * 4,
              "Destination sharing/copy accounting");
        auto command = [queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        [blit fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:0xa5];
        [blit endEncoding];
        Complete(command);
        buffer = nil;
        auto& result = key[0] == 'l' ? info.labels : info.base_margin_;
        Check(result.Shape(0) == 19 && result.Shape(1) == 7, "Preserved CUDA metadata shape");
        auto const& out = result.Data()->ConstHostVector();
        for (long r = 0; r < 19; ++r)
          for (long c = 0; c < 7; ++c)
            Same(out[r * 7 + c], float(values[400 + r * strides.first + c * strides.second]));
        ++cases;
      }
    }
  }
}
void Half(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<std::uint16_t> bits(65536);
  for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = i;
  auto buffer = Source(device, queue, bits.data(), bits.size() * 2, private_buffer);
  auto json = Descriptor(buffer, 256, 256, 0, 256, 1, 2, 16);
  MetaInfo info;
  metal::SetMetadata(&info, StringView{"base_margin"}, StringView{json});
  auto const& out = info.base_margin_.Data()->ConstHostVector();
  for (std::size_t i = 0; i < bits.size(); ++i) {
    __fp16 half;
    std::memcpy(&half, &bits[i], 2);
    Same(out[i], float(half));
  }
  ++cases;
}
template <class T>
void Qids(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  for (std::size_t n : {0, 1, 2, 255, 256, 257, 65536, 65537}) {
    for (std::size_t stride : {1, 11, 65537}) {
      @autoreleasepool {
        std::vector<T> values(n);
        std::vector<std::uint32_t> expected{0};
        for (std::size_t i = 0; i < n; ++i) {
          if constexpr (std::is_same_v<T, double>) values[i] = double(i / stride) + .75;
          else if constexpr (sizeof(T) == 8) values[i] = (1ULL << 32) + i / stride;
          else values[i] = i / stride;
          if (i && std::uint32_t(values[i]) != std::uint32_t(values[i - 1])) expected.push_back(i);
        }
        if (n) expected.push_back(n);
        auto buffer = Source(device, queue, values.data(), values.size() * sizeof(T), private_buffer);
        auto json = Descriptor(buffer, n, 1, 0, 1, 1,
                               std::is_floating_point_v<T> ? 2 : 1, sizeof(T) * 8, 1);
        MetaInfo info;
        metal::SetMetadata(&info, StringView{"qid"}, StringView{json});
        Check(info.group_ptr_.size() == expected.size(), "RLE group count");
        for (std::size_t i = 0; i < expected.size(); ++i)
          Check(info.group_ptr_[i] == expected[i], "Independent QID run boundaries");
        ++cases;
      }
    }
  }
}
void Edges(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  MetaInfo info;
  std::uint32_t patterns[]{0u, 0x80000000u, 1u, 0x80000001u, 0x3f800000u,
                           0xbf800000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
                           0xff800000u, 0x7fc00000u, 0xffc00000u};
  for (auto key : {"label", "weight", "feature_weights", "base_margin", "label_lower_bound", "label_upper_bound"}) {
    for (auto bits : patterns) {
      auto buffer = Source(device, queue, &bits, 4, true);
      auto json = Descriptor(buffer, 1, 1, 0, 1, 1, 2, 32, 0);
      bool valid = true;
      if (std::strcmp(key, "label") == 0) valid = (bits & 0x7fffffffu) < 0x7f800000u;
      if (std::strcmp(key, "weight") == 0 || std::strcmp(key, "feature_weights") == 0) {
        float value;
        std::memcpy(&value, &bits, 4);
        valid = std::isfinite(value) && !(value < 0);
      }
      auto set = [&] { metal::SetMetadata(&info, StringView{key}, StringView{json}); };
      if (!valid) Reject(set);
      else { set(); ++cases; }
      // CUDA publishes converted float metadata before validating it.
      auto value = std::strcmp(key, "label") == 0 ? info.labels.Data()->ConstHostVector()[0]
          : std::strcmp(key, "weight") == 0 ? info.weights_.ConstHostVector()[0]
          : std::strcmp(key, "feature_weights") == 0 ? info.feature_weights.ConstHostVector()[0]
          : std::strcmp(key, "base_margin") == 0 ? info.base_margin_.Data()->ConstHostVector()[0]
          : std::strcmp(key, "label_lower_bound") == 0 ? info.labels_lower_bound_.ConstHostVector()[0]
          : info.labels_upper_bound_.ConstHostVector()[0];
      Check(!std::memcmp(&value, &bits, 4), "Contiguous FP32 copy preserves every bit");
    }
  }
  std::vector<double> ids{-.5, .9, 2147483647.1, 2147483647.9, 4294967294.1, 4294967294.9};
  auto buffer = Source(device, queue, ids.data(), ids.size() * 8, true);
  auto json = Descriptor(buffer, ids.size(), 1, 0, 1, 1, 2, 64, 1);
  metal::SetMetadata(&info, StringView{"qid"}, StringView{json});
  Check(info.group_ptr_ == std::vector<std::uint32_t>({0, 2, 4, 6}), "Binary64 QID avoids FP32 narrowing");
  ++cases;
  for (double bad : {-1., 4294967296., std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()}) {
    auto b = Source(device, queue, &bad, 8, false);
    auto d = Descriptor(b, 1, 1, 0, 1, 1, 2, 64, 1);
    Reject([&] { metal::SetMetadata(&info, StringView{"qid"}, StringView{d}); });
    Check(info.group_ptr_ == std::vector<std::uint32_t>({0, 2, 4, 6}), "Invalid QID leaves groups intact");
  }
  std::uint32_t sizes[]{std::numeric_limits<std::uint32_t>::max(), 1};
  buffer = Source(device, queue, sizes, sizeof(sizes), false);
  json = Descriptor(buffer, 2, 1, 0, 1, 1, 1, 32, 1);
  Reject([&] { metal::SetMetadata(&info, StringView{"group"}, StringView{json}); });
  auto empty = Descriptor(nil, 5, 0, 0, 0, 1, 2, 32);
  metal::SetMetadata(&info, StringView{"label"}, StringView{empty});
  Check(info.labels.Shape(0) == 5 && info.labels.Shape(1) == 0 && info.labels.Size() == 0, "Empty matrix shape");
  Reject([&] { metal::SetMetadata(&info, StringView{"weight"}, StringView{empty}); });
  ++cases;
}
int main() {
  @autoreleasepool {
    try {
      auto device = MTLCreateSystemDefaultDevice();
      auto queue = [device newCommandQueue];
      for (bool private_buffer : {false, true}) {
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
        Half(device, queue, private_buffer);
        Qids<std::uint32_t>(device, queue, private_buffer);
        Qids<std::uint64_t>(device, queue, private_buffer);
        Qids<double>(device, queue, private_buffer);
        std::printf("Storage mode %d: %zu cases / %zu fields\n", private_buffer, cases, fields);
        std::fflush(stdout);
      }
      Edges(device, queue);
      auto stats = metal::GetStatistics();
      Check(stats.metadata_shared_output_bytes > 0, "Actual shared GPU metadata output");
      Check(stats.input_staging_bytes == 0, "No CPU input payload staging");
      std::printf("PASS metadata: %zu cases / %zu fields / %zu rejected; %llu shared output bytes / %llu host copy bytes\n",
                  cases, fields, rejected, (unsigned long long)stats.metadata_shared_output_bytes,
                  (unsigned long long)stats.metadata_host_copy_bytes);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
