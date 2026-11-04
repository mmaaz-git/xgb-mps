// Independent scalar casts and pair layout from CUDA CustomGradHessOp.
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
#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"

namespace metal = xgboost::common::metal;
using xgboost::StringView;
using Matrix = xgboost::linalg::Matrix<xgboost::GradientPair>;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++fields;
  if (!ok) throw std::runtime_error(message);
}
void Same(float a, float b) {
  Check((std::isnan(a) && std::isnan(b)) || !std::memcmp(&a, &b, 4), "Gradient pair cast bits");
}
void Complete(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted, "Producer command failed");
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
void Pack(std::string const& g, std::string const& h, std::size_t rows, Matrix* out) {
  auto before = metal::GetStatistics();
  metal::CopyNativeGradients(StringView{g}, StringView{h}, rows, out);
  auto after = metal::GetStatistics();
  auto n = out->Size();
  Check(after.gradient_import_pairs - before.gradient_import_pairs == n, "Pair count");
  Check(after.gradient_import_dispatches - before.gradient_import_dispatches == (n != 0), "Dispatch count");
  Check(after.gradient_import_shared_output_bytes - before.gradient_import_shared_output_bytes +
        after.gradient_import_host_copy_bytes - before.gradient_import_host_copy_bytes == n * 8,
        "Destination sharing/copy accounting");
  Check(after.input_staging_bytes == before.input_staging_bytes, "No CPU input staging");
  ++cases;
}
template <class T> int Code() {
  return std::is_floating_point_v<T> ? 2 : std::is_signed_v<T> ? 0 : 1;
}
template <class T, class U>
void Typed(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<T> grads(1024);
  std::vector<U> hess(1024);
  for (std::size_t i = 0; i < grads.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) grads[i] = T(int(i) - 512) / T(7);
    else grads[i] = T(std::uint64_t(i) * 0x9e3779b97f4a7c15ULL);
    if constexpr (std::is_floating_point_v<U>) hess[i] = U(int(i) - 512) / U(11);
    else hess[i] = U(std::uint64_t(i) * 0x517cc1b727220a95ULL);
  }
  for (auto gs : std::vector<std::pair<long, long>>{{17, 2}, {-17, -2}, {0, 0}, {1, 23}}) {
    for (auto hs : std::vector<std::pair<long, long>>{{19, 3}, {-19, 3}, {0, 0}}) {
      @autoreleasepool {
        auto gb = Source(device, queue, grads.data(), grads.size() * sizeof(T), private_buffer);
        auto hb = Source(device, queue, hess.data(), hess.size() * sizeof(U), !private_buffer);
        auto g = Descriptor(gb, 19, 7, 400 * sizeof(T), gs.first, gs.second, Code<T>(), sizeof(T) * 8);
        auto h = Descriptor(hb, 19, 7, 400 * sizeof(U), hs.first, hs.second, Code<U>(), sizeof(U) * 8);
        Matrix result;
        Pack(g, h, 19, &result);
        auto command = [queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        [blit fillBuffer:gb range:NSMakeRange(0, gb.length) value:0xa5];
        [blit fillBuffer:hb range:NSMakeRange(0, hb.length) value:0xa5];
        [blit endEncoding];
        Complete(command);
        gb = nil; hb = nil;
        Check(result.Shape(0) == 19 && result.Shape(1) == 7, "Pair matrix shape");
        auto const& out = result.Data()->ConstHostVector();
        for (long r = 0; r < 19; ++r) for (long c = 0; c < 7; ++c) {
          Same(out[r * 7 + c].GetGrad(), float(grads[400 + r * gs.first + c * gs.second]));
          Same(out[r * 7 + c].GetHess(), float(hess[400 + r * hs.first + c * hs.second]));
        }
      }
    }
  }
}
void Edges(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<std::uint16_t> bits(65536);
  for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = i;
  auto buffer = Source(device, queue, bits.data(), bits.size() * 2, private_buffer);
  auto g = Descriptor(buffer, 256, 256, 0, 256, 1, 2, 16);
  auto h = Descriptor(buffer, 256, 256, 65535 * 2, -256, -1, 2, 16);
  Matrix out;
  Pack(g, h, 256, &out);
  auto const& pairs = out.Data()->ConstHostVector();
  for (std::size_t i = 0; i < bits.size(); ++i) {
    __fp16 gv, hv;
    std::memcpy(&gv, &bits[i], 2);
    std::memcpy(&hv, &bits[65535 - i], 2);
    Same(pairs[i].GetGrad(), float(gv));
    Same(pairs[i].GetHess(), float(hv));
  }
  std::uint32_t patterns[]{0, 0x80000000u, 1, 0x80000001u, 0x7fffffu, 0x807fffffu,
                          0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu,
                          0x7f800000u, 0xff800000u, 0x7fc00123u, 0xffc00123u};
  for (auto bits32 : patterns) {
    auto b = Source(device, queue, &bits32, 4, private_buffer);
    for (int dimensions : {0, 1, 2}) {
      auto desc = Descriptor(b, 1, 1, 0, 1, 1, 2, 32, dimensions);
      Pack(desc, desc, 1, &out);
      float expected; std::memcpy(&expected, &bits32, 4);
      Same(out.Data()->ConstHostVector()[0].GetGrad(), expected);
      Same(out.Data()->ConstHostVector()[0].GetHess(), expected);
    }
  }
  for (auto shape : std::vector<std::pair<int, int>>{{0, 1}, {0, 3}, {5, 0}}) {
    auto desc = Descriptor(nil, shape.first, shape.second, 0, 1, 1, 2, 32);
    Pack(desc, desc, shape.first, &out);
    Check(out.Shape(0) == shape.first && out.Shape(1) == shape.second, "Empty shape");
  }
}
void Invalid(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  std::vector<float> values(16, 1.f);
  auto b = Source(device, queue, values.data(), values.size() * 4, true);
  auto good = Descriptor(b, 4, 2, 0, 2, 1, 2, 32);
  Matrix out;
  Pack(good, good, 4, &out);
  std::vector<std::string> bad{
    Descriptor(b, 3, 2, 0, 2, 1, 2, 32), Descriptor(b, 4, 3, 0, 3, 1, 2, 32),
    Descriptor(b, 4, 2, 0, -2, 1, 2, 32), Descriptor(b, 4, 2, 60, 2, 1, 2, 32),
    Descriptor(b, 4, 2, 1, 2, 1, 2, 32), Descriptor(b, 4, 2, 0, 2, 1, 2, 128),
    Descriptor(nil, 4, 2, 0, 2, 1, 2, 32)};
  auto wrong_device = good; wrong_device.replace(wrong_device.find("\"device_id\":0"), 13, "\"device_id\":1");
  bad.push_back(wrong_device);
  for (auto const& desc : bad) for (bool invalid_grad : {false, true}) {
    bool failed = false;
    try { metal::CopyNativeGradients(StringView{invalid_grad ? desc : good},
                                    StringView{invalid_grad ? good : desc}, 4, &out); }
    catch (std::exception const&) { failed = true; }
    Check(failed, "Invalid gradient input accepted");
    Check(out.Shape(0) == 4 && out.Shape(1) == 2, "Invalid input resized destination");
    for (auto const& pair : out.Data()->ConstHostVector()) {
      Same(pair.GetGrad(), 1.f); Same(pair.GetHess(), 1.f);
    }
    ++rejected;
  }
  bool failed = false;
  try { metal::CopyNativeGradients(StringView{good}, StringView{good}, 5, &out); }
  catch (std::exception const&) { failed = true; }
  Check(failed && out.Shape(0) == 4, "Training row count validation");
  ++rejected;
}
int main() {
  @autoreleasepool {
    try {
      auto device = MTLCreateSystemDefaultDevice();
      auto queue = [device newCommandQueue];
      for (bool p : {false, true}) {
        Typed<float, double>(device, queue, p);
        Typed<double, float>(device, queue, p);
        Typed<std::int8_t, std::uint64_t>(device, queue, p);
        Typed<std::uint8_t, std::int64_t>(device, queue, p);
        Typed<std::int16_t, std::uint32_t>(device, queue, p);
        Typed<std::uint16_t, std::int32_t>(device, queue, p);
        Typed<std::int32_t, std::uint16_t>(device, queue, p);
        Typed<std::uint32_t, std::int16_t>(device, queue, p);
        Typed<std::int64_t, std::uint8_t>(device, queue, p);
        Typed<std::uint64_t, std::int8_t>(device, queue, p);
        Edges(device, queue, p);
      }
      Invalid(device, queue);
      auto stats = metal::GetStatistics();
      std::printf("Native gradient import PASS: %zu cases, %zu fields, %zu invalid inputs; "
                  "%llu pairs, %llu shared output bytes, %llu host copy bytes\n",
                  cases, fields, rejected, (unsigned long long)stats.gradient_import_pairs,
                  (unsigned long long)stats.gradient_import_shared_output_bytes,
                  (unsigned long long)stats.gradient_import_host_copy_bytes);
    } catch (std::exception const& error) {
      std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
  }
}
