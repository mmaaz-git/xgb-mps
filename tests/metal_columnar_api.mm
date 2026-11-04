// Native columnar buffers: private allocations, signed strides, masks and exact
// names.
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_tensor.h"
#include "../xgboost/src/data/cat_container.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace metal = xgboost::common::metal;
using namespace xgboost;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool value, char const *msg) {
  ++fields;
  if (!value)
    throw std::runtime_error(msg);
}
void Complete(id<MTLCommandBuffer> c) {
  [c commit];
  [c waitUntilCompleted];
  Check(c.status == MTLCommandBufferStatusCompleted,
        "Producer GPU command failed");
}
id<MTLBuffer> Source(id<MTLDevice> d, id<MTLCommandQueue> q, void const *data,
                     std::size_t n) {
  if (!n)
    return nil;
  auto shared = [d newBufferWithBytes:data
                               length:n
                              options:MTLResourceStorageModeShared];
  auto out = [d newBufferWithLength:n options:MTLResourceStorageModePrivate];
  auto c = [q commandBuffer];
  auto b = [c blitCommandEncoder];
  [b copyFromBuffer:shared
           sourceOffset:0
               toBuffer:out
      destinationOffset:0
                   size:n];
  [b endEncoding];
  Complete(c);
  return out;
}
std::string Vector(id<MTLBuffer> buffer, long n, long offset, long stride,
                   int code, int bits) {
  std::ostringstream s;
  s << "{\"device_type\":8,\"device_id\":0,\"buffer\":"
    << reinterpret_cast<std::uintptr_t>((__bridge void *)buffer)
    << ",\"byte_offset\":" << offset << ",\"shape\":[" << n << "],\"strides\":["
    << stride << "],\"dtype_code\":" << code << ",\"dtype_bits\":" << bits
    << ",\"dtype_lanes\":1}";
  return s.str();
}
std::shared_ptr<metal::Tensor> Table(std::string const &columns) {
  auto json = "{\"columns\":[" + columns + "]}";
  return metal::ImportTensor(StringView{json});
}
template <typename Fn> void Reject(Fn fn) {
  bool caught = false;
  try {
    fn();
  } catch (std::exception const &) {
    caught = true;
  }
  Check(caught, "Invalid native columnar input accepted");
  ++rejected;
}
void Masks(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  std::vector<double> values(149);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = double(i) / 7;
  for (long stride : {1L, -1L, 0L})
    for (unsigned bit = 0; bit < 8; ++bit) {
      @autoreleasepool {
        std::vector<std::uint8_t> mask(20, 0);
        for (unsigned r = 0; r < 137; ++r)
          if (r % 5)
            mask[(r + bit) / 8] |= 1u << ((r + bit) % 8);
        auto v = Source(device, queue, values.data(), values.size() * 8);
        auto m = Source(device, queue, mask.data(), mask.size());
        long start = stride < 0 ? 140 : 4;
        auto column =
            "{\"values\":" + Vector(v, 137, start * 8, stride, 2, 64) +
            ",\"mask\":" + Vector(m, 20, 0, 1, 1, 8) +
            ",\"mask_bit\":" + std::to_string(bit) + "}";
        auto before = metal::GetStatistics();
        auto table = Table(column + "," + column);
        // Producer release/overwrite cannot change the snapshot, masks
        // included.
        auto c = [queue commandBuffer];
        auto b = [c blitCommandEncoder];
        [b fillBuffer:v range:NSMakeRange(0, v.length) value:0];
        [b fillBuffer:m range:NSMakeRange(0, m.length) value:0];
        [b endEncoding];
        Complete(c);
        v = nil;
        m = nil;
        Check(table->NumRows() == 137 && table->NumColumns() == 2,
              "Table shape");
        SparsePage all;
        metal::CopyTensorToSparsePage(*table, NAN, &all);
        for (unsigned begin : {0u, 1u, 7u, 8u, 93u, 137u}) {
          auto rows = std::min<unsigned>(137 - begin, 17);
          auto input = metal::PrepareInput(*table, NAN, begin, rows, 3, {});
          std::vector<float> actual(rows * 3);
          input->Readback(actual);
          for (unsigned r = 0; r < rows; ++r)
            for (unsigned f = 0; f < 3; ++f) {
              auto a = actual[r * 3 + f];
              if ((begin + r) % 5 == 0 || f == 2)
                Check(std::isnan(a), "Mask/window/padded feature");
              else
                Check(a == float(values[start + (begin + r) * stride]),
                      "Strided typed column value");
            }
        }
        // Partial first/last rows exercise the flattened CUDA sketch-window
        // API.
        for (unsigned begin : {1u, 15u, 187u}) {
          auto input = metal::PrepareAdapter(*table, NAN, begin, 47);
          SparsePage page;
          input->Readback(&page);
          std::size_t n = 0;
          for (unsigned i = begin; i < begin + 47; ++i) {
            auto row = i / 2, feature = i % 2;
            if (row % 5 == 0)
              continue;
            auto e = page.data.ConstHostVector().at(n++);
            Check(e.index == feature &&
                      e.fvalue == float(values[start + row * stride]),
                  "Flattened window entries");
          }
          Check(n == page.data.Size(), "Flattened window count");
        }
        auto after = metal::GetStatistics();
        Check(after.columnar_imports == before.columnar_imports + 1,
              "Native table dispatch accounting");
        Check(after.input_staging_bytes == before.input_staging_bytes,
              "Native payload host staging");
        ++cases;
      }
    }
}
template <typename T>
void Names(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  std::vector<T> values{std::numeric_limits<T>::lowest(), T(0), T(1), T(7),
                        std::numeric_limits<T>::max()};
  for (long stride : {-1L, 1L, 0L}) {
    auto b = Source(device, queue, values.data(), values.size() * sizeof(T));
    long start = stride < 0 ? 4 : 0;
    auto names = Vector(b, 5, start * sizeof(T), stride,
                        std::is_signed_v<T> ? 0 : 1, sizeof(T) * 8);
    auto codes = Vector(nil, 0, 0, 1, 0, 32);
    auto before = metal::GetStatistics();
    auto table =
        Table("{\"values\":" + codes + ",\"categories\":" + names + "}");
    auto view = table->Cats();
    Check(view.n_total_cats == 5, "Dictionary segment count");
    auto actual = std::get<common::Span<T const>>(view.columns[0]);
    for (long i = 0; i < 5; ++i)
      Check(actual[i] == values[start + i * stride],
            "Exact typed dictionary bits");
    auto after = metal::GetStatistics();
    Check(after.category_import_shared_bytes -
                  before.category_import_shared_bytes +
                  after.category_import_copy_bytes -
                  before.category_import_copy_bytes ==
              values.size() * sizeof(T),
          "Dictionary byte accounting");
    Check(after.category_import_dispatches ==
              before.category_import_dispatches + 1,
          "Dictionary GPU gather");
    ++cases;
  }
}
void Errors(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  std::uint32_t values[]{0, 1, 2, 3, 4, 5, 6, 7};
  auto b = Source(device, queue, values, sizeof(values));
  auto v = Vector(b, 8, 0, 1, 0, 32);
  Reject([&] { Table(""); });
  Reject([&] {
    Table("{\"values\":" + v + "},{\"values\":" + Vector(b, 7, 0, 1, 0, 32) +
          "}");
  });
  for (auto mask : {Vector(b, 0, 0, 1, 1, 8), Vector(b, 8, 0, 1, 0, 8),
                    Vector(b, 8, 0, 2, 1, 8)})
    Reject([&] {
      Table("{\"values\":" + v + ",\"mask\":" + mask + ",\"mask_bit\":0}");
    });
  for (int bit : {-1, 8})
    Reject([&] {
      Table("{\"values\":" + v + ",\"mask\":" + Vector(b, 8, 0, 1, 1, 8) +
            ",\"mask_bit\":" + std::to_string(bit) + "}");
    });
  for (auto names : {Vector(b, 8, 0, 1, 2, 32), Vector(nil, 0, 0, 1, 0, 32),
                     Vector(b, 8, 0, 1, 6, 8)})
    Reject(
        [&] { Table("{\"values\":" + v + ",\"categories\":" + names + "}"); });
  for (std::vector<std::int32_t> offsets :
       {std::vector<std::int32_t>{-1, 0}, {0, 33}, {2, 1}, {0}}) {
    auto o = Source(device, queue, offsets.data(), offsets.size() * 4);
    auto names = "{\"offsets\":" + Vector(o, offsets.size(), 0, 1, 0, 32) +
                 ",\"values\":" + Vector(b, 32, 0, 1, 0, 8) + "}";
    Reject(
        [&] { Table("{\"values\":" + v + ",\"categories\":" + names + "}"); });
  }
  // CPU ARM NaN-to-int may yield zero: dictionary mapping must not see NaN as
  // code 0.
  std::vector<std::int32_t> segments{0, 2}, mapping{1, 0};
  CatAccessor accessor{enc::MappingView{segments, mapping}};
  for (std::uint32_t pattern :
       {0x7fc00000u, 0xffc00000u, 0x7fffffffu, 0xffffffffu}) {
    float nan;
    std::memcpy(&nan, &pattern, 4);
    Check(std::isnan(accessor(nan, 0)),
          "CPU dictionary recoding converted missing to category zero");
  }
}
int main() {
  @autoreleasepool {
    auto d = MTLCreateSystemDefaultDevice();
    auto q = [d newCommandQueue];
    Masks(d, q);
    Names<std::int8_t>(d, q);
    Names<std::uint8_t>(d, q);
    Names<std::int16_t>(d, q);
    Names<std::uint16_t>(d, q);
    Names<std::int32_t>(d, q);
    Names<std::uint32_t>(d, q);
    Names<std::int64_t>(d, q);
    Names<std::uint64_t>(d, q);
    Errors(d, q);
    std::printf("PASS native columnar: %zu cases, %zu fields, %zu rejected "
                "descriptors\n",
                cases, fields, rejected);
  }
}
