// Explicit divergence witnesses for P5-N08/13/14; no NVIDIA execution implied.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "xgboost/host_device_vector.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_quantile.h"
#include "numerical_contract_fixtures.h"
#include "numerical_contract_report.h"

namespace fixture = numerical_contracts;
namespace metal = xgboost::common::metal;
using xgboost::HostDeviceVector;
using xgboost::DeviceOrd;
std::size_t checked = 0, witnesses = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
  ++checked;
}
std::uint32_t Bits(float x) {
  std::uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}
template <typename T> void Equal(T const* actual, std::vector<T> const& expected) {
  for (std::size_t i = 0; i < expected.size(); ++i)
    Check(std::memcmp(actual+i, &expected[i], sizeof(T)) == 0, "Exact witness payload differs");
}
struct GPU {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> cdf, select, prefix, add, reverse, fix;
  explicit GPU(char const* path) {
    Check(device != nil, "No Metal GPU");
    queue = [device newCommandQueue];
    std::ifstream file(path);
    std::string text{std::istreambuf_iterator<char>(file), {}};
    auto options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError* error = nil;
    auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:text.c_str()]
                                        options:options error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    auto pipeline = [&](NSString* name) {
      auto result = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                                          error:&error];
      Check(result != nil, "Missing numerical-contract pipeline");
      return result;
    };
    cdf = pipeline(@"xgb_quantile_weight_cdf");
    select = pipeline(@"xgb_quantile_select");
    prefix = pipeline(@"xgb_cs_prefix");
    add = pipeline(@"xgb_cs_prefix_add");
    reverse = pipeline(@"xgb_cs_scan_input");
    fix = pipeline(@"xgb_cs_fix");
  }
  id<MTLBuffer> Buffer(void const* input, std::size_t bytes) {
    auto result = [device newBufferWithLength:std::max<std::size_t>(bytes, 16)
                                      options:MTLResourceStorageModeShared];
    Check(result != nil, "Buffer allocation failed");
    std::memset(result.contents, 0xa5, result.length);
    if (input && bytes) std::memcpy(result.contents, input, bytes);
    return result;
  }
  template <typename T> id<MTLBuffer> Upload(std::vector<T> const& input) {
    return Buffer(input.data(), input.size()*sizeof(T));
  }
  id<MTLComputeCommandEncoder> Encoder(id<MTLCommandBuffer> command,
      id<MTLComputePipelineState> pipeline, std::initializer_list<id<MTLBuffer>> buffers) {
    auto e = [command computeCommandEncoder];
    Check(e != nil, "No encoder");
    [e setComputePipelineState:pipeline];
    unsigned slot = 0;
    for (auto buffer : buffers) [e setBuffer:buffer offset:0 atIndex:slot++];
    return e;
  }
  void Threads(id<MTLComputeCommandEncoder> e, std::size_t n) {
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  }
  void Finish(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    Check(command.status == MTLCommandBufferStatusCompleted, "GPU witness command failed");
  }
  id<MTLBuffer> Prefix(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                        id<MTLBuffer> ids, std::uint32_t count) {
    auto blocks = (count+255)/256;
    auto output = Buffer(nullptr, count*4), totals = Buffer(nullptr, blocks*4),
         keys = Buffer(nullptr, blocks*4);
    auto e = Encoder(command, prefix, {input, ids, output, totals, keys});
    [e setBytes:&count length:sizeof(count) atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    if (blocks > 1) {
      auto upper = Prefix(command, totals, keys, blocks);
      e = Encoder(command, add, {output, ids, upper, keys});
      Threads(e, count);
    }
    return output;
  }
};
void QuantileWitness(GPU& gpu, fixture::Report& report, unsigned rows, int exponent) {
  auto q = fixture::MakeQuantile(rows, exponent);
  Check(q.serial.back() != q.regrouped.back(), "CDF witness lost association difference");
  struct Record { std::uint32_t key, row; };
  std::vector<Record> order;
  for (unsigned i = 0; i < rows; ++i) order.push_back({0, i});
  auto records = gpu.Upload(order), weights = gpu.Upload(q.weights), values = gpu.Upload(q.values);
  auto cdf = gpu.Buffer(nullptr, rows*4), alternative = gpu.Upload(q.regrouped);
  metal::QuantileRequest request{0, 0, 1.};
  auto requests = gpu.Buffer(&request, sizeof(request));
  auto output = gpu.Buffer(nullptr, 16);  // Two result slots between two guards.
  auto command = [gpu.queue commandBuffer];
  auto e = gpu.Encoder(command, gpu.cdf, {records, weights, cdf});
  [e setBytes:&rows length:sizeof(rows) atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  struct { std::uint32_t rows, targets, weighted, requests; } p{rows, 1, 1, 1};
  struct { std::uint64_t rows, stride0, stride1; std::uint32_t targets, blocks; }
      shape{rows, 1, 1, 1, (rows+255)/256};
  for (unsigned i = 0; i < 2; ++i) {
    e = gpu.Encoder(command, gpu.select, {records, values, i ? alternative : cdf, requests, output});
    [e setBuffer:output offset:4*(i+1) atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e setBytes:&shape length:sizeof(shape) atIndex:6];
    gpu.Threads(e, 1);
  }
  gpu.Finish(command);
  Equal(static_cast<float const*>(cdf.contents), q.serial);
  auto results = static_cast<float const*>(output.contents);
  Check(results[1] == 0 && results[2] == float(rows-1), "CDF order did not change selected row");
  auto guards = static_cast<std::uint32_t const*>(output.contents);
  Check(guards[0] == 0xa5a5a5a5u && guards[3] == 0xa5a5a5a5u, "Quantile output guard changed");
  // Exercise the production sort/CDF/select API with native GPU ownership too.
  HostDeviceVector<float> v(q.values, DeviceOrd::MPS()), w(q.weights, DeviceOrd::MPS()),
      out(1, -1.f, DeviceOrd::MPS());
  std::size_t dimensions[]{rows, 1}, strides[]{1, 1};
  xgboost::linalg::MatrixView<float const> view{v.ConstDeviceSpan(), dimensions, strides, DeviceOrd::MPS()};
  auto before = metal::GetStatistics();
  metal::Quantiles(view, w.ConstDeviceSpan(), {&request, 1}, out.DeviceSpan());
  auto after = metal::GetStatistics();
  Check(after.quantile_dispatches == before.quantile_dispatches+1, "No production quantile dispatch");
  Check(after.vector_readback_bytes == before.vector_readback_bytes && !v.HostCanRead() &&
            !w.HostCanRead() && !out.HostCanRead(), "Quantile payload requested a host mirror");
  Check(out.ConstHostVector()[0] == 0, "Production quantile changed its declared CDF policy");
  report.Case("weighted_cdf", rows, exponent, 0,
              {{"values", fixture::FloatWords(q.values)}, {"weights", fixture::FloatWords(q.weights)}},
              {{"cdf", fixture::FloatWords(static_cast<float const*>(cdf.contents), rows)},
               {"selected", fixture::FloatWords(results+1, 1)},
               {"api_selected", fixture::FloatWords(out.ConstHostVector())}});
  std::printf("P5-N08 rows=%u exponent=%d totals=%08x/%08x selected=0/%u (Metal/regrouped)\n",
              rows, exponent, Bits(q.serial.back()), Bits(q.regrouped.back()), rows-1);
  ++witnesses;
}
void PrefixWitness(GPU& gpu, fixture::Report& report, unsigned rows, int exponent) {
  auto p = fixture::MakePrefix(rows, exponent);
  auto input = gpu.Upload(p.values), ids = gpu.Upload(p.ids);
  auto command = [gpu.queue commandBuffer];
  auto output = gpu.Prefix(command, input, ids, rows);
  gpu.Finish(command);
  Equal(static_cast<float const*>(output.contents), p.metal);
  unsigned differences = 0;
  for (unsigned i = 0; i < rows; ++i) differences += p.metal[i] != p.serial[i];
  Check(differences > 0, "Raw-prefix witness lost its sequential-scan difference");
  report.Case("raw_prefix", rows, exponent, 0,
              {{"weights", fixture::FloatWords(p.values)}, {"ids", {p.ids.begin(), p.ids.end()}}},
              {{"cdf", fixture::FloatWords(static_cast<float const*>(output.contents), rows)}});
  std::printf("P5-N14 raw-prefix rows=%u exponent=%d differing_CDF_entries=%u\n",
              rows, exponent, differences);
  ++witnesses;
}
void SummaryWitness(GPU& gpu, fixture::Report& report, int exponent, bool repair) {
  auto s = repair ? fixture::MakeRepair(exponent) : fixture::MakeReverse(exponent);
  Check(std::memcmp(s.metal.data(), s.alternative.data(), s.metal.size()*sizeof(fixture::Entry)) != 0,
        "Summary witness lost its order difference");
  auto data = gpu.Buffer(nullptr, (s.input.size()+2)*sizeof(fixture::Entry));
  std::memcpy(static_cast<fixture::Entry*>(data.contents)+1, s.input.data(),
              s.input.size()*sizeof(fixture::Entry));
  auto ptr = gpu.Upload(s.ptr);
  auto command = [gpu.queue commandBuffer];
  auto e = gpu.Encoder(command, repair ? gpu.fix : gpu.reverse, {data, ptr});
  [e setBuffer:data offset:sizeof(fixture::Entry) atIndex:0];
  gpu.Threads(e, s.ptr.size()-1);
  gpu.Finish(command);
  Equal(static_cast<fixture::Entry const*>(data.contents)+1, s.metal);
  auto guard = static_cast<std::uint32_t const*>(data.contents);
  for (unsigned i = 0; i < 4; ++i)
    Check(guard[i] == 0xa5a5a5a5u && guard[(s.input.size()+1)*4+i] == 0xa5a5a5a5u,
          "Summary guard changed");
  auto actual = static_cast<fixture::Entry const*>(data.contents)+1;
  std::vector<fixture::Entry> observed;
  fixture::Words observed_ptr{0};
  for (std::size_t c = 0; c+1 < s.ptr.size(); ++c) {
    for (auto i = s.ptr[c]; i < s.ptr[c+1]; ++i) {
      // CUDA ScanInput includes unique compaction. Project the observed Metal
      // reverse-scan output onto the same retained entries for comparison.
      // This is report processing, not an execution test of Metal compaction.
      if (repair || i == s.ptr[c] || actual[i].value != actual[i-1].value)
        observed.push_back(actual[i]);
    }
    observed_ptr.push_back(observed.size());
  }
  report.Case(repair ? "repair" : "reverse_unique", s.input.size(), exponent, 0,
              {{"entries", fixture::EntryWords(s.input.data(), s.input.size())}, {"ptr", s.ptr}},
              {{"entries", fixture::EntryWords(observed.data(), observed.size())}, {"ptr", observed_ptr}});
  std::printf("%s exponent=%d first_weight=%08x/%08x third_rank=%08x/%08x (Metal/alternative)\n",
              repair ? "P5-N13 repair" : "P5-N14 reverse", exponent,
              Bits(s.metal[0].wmin), Bits(s.alternative[0].wmin),
              Bits(s.metal[2].rmin), Bits(s.alternative[2].rmin));
  ++witnesses;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2 || (argc == 4 && std::string{argv[2]} == "--report"),
            "Pass production shader source [--report new-output.jsonl]");
      GPU gpu(argv[1]);
      fixture::Report report{argc == 4 ? argv[3] : ""};
      report.Metadata({{"backend", "metal"}, {"device", gpu.device.name.UTF8String},
                       {"compiler", __clang_version__}, {"shader", argv[1]},
                       {"reverse_unique", "host projection of observed GPU reverse scan"}});
      for (int exponent : {0, -149, 100}) {
        for (unsigned rows : {3u, 257u}) QuantileWitness(gpu, report, rows, exponent);
        for (unsigned rows : {3u, 261u, 513u}) PrefixWitness(gpu, report, rows, exponent);
        SummaryWitness(gpu, report, exponent, false);
        SummaryWitness(gpu, report, exponent, true);
      }
      report.Finish();
      std::printf("PASS %zu explicit numerical witnesses / %zu exact checks; NVIDIA untested\n",
                  witnesses, checked);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
