// Actual native quantile constructors must preserve device-only weight authority.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_cuda_sketch.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_metadata.h"
#include "../xgboost/src/data/gradient_index.h"
#include "../xgboost/src/data/proxy_dmatrix.h"
#include "cuda_sketch_reference.h"

using namespace xgboost;
namespace metal = common::metal;
namespace ref = cuda_sketch_reference;
std::size_t cases = 0, fields = 0, resident_batches = 0;
constexpr unsigned kColumns = 3, kBins = 17;

void Check(bool ok, char const* reason) {
  ++fields;
  if (!ok) throw std::runtime_error(reason);
}
void Complete(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted, "Producer command failed");
}
std::string Descriptor(id<MTLBuffer> buffer, std::size_t rows, std::size_t cols,
                       std::size_t offset = 0, std::size_t stride = 1) {
  std::ostringstream out;
  out << "{\"device_type\":8,\"device_id\":0,\"buffer\":"
      << reinterpret_cast<std::uintptr_t>((__bridge void*)buffer)
      << ",\"byte_offset\":" << offset * sizeof(float)
      << ",\"shape\":[" << rows << ',' << cols << "],\"strides\":["
      << cols * stride << ',' << stride
      << "],\"dtype_code\":2,\"dtype_bits\":32,\"dtype_lanes\":1}";
  return out.str();
}

struct Input {
  data::DMatrixProxy* proxy;
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLBuffer> data_buffer, weight_buffer;
  std::vector<std::size_t> sizes;
  // 0: unweighted; 1: row; 2: group; 3: all-zero row weights.
  unsigned kind;
  bool private_buffer, closed = false;
  std::size_t page = 0, base = 0;
  std::uint64_t readbacks = 0;
  bool pending = false;
  std::vector<FeatureType> types{FeatureType::kNumerical, FeatureType::kCategorical,
                                 FeatureType::kNumerical};

  static float Value(std::size_t r, unsigned f) {
    if ((r + f) % 13 == 0) return -99;
    return f == 1 ? float(r % 7) : float(int((r * 193 + f * 17) % 997) - 498) / 8;
  }
  std::vector<bst_group_t> Groups(std::size_t rows) const {
    return rows ? std::vector<bst_group_t>{0, 0, unsigned(rows / 2), unsigned(rows)}
                : std::vector<bst_group_t>{0};
  }
  std::vector<float> Weights(std::size_t rows, std::size_t begin) const {
    if (kind == 0 || !rows) return {};
    if (kind == 2) return {7.f, float(begin % 5) / 4, 1.75f};
    std::vector<float> result(rows);
    if (kind == 1)
      for (std::size_t i = 0; i < rows; ++i) result[i] = float((begin + i) % 9) / 4;
    return result;
  }
  void CheckPrior() {
    if (!pending) return;
    auto const& weights = proxy->Info().weights_;
    // Second-pass metadata may have been moved out of the proxy for a single batch.
    if (!weights.Empty()) {
      Check(weights.DeviceCanRead() && !weights.HostCanRead(), "Quantile batch weights left GPU");
      ++resident_batches;
    }
    Check(metal::GetStatistics().vector_readback_bytes == readbacks,
          "Constructor read back a device-only payload between input callbacks");
    pending = false;
  }
  void Write(id<MTLBuffer> __strong* destination, std::vector<float> const& values) {
    auto bytes = std::max<std::size_t>(4, values.size() * sizeof(float));
    if (!*destination || (*destination).length < bytes)
      *destination = [device newBufferWithLength:bytes options:private_buffer
          ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared];
    if (values.empty()) return;
    auto source = [device newBufferWithBytes:values.data() length:values.size() * sizeof(float)
                                    options:MTLResourceStorageModeShared];
    auto command = [queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:*destination destinationOffset:0
                   size:values.size() * sizeof(float)];
    [blit endEncoding];
    Complete(command);
  }
  static void Reset(DataIterHandle handle) {
    auto& self = *static_cast<Input*>(handle);
    Check(!self.closed, "Constructor retained released input callbacks");
    self.CheckPrior();
    self.page = self.base = 0;
  }
  static int Next(DataIterHandle handle) {
    auto& self = *static_cast<Input*>(handle);
    Check(!self.closed, "Cache reread released input callbacks");
    self.CheckPrior();
    if (self.page == self.sizes.size()) return 0;
    auto rows = self.sizes[self.page++];
    std::vector<float> values(rows * kColumns);
    for (std::size_t r = 0; r < rows; ++r)
      for (unsigned f = 0; f < kColumns; ++f)
        values[r * kColumns + f] = Value(self.base + r, f);
    self.Write(&self.data_buffer, values);
    // Discard old metadata without requesting its host mirror in the test callback.
    self.proxy->Info() = MetaInfo{};
    self.proxy->SetMetalArray(StringView{Descriptor(self.data_buffer, rows, kColumns)});
    auto& info = self.proxy->Info();
    info.feature_types.HostVector() = self.types;
    if (self.kind == 2) info.group_ptr_ = self.Groups(rows);
    auto weights = self.Weights(rows, self.base);
    // Reuse an offset, strided producer allocation, then poison it after import.
    std::vector<float> storage(7 + weights.size() * 2, -123.f);
    for (std::size_t i = 0; i < weights.size(); ++i) storage[7 + i * 2] = weights[i];
    self.Write(&self.weight_buffer, storage);
    metal::SetMetadata(&info, StringView{"weight"},
                       StringView{Descriptor(self.weight_buffer, weights.size(), 1, 7, 2)});
    auto command = [self.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    [blit fillBuffer:self.weight_buffer range:NSMakeRange(0, self.weight_buffer.length) value:0xa5];
    [blit fillBuffer:self.data_buffer range:NSMakeRange(0, self.data_buffer.length) value:0xa5];
    [blit endEncoding];
    Complete(command);
    self.readbacks = metal::GetStatistics().vector_readback_bytes;
    self.pending = true;
    self.base += rows;
    return 1;
  }
  common::HistogramCuts ExpectedCuts() const {
    ref::Sketch oracle(kColumns, kBins, {false, true, false});
    std::size_t begin = 0;
    for (auto rows : sizes) {
      std::vector<ref::Raw> raw;
      auto weights = Weights(rows, begin);
      for (std::size_t r = 0; r < rows; ++r) {
        auto w = weights.empty() ? 1.f : weights[kind == 2 ? (r < rows / 2 ? 1 : 2) : r];
        for (unsigned f = 0; f < kColumns; ++f) {
          auto value = Value(begin + r, f);
          if (value != -99) raw.push_back({f, value, w});
        }
      }
      oracle.Push(std::move(raw), rows, !weights.empty());
      begin += rows;
    }
    metal::CudaSketchSummary summary;
    summary.ptr.push_back(0);
    for (auto const& column : oracle.columns) {
      for (auto e : column) summary.entries.emplace_back(e.rmin, e.rmax, e.wmin, e.value);
      summary.ptr.push_back(summary.entries.size());
    }
    // Sketch arithmetic is independent. The existing final-cut function is shared;
    // Python's CUDA-source oracle separately checks end-to-end cut extraction.
    return metal::MakeCudaCuts(summary, kColumns, kBins, types);
  }
};

void Case(id<MTLDevice> device, id<MTLCommandQueue> queue, std::filesystem::path const& dir,
          unsigned constructor, unsigned kind, bool private_buffer,
          std::vector<std::size_t> const& sizes) {
  std::shared_ptr<DMatrix> proxy = std::make_shared<data::DMatrixProxy>();
  Input input{static_cast<data::DMatrixProxy*>(proxy.get()), device, queue, nil, nil,
              sizes, kind, private_buffer};
  auto expected = input.ExpectedCuts();
  auto handle = static_cast<DataIterHandle>(&input);
  auto proxy_handle = static_cast<DMatrixHandle>(&proxy);
  std::shared_ptr<DMatrix> reference;
  for (bool use_ref : {false, true}) {
    auto before = metal::GetStatistics();
    std::shared_ptr<DMatrix> matrix;
    if (constructor == 0) {
      matrix.reset(DMatrix::Create(handle, proxy_handle, reference, Input::Reset, Input::Next,
                                   -99.f, 2, kBins, std::string{"mps"}));
    } else {
      ExtMemConfig config{(dir / (use_ref ? "validation" : "training")).string(),
                           constructor == 1, std::numeric_limits<float>::quiet_NaN(), 0, -99.f, 2};
      config.device = "mps";
      matrix.reset(DMatrix::Create(handle, proxy_handle, reference, Input::Reset, Input::Next,
                                   kBins, config));
    }
    auto after = metal::GetStatistics();
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Constructor vector readback");
    Check(after.input_staging_bytes == before.input_staging_bytes, "Native input staged on CPU");
    Check(after.metadata_host_copy_bytes == before.metadata_host_copy_bytes, "Metadata host copy");
    Check(after.adapter_debug_readback_bytes == before.adapter_debug_readback_bytes, "Adapter readback");
    std::size_t expanded = 0, dispatches = 0, group_bytes = 0;
    if (kind == 2 && !use_ref) {
      for (auto rows : sizes) if (rows) {
        expanded += rows * sizeof(float);
        ++dispatches;
        group_bytes += input.Groups(rows).size() * sizeof(bst_group_t);
      }
    }
    Check(after.sketch_weight_dispatches - before.sketch_weight_dispatches == dispatches,
          "Only nonempty weighted ranking sketches expand weights");
    Check(after.gradient_state_shared_output_bytes - before.gradient_state_shared_output_bytes == expanded,
          "Expanded group weights must have a GPU destination");
    Check(after.gradient_state_staging_bytes - before.gradient_state_staging_bytes <= group_bytes,
          "Only source-defined group boundaries may need CPU upload");
    if (use_ref) Check(after.cuda_sketch_batches == before.cuda_sketch_batches,
                       "Reference cuts must bypass sketching");
    auto const& weights = matrix->Info().weights_;
    if (!weights.Empty())
      Check(weights.DeviceCanRead() && !weights.HostCanRead(), "Final weights lost GPU authority");

    // Drop/overwrite all producer allocations before consuming the completed matrix.
    input.closed = true;
    input.data_buffer = input.weight_buffer = nil;
    Context ctx;
    ctx.Init({{"device", "mps"}, {"nthread", "2"}});
    for (auto const& page : matrix->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{})) {
      Check(!page.HostReadable(), "Native quantized page materialized on CPU");
      Check(page.cut.Ptrs() == expected.Ptrs(), "Independent sketch cut pointers differ");
      auto a = page.cut.Values(), b = expected.Values();
      Check(a.size() == b.size(), "Independent sketch cut count differs");
      for (std::size_t i = 0; i < a.size(); ++i)
        Check(!std::memcmp(&a[i], &b[i], sizeof(float)), "Independent sketch cut bits differ");
    }
    std::vector<float> expected_weights;
    std::size_t base = 0;
    for (auto rows : sizes) {
      auto batch = input.Weights(rows, base);
      expected_weights.insert(expected_weights.end(), batch.begin(), batch.end());
      base += rows;
    }
    Check(weights.ConstHostVector() == expected_weights, "Owned metadata changed with producer");
    Check(matrix->Info().num_row_ == base, "Matrix rows differ");
    input.closed = false;
    if (!use_ref) reference = std::move(matrix);
    ++cases;
  }
}

int main() {
  @autoreleasepool {
    auto device = MTLCreateSystemDefaultDevice();
    auto queue = [device newCommandQueue];
    auto dir = std::filesystem::temp_directory_path() /
               ("xgb-quantile-weights-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    try {
      for (unsigned constructor = 0; constructor < 3; ++constructor) {
        for (auto private_buffer : {false, true})
          for (unsigned kind = 0; kind < 4; ++kind)
            for (auto const& sizes : std::vector<std::vector<std::size_t>>{
                   {0}, {257}, {0, 131, 0, 257, 131, 0}})
              Case(device, queue, dir, constructor, kind, private_buffer, sizes);
        std::printf("quantile weights constructor %u: %zu cases checked\n", constructor, cases);
        std::fflush(stdout);
      }
      Check(std::filesystem::is_empty(dir), "External cache files survived matrix destruction");
      std::filesystem::remove(dir);
      std::printf("PASS %zu constructor cases, %zu resident batches, %zu checks\n",
                  cases, resident_batches, fields);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL after %zu cases: %s\n", cases, e.what());
      std::filesystem::remove_all(dir);
      return 1;
    }
  }
}
