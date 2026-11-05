// Native ordinary-matrix import: independent CPU typed casts and CSR filtering.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "../xgboost/src/common/metal_tensor.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_cuda_sketch.h"
#include "../xgboost/src/common/metal_ellpack.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/simple_dmatrix.h"
#include "../xgboost/src/data/gradient_index.h"

namespace metal = xgboost::common::metal;
using xgboost::StringView;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++fields;
  if (!ok) throw std::runtime_error(message);
}
void Same(float a, float b) { Check(!std::memcmp(&a, &b, 4), "CSR typed cast bits"); }
void Complete(id<MTLCommandBuffer> command) {
  [command commit]; [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted, "Producer command failed");
}
id<MTLBuffer> Source(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     void const* data, std::size_t bytes, bool private_buffer) {
  if (!bytes) return nil;
  auto shared = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
  if (!private_buffer) return shared;
  auto result = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  auto command = [queue commandBuffer]; auto blit = [command blitCommandEncoder];
  [blit copyFromBuffer:shared sourceOffset:0 toBuffer:result destinationOffset:0 size:bytes];
  [blit endEncoding]; Complete(command);
  return result;
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
void Import(std::shared_ptr<metal::Tensor> const& tensor, std::vector<float> const& expected,
             float missing, xgboost::SparsePage* output) {
  auto before = metal::GetStatistics();
  metal::CopyTensorToSparsePage(*tensor, missing, output);
  auto after = metal::GetStatistics();
  Check(output->data.Device().IsMPS() && output->data.DeviceCanWrite() &&
            !output->data.HostCanRead(), "CSR entries lost Metal authority");
  Check(output->offset.Device().IsMPS() && output->offset.HostCanWrite(),
        "CSR offsets must publish CUDA's host view");
  Check(after.vector_readback_bytes - before.vector_readback_bytes ==
            (tensor->NumRows() + 1) * sizeof(xgboost::bst_idx_t),
        "Import read back more than row offsets");
  Check(after.tensor_csr_host_copy_bytes == before.tensor_csr_host_copy_bytes,
        "CSR destination did not bind owned Metal buffers");
  auto const& ptr = output->offset.ConstHostVector();
  auto const& entries = output->data.ConstHostVector();
  std::size_t n = 0;
  Check(ptr.size() == tensor->NumRows() + 1 && ptr[0] == 0, "CSR pointer shape");
  for (std::size_t r = 0; r < tensor->NumRows(); ++r) {
    for (std::size_t c = 0; c < tensor->NumColumns(); ++c) {
      auto value = expected[r * tensor->NumColumns() + c];
      if (std::isnan(value) || value == missing) continue;
      Check(n < entries.size() && entries[n].index == c, "Ascending CSR feature index");
      Same(entries[n++].fvalue, value);
    }
    Check(ptr[r + 1] == n, "Independent CSR row prefix");
  }
  Check(n == entries.size(), "CSR nonmissing count");
  Check(after.tensor_csr_values - before.tensor_csr_values == n, "CSR value accounting");
  Check(after.tensor_csr_dispatches == before.tensor_csr_dispatches + 1, "CSR dispatch");
  Check(after.tensor_csr_shared_output_bytes - before.tensor_csr_shared_output_bytes +
        after.tensor_csr_host_copy_bytes - before.tensor_csr_host_copy_bytes == (ptr.size() + n) * 8,
        "CSR sharing/copy accounting");
  Check(after.input_staging_bytes == before.input_staging_bytes, "No CPU native input staging");
  Check(after.adapter_debug_readback_bytes == before.adapter_debug_readback_bytes, "No diagnostic readback");
  ++cases;
}
template <class T>
void Typed(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<T> values(1024);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) values[i] = i % 31 ? T(int(i) - 512) / T(7) : T(NAN);
    else values[i] = T(std::uint64_t(i) * 0x9e3779b97f4a7c15ULL);
  }
  for (auto stride : std::vector<std::pair<long, long>>{{17, 2}, {-17, -2}, {0, 0}, {1, 23}}) {
    @autoreleasepool {
      auto b = Source(device, queue, values.data(), values.size() * sizeof(T), private_buffer);
      auto desc = Descriptor(b, 19, 7, 400 * sizeof(T), stride.first, stride.second,
                             std::is_floating_point_v<T> ? 2 : std::is_signed_v<T> ? 0 : 1, sizeof(T) * 8);
      auto tensor = metal::ImportTensor(StringView{desc});
      auto command = [queue commandBuffer]; auto blit = [command blitCommandEncoder];
      [blit fillBuffer:b range:NSMakeRange(0, b.length) value:0xa5];
      [blit endEncoding]; Complete(command); b = nil;
      std::vector<float> expected;
      for (long r = 0; r < 19; ++r) for (long c = 0; c < 7; ++c)
        expected.push_back(float(values[400 + r * stride.first + c * stride.second]));
      xgboost::SparsePage output;
      output.base_rowid = 73;
      Import(tensor, expected, NAN, &output);
      Import(tensor, expected, expected[0], &output);
      Check(output.base_rowid == 73, "Caller row base preserved");
    }
  }
}
void Half(id<MTLDevice> device, id<MTLCommandQueue> queue, bool private_buffer) {
  std::vector<std::uint16_t> bits(65536);
  std::vector<float> expected(65536);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    bits[i] = i;
    // Nonmissing infinities must reject, so they have separate predicate tests
    // below instead of entering this valid CSR page.
    if ((i & 0x7fff) == 0x7c00) bits[i] |= 1;
    __fp16 value; std::memcpy(&value, &bits[i], 2); expected[i] = float(value);
  }
  auto b = Source(device, queue, bits.data(), bits.size() * 2, private_buffer);
  auto desc = Descriptor(b, 256, 256, 0, 256, 1, 2, 16);
  xgboost::SparsePage output;
  Import(metal::ImportTensor(StringView{desc}), expected, NAN, &output);
  Import(metal::ImportTensor(StringView{desc}), expected, -0.f, &output);
  for (std::uint16_t infinity : {std::uint16_t{0x7c00}, std::uint16_t{0xfc00}}) {
    auto ib = Source(device, queue, &infinity, 2, private_buffer);
    auto idesc = Descriptor(ib, 1, 1, 0, 1, 1, 2, 16);
    auto input = metal::ImportTensor(StringView{idesc});
    float value = infinity == 0x7c00 ? INFINITY : -INFINITY;
    Import(input, {value}, value, &output);
    bool failed = false;
    try { metal::CopyTensorToSparsePage(*input, NAN, &output); }
    catch (std::exception const&) { failed = true; }
    Check(failed && output.data.Size() == 0, "Half nonmissing infinity validation");
    ++rejected;
  }
}
void Edges(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  for (auto shape : std::vector<std::pair<int, int>>{{0, 7}, {0, 0}, {7, 0}}) {
    auto desc = Descriptor(nil, shape.first, shape.second, 0, shape.second, 1, 2, 32);
    xgboost::SparsePage output;
    auto tensor = metal::ImportTensor(StringView{desc});
    Import(tensor, {}, NAN, &output);
    auto adapter = metal::PrepareAdapter(*tensor, NAN, 0, 0);
    Check(adapter->NonMissing() == 0, "Empty native adapter");
  }
  std::uint32_t patterns[]{0, 0x80000000u, 1, 0x80000001u, 0x7fffffu, 0x807fffffu,
                          0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu,
                          0x7f800000u, 0xff800000u, 0x7fc00123u, 0xffc00123u};
  for (auto bits : patterns) {
    float value; std::memcpy(&value, &bits, 4);
    auto b = Source(device, queue, &bits, 4, true);
    auto desc = Descriptor(b, 1, 1, 0, 1, 1, 2, 32);
    auto tensor = metal::ImportTensor(StringView{desc});
    for (float missing : {NAN, value}) {
      xgboost::SparsePage output;
      output.offset.HostVector() = {0, 1};
      output.data.HostVector() = {xgboost::Entry{7, 13.f}};
      if (std::isinf(value) && value != missing) {
        bool failed = false;
        try { metal::CopyTensorToSparsePage(*tensor, missing, &output); }
        catch (std::exception const&) { failed = true; }
        Check(failed && output.offset.ConstHostVector() == std::vector<xgboost::bst_idx_t>{0, 1},
              "Invalid nonmissing infinity accepted or changed destination");
        Check(output.data.ConstHostVector()[0].index == 7, "Invalid import overwrote data");
        Same(output.data.ConstHostVector()[0].fvalue, 13.f);
        ++rejected;
      } else {
        Import(tensor, {value}, missing, &output);
      }
    }
  }
}
void SharedAndFallbackSketch() {
  for (bool readonly : {false, true}) {
    auto bytes = std::size_t(getpagesize());
    auto mem = static_cast<char*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
    Check(mem != MAP_FAILED, "mmap");
    auto rows = reinterpret_cast<xgboost::bst_idx_t*>(mem + 8);
    auto entries = reinterpret_cast<xgboost::Entry*>(mem + 256);
    for (int i = 0; i < 17; ++i) rows[i] = i;
    for (int i = 0; i < 16; ++i) entries[i] = xgboost::Entry{0, float(16 - i)};
    if (readonly) Check(mprotect(mem, bytes, PROT_READ) == 0, "mprotect");
    auto sketch = metal::MakeCudaSketch(1, 32, {}, {});
    auto before = metal::GetStatistics();
    sketch->Push({entries, 16}, {rows, 17}, 0, 0, 16);
    auto after = metal::GetStatistics();
    Check((readonly ? after.csr_input_copy_bytes - before.csr_input_copy_bytes
                    : after.csr_shared_input_bytes - before.csr_shared_input_bytes) == 33 * 8,
          "Sparse sketch shared/copy branch");
    Check(munmap(mem, bytes) == 0, "munmap");
    auto summary = sketch->Read();
    Check(summary.ptr == std::vector<std::uint64_t>{0, 16}, "Independent complete summary size");
    for (int i = 0; i < 16; ++i) Same(summary.entries[i].value, float(i + 1));
    ++cases;
  }
}
void ResidentConsumers(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  using namespace xgboost;
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "2"}});
  auto cpu = ctx.MakeCPU();
  for (std::size_t rows : {0, 33, 257}) for (std::size_t columns : {1, 7}) {
    for (int kind : {0, 1, 2}) for (bool weighted : {false, true}) {
      std::vector<float> values(rows * columns);
      for (std::size_t r = 0; r < rows; ++r) for (std::size_t c = 0; c < columns; ++c)
        values[r * columns + c] = kind == 2 || (kind == 1 && (r % 7 == 0 || c % 3 == 2))
                                     ? NAN : float((r * 13 + c * 7) % 31) / 4;
      auto buffer = Source(device, queue, values.data(), values.size() * 4, weighted);
      auto descriptor = Descriptor(buffer, rows, columns, 0, columns, 1, 2, 32);
      auto tensor = metal::ImportTensor(StringView{descriptor});
      data::SimpleDMatrix actual{*tensor, NAN, 2};
      tensor.reset(); buffer = nil;
      data::DenseAdapter adapter{values.data(), rows, columns};
      data::SimpleDMatrix expected{&adapter, NAN, 2};
      auto raw_batches = actual.GetBatches<SparsePage>();
      auto const& raw = *raw_batches.begin();
      Check(raw.data.Device().IsMPS() && !raw.data.HostCanRead(),
            "Native constructor materialized entry payloads");
      if (weighted) {
        std::vector<float> weights(rows);
        for (std::size_t r = 0; r < rows; ++r) weights[r] = float(r % 5 + 1) / 4;
        actual.Info().weights_.HostVector() = weights;
        expected.Info().weights_.HostVector() = weights;
      }
      std::vector<float> hessian(rows, 2.f);
      // Hist/approx regeneration and cache reuse must preserve entry authority.
      for (bool approx : {false, true}) for (bst_bin_t bins : {9, 17, 17}) {
        auto param = approx ? BatchParam{bins, hessian, true} : BatchParam{bins, .2};
        auto before = metal::GetStatistics();
        auto batches = actual.GetBatches<GHistIndexMatrix>(&ctx, param);
        auto const& gpu = *batches.begin();
        auto after = metal::GetStatistics();
        Check(raw.data.DeviceCanRead() && !raw.data.HostCanRead(),
              "Sketch/compression published entry host mirror");
        Check(after.vector_readback_bytes == before.vector_readback_bytes,
              "Sketch/compression implicitly read back vector payloads");
        Check(after.vector_upload_bytes == before.vector_upload_bytes || weighted,
              "Resident entries were uploaded again");
        Check(after.csr_input_copy_bytes == before.csr_input_copy_bytes,
              "Resident CSR consumer copied input through CPU");
        Check(gpu.metal_page != nullptr, "Native matrix lost compressed GPU representation");
        Check(metal::CompressedNonMissing(gpu) == actual.Info().num_nonzero_,
              "Compressed nonmissing count differs");
        auto references = expected.GetBatches<GHistIndexMatrix>(&ctx, param);
        auto const& reference = *references.begin();
        Check(gpu.cut.Ptrs() == reference.cut.Ptrs(), "Host/native cut offsets differ");
        Check(gpu.cut.Values() == reference.cut.Values(), "Host/native cut values differ");
        auto oracle_rows = expected.GetBatches<SparsePage>();
        GHistIndexMatrix oracle{&cpu, *oracle_rows.begin(), {}, gpu.cut, bins,
                               actual.IsDense(), param.sparse_thresh};
        auto host_batches = actual.GetBatches<GHistIndexMatrix>(&cpu, BatchParam{});
        auto const& host = *host_batches.begin();
        Check(host.index.Size() == oracle.index.Size(), "CPU bin count differs");
        for (std::size_t r = 0; r <= rows; ++r)
          Check(host.row_ptr[r] == oracle.row_ptr[r], "CPU row offset differs");
        for (std::size_t k = 0; k < oracle.index.Size(); ++k)
          Check(host.index[k] == oracle.index[k], "Independent CPU bin search differs");
        Check(!raw.data.HostCanRead(), "Reference comparison read native entries");
        ++cases;
      }
      // Explicit CPU access remains usable after the entire GPU pipeline.
      auto const& host = raw.data.ConstHostVector();
      Check(raw.data.HostCanRead() && host.size() == actual.Info().num_nonzero_,
            "Explicit CPU entry handoff failed");
    }
  }
}
int main() {
  @autoreleasepool {
    try {
      auto device = MTLCreateSystemDefaultDevice(); auto queue = [device newCommandQueue];
      for (bool p : {false, true}) {
        Typed<float>(device, queue, p); Typed<double>(device, queue, p);
        Typed<std::int8_t>(device, queue, p); Typed<std::uint8_t>(device, queue, p);
        Typed<std::int16_t>(device, queue, p); Typed<std::uint16_t>(device, queue, p);
        Typed<std::int32_t>(device, queue, p); Typed<std::uint32_t>(device, queue, p);
        Typed<std::int64_t>(device, queue, p); Typed<std::uint64_t>(device, queue, p);
        Half(device, queue, p);
      }
      Edges(device, queue); SharedAndFallbackSketch();
      ResidentConsumers(device, queue);
      auto s = metal::GetStatistics();
      std::printf("Native tensor CSR PASS: %zu cases / %zu fields / %zu rejected; "
                  "%llu entries / %llu shared output bytes / %llu host copies; "
                  "CSR inputs %llu shared bytes / %llu copied bytes\n", cases, fields, rejected,
                  (unsigned long long)s.tensor_csr_values, (unsigned long long)s.tensor_csr_shared_output_bytes,
                  (unsigned long long)s.tensor_csr_host_copy_bytes, (unsigned long long)s.csr_shared_input_bytes,
                  (unsigned long long)s.csr_input_copy_bytes);
    } catch (std::exception const& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
  }
}
