// Ordered, variable GPU payload reduction and distributed CUDA-derived sketches.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/collective/allreduce.h"
#include "../xgboost/src/collective/allreduce_v_metal.h"
#include "../xgboost/src/common/hist_util.h"
#include "../xgboost/src/common/metal_collective.h"
#include "../xgboost/src/common/metal_cuda_sketch.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "cuda_sketch_reference.h"
#include "xgboost/c_api.h"

using namespace xgboost;
namespace metal = common::metal;
namespace coll = collective;
namespace ref = cuda_sketch_reference;
std::size_t cases = 0, fields = 0;
bool federated = false;
std::string current;
void Check(bool ok, char const* why) {
  if (!ok) throw std::runtime_error(current + ": " + why);
}
void Api(int status) {
  if (status) throw std::runtime_error(XGBGetLastError());
}
void Same(float a, float b) {
  ++fields;
  Check((std::isnan(a) && std::isnan(b)) || !std::memcmp(&a, &b, 4), "Float bits differ");
}
void Verify(metal::CudaSketchSummary const& actual, ref::Sketch const& expected) {
  std::size_t offset = 0;
  Check(actual.ptr.size() == expected.columns.size() + 1, "Pointer count");
  for (std::size_t f = 0; f < expected.columns.size(); ++f) {
    Check(actual.ptr[f] == offset, "Column pointer");
    ++fields;
    for (auto e : expected.columns[f]) {
      Check(offset < actual.entries.size(), "Missing entry");
      auto a = actual.entries[offset++];
      Same(a.rmin, e.rmin);
      Same(a.rmax, e.rmax);
      Same(a.wmin, e.wmin);
      Same(a.value, e.value);
    }
  }
  Check(actual.entries.size() == offset && actual.ptr.back() == offset, "Summary size");
}
// Independent recursive grouping: split a range at its largest complete power
// of two. No production topology helpers or transport are used by the oracle.
template <typename T, typename Fn>
T Tree(std::vector<T> const& inputs, std::size_t begin, std::size_t count, Fn merge) {
  if (count == 1) return inputs[begin];
  std::size_t split = 1;
  while (split * 2 < count) split *= 2;
  return merge(Tree(inputs, begin, split, merge),
               Tree(inputs, begin + split, count - split, merge));
}
template <typename T, typename Fn>
T Ordered(std::vector<T> const& inputs, Fn merge) {
  if (!federated) return Tree(inputs, 0, inputs.size(), merge);
  // CUDA's federated fallback folds on rank zero in ascending peer order.
  T result = inputs.front();
  for (std::size_t peer = 1; peer < inputs.size(); ++peer) result = merge(result, inputs[peer]);
  return result;
}
common::Span<std::int8_t> Bytes(metal::VectorBuffer const& v) {
  return {static_cast<std::int8_t*>(v.Data()), v.SizeBytes()};
}
void Variable(Context const& ctx, int rank, int world) {
  std::size_t order_witnesses = 0;
  coll::metal_impl::AllreduceVScratch scratch;
  scratch.Reserve(65536);
  auto data = metal::MakeVectorBuffer();
  for (int mode = 0; mode < 3; ++mode)
    for (unsigned layout = 0; layout < 5; ++layout)
      for (std::size_t n : {0, 1, 257, 4099}) {
        current = "variable/" + std::to_string(mode) + "/" + std::to_string(layout) + "/" +
                  std::to_string(n);
        std::vector<std::vector<std::uint32_t>> inputs(world);
        for (int r = 0; r < world; ++r) {
          auto size = layout == 0 || (layout == 1 && r == 0) || (layout == 2 && r != 0) ? 0 : n + r;
          inputs[r].resize(size);
          for (std::size_t i = 0; i < size; ++i) inputs[r][i] = (r + 1) * 10000 + i;
        }
        data->Resize(inputs[rank].size() * 4);
        data->Import(inputs[rank].data(), 0, data->SizeBytes());
        auto merge = [&](auto const& a, auto const& b) {
          std::size_t left = mode == 1 ? a.size() / 2 : a.size(),
                      right = mode == 1 ? b.size() / 2 : b.size();
          std::vector<std::uint32_t> out;
          if (mode == 2 || (a.empty() && b.empty())) return out;
          out.insert(out.end(), a.begin(), a.begin() + left);
          out.push_back(0xa0000000u | (std::uint32_t(a.size()) << 12) | std::uint32_t(b.size()));
          out.insert(out.end(), b.begin(), b.begin() + right);
          return out;
        };
        auto expected = Ordered(inputs, merge);
        if (federated && expected != Tree(inputs, 0, world, merge)) ++order_witnesses;
        auto before = metal::GetCollectiveStatistics();
        auto state = metal::GetStatistics();
        coll::SafeColl(coll::metal_impl::AllreduceV(
            &ctx, &data, &scratch,
            [&](metal::VectorBuffer const& a, metal::VectorBuffer const& b,
                metal::VectorBuffer* out) {
              auto left = a.SizeBytes() / 4, right = b.SizeBytes() / 4;
              if (mode == 2 || (!left && !right)) {
                out->Resize(0);
                return;
              }
              auto n_left = mode == 1 ? left / 2 : left, n_right = mode == 1 ? right / 2 : right;
              out->Resize((n_left + n_right + 1) * 4);
              metal::CollectiveCopy(Bytes(a).first(n_left * 4), Bytes(*out).first(n_left * 4));
              std::uint32_t marker =
                  0xa0000000u | (std::uint32_t(left) << 12) | std::uint32_t(right);
              out->Fill({reinterpret_cast<std::byte*>(&marker), 4}, n_left * 4, 4);
              metal::CollectiveCopy(Bytes(b).first(n_right * 4),
                                    Bytes(*out).subspan((n_left + 1) * 4));
            }));
        auto after = metal::GetCollectiveStatistics();
        Check(after.staging_bytes == before.staging_bytes, "Variable payload staged");
        Check(metal::GetStatistics().vector_readback_bytes == state.vector_readback_bytes,
              "Variable payload host mirror");
        Check(data->SizeBytes() == expected.size() * 4, "Variable result size");
        std::vector<std::uint32_t> actual(expected.size());
        data->Readback(actual.data(), data->SizeBytes());
        Check(actual == expected, "Variable callback order/result differs");
        fields += actual.size();
        ++cases;
      }
  if (federated && world > 3) Check(order_witnesses > 0, "Missing fold/tree order witnesses");
}
struct Input {
  std::vector<float> weights, dense;
  std::vector<bst_idx_t> ptr{0};
  std::vector<xgboost::Entry> entries;
  std::vector<ref::Raw> raw;
  std::size_t rows;
};
Input MakeInput(int rank, unsigned mode, unsigned layout, std::size_t columns, std::size_t rows) {
  Input out;
  out.rows = layout == 1 || (layout == 2 && rank == 0) || (layout == 3 && rank != 0)
                 ? 0
                 : rows + rank * 13;
  bool weighted = mode >= 2;
  if (weighted) out.weights.resize(out.rows);
  out.dense.assign(out.rows * columns, NAN);
  for (std::size_t row = 0; row < out.rows; ++row) {
    float w = mode == 4 ? 0.f : float(1 + (row * 7 + rank) % 15) / 8;
    if (mode == 6) w = std::ldexp(w, -140);
    if (mode == 7) w = std::ldexp(w, 110);
    if (weighted) out.weights[row] = w;
    for (std::size_t f = 0; f < columns; ++f) {
      if (layout == 4 || f == 3 || row % 19 == 0 || (row + f) % 7 == 0) continue;
      bool cat = mode % 2 && f == 1;
      float value = cat ? float((row * 11 + rank * 3) % 109)
                        : float(int((row * 353 + f * 107 + rank * 233) % 4001) - 2000) / 8;
      if (mode == 5 && f == 0) value = row % 2 ? -INFINITY : INFINITY;
      out.entries.emplace_back(f, value);
      out.raw.push_back({std::uint32_t(f), value, w});
      out.dense[row * columns + f] = value;
    }
    out.ptr.push_back(out.entries.size());
  }
  return out;
}
void Push(Input const& input, std::size_t window, bool weighted, metal::CudaSketch* gpu,
          ref::Sketch* oracle) {
  for (std::size_t begin = 0; begin < input.entries.size(); begin += window) {
    auto end = std::min(input.entries.size(), begin + window);
    auto first =
        std::upper_bound(input.ptr.begin(), input.ptr.end(), begin) - input.ptr.begin() - 1;
    auto last = std::lower_bound(input.ptr.begin(), input.ptr.end(), end) - input.ptr.begin();
    auto rows = std::max<std::size_t>(1, last - first);
    if (gpu) gpu->Push({input.entries.data() + begin, end - begin}, input.ptr, begin, 0, rows);
    oracle->Push({input.raw.begin() + begin, input.raw.begin() + end}, rows, weighted);
  }
}
ref::Sketch Distributed(std::vector<ref::Sketch> inputs, int world) {
  if (world == 1) return inputs.front();
  std::size_t rows = 0;
  for (auto const& input : inputs) rows += input.seen;
  auto budget = ref::Budget(rows, inputs.front().bins);
  auto prune = [&](ref::Sketch* sketch) {
    for (std::size_t f = 0; f < sketch->columns.size(); ++f)
      sketch->columns[f] =
          ref::Unique(ref::Prune(sketch->columns[f], budget, sketch->categorical[f], true));
  };
  for (auto& input : inputs) prune(&input);
  return Ordered(inputs, [&](auto a, auto const& b) {
    for (std::size_t f = 0; f < a.columns.size(); ++f)
      a.columns[f] = ref::Merge(a.columns[f], b.columns[f]);
    prune(&a);
    return a;
  });
}
void Sketches(Context const& ctx, int rank, int world, unsigned mode, unsigned layout,
              std::size_t columns, int bins, std::size_t window, std::size_t rows) {
  current = "sketch/" + std::to_string(mode) + "/" + std::to_string(layout) + "/" +
            std::to_string(columns) + "/" + std::to_string(bins) + "/" + std::to_string(window);
  std::vector<FeatureType> types(columns, FeatureType::kNumerical);
  if (mode % 2 && columns > 1) types[1] = FeatureType::kCategorical;
  std::vector<bool> cats;
  for (auto type : types) cats.push_back(type == FeatureType::kCategorical);
  std::vector<ref::Sketch> inputs;
  auto own = MakeInput(rank, mode, layout, columns, rows);
  auto gpu = metal::MakeCudaSketch(columns, bins, types, own.weights);
  for (int r = 0; r < world; ++r) {
    auto input = MakeInput(r, mode, layout, columns, rows);
    inputs.emplace_back(columns, bins, cats);
    Push(input, window, mode >= 2, r == rank ? gpu.get() : nullptr, &inputs.back());
  }
  Verify(gpu->Read(), inputs[rank]);
  auto expected = Distributed(inputs, world);
  auto before = metal::GetCollectiveStatistics();
  auto state = metal::GetStatistics();
  gpu->AllReduce(&ctx);
  auto after = metal::GetCollectiveStatistics();
  Check(after.staging_bytes == before.staging_bytes, "Sketch payload staged");
  Check((after.copy_dispatches > before.copy_dispatches) == (world > 1),
        "Missing sketch GPU copies");
  Check(metal::GetStatistics().vector_readback_bytes == state.vector_readback_bytes,
        "Sketch payload mirrored");
  Verify(gpu->Read(), expected);
  ++cases;
  if (columns == 5 && mode < 5 && layout < 5 && window == (1u << 26)) {
    // Production ordinary-matrix caller must perform the same distributed step.
    DMatrixHandle handle = nullptr;
    Api(XGDMatrixCreateFromMat(own.dense.data(), own.rows, columns, NAN, &handle));
    std::vector<char const*> names;
    for (auto cat : cats) names.push_back(cat ? "c" : "q");
    Api(XGDMatrixSetStrFeatureInfo(handle, "feature_type", names.data(), names.size()));
    if (mode >= 2)
      Api(XGDMatrixSetFloatInfo(handle, "weight", own.weights.data(), own.weights.size()));
    auto matrix = *static_cast<std::shared_ptr<DMatrix>*>(handle);
    auto cuts = metal::CudaSketchOnDMatrix(&ctx, matrix.get(), bins, {});
    auto final = gpu->Read();
    auto oracle_cuts = metal::MakeCudaCuts(final, columns, bins, types);
    Check(cuts.Ptrs() == oracle_cuts.Ptrs(), "Ordinary caller cut pointers");
    Check(cuts.Values().size() == oracle_cuts.Values().size(), "Ordinary caller cut count");
    for (std::size_t i = 0; i < cuts.Values().size(); ++i)
      Same(cuts.Values()[i], oracle_cuts.Values()[i]);
    Api(XGDMatrixFree(handle));
    ++cases;
  }
}

std::unique_ptr<metal::VectorBuffer> PrepareDisconnect() {
  auto data = metal::MakeVectorBuffer();
  data->Resize(4097);
  std::byte marker{0x37};
  data->Fill({&marker, 1}, 0, data->SizeBytes());
  return data;
}
void Disconnect(Context const& ctx, int rank, bool root,
                std::unique_ptr<metal::VectorBuffer> prepared) {
  current = root ? "AllreduceV broadcast disconnect" : "AllreduceV receive disconnect";
  {
    auto data = prepared ? std::move(prepared) : PrepareDisconnect();
    coll::metal_impl::AllreduceVScratch scratch;
    int ready = 1;
    auto cpu = ctx.MakeCPU();
    coll::SafeColl(coll::Allreduce(&cpu, &ready, coll::Op::kSum));
    if (!root && rank == 1) {
      std::puts("Deliberately disconnecting child before variable payload receive");
      std::fflush(stdout);
      std::_Exit(0);
    }
    auto result = coll::metal_impl::AllreduceV(
        &ctx, &data, &scratch,
        [&](metal::VectorBuffer const&, metal::VectorBuffer const&, metal::VectorBuffer*) {
          Check(root && rank == 0, "Unexpected failure-test callback");
          std::puts("Deliberately disconnecting root before variable result broadcast");
          std::fflush(stdout);
          std::_Exit(0);
        });
    Check(!result.OK(), "AllreduceV disconnected peer returned success");
  }
  Check(metal::GetStatistics().vector_live_buffers == 0, "Failure leaked vector buffers");
  Api(XGCommunicatorFinalize());
  std::printf("PASS %s: failure propagated and buffers released\n", current.c_str());
}
int main(int argc, char** argv) {
  try {
    Check(argc == 2 || argc == 3, "Communicator JSON required");
    std::string config = argv[1];
    std::unique_ptr<metal::VectorBuffer> prepared;
    if (config == "-") {
      Check(argc == 3, "Deferred configuration is only for disconnect checks");
      // GPU initialization can exceed the short fault-injection deadline. The
      // launcher starts the tracker only after every worker owns its payload.
      prepared = PrepareDisconnect();
      std::puts("READY sketch disconnect");
      std::fflush(stdout);
      Check(bool(std::getline(std::cin, config)), "Missing communicator configuration");
    }
    Api(XGCommunicatorInit(config.c_str()));
    int rank = XGCommunicatorGetRank(), world = XGCommunicatorGetWorldSize();
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    federated = coll::GlobalCommGroup()->Ctx(&ctx, DeviceOrd::MPS()).IsFederated();
    if (argc == 3) {
      Check(world == 2, "Disconnect test needs 2 workers");
      Disconnect(ctx, rank, std::string(argv[2]) == "--disconnect-root", std::move(prepared));
      return 0;
    }
    Variable(ctx, rank, world);
    for (unsigned mode = 0; mode < 8; ++mode) {
      for (unsigned layout = 0; layout < 5; ++layout)
        for (std::size_t columns : {1, 2, 5})
          Sketches(ctx, rank, world, mode, layout, columns, 7, 1u << 26, 513);
      Sketches(ctx, rank, world, mode, 0, 5, 2, 97, 257);
      std::printf("PASS world%d rank%d mode%d: %zu cases / %zu fields\n", world, rank, mode, cases,
                  fields);
      std::fflush(stdout);
    }
    Check(metal::GetStatistics().vector_live_buffers == 0, "Leaked vector buffers");
    Api(XGCommunicatorFinalize());
    std::printf("PASS world%d rank%d: %zu sketch collective cases / %zu fields\n", world, rank,
                cases, fields);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), e.what());
    return 1;
  }
}
