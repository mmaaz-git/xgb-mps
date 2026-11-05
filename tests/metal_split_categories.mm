// Category assembly: independent upstream LBitField32 oracle and production API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "../xgboost/src/common/categorical.h"
#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_model.h"
#include "../xgboost/src/common/metal_split.h"
#include "../xgboost/src/data/gradient_index.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t direct_nodes = 0, words_checked = 0, api_calls = 0;
std::size_t weight_fields = 0, weight_storage_cases = 0;
std::size_t category_storage_cases = 0;
void Check(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
template <typename F>
void Reject(F fn) {
  bool caught = false;
  try {
    fn();
  } catch (std::exception const&) {
    caught = true;
  }
  Check(caught, "Invalid category API request accepted");
}
void Direct(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
            bool multi, unsigned targets, unsigned count, bool maximum, unsigned bad) {
  std::vector<metal::QuantizedSplitTask> scalar_tasks;
  std::vector<metal::QuantizedMultiSplitTask> vector_tasks;
  std::vector<metal::QuantizedSplitResult> scalar_results;
  std::vector<metal::QuantizedMultiSplitResult> vector_results;
  std::vector<metal::SplitCategoryTask> requests;
  std::vector<unsigned> order;
  std::vector<float> cuts;
  unsigned largest = maximum ? 16777215u : count * 97 + 128;
  unsigned words = common::CatBitField::ComputeStorageSize(largest + 1);
  std::vector<unsigned> expected(8 * words, 0);
  std::mt19937 rng{10291};
  for (unsigned k = 0; k < 8; ++k) {
    unsigned begin = order.size(), cut_begin = cuts.size(), kind = k == 0 ? 0 : k < 4 ? 1 : 2;
    order.resize(begin + count * targets + 7, UINT32_MAX);
    std::iota(order.begin() + begin, order.begin() + begin + count, 0u);
    std::shuffle(order.begin() + begin, order.begin() + begin + count, rng);
    for (unsigned j = 0; j < count; ++j)
      cuts.push_back(maximum && j == count - 1 ? float(largest) : float(j * 97 + j % 3));
    if (count > 6) {
      cuts[cut_begin] = -0.f;
      cuts[cut_begin + 1] = 31.75f;
      cuts[cut_begin + 2] = 32.f;
      cuts[cut_begin + 3] = 63.875f;
      cuts[cut_begin + 4] = 64.f;
      cuts[cut_begin + 5] = std::numeric_limits<float>::denorm_min();
    }
    cuts.resize(cuts.size() + 5, NAN);  // Unselected padding must not be read.
    unsigned missing = k % 2, partition = k == 4 ? 1 : k == 5 ? count : count / 2;
    int bin = kind == 1 ? (k == 2 ? 0 : int(count - 1)) : int(partition - missing);
    if (k == 1 || (kind == 2 && (!partition || bin >= int(count)))) bin = -1;
    scalar_tasks.push_back({begin, begin + count, kind, count, {}, -INFINITY, INFINITY, 0});
    vector_tasks.push_back({begin, begin + count * targets, kind, count, 0, targets, 0, 0});
    metal::QuantizedSplitResult sr{};
    sr.bin = bin;
    sr.default_left = missing;
    sr.partition = partition;
    scalar_results.push_back(sr);
    vector_results.push_back({0.f, bin, missing, partition});
    requests.push_back({k, cut_begin});
    common::CatBitField bits{{expected.data() + k * words, words}};
    if (kind && bin >= 0) {
      if (kind == 1)
        bits.Set(common::AsCat(cuts[cut_begin + bin]));
      else
        for (unsigned j = 0; j < partition; ++j)
          bits.Set(common::AsCat(cuts[cut_begin + order[begin + j]]));
    }
  }
  // Reverse requests so the kernel cannot assume candidate index == output slot.
  std::reverse(requests.begin(), requests.end());
  for (unsigned a = 0; a < 4; ++a)
    std::swap_ranges(expected.begin() + a * words, expected.begin() + (a + 1) * words,
                     expected.begin() + (7 - a) * words);
  if (bad) {
    auto& r = scalar_results[2];
    auto cut = requests[5].cut_begin;  // Candidate 2, one-hot at bin zero.
    if (bad <= 5) {
      std::uint32_t values[]{0xbf800000, 0x80000001, 0x7f800000, 0x7fc12345, 0x4b800000};
      std::memcpy(&cuts[cut], &values[bad - 1], 4);
    } else if (bad == 6) {
      cuts[cut] = float(words * 32);  // Legal category, insufficient bitset extent.
    } else if (bad == 7) {
      r.bin = vector_results[2].bin = count;
    } else if (bad == 8) {
      scalar_results[4].partition = vector_results[4].partition = count + 1;
    } else {
      order[scalar_tasks[4].begin] = count;
    }
  }
  auto upload = [&](void const* data, std::size_t bytes) {
    auto buffer = [device newBufferWithLength:std::max<std::size_t>(32, bytes)
                                      options:MTLResourceStorageModeShared];
    Check(buffer != nil, "Buffer allocation failed");
    std::memset(buffer.contents, 0xa5, buffer.length);
    if (bytes && data) std::memcpy(buffer.contents, data, bytes);
    return buffer;
  };
  auto input = multi ? upload(vector_tasks.data(), vector_tasks.size() * sizeof(vector_tasks[0]))
                     : upload(scalar_tasks.data(), scalar_tasks.size() * sizeof(scalar_tasks[0]));
  auto result =
      multi ? upload(vector_results.data(), vector_results.size() * sizeof(vector_results[0]))
            : upload(scalar_results.data(), scalar_results.size() * sizeof(scalar_results[0]));
  auto request = upload(requests.data(), requests.size() * sizeof(requests[0]));
  auto sorted = upload(order.data(), order.size() * 4);
  auto values = upload(cuts.data(), cuts.size() * 4);
  auto output = upload(nullptr, expected.size() * 4 + 32);
  unsigned zero = 0;
  auto error = upload(&zero, 4);
  struct {
    unsigned words, multi;
  } params{words, multi};
  for (unsigned repeat = 0; repeat < (bad ? 1u : 2u); ++repeat) {
    if (repeat) {
      // Reuse dirty output without clearing it on the host. Invalid candidates
      // must clear every word, including long suffixes and numeric slots.
      if (multi) {
        auto r = static_cast<metal::QuantizedMultiSplitResult*>(result.contents);
        for (unsigned i = 0; i < 8; ++i) r[i].bin = -1;
      } else {
        auto r = static_cast<metal::QuantizedSplitResult*>(result.contents);
        for (unsigned i = 0; i < 8; ++i) r[i].bin = -1;
      }
      std::fill(expected.begin(), expected.end(), 0);
    }
    auto command = [queue commandBuffer];
    auto encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:request offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:result offset:0 atIndex:2];
    [encoder setBuffer:sorted offset:0 atIndex:3];
    [encoder setBuffer:values offset:0 atIndex:4];
    [encoder setBuffer:output offset:16 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder setBuffer:error offset:0 atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    Check(command.status == MTLCommandBufferStatusCompleted, "Category shader failed");
    Check((*static_cast<unsigned*>(error.contents) != 0) == (bad != 0), "Category error flag");
    auto actual = static_cast<unsigned char*>(output.contents);
    for (unsigned i = 0; i < 16; ++i)
      Check(actual[i] == 0xa5 && actual[output.length - 16 + i] == 0xa5,
            "Bitset guard overwritten");
    if (!bad) {
      Check(!std::memcmp(actual + 16, expected.data(), expected.size() * 4),
            "Category bits differ");
      words_checked += expected.size();
    }
    Check(!std::memcmp(values.contents, cuts.data(), cuts.size() * 4), "Cuts changed");
    Check(!std::memcmp(sorted.contents, order.data(), order.size() * 4), "Sorted indices changed");
    direct_nodes += 8;
  }
}

void WeightStorage(unsigned targets, metal::SplitWorkspace* workspace,
                   std::vector<GradientPairPrecise> const& sums,
                   std::vector<GradientPairPrecise> const& children) {
  std::vector<metal::SplitFinalizeNode> nodes;
  std::vector<metal::SplitFinalizeFeature> features;
  std::vector<GradientPairPrecise> parents;
  for (unsigned k = 0; k < 3; ++k) {
    nodes.push_back({k, k + 1, k * targets, targets});
    features.push_back({k, 0, 2 * k * targets});
    parents.insert(parents.end(), sums.begin(), sums.end());
  }
  std::vector<metal::SplitBounds> bounds(parents.size(), {-INFINITY, INFINITY});
  std::vector<metal::SplitNodeResult> result;
  auto storage = metal::MakeSplitWeightStorage(targets);
  auto before = metal::GetStatistics();
  metal::FinalizeSplitNodes(nodes, features, parents, bounds, {1, 0, 0, 0}, true, &result,
                            nullptr, workspace);
  std::vector<metal::SplitWeightCopy> copies{{2, 5}, {0, 0}, {1, 17}};
  storage->Store(*workspace, copies);
  for (unsigned k = 0; k < 3; ++k)
    Check(result[k].winner == k, "Weight-storage fixture did not produce a split");
  // Independent CPU weight expression for this unbounded, lambda=1 fixture.
  auto weight = [](GradientPairPrecise p) { return float(-p.GetGrad() / (p.GetHess() + 1.)); };
  std::vector<float> expected;
  for (unsigned k : {0u, 2u}) {
    for (auto p : sums) expected.push_back(weight(p));
    for (unsigned t = 0; t < 2 * targets; ++t)
      expected.push_back(weight(children[2 * k * targets + t]));
  }
  // Reuse the workspace, overwrite one node, and grow past 256 node IDs. The
  // previously queued nodes must survive with their original finalized weights.
  metal::SplitFinalizeNode root{0, 0, 0, targets};
  std::vector<GradientPairPrecise> root_sum(targets, {12., 3.});
  std::vector<metal::SplitBounds> root_bounds(targets, {-INFINITY, INFINITY});
  metal::FinalizeSplitNodes({&root, 1}, {}, root_sum, root_bounds, {1, 0, 0, 0}, true,
                            &result, nullptr, workspace);
  copies = {{0, 257}, {0, 17}};
  storage->Store(*workspace, copies);
  for (unsigned k = 0; k < 2; ++k) {
    expected.insert(expected.end(), targets, -3.f);
    expected.insert(expected.end(), 2 * targets, 0.f);
  }
  auto invalid_before = metal::GetStatistics();
  Reject([&] { storage->Base(1); });
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitWeightCopy>{{1, 0}}); });
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitWeightCopy>{{0, 0}, {0, 0}}); });
  Reject([&] { metal::MakeSplitWeightStorage(targets + 1)->Store(*workspace, copies); });
  Reject([&] { metal::MakeSplitWeightStorage(0); });
  storage->Store(*workspace, {});
  metal::FinalizeSplitNodes({}, {}, {}, {}, {1, 0, 0, 0}, true, &result, nullptr, workspace);
  Reject([&] { storage->Store(*workspace, copies); });
  Check(metal::GetStatistics().split_weight_store_dispatches ==
            invalid_before.split_weight_store_dispatches, "Invalid storage request dispatched");
  // Consume every view on the GPU before the first CPU payload comparison.
  HostDeviceVector<float> consumer(expected.size(), 0.f, DeviceOrd::MPS());
  std::vector<metal::ModelCopy> gather;
  for (unsigned node : {0u, 5u, 17u, 257u})
    for (auto view : {storage->Base(node), storage->Left(node), storage->Right(node)})
      gather.push_back({view.data(), gather.size() * targets, targets});
  metal::CopyModelBuffer(consumer.DeviceSpan().data(), consumer.Size(), gather,
                         metal::ModelBufferKind::kWeights);
  metal::ScaleModelWeights(consumer.DeviceSpan(), .25f);
  auto after = metal::GetStatistics();
  Check(after.split_weight_store_dispatches - before.split_weight_store_dispatches == 2,
        "Weight store dispatch count");
  Check(after.split_weight_readback_bytes == before.split_weight_readback_bytes &&
            after.model_host_copy_bytes == before.model_host_copy_bytes &&
            after.vector_upload_bytes == before.vector_upload_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes,
        "Weight pipeline copied resident payload through CPU memory");
  Check(consumer.DeviceCanWrite() && !consumer.HostCanRead(), "Weight consumer is not resident");
  for (auto& value : expected) value *= .25f;
  auto const& actual = consumer.ConstHostVector();
  Check(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) == 0,
        "Queued/overwritten/grown GPU weights differ from CPU expression");
  weight_fields += expected.size();
  ++weight_storage_cases;
}

void CategoryStorage(std::vector<float> const& values,
                      std::vector<metal::SplitCategoryTask> const& requests, unsigned words,
                      std::vector<unsigned> const& expected, metal::SplitWorkspace* workspace) {
  auto storage = metal::MakeSplitCategoryStorage(words);
  auto before = metal::GetStatistics();
  metal::BuildSplitCategories(values, requests, words, nullptr, workspace);
  storage->Store(*workspace, std::vector<metal::SplitCategoryCopy>{{0, 5}, {1, 0}, {2, 17}});
  // Reuse the scratch payload, overwrite numeric node17 with one-hot words,
  // and grow past node256. Node0/5 must retain their earlier selected bitsets.
  std::vector<metal::SplitCategoryTask> next{requests[1], requests[2]};
  metal::BuildSplitCategories(values, next, words, nullptr, workspace);
  storage->Store(*workspace, std::vector<metal::SplitCategoryCopy>{{0, 17}, {1, 257}});
  Reject([&] { storage->Get(1); });
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitCategoryCopy>{{2, 1}}); });
  Reject([&] {
    storage->Store(*workspace, std::vector<metal::SplitCategoryCopy>{{0, 1}, {1, 1}});
  });
  Reject([&] { metal::MakeSplitCategoryStorage(0); });
  Reject([&] {
    metal::MakeSplitCategoryStorage(words + 1)->Store(
        *workspace, std::vector<metal::SplitCategoryCopy>{{0, 0}});
  });
  storage->Store(*workspace, {});
  metal::BuildSplitCategories({}, {}, 0, nullptr, workspace);
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitCategoryCopy>{{0, 0}}); });
  HostDeviceVector<unsigned> output(4 * words, 0u, DeviceOrd::MPS());
  std::vector<metal::ModelCopy> copies;
  for (unsigned node : {0u, 5u, 17u, 257u})
    copies.push_back({storage->Get(node).data(), copies.size() * words, words});
  metal::CopyModelBuffer(output.DeviceSpan().data(), output.Size(), copies,
                         metal::ModelBufferKind::kCategories);
  auto after = metal::GetStatistics();
  Check(after.split_category_store_dispatches - before.split_category_store_dispatches == 2,
        "Selected category stores/invalid request accounting");
  Check(after.split_category_readback_bytes == before.split_category_readback_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes &&
            after.vector_upload_bytes == before.vector_upload_bytes &&
            after.model_host_copy_bytes == before.model_host_copy_bytes,
        "Selected categories passed through host payload storage");
  Check(output.DeviceCanWrite() && !output.HostCanRead(), "Category consumer lost GPU authority");
  std::vector<unsigned> reference;
  for (unsigned source : {1u, 0u, 1u, 2u})
    reference.insert(reference.end(), expected.begin() + source * words,
                      expected.begin() + (source + 1) * words);
  Check(output.ConstHostVector() == reference, "Queued/overwritten/grown categories differ");
  ++category_storage_cases;
}

void Production(bool multi, unsigned targets) {
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "1"}});
  unsigned n = 8;
  SparsePage page;
  page.offset.HostVector().push_back(0);
  common::HistogramCuts cuts{1};
  cuts.cut_ptrs_.HostVector() = {0, n};
  for (unsigned i = 0; i < n; ++i) {
    page.data.HostVector().emplace_back(0, float(i));
    page.offset.HostVector().push_back(i + 1);
    cuts.cut_values_.HostVector().push_back(float(i + 1));
  }
  GHistIndexMatrix index{&ctx, page, {}, std::move(cuts), 16, true, .2};
  std::vector<GradientPair> raw;
  for (unsigned i = 0; i < n; ++i)
    for (unsigned t = 0; t < targets; ++t) raw.emplace_back(float(int(i) - 4) * (t + 1), 1.f);
  auto state = metal::MakeQuantizedGradients(linalg::MakeTensorView(&ctx, raw, n, targets));
  std::vector<GradientPairPrecise> sums(targets);
  std::vector<GradientPairInt64> integers(targets);
  state->Sum(linalg::MakeVec(DeviceOrd::CPU(), common::Span{sums}), integers);
  common::RowSetCollection rows;
  rows.Data()->resize(n);
  std::iota(rows.Data()->begin(), rows.Data()->end(), 0);
  rows.Init();
  std::vector<std::unique_ptr<metal::HistogramBuffer>> caches;
  std::vector<metal::HistogramInput> inputs;
  for (unsigned t = 0; t < targets; ++t) {
    auto hist = metal::MakeHistogram();
    hist->BindQuantized(state, t);
    auto cache = metal::MakeHistogramBuffer();
    cache->Resize(n);
    std::vector<GradientPair> column(n);
    for (unsigned i = 0; i < n; ++i) column[i] = raw[i * targets + t];
    hist->Build(0, index, rows, {0}, column, {cache->Data()}, cache.get());
    inputs.push_back({cache.get(), {}});
    for (unsigned k = 0; k < 3; ++k)
      inputs.back().segments.push_back({0, k * n * targets + t, n, targets, 0});
    caches.push_back(std::move(cache));
  }
  std::vector<metal::QuantizedSplitTask> scalar;
  std::vector<metal::QuantizedMultiSplitTask> vector;
  std::vector<GradientPairInt64> parents;
  std::vector<metal::SplitBounds> bounds;
  std::vector<float> values{NAN, NAN};
  for (unsigned k = 0; k < 3; ++k) {
    scalar.push_back({k * n, (k + 1) * n, k, n, integers[0], -INFINITY, INFINITY, 0});
    vector.push_back(
        {k * n * targets, (k + 1) * n * targets, k, n, unsigned(parents.size()), targets, 0, 0});
    parents.insert(parents.end(), integers.begin(), integers.end());
    bounds.insert(bounds.end(), targets, {-INFINITY, INFINITY});
    for (unsigned i = 0; i < n; ++i) values.push_back(float(i * 31 + k));
  }
  auto ws = metal::MakeSplitWorkspace();
  std::vector<metal::QuantizedSplitResult> sr;
  std::vector<metal::QuantizedMultiSplitResult> vr;
  std::vector<GradientPairInt64> exact;
  std::vector<GradientPairPrecise> child;
  std::vector<unsigned> order;
  auto evaluate = [&](std::vector<unsigned>* sorted) {
    if (multi)
      metal::EvaluateQuantizedMultiSplits(inputs, vector, parents, bounds, {1, 0, 0, 0}, &vr,
                                          &exact, &child, sorted, ws.get());
    else
      metal::EvaluateQuantizedSplits(inputs, scalar, {1, 0, 0, 0}, &sr, sorted, ws.get());
  };
  evaluate(&order);
  if (multi) WeightStorage(targets, ws.get(), sums, child);
  std::vector<metal::SplitCategoryTask> requests{
      {2, 2 + 2 * n}, {1, 2 + n}, {0, UINT32_MAX}, {2, 2 + 2 * n}};
  unsigned words = 16;
  std::vector<unsigned> expected(requests.size() * words, 0), output{123};
  for (unsigned i = 0; i < requests.size(); ++i) {
    auto k = requests[i].candidate;
    if (!k) continue;
    int bin = multi ? vr[k].bin : sr[k].bin;
    unsigned partition = multi ? vr[k].partition : sr[k].partition;
    Check(bin >= 0, "Production category candidate invalid");
    common::CatBitField bits{{expected.data() + i * words, words}};
    if (k == 1)
      bits.Set(common::AsCat(values[requests[i].cut_begin + bin]));
    else
      for (unsigned j = 0; j < partition; ++j)
        bits.Set(common::AsCat(values[requests[i].cut_begin + order[k * n * targets + j]]));
  }
  if (multi) CategoryStorage(values, requests, words, expected, ws.get());
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    if (repeat) evaluate(nullptr);  // Production path skips sorted-index readback.
    std::fill(order.begin(), order.end(), UINT32_MAX);
    for (auto& r : sr) r.bin = -1;
    for (auto& r : vr) r.bin = -1;
    auto before = metal::GetStatistics();
    metal::BuildSplitCategories(values, requests, words, &output, ws.get());
    Check(output == expected, "Production bitsets differ or used modified host readbacks");
    auto after = metal::GetStatistics();
    Check(after.split_category_dispatches - before.split_category_dispatches == 1 &&
              after.split_category_nodes - before.split_category_nodes == requests.size(),
          "Category dispatch accounting");
    ++api_calls;
  }
  Reject([&] { metal::BuildSplitCategories(values, requests, 0, &output, ws.get()); });
  Reject([&] { metal::BuildSplitCategories(values, requests, words, &output, nullptr); });
  Reject([&] { metal::BuildSplitCategories({}, requests, words, &output, ws.get()); });
  auto bad = requests;
  bad[0].candidate = 3;
  Reject([&] { metal::BuildSplitCategories(values, bad, words, &output, ws.get()); });
  auto before_output = output;
  Reject([&] { metal::BuildSplitCategories(values, requests, 1, &output, ws.get()); });
  Check(output == before_output, "GPU error changed caller output");
  metal::BuildSplitCategories(values, requests, words, &output, ws.get());
  Check(output == expected, "Failed category call poisoned workspace");
  ++api_calls;
  // Empty quantized evaluation invalidates old resident candidates, even though
  // the scratch allocation still has capacity for them.
  if (multi)
    metal::EvaluateQuantizedMultiSplits({}, {}, {}, {}, {1, 0, 0, 0}, &vr, &exact, &child, nullptr,
                                        ws.get());
  else
    metal::EvaluateQuantizedSplits({}, {}, {1, 0, 0, 0}, &sr, nullptr, ws.get());
  Reject([&] { metal::BuildSplitCategories(values, requests, words, &output, ws.get()); });
  auto before = metal::GetStatistics();
  metal::BuildSplitCategories({}, {}, 0, &output, nullptr);
  Check(output.empty() &&
            metal::GetStatistics().split_category_dispatches == before.split_category_dispatches,
        "Empty category request dispatched");
}

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production shader source");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "Metal unavailable");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [device
          newComputePipelineStateWithFunction:[library
                                                  newFunctionWithName:@"xgb_build_split_categories"]
                                        error:&error];
      Check(pipeline != nil, "Missing category pipeline");
      auto queue = [device newCommandQueue];
      for (bool multi : {false, true}) {
        for (unsigned targets : {1u, 3u, 33u}) {
          if (!multi && targets != 1) continue;
          for (unsigned count : {1u, 2u, 31u, 32u, 33u, 65u, 257u, 513u})
            Direct(device, queue, pipeline, multi, targets, count, false, 0);
          Direct(device, queue, pipeline, multi, targets, 16, true, 0);
          for (unsigned bad = 1; bad <= 9; ++bad)
            Direct(device, queue, pipeline, multi, targets, 33, false, bad);
          Production(multi, targets);
          std::printf("Checked %zu category slots / %zu words / %zu API calls\n", direct_nodes,
                      words_checked, api_calls);
        }
      }
      Production(true, 129);
      std::printf("PASS %zu category slots, %zu exact words, %zu successful production API calls; "
                  "%zu weight-storage chains / %zu exact weight fields; %zu category chains\n",
                  direct_nodes, words_checked, api_calls, weight_storage_cases, weight_fields,
                  category_storage_cases);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
