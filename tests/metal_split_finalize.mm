// Independent CUDA-source finalization checks; no NVIDIA execution implied.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_split.h"
#include "cuda_multi_split_reference.h"
namespace metal = xgboost::common::metal;
namespace ref = cuda_multi_split_reference;
using xgboost::GradientPairPrecise;
void Check(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
bool Bits(void const* a, void const* b, std::size_t n) { return std::memcmp(a, b, n) == 0; }
ref::Floating Pair(GradientPairPrecise v) { return {v.GetGrad(), v.GetHess()}; }
std::size_t checked = 0, fields = 0;
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production shader source");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No Metal GPU");
      std::ifstream f(argv[1]);
      std::string source{std::istreambuf_iterator<char>(f), {}};
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [device
          newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xgb_finalize_split_nodes"]
                                        error:&error];
      Check(pipeline != nil, "Missing finalization kernel");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* p, std::size_t n) {
        auto b = [device newBufferWithLength:std::max<std::size_t>(n, 32)
                                     options:MTLResourceStorageModeShared];
        std::memset(b.contents, 0xa5, b.length);
        if (p && n) std::memcpy(b.contents, p, n);
        return b;
      };
      std::mt19937_64 rng(7822);
      for (bool multi : {false, true})
        for (std::uint32_t nt : {1u, 2u, 3u, 17u, 33u, 129u}) {
          if (!multi && nt != 1) continue;
          for (unsigned variant = 0; variant < 8; ++variant) {
            std::vector<metal::SplitFinalizeNode> nodes;
            std::vector<metal::SplitFinalizeFeature> features;
            std::vector<GradientPairPrecise> parents, children;
            std::vector<metal::SplitBounds> bounds;
            std::vector<metal::QuantizedSplitResult> scalar;
            std::vector<metal::QuantizedMultiSplitResult> vector;
            for (std::uint32_t n : {0u, 1u, 2u, 31u, 32u, 33u, 65u, 129u}) {
              std::uint32_t pb = parents.size(), begin = features.size();
              for (std::uint32_t t = 0; t < nt; ++t) {
                double scale = variant == 6   ? std::ldexp(1., -140)
                               : variant == 7 ? std::ldexp(1., 30)
                                              : 1.;
                parents.emplace_back((int(rng() % 31) - 15) * scale,
                                     variant == 4 && t % 3 == 0 ? 0. : double(rng() % 17 + 1));
                bounds.push_back(variant == 4 ? metal::SplitBounds{.25f, .75f}
                                              : metal::SplitBounds{-.5f, .5f});
              }
              for (std::uint32_t i = 0; i < n; ++i) {
                std::uint32_t child_begin = children.size();
                for (std::uint32_t t = 0; t < nt; ++t)
                  children.emplace_back(parents[pb + t].GetGrad() * .25 + (int(rng() % 7) - 3),
                                        parents[pb + t].GetHess() * .25);
                for (std::uint32_t t = 0; t < nt; ++t)
                  children.emplace_back(
                      parents[pb + t].GetGrad() - children[child_begin + t].GetGrad(),
                      parents[pb + t].GetHess() - children[child_begin + t].GetHess());
                float gain =
                    float(int(rng() % 5) - 2);  // Frequent exact ties, shuffled feature IDs.
                if (variant == 5) gain = std::ldexp(gain, -145);
                int bin = variant == 3 || i % 7 == 6 ? -1 : 0;
                features.push_back({n - 1 - i, int(variant % 3) - 1, child_begin});
                metal::QuantizedSplitResult r{};
                r.gain = gain;
                r.bin = bin;
                r.left_sum = children[child_begin];
                r.right_sum = children[child_begin + nt];
                scalar.push_back(r);
                vector.push_back({gain, bin, 0, 0});
              }
              nodes.push_back({begin, std::uint32_t(features.size()), pb, nt});
            }
            metal::SplitParameters p{variant % 4 == 0 ? 0.f : 1.25f, variant % 4 == 1 ? .3f : 0.f,
                                     variant % 4 == 2 ? .75f : 0.f, 0};
            ref::Parameters rp{p.reg_lambda, p.reg_alpha, p.max_delta_step, 0};
            std::vector<metal::SplitNodeResult> expected;
            std::vector<metal::SplitNodeWeights> ew(parents.size());
            for (auto node : nodes) {
              std::uint32_t winner = UINT32_MAX;
              for (auto i = node.begin; i < node.end; ++i)
                if (vector[i].bin >= 0) {
                  if (winner == UINT32_MAX || vector[i].gain > vector[winner].gain ||
                      (multi && vector[i].gain == vector[winner].gain &&
                       features[i].feature < features[winner].feature))
                    winner = i;
                }
              double pg = 0, ph = 0, lh = 0, rh = 0;
              for (std::uint32_t t = 0; t < nt; ++t) {
                auto idx = node.parent_begin + t;
                auto b = bounds[idx];
                float bw = ref::VectorWeight(Pair(parents[idx]), rp, {b.lower, b.upper}), lw = 0,
                      rw = 0;
                if (multi)
                  pg += ref::GainWeight(Pair(parents[idx]), bw, rp);
                else
                  pg = cuda_split_reference::Gain(Pair(parents[idx]), bw, rp);
                ph += parents[idx].GetHess();
                if (winner != UINT32_MAX) {
                  auto off = features[winner].child_begin;
                  auto l = Pair(children[off + t]), r = Pair(children[off + nt + t]);
                  lh += l.h;
                  rh += r.h;
                  lw = ref::VectorWeight(l, rp, {b.lower, b.upper});
                  rw = ref::VectorWeight(r, rp, {b.lower, b.upper});
                  auto c = features[winner].monotone;
                  if (multi && ((c > 0 && !(lw <= rw)) || (c < 0 && !(lw >= rw)))) {
                    auto pooled = rp;
                    pooled.lambda = 2.f * rp.lambda;
                    pooled.alpha = 2.f * rp.alpha;
                    lw = rw = ref::VectorWeight({l.g + r.g, l.h + r.h}, pooled, {b.lower, b.upper});
                  }
                }
                ew[idx] = {bw, lw, rw, 0};
              }
              float loss = winner == UINT32_MAX ? 0.f
                           : multi              ? float(double(vector[winner].gain) - pg)
                                                : vector[winner].gain - float(pg);
              expected.push_back({winner, loss, pg, ph, lh, rh});
            }
            auto c = multi ? upload(vector.data(), vector.size() * sizeof(vector[0]))
                           : upload(scalar.data(), scalar.size() * sizeof(scalar[0]));
            auto feature = upload(features.data(), features.size() * sizeof(features[0]));
            auto node = upload(nodes.data(), nodes.size() * sizeof(nodes[0]));
            auto parent = upload(parents.data(), parents.size() * sizeof(parents[0]));
            auto bound = upload(bounds.data(), bounds.size() * sizeof(bounds[0]));
            auto child = upload(children.data(), children.size() * sizeof(children[0]));
            auto out = upload(nullptr, (nodes.size() + 2) * sizeof(metal::SplitNodeResult));
            auto weight = upload(nullptr, (parents.size() + 2) * sizeof(metal::SplitNodeWeights));
            std::uint32_t mode = multi;
            auto command = [queue commandBuffer];
            auto e = [command computeCommandEncoder];
            [e setComputePipelineState:pipeline];
            [e setBuffer:c offset:0 atIndex:0];
            [e setBuffer:feature offset:0 atIndex:1];
            [e setBuffer:node offset:0 atIndex:2];
            [e setBytes:&p length:sizeof(p) atIndex:3];
            [e setBuffer:parent offset:0 atIndex:4];
            [e setBuffer:bound offset:0 atIndex:5];
            [e setBuffer:child offset:0 atIndex:6];
            [e setBuffer:out offset:sizeof(metal::SplitNodeResult) atIndex:7];
            [e setBuffer:weight offset:sizeof(metal::SplitNodeWeights) atIndex:8];
            [e setBytes:&mode length:sizeof(mode) atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake(nodes.size(), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
              throw std::runtime_error(command.error.localizedDescription.UTF8String);
            auto actual = static_cast<metal::SplitNodeResult const*>(out.contents) + 1;
            for (std::size_t i = 0; i < nodes.size(); ++i) {
              if (!Bits(actual + i, &expected[i], sizeof(expected[i]))) {
                auto a = actual[i], b = expected[i];
                std::fprintf(stderr,
                             "mode=%u nt=%u variant=%u node=%zu winner=%u/%u gain=%a/%a "
                             "parent=%a/%a h=%a/%a\n",
                             mode, nt, variant, i, a.winner, b.winner, a.loss_chg, b.loss_chg,
                             a.parent_gain, b.parent_gain, a.parent_hess, b.parent_hess);
                throw std::runtime_error("Final candidate/gain/trace mismatch");
              }
            }
            Check(Bits(static_cast<metal::SplitNodeWeights*>(weight.contents) + 1, ew.data(),
                       ew.size() * sizeof(ew[0])),
                  "Final weights mismatch");
            for (auto [buffer, size] :
                 {std::pair<id<MTLBuffer>, std::size_t>{out, sizeof(metal::SplitNodeResult)},
                  {weight, sizeof(metal::SplitNodeWeights)}}) {
              auto data = static_cast<unsigned char const*>(buffer.contents);
              for (std::size_t i = 0; i < size; ++i)
                Check(data[i] == 0xa5 && data[buffer.length - size + i] == 0xa5,
                      "Output guard changed");
            }
            checked += nodes.size();
            fields += parents.size();
          }
          std::printf("Checked %zu finalized nodes / %zu target weights (multi=%d targets=%u)\n",
                      checked, fields, multi, nt);
          std::fflush(stdout);
        }
      // Production root finalizer: no prior candidates, no host weight arithmetic.
      {
        auto ws = metal::MakeSplitWorkspace();
        std::vector<metal::SplitFinalizeNode> nodes{{0, 0, 0, 3}};
        std::vector<GradientPairPrecise> parent{{0, 16777216.}, {0, 1}, {0, 1}};
        std::vector<metal::SplitBounds> bounds(3, {-INFINITY, INFINITY});
        std::vector<metal::SplitNodeResult> result;
        std::vector<metal::SplitNodeWeights> weights;
        auto before = metal::GetStatistics();
        metal::FinalizeSplitNodes(nodes, {}, parent, bounds, {1, 0, 0, 1}, true, &result, &weights,
                                  ws.get());
        Check(result[0].winner == UINT32_MAX && result[0].parent_hess == 16777218.,
              "Root trace lost FP32 contributions");
        Check(metal::GetStatistics().split_finalize_dispatches ==
                  before.split_finalize_dispatches + 1,
              "Root dispatch missing");
      }
      std::size_t bounds_cases = 0;
      for (bool multi : {false, true})
        for (std::uint32_t nt : {1u, 2u, 33u, 129u}) {
          if (!multi && nt != 1) continue;
          for (int c : {-1, 0, 1})
            for (bool alias : {false, true})
              for (int exponent : {-145, 0, 30}) {
                auto count = 16 * nt;
                std::vector<float> lo(count + 2, -2.f), hi(count + 2, 3.f);
                std::vector<float> lw(nt), rw(nt);
                for (std::uint32_t t = 0; t < nt; ++t) {
                  lw[t] = std::ldexp(float(int(t % 5) - 2), exponent);
                  rw[t] = std::ldexp(float(int(t % 7) - 3), exponent);
                }
                auto left = xgboost::common::Span<float const>{
                    alias ? lo.data() + 1 + 2 * nt : lw.data(), nt};
                auto right = xgboost::common::Span<float const>{
                    alias ? hi.data() + 1 + 2 * nt : rw.data(), nt};
                auto elo = lo, ehi = hi;
                for (std::uint32_t t = 0; t < nt; ++t) {
                  float mid =
                      multi ? left[t] + .5f * (right[t] - left[t]) : (left[t] + right[t]) / 2.f;
                  elo[1 + 8 * nt + t] = c < 0 ? mid : lo[1 + 2 * nt + t];
                  ehi[1 + 8 * nt + t] = c > 0 ? mid : hi[1 + 2 * nt + t];
                  elo[1 + 15 * nt + t] = c > 0 ? mid : lo[1 + 2 * nt + t];
                  ehi[1 + 15 * nt + t] = c < 0 ? mid : hi[1 + 2 * nt + t];
                }
                auto before = metal::GetStatistics();
                metal::UpdateSplitBounds({lo.data() + 1, count}, {hi.data() + 1, count}, 2, 8, 15,
                                         c, left, right, multi);
                Check(Bits(lo.data(), elo.data(), lo.size() * sizeof(float)) &&
                          Bits(hi.data(), ehi.data(), hi.size() * sizeof(float)),
                      "Bound update/guard mismatch");
                Check(metal::GetStatistics().split_bounds_dispatches ==
                          before.split_bounds_dispatches + 1,
                      "Bounds dispatch missing");
                ++bounds_cases;
              }
        }
      std::printf(
          "PASS %zu exact finalized nodes / %zu target weights / %zu production bound updates\n",
          checked, fields, bounds_cases);
      // Validation must reject bad layouts before dispatch, preserving bound storage.
      auto reject = [](auto call) {
        bool failed = false;
        try {
          call();
        } catch (std::exception const&) {
          failed = true;
        }
        Check(failed, "Invalid finalization/bounds input accepted");
      };
      {
        auto ws = metal::MakeSplitWorkspace();
        std::vector<metal::SplitNodeResult> result;
        std::vector<metal::SplitNodeWeights> weight;
        metal::SplitFinalizeNode node{0, 1, 0, 1};
        metal::SplitFinalizeFeature feature{0, 0, 0};
        GradientPairPrecise parent{0, 1};
        metal::SplitBounds bound{-1, 1};
        auto before = metal::GetStatistics();
        reject([&] {
          metal::FinalizeSplitNodes({&node, 1}, {&feature, 1}, {&parent, 1}, {&bound, 1},
                                    {1, 0, 0, 1}, false, &result, &weight, ws.get());
        });
        node = {0, 0, 1, 1};
        reject([&] {
          metal::FinalizeSplitNodes({&node, 1}, {}, {&parent, 1}, {&bound, 1}, {1, 0, 0, 1}, false,
                                    &result, &weight, ws.get());
        });
        Check(metal::GetStatistics().split_finalize_dispatches == before.split_finalize_dispatches,
              "Invalid finalizer dispatched");
      }
      {
        std::vector<float> lo(6, -1), hi(6, 1);
        float a = -.5f, b = .5f;
        auto before = metal::GetStatistics();
        reject([&] { metal::UpdateSplitBounds(lo, lo, 0, 1, 2, 1, {&a, 1}, {&b, 1}, false); });
        reject([&] { metal::UpdateSplitBounds(lo, hi, 0, 1, 6, 1, {&a, 1}, {&b, 1}, false); });
        reject([&] { metal::UpdateSplitBounds(lo, hi, 0, 0, 2, 1, {&a, 1}, {&b, 1}, false); });
        Check(metal::GetStatistics().split_bounds_dispatches == before.split_bounds_dispatches,
              "Invalid bounds dispatched");
        a = std::numeric_limits<float>::quiet_NaN();
        reject([&] { metal::UpdateSplitBounds(lo, hi, 0, 1, 2, 1, {&a, 1}, {&b, 1}, false); });
        Check(std::all_of(lo.begin(), lo.end(), [](float v) { return v == -1; }) &&
                  std::all_of(hi.begin(), hi.end(), [](float v) { return v == 1; }),
              "Rejected bound update changed values");
      }
      std::printf("PASS rejected finalization layouts, overlapping/out-of-range bounds, and NaN "
                  "midpoints\n");
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
