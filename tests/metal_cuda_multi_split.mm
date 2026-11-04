// Production Metal vector split kernel vs independent CUDA-source host formulas.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include "cuda_multi_split_reference.h"
namespace ref = cuda_multi_split_reference;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
std::size_t cases = 0, checked = 0;
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No Metal GPU");
      std::ifstream f(argv[1]);
      std::string source{std::istreambuf_iterator<char>(f), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [device newComputePipelineStateWithFunction:
                                  [lib newFunctionWithName:@"xgb_evaluate_quantized_multi_splits"]
                                                            error:&error];
      Check(pipeline != nil, "Missing vector split pipeline");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, std::size_t bytes) {
        auto b = [device newBufferWithLength:std::max<std::size_t>(bytes, 32)
                                     options:MTLResourceStorageModeShared];
        std::memset(b.contents, 0xa5, b.length);
        if (data && bytes) std::memcpy(b.contents, data, bytes);
        return b;
      };
      auto run = [&](std::vector<ref::Pair> const& hist, std::vector<ref::Pair> const& parents,
                     std::vector<ref::Task> const& tasks, std::vector<ref::Scale> const& scales,
                     std::vector<ref::Bounds> const& bounds, ref::Parameters params) {
        std::vector<ref::Result> expected;
        std::vector<std::vector<std::uint32_t>> orders(tasks.size());
        std::vector<std::vector<ref::Pair>> children(tasks.size());
        for (std::size_t i = 0; i < tasks.size(); ++i)
          expected.push_back(ref::Evaluate(hist, parents, scales, bounds, tasks[i], params,
                                           &orders[i], &children[i]));
        auto h = upload(hist.data(), hist.size() * sizeof(ref::Pair));
        auto t = upload(tasks.data(), tasks.size() * sizeof(ref::Task));
        auto o = upload(nullptr, (tasks.size() + 2) * sizeof(ref::Result));
        auto order = upload(nullptr, (hist.size() + 2) * sizeof(std::uint32_t));
        auto scores = upload(nullptr, hist.size() * sizeof(double));
        auto scans = upload(nullptr, hist.size() * 2 * sizeof(ref::Pair));
        auto parent = upload(parents.data(), parents.size() * sizeof(ref::Pair));
        auto bound = upload(bounds.data(), bounds.size() * sizeof(ref::Bounds));
        auto scale = upload(scales.data(), scales.size() * sizeof(ref::Scale));
        auto child = upload(nullptr, (parents.size() * 2 + 2) * sizeof(ref::Floating));
        auto exact = upload(nullptr, (parents.size() * 2 + 2) * sizeof(ref::Pair));
        auto command = [queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:pipeline];
        [e setBuffer:h offset:0 atIndex:0];
        [e setBuffer:t offset:0 atIndex:1];
        [e setBytes:&params length:sizeof(params) atIndex:2];
        [e setBuffer:o offset:sizeof(ref::Result) atIndex:3];
        [e setBuffer:order offset:sizeof(std::uint32_t) atIndex:4];
        [e setBuffer:scores offset:0 atIndex:5];
        [e setBuffer:scans offset:0 atIndex:6];
        [e setBuffer:parent offset:0 atIndex:7];
        [e setBuffer:bound offset:0 atIndex:8];
        [e setBuffer:child offset:sizeof(ref::Floating) atIndex:9];
        [e setBuffer:scale offset:0 atIndex:10];
        [e setBuffer:exact offset:sizeof(ref::Pair) atIndex:11];
        [e dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
          throw std::runtime_error(command.error.localizedDescription.UTF8String);
        auto actual = static_cast<ref::Result const*>(o.contents) + 1;
        auto ic = static_cast<ref::Pair const*>(exact.contents) + 1;
        auto fc = static_cast<ref::Floating const*>(child.contents) + 1;
        auto sorted = static_cast<std::uint32_t const*>(order.contents) + 1;
        for (std::size_t i = 0; i < tasks.size(); ++i) {
          auto const& a = actual[i];
          auto const& b = expected[i];
          auto const& task = tasks[i];
          if (std::memcmp(&a, &b, sizeof(a))) {
            std::fprintf(
                stderr,
                "case=%zu targets=%u bins=%u kind=%u constraint=%d/%u onepass=%u "
                "actual gain=%a bin=%d dir=%u part=%u expected gain=%a bin=%d dir=%u part=%u\n",
                cases + i, task.targets, (task.end - task.begin) / task.targets, task.kind,
                task.monotone, task.constrained, task.one_pass, a.gain, a.bin, a.missing,
                a.partition, b.gain, b.bin, b.missing, b.partition);
            throw std::runtime_error("Vector CUDA chunk winner differs");
          }
          for (std::size_t j = 0; j < children[i].size(); ++j) {
            auto value = children[i][j];
            auto precise = ref::Dequantize(value, scales[j % task.targets]);
            auto offset = 2 * task.parent_begin + j;
            if (std::memcmp(&ic[offset], &value, sizeof(value)) ||
                std::memcmp(&fc[offset], &precise, sizeof(precise)))
              throw std::runtime_error("Exact vector child or materialized sum differs");
            ++checked;
          }
          if (task.kind == 2)
            Check(std::equal(orders[i].begin(), orders[i].end(), sorted + task.begin),
                  "Vector category order differs");
        }
        for (auto buffer : {o, child, exact}) {
          auto guard = static_cast<unsigned char const*>(buffer.contents);
          for (unsigned i = 0; i < 16; ++i)
            Check(guard[i] == 0xa5 && guard[buffer.length - 16 + i] == 0xa5,
                  "Output guard changed");
        }
        Check(sorted[-1] == 0xa5a5a5a5u && sorted[hist.size()] == 0xa5a5a5a5u,
              "Category guard changed");
        cases += tasks.size();
      };
      std::mt19937_64 rng(617893);
      for (std::uint32_t nt : {1u, 2u, 3u, 17u, 33u, 65u, 129u})
        for (unsigned variant = 0; variant < 8; ++variant) {
          std::vector<ref::Scale> scales;
          for (std::uint32_t t = 0; t < nt; ++t)
            scales.push_back({int(40 + t % 3 * 8), int(40 + t % 2 * 8)});
          std::vector<ref::Pair> hist, parents;
          std::vector<ref::Bounds> bounds;
          std::vector<ref::Task> tasks;
          for (std::uint32_t n : {0u, 1u, 2u, 3u, 31u, 32u, 33u, 65u, 129u})
            for (std::uint32_t kind = 0; kind < 3; ++kind) {
              auto begin = std::uint32_t(hist.size()), pb = std::uint32_t(parents.size());
              std::vector<ref::Pair> total(nt);
              for (std::uint32_t i = 0; i < n; ++i)
                for (std::uint32_t t = 0; t < nt; ++t) {
                  ref::Pair value{
                      (std::int64_t(rng() % 17) - 8) * (std::int64_t{1} << scales[t].ge) +
                          std::int64_t(rng() % 7),
                      std::int64_t(rng() % 4) * (std::int64_t{1} << scales[t].he) +
                          std::int64_t(rng() % 3)};
                  if (variant == 6 && t % 3 == 0)
                    value = {value.g, 0};  // zero target-H, valid average trace
                  hist.push_back(value);
                  total[t] = ref::Add(total[t], value);
                }
              for (std::uint32_t t = 0; t < nt; ++t) {
                if (variant % 2)
                  total[t] = ref::Add(total[t], {(std::int64_t{1} << scales[t].ge) + 7,
                                                 (std::int64_t{1} << scales[t].he) + 3});
                parents.push_back(total[t]);
                bounds.push_back(variant == 6 ? ref::Bounds{.25f, .75f} : ref::Bounds{-.5f, .5f});
              }
              tasks.push_back({begin, std::uint32_t(hist.size()), kind,
                               variant % 4 == 0   ? 1u
                               : variant % 4 == 1 ? 3u
                                                  : 128u,
                               pb, nt, int(variant % 3) - 1, variant % 2,
                               variant == 4   ? 1u
                               : variant == 5 ? 2u
                                              : 0u});
              if (variant == 6) tasks.back().constrained = 1;
            }
          ref::Parameters params{variant % 4 == 0 ? 0.f : 1.25f, variant % 4 == 1 ? .3f : 0.f,
                                 variant % 4 == 2 ? .75f : 0.f, variant % 4 == 3 ? 2.f : 0.f};
          run(hist, parents, tasks, scales, bounds, params);
          if (variant == 7) {
            std::printf("Checked %zu vector tasks / %zu child pairs (targets=%u)\n", cases, checked,
                        nt);
            std::fflush(stdout);
          }
        }
      // Within a warp, bins 0/1 tie in FP32 but bin1 wins in FP64. The later
      // chunk's bin33 has greater FP64 gain, but ties the stored FP32 winner.
      {
        constexpr std::int64_t unit = std::int64_t{1} << 40;
        std::vector<ref::Pair> hist(65, {unit, unit});
        hist[1].g -= std::int64_t{1} << 16;
        hist[33].g -= std::int64_t{1} << 17;
        std::vector<ref::Pair> parents{{0, 75 * unit}};
        std::vector<ref::Scale> scales{{40, 40}};
        std::vector<ref::Bounds> bounds{{-INFINITY, INFINITY}};
        std::vector<ref::Task> tasks{{0, 65, 1, 64, 0, 1, 0, 0}};
        std::vector<std::uint32_t> order;
        std::vector<ref::Pair> child;
        auto reference =
            ref::Evaluate(hist, parents, scales, bounds, tasks[0], {1, 0, 0, 0}, &order, &child);
        Check(reference.bin == 1 && reference.missing == 0,
              "Chunk-rounding fixture did not distinguish both selection stages");
        run(hist, parents, tasks, scales, bounds, {1, 0, 0, 0});
      }
      // Forward/backward cancellation and heterogeneous target scales.
      for (std::uint32_t kind = 0; kind < 3; ++kind)
        for (std::uint32_t offset : {0u, 30u, 31u, 32u, 127u}) {
          std::vector<ref::Pair> hist(offset * 3, {0, 0});
          for (std::int64_t g :
               {std::int64_t{1} << 60, std::int64_t{1}, -(std::int64_t{1} << 60), std::int64_t{-2}})
            for (unsigned t = 0; t < 3; ++t) hist.push_back({g, 1});
          std::vector<ref::Task> tasks{{0, std::uint32_t(hist.size()), kind, 256, 0, 3, 0, 0}};
          run(hist, {{-1, 4}, {-1, 4}, {-1, 4}}, tasks, {{140, 0}, {60, 0}, {6, 0}},
              {{-INFINITY, INFINITY}, {-INFINITY, INFINITY}, {-INFINITY, INFINITY}}, {1, 0, 0, 0});
        }
      std::printf("PASS %zu vector CUDA-source tasks / %zu exact child pairs; chunk precision, "
                  "directions, ordering and guards\n",
                  cases, checked);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
