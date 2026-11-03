// Check objective sharing, copied fallbacks and alias semantics through the real library.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"
#include "../xgboost/src/common/metal_training.h"
void Check(bool ok, char const* msg) {
  if (!ok) throw std::runtime_error(msg);
}
int main() {
  @autoreleasepool {
    try {
      using namespace xgboost;
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No GPU");
      bool unified = device.hasUnifiedMemory;
      size_t cases = 0, values = 0;
      for (size_t rows : {1, 257, 4097, 65537})
        for (size_t targets : {1, 3})
          for (size_t offset : {0, 1, 3})
            for (bool weighted : {false, true}) {
              size_t n = rows * targets;
              std::vector<float> prediction(n + offset), label(n + offset),
                  weight(weighted ? rows : 0);
              std::vector<GradientPair> output(n + 2, GradientPair{123456, 654321});
              for (size_t i = 0; i < n; ++i) {
                prediction[offset + i] = float(int(i % 127) - 63);
                label[offset + i] = float(i % 31);
              }
              for (size_t i = 0; i < weight.size(); ++i) weight[i] = float(i % 7) / 4;
              auto before = common::metal::GetStatistics();
              common::metal::Gradients(common::metal::Objective::kSquare,
                                       {prediction.data() + offset, n}, {label.data() + offset, n},
                                       weight, targets, 0, 1, {}, {output.data() + 1, n});
              auto after = common::metal::GetStatistics();
              Check(after.objective_dispatches - before.objective_dispatches == 1,
                    "No GPU objective dispatch");
              auto shared =
                  after.objective_shared_input_bytes - before.objective_shared_input_bytes +
                  after.objective_shared_output_bytes - before.objective_shared_output_bytes;
              auto staged = after.objective_staging_bytes - before.objective_staging_bytes;
              Check(shared + staged == n * 16 + weight.size() * 4,
                    "Objective memory accounting differs from logical byte counts");
              if (unified)
                Check(shared > 0, "Unified-memory objective did not share any host storage");
              else
                Check(shared == 0, "Discrete GPU should use staging");
              for (size_t i = 0; i < n; ++i) {
                float w = weighted ? weight[i / targets] : 1;
                Check(output[i + 1].GetGrad() == (prediction[offset + i] - label[offset + i]) * w &&
                          output[i + 1].GetHess() == w,
                      "Shared objective differs from CPU");
              }
              Check(output.front().GetGrad() == 123456 && output.front().GetHess() == 654321 &&
                        output.back().GetGrad() == 123456 && output.back().GetHess() == 654321,
                    "Output guard overwritten");
              ++cases;
              values += n;
            }
      auto page = size_t(getpagesize());
      for (bool split_region : {false, true}) {
        size_t n = 2 * page / sizeof(float);
        auto mapping = static_cast<float*>(
            mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
        Check(mapping != MAP_FAILED, "mmap failed");
        for (size_t i = 0; i < n; ++i) mapping[i] = float(i % 127);
        auto protect_start = split_region ? reinterpret_cast<char*>(mapping) + page
                                          : reinterpret_cast<char*>(mapping);
        Check(mprotect(protect_start, split_region ? page : 2 * page, PROT_READ) == 0,
              "mprotect failed");
        std::vector<float> labels(n, 3);
        std::vector<GradientPair> output(n);
        auto before = common::metal::GetStatistics();
        common::metal::Gradients(common::metal::Objective::kSquare, {mapping, n}, labels, {}, 1, 0,
                                 1, {}, output);
        auto after = common::metal::GetStatistics();
        Check(after.objective_staging_bytes - before.objective_staging_bytes >= n * 4,
              "Unsafe input must use copied staging");
        for (size_t i = 0; i < n; ++i)
          Check(output[i].GetGrad() == mapping[i] - 3 && output[i].GetHess() == 1,
                "Copied input fallback differs from CPU");
        Check(munmap(mapping, 2 * page) == 0, "munmap failed");
        ++cases;
        values += n;
      }
      // Input and output overlap logically. Snapshot semantics require separate GPU
      // output storage until computation completes, even when the memory is writable.
      std::vector<GradientPair> alias{{7, 9}, {13, 17}, {19, 23}, {29, 31}};
      std::vector<float> labels(3, 3);
      auto before = common::metal::GetStatistics();
      common::metal::Gradients(common::metal::Objective::kSquare,
                               {reinterpret_cast<float const*>(alias.data()), 3}, labels, {}, 1, 0,
                               1, {}, {alias.data() + 1, 3});
      auto after = common::metal::GetStatistics();
      Check(after.objective_shared_output_bytes == before.objective_shared_output_bytes,
            "Overlapping output must not be shared");
      Check(after.objective_staging_bytes - before.objective_staging_bytes >=
                3 * sizeof(GradientPair),
            "Overlapping output staging was not recorded");
      for (size_t i = 0; i < 3; ++i)
        Check(alias[i + 1].GetGrad() == float(i == 0   ? 4
                                              : i == 1 ? 6
                                                       : 10) &&
                  alias[i + 1].GetHess() == 1,
              "Overlapping objective lost input snapshot semantics");
      ++cases;
      values += 3;
      std::printf("PASS %zu production objective memory cases / %zu gradient pairs, including "
                  "read-only, split-region and overlapping storage\n",
                  cases, values);
      size_t leaf_cases = 0, leaf_values = 0;
      for (size_t n : {1, 257, 65537})
        for (size_t base : {0, 17})
          for (size_t targets : {1, 3})
            for (auto order : {linalg::Order::kC, linalg::Order::kF})
              for (bool readonly : {false, true}) {
                size_t total = base + n + 13;
                size_t shape[2]{total, targets};
                std::vector<GradientPair> gradients(total * targets);
                linalg::MatrixView<GradientPair> view{gradients, shape, DeviceOrd::CPU(), order};
                for (size_t r = 0; r < total; ++r)
                  for (size_t t = 0; t < targets; ++t)
                    view(r, t) = GradientPair{
                        float(r % 19),
                        r % 5 && t + 1 == targets
                            ? (r % 7 == 0 ? std::numeric_limits<float>::denorm_min() : 1.0f)
                            : 0.0f};
                size_t bytes = gradients.size() * sizeof(GradientPair);
                size_t mapped_bytes = (bytes + page - 1) / page * page;
                void* mapping = nullptr;
                auto input = gradients.data();
                if (readonly) {
                  mapping = mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANON, -1, 0);
                  Check(mapping != MAP_FAILED, "Leaf mmap failed");
                  std::memcpy(mapping, input, bytes);
                  Check(mprotect(mapping, mapped_bytes, PROT_READ) == 0, "Leaf mprotect failed");
                  input = static_cast<GradientPair*>(mapping);
                }
                linalg::MatrixView<GradientPair const> const_view{
                    {input, gradients.size()}, shape, DeviceOrd::CPU(), order};
                constexpr bst_node_t sentinel = -123456;
                std::vector<bst_node_t> output(total + 2, sentinel);
                // Unordered ranges include an empty leaf at the shared boundary.
                std::vector<common::metal::LeafRange> leaves{
                    {n / 2, n, 71}, {0, n / 2, 23}, {n / 2, n / 2, 99}};
                auto rows = common::metal::MakeDeviceRows();
                rows->Reset(n, base);
                auto before = common::metal::GetStatistics();
                rows->LeafPositions(leaves, const_view, {output.data() + 1, total});
                auto after = common::metal::GetStatistics();
                Check(after.leaf_partition_dispatches - before.leaf_partition_dispatches == 1,
                      "Leaf assignment did not dispatch on GPU");
                auto shared_in =
                    after.leaf_shared_gradient_bytes - before.leaf_shared_gradient_bytes;
                auto shared_out = after.leaf_shared_output_bytes - before.leaf_shared_output_bytes;
                if (!unified || readonly)
                  Check(shared_in == 0, "Ineligible leaf input was shared");
                else
                  Check(shared_in == bytes, "Leaf gradient sharing byte count differs");
                Check(shared_out == (unified ? n * sizeof(bst_node_t) : 0),
                      "Leaf output sharing byte count differs");
                for (size_t r = 0; r < total + 2; ++r) {
                  bst_node_t expected = sentinel;
                  if (r > base && r <= base + n) {
                    expected = r - base - 1 < n / 2 ? 23 : 71;
                    if ((r - 1) % 5 == 0) expected = ~expected;
                  }
                  Check(output[r] == expected, "Leaf output or guard differs from CPU");
                }
                if (readonly) Check(munmap(mapping, mapped_bytes) == 0, "Leaf munmap failed");
                ++leaf_cases;
                leaf_values += n;
              }
      {
        // A writable aliased output still requires staging so no GPU invocation
        // overwrites gradient input while another invocation is reading it.
        size_t n = 65537, shape[2]{n, 1};
        std::vector<GradientPair> gradients(n, GradientPair{3, 1});
        auto rows = common::metal::MakeDeviceRows();
        rows->Reset(n, 0);
        linalg::MatrixView<GradientPair const> view{gradients, shape, DeviceOrd::CPU()};
        auto before = common::metal::GetStatistics();
        rows->LeafPositions({{n / 2, n, 71}, {0, n / 2, 23}}, view,
                            {reinterpret_cast<bst_node_t*>(gradients.data()), n});
        auto after = common::metal::GetStatistics();
        Check(after.leaf_shared_output_bytes == before.leaf_shared_output_bytes,
              "Aliased leaf output must use staging");
        std::vector<bst_node_t> result(n);
        std::memcpy(result.data(), gradients.data(), n * sizeof(bst_node_t));
        for (size_t i = 0; i < n; ++i)
          Check(result[i] == (i < n / 2 ? 23 : 71), "Aliased leaf assignment corrupted input");
        ++leaf_cases;
        leaf_values += n;
      }
      std::printf("PASS %zu production leaf memory cases / %zu row positions, including "
                  "strided gradients, page offsets, read-only inputs, empty leaves and aliases\n",
                  leaf_cases, leaf_values);
      return 0;
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
