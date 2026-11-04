// Independent floating-point reference for the adaptive robust objective pipeline.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Params {
  uint64_t rows;
  uint32_t targets, label_targets, weighted, kind, blocks, padding;
};
struct Pair {
  float g, h;
};
void Require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
bool Close(float actual, float expected) {
  return std::abs(double(actual) - expected) <= 3e-6 * std::max(1e-30, double(std::abs(expected)));
}
struct MetricParams {
  uint64_t count;
  uint32_t targets, levels, weighted, expectile;
};
void CheckMetrics(id<MTLDevice> device, id<MTLLibrary> lib, id<MTLCommandQueue> queue) {
  NSError* error = nil;
  auto f = [lib newFunctionWithName:@"xgb_quantile_metric"];
  auto pipeline = [device newComputePipelineStateWithFunction:f error:&error];
  Require(pipeline != nil, "No quantile metric pipeline");
  auto upload = [&](void const* data, size_t bytes) {
    auto b = [device newBufferWithLength:std::max<size_t>(bytes, 16)
                                 options:MTLResourceStorageModeShared];
    if (data && bytes) memcpy(b.contents, data, bytes);
    return b;
  };
  std::mt19937 rng(771);
  std::uniform_real_distribution<float> noise(-2, 2), positive(0.1, 2);
  size_t cases = 0, values = 0;
  for (uint32_t kind : {0u, 1u})
    for (uint32_t rows : {1u, 257u, 1027u})
      for (uint32_t targets : {1u, 3u})
        for (uint32_t levels : {1u, 3u, 17u})
          for (uint32_t scenario = 0; scenario < 8; ++scenario) {
            MetricParams p{uint64_t(rows) * targets * levels, targets, levels, scenario != 0, kind};
            std::vector<float> predictions(p.count), labels(rows * targets), weights(rows),
                alpha(levels);
            for (auto& v : predictions) v = noise(rng);
            for (auto& v : labels) v = noise(rng);
            for (auto& v : weights)
              v = positive(rng) * (scenario == 2
                                       ? 0.0f
                                       : (scenario == 3 ? 1e-30f : (scenario == 4 ? 1e30f : 1.0f)));
            for (uint32_t j = 0; j < levels; ++j) alpha[j] = float(j + 1) / (levels + 1);
            if (scenario == 5 || scenario == 6) {
              float response = scenario == 5 ? 1e-20f : 1e-40f;
              for (auto& value : predictions) value *= response;
              for (auto& value : labels) value *= response;
              for (auto& value : weights) value *= 1e30f;
            }
            if (scenario == 7)
              for (auto& value : alpha) value *= 1e-40f;

            auto pred = upload(predictions.data(), predictions.size() * 4),
                 label = upload(labels.data(), labels.size() * 4),
                 weight = upload(weights.data(), weights.size() * 4),
                 a = upload(alpha.data(), alpha.size() * 4);
            auto blocks = (p.count + 255) / 256;
            auto out = upload(nullptr, blocks * 16);
            auto cmd = [queue commandBuffer];
            auto e = [cmd computeCommandEncoder];
            [e setComputePipelineState:pipeline];
            [e setBuffer:pred offset:0 atIndex:0];
            [e setBuffer:label offset:0 atIndex:1];
            [e setBuffer:weight offset:0 atIndex:2];
            [e setBuffer:a offset:0 atIndex:3];
            [e setBuffer:out offset:0 atIndex:4];
            [e setBytes:&p length:sizeof(p) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            Require(cmd.status == MTLCommandBufferStatusCompleted, "Quantile metric GPU failure");
            double expected_loss = 0, expected_weight = 0, actual_loss = 0, actual_weight = 0;
            auto actual = static_cast<double const*>(out.contents);
            for (size_t i = 0; i < p.count; ++i) {
              size_t row = i / (targets * levels), level = (i / targets) % levels,
                     target = i % targets;
              float pred = predictions[i], label = labels[row * targets + target],
                    w = p.weighted ? weights[row] : 1.0f, value;
              if (kind) {
                float diff = pred - label, scale = diff >= 0 ? 1.0f - alpha[level] : alpha[level];
                value = scale * diff * diff * w;
              } else {
                float diff = label - pred, sign = diff >= 0 ? 1.0f : 0.0f;
                value =
                    ((alpha[level] * sign * diff) - (1.0f - alpha[level]) * (1.0f - sign) * diff) *
                    w;
              }
              expected_loss += value;
              expected_weight += w;
            }
            for (size_t i = 0; i < blocks; ++i) {
              actual_loss += actual[2 * i];
              actual_weight += actual[2 * i + 1];
            }
            if (std::abs(actual_loss - expected_loss) > 3e-13 * std::max(1e-30, expected_loss))
              fprintf(stderr,
                      "metric kind=%u rows=%u targets=%u levels=%u scenario=%u actual=%.17g "
                      "expected=%.17g relative=%g weights=%.17g/%.17g\n",
                      kind, rows, targets, levels, scenario, actual_loss, expected_loss,
                      (actual_loss - expected_loss) / expected_loss, actual_weight,
                      expected_weight);
            Require(std::abs(actual_loss - expected_loss) <= 3e-13 * std::max(1e-30, expected_loss),
                    "Quantile metric loss sum mismatch");
            Require(std::abs(actual_weight - expected_weight) <=
                        3e-13 * std::max(1e-30, expected_weight),
                    "Quantile metric weight sum mismatch");
            ++cases;
            values += p.count;
          }
  printf("PASS %zu quantile/expectile metric cases, %zu loss/weight pairs including multi-label "
         "layouts\n",
         cases, values);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [&](NSString* name) {
        auto f = [lib newFunctionWithName:name];
        auto p = [device newComputePipelineStateWithFunction:f error:&error];
        Require(p != nil, "No pipeline");
        return p;
      };
      auto reduce = pipeline(@"xgb_residual_scale_reduce"),
           finish = pipeline(@"xgb_residual_scale_finish"),
           stats_reduce = pipeline(@"xgb_residual_stats_reduce"),
           apply = pipeline(@"xgb_residual_scale_apply"),
           gradient = pipeline(@"xgb_adaptive_objective");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, size_t bytes) {
        auto b = [device newBufferWithLength:std::max<size_t>(bytes, 16)
                                     options:MTLResourceStorageModeShared];
        if (data && bytes) memcpy(b.contents, data, bytes);
        return b;
      };
      std::mt19937 rng(4537);
      std::uniform_real_distribution<float> noise(-2, 2), positive(0.1, 2);
      size_t cases = 0, pairs = 0, scales_checked = 0;
      for (uint32_t kind : {0u, 1u})
        for (uint32_t rows : {1u, 255u, 256u, 257u, 1027u})
          for (uint32_t targets : {1u, 3u, 17u})
            for (uint32_t scenario = 0; scenario < 6; ++scenario) {
              Params p{
                  rows, targets, kind == 1 ? 1u : targets, scenario != 0, kind, (rows + 255) / 256,
                  0};
              std::vector<float> predictions(rows * targets), labels(rows * p.label_targets),
                  weights(rows), alpha(targets);
              float response = scenario == 4 ? 1e18f : (scenario == 5 ? 1e-18f : 1.0f);
              for (auto& v : predictions) v = noise(rng) * response;
              for (auto& v : labels) v = noise(rng) * response;
              for (auto& v : weights)
                v = scenario == 2 ? 0.0f : (scenario == 3 ? 1e-12f : positive(rng));
              if (scenario == 1) {
                std::fill(predictions.begin(), predictions.end(), 0);
                std::fill(labels.begin(), labels.end(), 0);
              }
              for (uint32_t j = 0; j < targets; ++j) alpha[j] = float(j + 1) / (targets + 1);
              auto pred = upload(predictions.data(), predictions.size() * 4),
                   label = upload(labels.data(), labels.size() * 4),
                   weight = upload(weights.data(), weights.size() * 4),
                   levels = upload(alpha.data(), alpha.size() * 4);
              auto partial = upload(nullptr, p.blocks * targets * 16),
                   statistics = upload(nullptr, (targets + 1) * 8),
                   scale = upload(nullptr, targets * 4),
                   output = upload(nullptr, predictions.size() * 8);
              auto command = [queue commandBuffer];
              auto e = [command computeCommandEncoder];
              [e setComputePipelineState:reduce];
              [e setBuffer:pred offset:0 atIndex:0];
              [e setBuffer:label offset:0 atIndex:1];
              [e setBuffer:weight offset:0 atIndex:2];
              [e setBuffer:partial offset:0 atIndex:3];
              [e setBytes:&p length:sizeof(p) atIndex:4];
              [e dispatchThreadgroups:MTLSizeMake(p.blocks, targets, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              [e endEncoding];
              while (p.blocks > 1) {
                auto blocks = (p.blocks + 255) / 256;
                auto next = upload(nullptr, blocks * targets * 16);
                e = [command computeCommandEncoder];
                [e setComputePipelineState:stats_reduce];
                [e setBuffer:partial offset:0 atIndex:0];
                [e setBuffer:next offset:0 atIndex:1];
                [e setBytes:&p length:sizeof(p) atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(blocks, targets, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                partial = next;
                p.blocks = blocks;
              }
              e = [command computeCommandEncoder];
              [e setComputePipelineState:finish];
              [e setBuffer:partial offset:0 atIndex:0];
              [e setBuffer:statistics offset:0 atIndex:1];
              [e setBytes:&p length:sizeof(p) atIndex:2];
              [e dispatchThreads:MTLSizeMake(targets, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
              [e endEncoding];
              e = [command computeCommandEncoder];
              [e setComputePipelineState:apply];
              [e setBuffer:statistics offset:0 atIndex:0];
              [e setBuffer:scale offset:0 atIndex:1];
              [e setBytes:&p length:sizeof(p) atIndex:2];
              [e dispatchThreads:MTLSizeMake(targets, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
              [e endEncoding];
              e = [command computeCommandEncoder];
              [e setComputePipelineState:gradient];
              [e setBuffer:pred offset:0 atIndex:0];
              [e setBuffer:label offset:0 atIndex:1];
              [e setBuffer:weight offset:0 atIndex:2];
              [e setBuffer:scale offset:0 atIndex:3];
              [e setBuffer:levels offset:0 atIndex:4];
              [e setBuffer:output offset:0 atIndex:5];
              [e setBytes:&p length:sizeof(p) atIndex:6];
              [e dispatchThreads:MTLSizeMake(predictions.size(), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              [e endEncoding];
              [command commit];
              [command waitUntilCompleted];
              Require(command.status == MTLCommandBufferStatusCompleted, "GPU command failed");
              std::vector<float> expected_scale(targets);
              auto actual_scale = static_cast<float const*>(scale.contents);
              for (uint32_t j = 0; j < targets; ++j) {
                double root = 0, total = 0;
                for (uint32_t i = 0; i < rows; ++i) {
                  float residual = predictions[i * targets + j] -
                                   labels[i * p.label_targets + (p.label_targets == 1 ? 0 : j)];
                  float w = p.weighted ? weights[i] : 1.0f;
                  float value = w * std::sqrt(std::abs(residual));
                  root += value;
                  total += w;
                }
                double mean = root / total;
                expected_scale[j] = std::abs(total) < 1e-6 ? 0.0f : float(mean * mean);
                Require(Close(actual_scale[j], expected_scale[j]),
                        "Residual scale differs from CPU reference");
                ++scales_checked;
              }
              auto actual = static_cast<Pair const*>(output.contents);
              for (size_t i = 0; i < predictions.size(); ++i) {
                uint32_t row = i / targets, j = i % targets;
                float residual = predictions[i] -
                                 labels[row * p.label_targets + (p.label_targets == 1 ? 0 : j)],
                      s = expected_scale[j], w = p.weighted ? weights[row] : 1.0f, g, h;
                if (kind == 0) {
                  float norm = std::hypot(s, residual), curvature = norm > 0 ? s / norm : 1;
                  g = w * residual * curvature;
                  h = w * curvature;
                } else if (!(s > 0) || w == 0) {
                  g = 0;
                  h = 0;
                } else {
                  float x = residual / (0.04f * s), t = std::tanh(x), ratio = x == 0 ? 1 : t / x;
                  ratio = std::max(ratio, 3e-4f);
                  float grad = 0.5f * s * (t + 1 - 2 * alpha[j]);
                  float hess = 0.5f / 0.04f * ratio;
                  g = w * grad;
                  h = w * hess;
                }
                // Quantile gradients near zero need an absolute error scaled by the response.
                bool grad_ok = Close(actual[i].g, g) ||
                               std::abs(double(actual[i].g) - g) <= 2e-6 * double(s) * w;
                if (!grad_ok || !Close(actual[i].h, h)) {
                  fprintf(stderr,
                          "kind=%u rows=%u targets=%u scenario=%u i=%zu actual=(%g,%g) "
                          "expected=(%g,%g)\n",
                          kind, rows, targets, scenario, i, actual[i].g, actual[i].h, g, h);
                  throw std::runtime_error("Adaptive gradients differ");
                }
                ++pairs;
              }
              ++cases;
            }
      printf("PASS %zu adaptive objective cases, %zu scales and %zu gradient pairs on %s\n", cases,
             scales_checked, pairs, device.name.UTF8String);
      CheckMetrics(device, lib, queue);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
