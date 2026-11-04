// Independent CPU mathematical reference for the production gradient kernels.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Pair {
  float g, h;
};
struct Params {
  uint64_t n;
  uint32_t targets, kind, weighted, label_targets;
  float parameter, scale;
};
static_assert(sizeof(Params) == 32);
void Require(bool ok, char const* msg) {
  if (!ok) throw std::runtime_error(msg);
}
double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(std::min(-x, 88.7))); }
double Softplus(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x))); }
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production shader path");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal device");
      std::ifstream file(argv[1]);
      Require(file.good(), "Cannot read shader");
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto function = [library newFunctionWithName:@"xgb_objective"];
      auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      Require(pipeline != nil, "Cannot create objective pipeline");
      auto multiclass_function = [library newFunctionWithName:@"xgb_multiclass_objective"];
      auto multiclass_pipeline = [device newComputePipelineStateWithFunction:multiclass_function
                                                                       error:&error];
      Require(multiclass_pipeline != nil, "Cannot create multiclass pipeline");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, size_t size) {
        auto b = [device newBufferWithLength:std::max<size_t>(size, 16)
                                     options:MTLResourceStorageModeShared];
        Require(b != nil, "Allocation failed");
        if (size) memcpy(b.contents, data, size);
        return b;
      };
      size_t cases = 0, values = 0;
      for (uint32_t kind = 0; kind < 10; ++kind)
        for (bool weighted : {false, true}) {
          constexpr uint32_t rows = 1031;
          uint32_t targets = kind == 8 ? 37 : 3;
          Params p{uint64_t(rows) * targets,
                   targets,
                   kind,
                   uint32_t(weighted),
                   kind >= 8 ? 1u : targets,
                   kind == 5 ? 0.4f : 1.3f,
                   kind < 4 ? 2.1f : 1.0f};
          std::mt19937 random(431);
          std::uniform_real_distribution<float> dist(-4, 4), positive(0.05, 2);
          std::vector<float> preds(p.n), labels(rows * p.label_targets), weights(rows),
              alpha(targets);
          for (auto& v : preds) v = dist(random);
          for (auto& v : labels) v = kind == 8 ? float(random() % targets) : positive(random);
          for (auto& v : weights) v = positive(random);
          for (uint32_t i = 0; i < rows; i += 17) weights[i] = 0;
          for (uint32_t i = 0; i < targets; ++i) alpha[i] = float(i + 1) / (targets + 1);
          if (kind == 1 || kind == 7)
            for (auto& v : labels) v = float(random() % 2);
          // Squared-log near its singularity can amplify any log implementation's ULP error.
          // Keep this mathematical accuracy test away from the singularity; integrated
          // tests separately exercise XGBoost's domain clamp.
          if (kind == 2)
            for (auto& v : preds) v = positive(random) - 0.5f;
          auto pred = upload(preds.data(), preds.size() * 4),
               label = upload(labels.data(), labels.size() * 4);
          auto weight = upload(weights.data(), weights.size() * 4),
               level = upload(alpha.data(), alpha.size() * 4);
          std::vector<Pair> zero(p.n);
          auto output = upload(zero.data(), zero.size() * sizeof(Pair));
          auto command = [queue commandBuffer];
          auto encoder = [command computeCommandEncoder];
          [encoder setComputePipelineState:kind == 8 ? multiclass_pipeline : pipeline];
          [encoder setBuffer:pred offset:0 atIndex:0];
          [encoder setBuffer:label offset:0 atIndex:1];
          [encoder setBuffer:weight offset:0 atIndex:2];
          [encoder setBuffer:level offset:0 atIndex:3];
          [encoder setBuffer:output offset:0 atIndex:4];
          [encoder setBytes:&p length:sizeof(p) atIndex:5];
          [encoder dispatchThreads:MTLSizeMake(kind == 8 ? rows : p.n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [encoder endEncoding];
          [command commit];
          [command waitUntilCompleted];
          Require(command.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
          auto actual = static_cast<Pair const*>(output.contents);
          for (uint64_t idx = 0; idx < p.n; ++idx) {
            auto row = idx / targets, target = idx % targets;
            double raw = preds[idx],
                   y = labels[row * p.label_targets + (p.label_targets == 1 ? 0 : target)];
            double w = weighted ? weights[row] : 1;
            if (y == 1) w *= p.scale;
            double g = 0, h = 0;
            switch (kind) {
              case 0:
                g = raw - y;
                h = 1;
                break;
              case 1: {
                auto prob = Sigmoid(raw);
                g = prob - y;
                h = std::max(prob * (1 - prob), 1e-16);
                break;
              }
              case 2: {
                auto diff = std::log1p(raw) - std::log1p(y);
                g = diff / (raw + 1);
                h = std::max((1 - diff) / std::pow(raw + 1, 2), 1e-6);
                break;
              }
              case 3:
                h = y / std::exp(raw);
                g = 1 - h;
                break;
              case 4: {
                double z = raw - y, a = p.parameter * p.parameter, root = std::sqrt(1 + z * z / a);
                g = z / root;
                h = a / ((a + z * z) * root);
                break;
              }
              case 5:
                g = std::exp(raw) - y;
                h = std::exp(raw + p.parameter);
                break;
              case 6: {
                double a = 1.0f - p.parameter, b = 2.0f - p.parameter;
                g = -y * std::exp(a * raw) + std::exp(b * raw);
                h = -y * a * std::exp(a * raw) + b * std::exp(b * raw);
                break;
              }
              case 7:
                if (raw * (2 * y - 1) < 1) {
                  g = -(2 * y - 1);
                  h = 1;
                } else {
                  g = 0;
                  h = 0;
                }
                break;
              case 8: {
                double sum = 0;
                for (uint32_t k = 0; k < targets; ++k) sum += std::exp(preds[row * targets + k]);
                double prob = std::exp(raw) / sum;
                g = prob - (y == target);
                h = 2 * prob * (1 - prob);
                break;
              }
              case 9: {
                double pred = preds[row * targets];
                for (uint32_t k = 0; k < targets; ++k) {
                  if (k) pred += 1e-6 + Softplus(preds[row * targets + k]);
                  if (k >= target) {
                    double diff = pred - y, a = diff >= 0 ? 1 - alpha[k] : alpha[k];
                    g += a * diff;
                    h += a;
                  }
                }
                double a = target ? Sigmoid(raw) : 1;
                g *= a;
                h *= a * a;
                break;
              }
            }
            g *= w;
            h *= w;
            if (kind == 8) h = std::max(h, 1e-16);
            if (kind == 7 && raw * (2 * y - 1) >= 1) h = std::numeric_limits<float>::min();
            if (std::abs(actual[idx].g - g) > 3e-5 * std::max(1.0, std::abs(g)) ||
                std::abs(actual[idx].h - h) > 3e-5 * std::max(1.0, std::abs(h))) {
              fprintf(stderr,
                      "kind=%u weighted=%d row=%llu target=%llu actual=(%g,%g) reference=(%g,%g)\n",
                      kind, weighted, row, target, actual[idx].g, actual[idx].h, g, h);
              throw std::runtime_error("Gradient differs from independent CPU reference");
            }
          }
          ++cases;
          values += p.n;
          printf("PASS objective %u weighted=%d: %llu gradient pairs\n", kind, weighted, p.n);
        }
      printf("%zu objective cases, %zu gradient pairs passed on %s\n", cases, values,
             device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
