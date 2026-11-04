// Exact production rounding/truncation against scalar CUDA-source arithmetic.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_ranking_rounding.h"
template <typename T, typename U>
T Cast(U value) {
  static_assert(sizeof(T) == sizeof(U));
  T out;
  std::memcpy(&out, &value, sizeof(T));
  return out;
}
struct Query {
  std::uint32_t maximum, rows, value, padding;
  double cost, minimum, term;
};
struct Result {
  std::uint32_t factor, positive, negative, padding;
  double cost_factor, cost_positive, cost_negative;
};
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::runtime_error("Pass production shader");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
struct RoundQuery { uint maximum, rows, value, padding; ulong cost, minimum, term; };
struct RoundResult { uint factor, positive, negative, padding; ulong cost_factor, cost_positive, cost_negative; };
kernel void test_ranking_rounding(device const RoundQuery* input [[buffer(0)]],
                                   device RoundResult* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  using namespace xgb_survival;
  RoundQuery q = input[tid];
  uint m = xgb_rank_rounding(q.maximum, q.rows);
  D cm{xgb_rank_cost_rounding(D{q.cost}, q.rows, D{q.minimum})}, term{q.term};
  output[tid] = {m, xgb_rank_truncate(m, q.value), xgb_rank_truncate(m, q.value ^ 0x80000000u), 0,
                 cm.bits, ((cm + term) - cm).bits, ((cm - term) - cm).bits};
})";
      auto device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("Metal required");
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [device
          newComputePipelineStateWithFunction:[library newFunctionWithName:@"test_ranking_rounding"]
                                        error:&error];
      if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
      std::vector<Query> queries;
      std::vector<float> maximums{0.f,      0x1p-149f, 0x1p-126f, 0x1p-23f,
                                  .5f,      1.f,       100.f,     std::numeric_limits<float>::max(),
                                  INFINITY, NAN};
      for (int exponent = -148; exponent < 128; ++exponent) {
        float x = std::ldexp(1.f, exponent);
        maximums.insert(maximums.end(), {x, std::nextafter(x, 0.f), std::nextafter(x, INFINITY)});
      }
      for (float maximum : maximums)
        for (std::uint32_t rows : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 4097u, 65537u,
                                   4194303u, 4194304u, 4194305u, 0xffffffffu})
          for (float fraction : {0.f, 0x1p-24f, .25f, .5f, 1.f})
            queries.push_back({Cast<std::uint32_t>(maximum), rows,
                               Cast<std::uint32_t>(maximum * fraction), 0, double(maximum), 1.,
                               double(maximum) * fraction});
      // Deliberately hit different positive/negative truncation grids at M=2^e.
      queries.push_back(
          {Cast<std::uint32_t>(.25f), 1, Cast<std::uint32_t>(0x1p-26f), 0, .25, 1., 0x1p-55});
      std::mt19937_64 rng(23843);
      for (int i = 0; i < 100000; ++i)
        queries.push_back({std::uint32_t(rng()) & 0x7fffffffu, std::uint32_t(rng()),
                           std::uint32_t(rng()), 0, Cast<double>(rng() & 0x7fffffffffffffffULL),
                           i % 7 ? Cast<double>(rng() & 0x7fffffffffffffffULL) : 0.,
                           Cast<double>(rng())});
      auto input = [device newBufferWithBytes:queries.data()
                                       length:queries.size() * sizeof(Query)
                                      options:MTLResourceStorageModeShared];
      auto output = [device newBufferWithLength:queries.size() * sizeof(Result)
                                        options:MTLResourceStorageModeShared];
      auto queue = [device newCommandQueue];
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:output offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(queries.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      auto actual = static_cast<Result const*>(output.contents);
      std::size_t asymmetric = 0, double_asymmetric = 0;
      for (std::size_t i = 0; i < queries.size(); ++i) {
        auto q = queries[i];
        float maximum = Cast<float>(q.maximum), value = Cast<float>(q.value);
        float factor = cuda_reference::RankingRounding(float(q.rows) * maximum, q.rows);
        double cf =
            cuda_reference::RankingRounding(double(float(q.rows)) * (q.cost / q.minimum), q.rows);
        float expected[] = {factor, cuda_reference::RankingTruncate(factor, value),
                            cuda_reference::RankingTruncate(factor, -value)};
        double de[] = {cf, cuda_reference::RankingTruncate(cf, q.term),
                       cuda_reference::RankingTruncate(cf, -q.term)};
        float av[] = {Cast<float>(actual[i].factor), Cast<float>(actual[i].positive),
                      Cast<float>(actual[i].negative)};
        double da[] = {actual[i].cost_factor, actual[i].cost_positive, actual[i].cost_negative};
        for (int f = 0; f < 3; ++f) {
          bool ok = (std::isnan(av[f]) && std::isnan(expected[f])) ||
                    Cast<std::uint32_t>(av[f]) == Cast<std::uint32_t>(expected[f]);
          bool dok = (std::isnan(da[f]) && std::isnan(de[f])) ||
                     Cast<std::uint64_t>(da[f]) == Cast<std::uint64_t>(de[f]);
          if (!ok || !dok) {
            std::fprintf(stderr,
                         "case=%zu rows=%u max=%a value=%a field=%d float=%a expected=%a double=%a "
                         "expected=%a\n",
                         i, q.rows, maximum, value, f, av[f], expected[f], da[f], de[f]);
            throw std::runtime_error("Rounding/truncation source mismatch");
          }
        }
        asymmetric += std::isfinite(expected[1]) && expected[1] != -expected[2];
        double_asymmetric += std::isfinite(de[1]) && de[1] != -de[2];
      }
      if (!asymmetric || !double_asymmetric)
        throw std::runtime_error("Asymmetric rounding not exercised");
      std::printf("PASS %zu exact rounding cases / %zu float and double fields; %zu float and %zu "
                  "double asymmetric terms\n",
                  queries.size(), queries.size() * 6, asymmetric, double_asymmetric);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
