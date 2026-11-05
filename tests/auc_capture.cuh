// Test-only readback of values produced by the original CUDA AUC expressions.
// Every captured vector has an explicit presence flag: skipped/empty source
// stages must not be represented by fabricated zero-filled observations.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "xgboost/span.h"

namespace xgboost { class Context; }

namespace auc_capture {
struct Observation {
  bool trace{false};
  bool terms_present{false}, prefixes_present{false}, totals_present{false};
  bool curves_present{false}, normalization_present{false};
  std::vector<std::uint64_t> sorted_indices;
  std::vector<double> terms, prefixes, totals, curves, normalization;
};
inline thread_local Observation* sink = nullptr;
class Scope {
  Observation* previous_;
 public:
  explicit Scope(Observation* current) : previous_{sink} { sink = current; }
  ~Scope() { sink = previous_; }
  Scope(Scope const&) = delete;
  Scope& operator=(Scope const&) = delete;
};
inline void Cuda(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template <typename T>
std::vector<std::remove_const_t<T>> Read(xgboost::common::Span<T> input) {
  // Some original AUC launches use the default stream. Device synchronization
  // observes all original producers, rather than assuming they share a stream.
  Cuda(cudaDeviceSynchronize());
  std::vector<std::remove_const_t<T>> output(input.size());
  if (!output.empty())
    Cuda(cudaMemcpy(output.data(), input.data(), input.size_bytes(), cudaMemcpyDeviceToHost));
  return output;
}
template <typename Pair>
std::vector<double> Pairs(xgboost::common::Span<Pair> values, bool reverse = false) {
  auto input = Read(values);
  std::vector<double> result;
  result.reserve(2 * input.size());
  for (auto const& pair : input) {
    result.push_back(reverse ? pair.second : pair.first);
    result.push_back(reverse ? pair.first : pair.second);
  }
  return result;
}
template <typename Pair, typename Index>
void Terms(xgboost::Context const*, xgboost::common::Span<Pair> values,
           xgboost::common::Span<Index> sorted) {
  if (!sink || !sink->trace) return;
  if (sink->terms_present) throw std::runtime_error("Repeated AUC terms producer");
  sink->terms = Pairs(values);
  auto indices = Read(sorted);
  sink->sorted_indices.assign(indices.begin(), indices.end());
  sink->terms_present = true;
}
template <typename Pair>
void Prefixes(xgboost::Context const*, xgboost::common::Span<Pair> values) {
  if (!sink || !sink->trace) return;
  if (sink->prefixes_present) throw std::runtime_error("Repeated AUC prefix producer");
  sink->prefixes = Pairs(values);
  sink->prefixes_present = true;
}
inline void BinaryTotals(double fp, double tp) {
  if (!sink) return;
  if (sink->totals_present) throw std::runtime_error("Repeated AUC totals producer");
  sink->totals = {fp, tp};
  sink->totals_present = true;
}
template <typename Pair>
void MultiTotals(xgboost::Context const*, xgboost::common::Span<Pair> values) {
  if (!sink) return;
  if (sink->totals_present) throw std::runtime_error("Repeated AUC totals producer");
  sink->totals = Pairs(values, true);  // Source stores (positive, negative).
  sink->totals_present = true;
}
template <typename Pair>
void Curves(xgboost::Context const*, xgboost::common::Span<Pair> prefixes,
            xgboost::common::Span<double> areas, std::size_t rows) {
  if (!sink) return;
  if (sink->curves_present || !rows || prefixes.size() != rows * areas.size())
    throw std::runtime_error("Invalid AUC curve observation boundary");
  auto points = Read(prefixes);
  auto integral = Read(areas);
  sink->curves.reserve(3 * integral.size());
  for (std::size_t t = 0; t < integral.size(); ++t) {
    auto last = points[(t + 1) * rows - 1];
    sink->curves.insert(sink->curves.end(), {last.first, last.second, integral[t]});
  }
  sink->curves_present = true;
}
inline void Normalization(xgboost::Context const*, xgboost::common::Span<double> areas,
                           xgboost::common::Span<double> weights,
                           xgboost::common::Span<double> integrals) {
  if (!sink) return;
  if (sink->normalization_present || areas.size() != weights.size() || areas.size() != integrals.size())
    throw std::runtime_error("Invalid AUC normalization observation boundary");
  auto a = Read(areas), w = Read(weights), v = Read(integrals);
  sink->normalization.reserve(3 * a.size());
  for (std::size_t t = 0; t < a.size(); ++t)
    sink->normalization.insert(sink->normalization.end(), {a[t], w[t], v[t]});
  sink->normalization_present = true;
}
}  // namespace auc_capture
