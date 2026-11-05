/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>
#include <cstdint>

#include "../collective/comm.h"
#include "../data/array_interface.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Synchronous operations over shared Metal allocations (or guarded host views).
// The caller retains all allocations until return and completes foreign writes
// before entry. Network receives write directly into owned shared allocations.
void CollectiveCopy(Span<std::int8_t const> input, Span<std::int8_t> output);
void CollectiveReduce(Span<std::int8_t const> input, Span<std::int8_t> output,
                      ArrayInterfaceHandler::Type type, collective::Op op, std::size_t ranks,
                      bool initialize);
struct CollectiveStatistics {
  std::uint64_t reduce_dispatches, copy_dispatches, shared_input_bytes, shared_output_bytes,
      staging_bytes, gpu_time_ns;
};
CollectiveStatistics GetCollectiveStatistics();
}  // namespace xgboost::common::metal
