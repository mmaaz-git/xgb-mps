/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include "numeric.h"

#include <type_traits>  // std::is_same_v

#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

#if defined(XGBOOST_USE_METAL)
#include "metal_reduction.h"
#endif

namespace xgboost {
namespace common {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  // The additive identity also avoids requesting zero OpenMP threads on CPU.
  if (values.Empty()) return 0.0;
#if defined(XGBOOST_USE_METAL)
  if (ctx->IsMPS()) return metal::Sum(values.ConstHostSpan());
#endif
  if (ctx->IsCUDA()) {
    return cuda_impl::Reduce(ctx, values);
  } else {
    auto const& h_values = values.ConstHostVector();
    auto result = cpu_impl::Reduce(ctx, h_values.cbegin(), h_values.cend(), 0.0);
    static_assert(std::is_same_v<decltype(result), double>);
    return result;
  }
}
}  // namespace common
}  // namespace xgboost
