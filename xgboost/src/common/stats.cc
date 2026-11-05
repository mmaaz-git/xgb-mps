/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include "stats.h"

#include <cstddef>  // std::size_t
#include <cstdint>  // std::int32_t
#include <numeric>  // std::accumulate

#include "../collective/aggregator.h"    // for GlobalMax, GlobalSum
#include "linalg_op.h"                   // for Matrix
#include "optional_weight.h"             // OptionalWeights
#include "threading_utils.h"             // ParallelFor, MemStackAllocator
#include "transform_iterator.h"          // MakeIndexTransformIter
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // Tensor, UnravelIndex, Apply
#include "xgboost/logging.h"             // CHECK_EQ

#if defined(XGBOOST_USE_METAL)
#include "metal_quantile.h"
#include "metal_reduction.h"
#include "metal_vector.h"
#endif

namespace xgboost::common {
void Median(Context const* ctx, linalg::Matrix<float> const& t,
            HostDeviceVector<float> const& weights, linalg::Tensor<float, 1>* out) {
#if defined(XGBOOST_USE_METAL)
  if (ctx->IsMPS()) {
    out->SetDevice(DeviceOrd::MPS());
    out->Reshape(t.Shape(1));
    std::vector<metal::QuantileRequest> requests;
    for (std::uint32_t i = 0; i < t.Shape(1); ++i) requests.push_back({i, 0, 0.5});
    metal::Quantiles(t.View(DeviceOrd::MPS()), metal::ConstDeviceSpan(weights), requests,
                     out->View(DeviceOrd::MPS()).Values());
    return;
  }
#endif
  if (ctx->IsCUDA()) {
    weights.SetDevice(ctx->Device());
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->Device());
    cuda_impl::Median(ctx, t_v, opt_weights, out);
    return;
  }

  auto opt_weights = OptionalWeights(weights.ConstHostSpan());
  auto t_v = t.HostView();
  out->Reshape(t.Shape(1));
  auto h_out = out->HostView();
  for (std::size_t i{0}; i < t.Shape(1); ++i) {
    auto ti_v = t_v.Slice(linalg::All(), i);
    auto iter = linalg::cbegin(ti_v);
    float q{0};
    if (opt_weights.Empty()) {
      q = common::Quantile(ctx, 0.5, iter, iter + ti_v.Size());
    } else {
      CHECK_NE(t_v.Shape(1), 0);
      auto w_it = common::MakeIndexTransformIter([&](std::size_t i) { return opt_weights[i]; });
      q = common::WeightedQuantile(ctx, 0.5, iter, iter + ti_v.Size(), w_it);
    }
    h_out(i) = q;
  }
}

void Mean(Context const* ctx, linalg::VectorView<float const> v, linalg::Vector<float>* out) {
#if defined(XGBOOST_USE_METAL)
  if (ctx->IsMPS()) {
    out->SetDevice(DeviceOrd::MPS());
    out->Reshape(1);
    std::size_t shape[2]{v.Size(), 1}, strides[2]{v.Stride(0), 1};
    linalg::MatrixView<float const> matrix{v.Values(), shape, strides, v.Device()};
    metal::Mean(metal::MeanKind::kFloat32, matrix, {}, out->View(DeviceOrd::MPS()), ctx);
    return;
  }
#endif
  out->SetDevice(ctx->Device());
  out->Reshape(1);
  if (ctx->IsCUDA()) {
    cuda_impl::Mean(ctx, v, out->View(ctx->Device()));
  } else if (ctx->IsSycl()) {
    sycl_impl::Mean(ctx, v, out->View(ctx->Device()));
  } else {
    auto h_v = v;
    float n = v.Size();
    MemStackAllocator<float, DefaultMaxThreads()> tloc(ctx->Threads(), 0.0f);
    ParallelFor(v.Size(), ctx->Threads(),
                [&](auto i) { tloc[omp_get_thread_num()] += h_v(i) / n; });
    auto ret = std::accumulate(tloc.cbegin(), tloc.cend(), .0f);
    out->HostView()(0) = ret;
  }
}

void SampleMean(Context const* ctx, linalg::Matrix<float> const& v, linalg::Vector<float>* out) {
#if defined(XGBOOST_USE_METAL)
  if (ctx->IsMPS()) {
    *out = linalg::Vector<float>({std::max<std::size_t>(v.Shape(1), 1)}, DeviceOrd::MPS());
    metal::Mean(metal::MeanKind::kSample, v.View(DeviceOrd::MPS()), {}, out->View(DeviceOrd::MPS()),
                ctx);
    return;
  }
#endif
  *out = linalg::Zeros<float>(ctx, std::max(v.Shape(1), decltype(v.Shape(1)){1}));
  if (!ctx->IsCUDA()) {
    auto h_v = v.HostView();
    CHECK(h_v.CContiguous());
    std::int64_t n_samples = v.Shape(0);
    SafeColl(collective::GlobalSum(ctx, linalg::MakeVec(&n_samples, 1)));
    auto n_columns = v.Shape(1);
    auto h_out = out->HostView();

    auto n_rows_f64 = static_cast<double>(n_samples);
    for (std::size_t j = 0; j < n_columns; ++j) {
      MemStackAllocator<double, DefaultMaxThreads()> mean_tloc(ctx->Threads(), 0.0);
      ParallelFor(v.Shape(0), ctx->Threads(),
                  [&](auto i) { mean_tloc[omp_get_thread_num()] += (h_v(i, j) / n_rows_f64); });
      auto mean = std::accumulate(mean_tloc.cbegin(), mean_tloc.cend(), 0.0);
      h_out(j) = mean;
    }
    SafeColl(collective::GlobalSum(ctx, h_out));
  } else {
    auto d_v = v.View(ctx->Device());
    auto d_out = out->View(ctx->Device());
    cuda_impl::SampleMean(ctx, d_v, d_out);
  }
}

void WeightedSampleMean(Context const* ctx, linalg::Matrix<float> const& v,
                        HostDeviceVector<float> const& w, linalg::Vector<float>* out) {
  CHECK_EQ(v.Shape(0), w.Size());
#if defined(XGBOOST_USE_METAL)
  if (ctx->IsMPS()) {
    *out = linalg::Vector<float>({std::max<std::size_t>(v.Shape(1), 1)}, DeviceOrd::MPS());
    metal::Mean(metal::MeanKind::kWeightedSample, v.View(DeviceOrd::MPS()),
                metal::ConstDeviceSpan(w), out->View(DeviceOrd::MPS()), ctx);
    return;
  }
#endif
  *out = linalg::Zeros<float>(ctx, std::max(v.Shape(1), decltype(v.Shape(1)){1}));
  if (!ctx->IsCUDA()) {
    auto h_v = v.HostView();
    auto h_w = w.ConstHostSpan();
    auto sum_w = std::accumulate(h_w.data(), h_w.data() + h_w.size(), 0.0);
    SafeColl(collective::GlobalSum(ctx, linalg::MakeVec(&sum_w, 1)));
    CHECK_GT(sum_w, 0.0) << "weights must contain at least one non-zero value.";
    auto h_out = out->HostView();
    for (std::size_t j = 0; j < v.Shape(1); ++j) {
      MemStackAllocator<double, DefaultMaxThreads()> mean_tloc(ctx->Threads(), 0.0);
      ParallelFor(v.Shape(0), ctx->Threads(),
                  [&](auto i) { mean_tloc[omp_get_thread_num()] += (h_v(i, j) / sum_w * h_w(i)); });
      auto mean = std::accumulate(mean_tloc.cbegin(), mean_tloc.cend(), 0.0);
      h_out(j) = mean;
    }
    SafeColl(collective::GlobalSum(ctx, h_out));
  } else {
    auto d_v = v.View(ctx->Device());
    w.SetDevice(ctx->Device());
    auto d_w = w.ConstDeviceSpan();
    auto d_out = out->View(ctx->Device());
    cuda_impl::WeightedSampleMean(ctx, d_v, d_w, d_out);
  }
}
void OptionalWeightedSampleMean(Context const* ctx, linalg::Matrix<float> const& v,
                                HostDeviceVector<float> const& w, linalg::Vector<float>* out) {
  auto weighted = collective::GlobalMax(ctx, static_cast<std::int32_t>(!w.Empty()));
  if (!weighted) {
    SampleMean(ctx, v, out);
  } else if (w.Empty() && v.Shape(0) != 0) {
    // A populated rank without explicit weights still contributes unit weights.
    // Keep these implicit weights on the same device as the mean calculation.
    auto device = ctx->Device();
#if defined(XGBOOST_USE_METAL)
    if (ctx->IsMPS()) device = DeviceOrd::MPS();
#endif
    HostDeviceVector<float> unit_weights(v.Shape(0), 1.0f, device);
    WeightedSampleMean(ctx, v, unit_weights, out);
  } else {
    // Empty ranks participate in the weighted denominator/output reductions too.
    WeightedSampleMean(ctx, v, w, out);
  }
}
}  // namespace xgboost::common
