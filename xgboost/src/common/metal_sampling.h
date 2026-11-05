/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>
#include <cstdint>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"
namespace xgboost::common::metal {
// CUDA column sampling: modern Philox weighted priorities, Thrust Feistel shuffle
// for unweighted samples. Optional diagnostics expose GPU uniforms/keys or the
// full unweighted permutation of input positions. Empty selections dispatch nothing.
// Bind addressable Metal spans directly, including offsets. Sampling publishes
// from scratch only after validation, so output may alias either input. Host
// diagnostic spans remain supported; unborrowable memory uses counted copies.
void InitFeatureSet(Span<bst_feature_t> features);
void SampleFeatures(Span<bst_feature_t const> features, Span<float const> weights,
                    std::uint32_t seed, Span<bst_feature_t> selected, Span<float> uniforms = {},
                    Span<float> keys = {}, Span<bst_feature_t> permutation = {});
void UniformSample(linalg::MatrixView<GradientPair> gradients, float probability,
                   std::uint64_t seed, std::uint32_t threads);
// Compute probabilities from the original split gradients, then sample/rescale
// output gradients. Different target counts support reduced-gradient replay.
// When 0 < sample_rows < rows, optional diagnostics read back GPU-computed norms
// and the selected threshold. Otherwise diagnostics are left untouched.
void GradientSample(linalg::MatrixView<GradientPair const> split_gradients,
                    linalg::MatrixView<GradientPair> output, std::size_t sample_rows,
                    std::uint64_t seed, std::uint32_t threads, Span<float> norms = {},
                    float* threshold = nullptr);
}  // namespace xgboost::common::metal
