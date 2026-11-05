// Compile unchanged CUDA method bodies with isolated registration and symbols.
// Explicit member-pointer instantiations give this test adapter access to owned
// state; the source class, access specifiers and algorithms are not rewritten.
// Compile this translation unit with hidden symbol/inline visibility.
#pragma once
#include "xgboost/linear_updater.h"
#include <dmlc/registry.h>
#pragma push_macro("XGBOOST_REGISTER_LINEAR_UPDATER")
#pragma push_macro("DMLC_REGISTRY_FILE_TAG")
#undef XGBOOST_REGISTER_LINEAR_UPDATER
#undef DMLC_REGISTRY_FILE_TAG
#define XGBOOST_REGISTER_LINEAR_UPDATER(Id, Name) \
  static ::xgboost::LinearUpdaterReg linear_probe_storage_##Id; \
  static auto& linear_probe_config_##Id = linear_probe_storage_##Id
#define DMLC_REGISTRY_FILE_TAG(Tag)
#include "../xgboost/src/linear/updater_gpu_coordinate.cu"
#pragma pop_macro("DMLC_REGISTRY_FILE_TAG")
#pragma pop_macro("XGBOOST_REGISTER_LINEAR_UPDATER")

namespace cuda_linear_trace {
using Source = xgboost::linear::GPUCoordinateUpdater;
template <typename Tag, typename Tag::Type Member> struct StateAccess {
  friend typename Tag::Type StateMember(Tag) { return Member; }
};
struct RowOffsets {
  using Type = std::vector<std::size_t> Source::*;
  friend Type StateMember(RowOffsets);
};
struct Entries {
  using Type = ::dh::device_vector<xgboost::Entry> Source::*;
  friend Type StateMember(Entries);
};
struct Gradients {
  using Type = ::dh::caching_device_vector<xgboost::GradientPair> Source::*;
  friend Type StateMember(Gradients);
};
struct Rows {
  using Type = std::size_t Source::*;
  friend Type StateMember(Rows);
};
struct Upload {
  using Type = void (Source::*)(std::vector<xgboost::GradientPair> const&);
  friend Type StateMember(Upload);
};
template struct StateAccess<RowOffsets, &Source::row_ptr_>;
template struct StateAccess<Entries, &Source::data_>;
template struct StateAccess<Gradients, &Source::gpair_>;
template struct StateAccess<Rows, &Source::num_row_>;
template struct StateAccess<Upload, &Source::UpdateGpair>;

class Probe : public Source {
 public:
  explicit Probe(xgboost::Context const* ctx) { this->ctx_ = ctx; }
  auto& Offsets() { return this->*StateMember(RowOffsets{}); }
  auto& Data() { return this->*StateMember(Entries{}); }
  auto& Residuals() { return this->*StateMember(Gradients{}); }
  void SetRows(std::size_t rows) { this->*StateMember(Rows{}) = rows; }
  void SetGradients(std::vector<xgboost::GradientPair> const& pairs) {
    (this->*StateMember(Upload{}))(pairs);
  }
};
}  // namespace cuda_linear_trace
