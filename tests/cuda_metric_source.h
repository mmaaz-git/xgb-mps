// Test-only access to the unchanged metric policies and reduction consumers.
// Compile this translation unit with hidden symbol/inline visibility so these
// definitions cannot interpose on libxgboost. Registration expressions populate
// private, unregistered entries; no production registry or file tags are changed.
#pragma once
#include "xgboost/metric.h"
#include <dmlc/registry.h>
#pragma push_macro("XGBOOST_REGISTER_METRIC")
#pragma push_macro("DMLC_REGISTRY_FILE_TAG")
#undef XGBOOST_REGISTER_METRIC
#undef DMLC_REGISTRY_FILE_TAG
#define XGBOOST_REGISTER_METRIC(Id, Name) \
  static ::xgboost::MetricReg probe_storage_##Id; \
  static auto& probe_config_##Id = probe_storage_##Id
#define DMLC_REGISTRY_FILE_TAG(Tag)
#include "../xgboost/src/metric/elementwise_metric.cu"
#include "../xgboost/src/metric/multiclass_metric.cu"
#include "../xgboost/src/metric/survival_metric.cu"
#pragma pop_macro("DMLC_REGISTRY_FILE_TAG")
#pragma pop_macro("XGBOOST_REGISTER_METRIC")
