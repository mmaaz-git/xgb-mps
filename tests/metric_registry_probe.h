// A host-only registry signature; no device/context initialization or GPU calls.
#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <dmlc/registry.h>
#include "xgboost/metric.h"
inline void MetricRegistrySignature() {
  std::vector<std::string> registry;
  for (auto* r : dmlc::Registry<xgboost::MetricReg>::List())
    registry.push_back(r->name+" "+r->body.target_type().name());
  std::sort(registry.begin(),registry.end());
  for (auto const& r : registry) std::printf("%s\n",r.c_str());
}
