// Host-only registry/factory signature: no device/context initialization.
#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <dmlc/registry.h>
#include "xgboost/linear_updater.h"
inline void LinearRegistrySignature() {
  std::vector<std::string> entries;
  for (auto* entry : dmlc::Registry<xgboost::LinearUpdaterReg>::List())
    entries.push_back(entry->name+" "+entry->body.target_type().name());
  std::sort(entries.begin(),entries.end());
  for (auto const& entry : entries) std::printf("%s\n",entry.c_str());
}
