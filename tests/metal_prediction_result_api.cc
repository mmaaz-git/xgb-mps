// Public prediction pointers must be ready, learner-owned and independently
// snapshotable without materializing a host mirror of a Metal result.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/api_entry.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_output.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/proxy_dmatrix.h"
#include "xgboost/c_api.h"
#include "xgboost/learner.h"

using namespace xgboost;
namespace metal = common::metal;
std::string current;
std::size_t cases = 0, checks = 0;
void Check(bool ok, char const* message) {
  ++checks;
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void API(int code) { Check(code == 0, XGBGetLastError()); }
void Compare(common::Span<float const> actual, common::Span<float const> expected) {
  Check(actual.size() == expected.size(), "Result size");
  for (std::size_t i = 0; i < actual.size(); ++i)
    Check(actual[i] == expected[i] ||
              (std::isfinite(actual[i]) && std::isfinite(expected[i]) &&
               std::abs(actual[i] - expected[i]) <= 2e-6f * (1.f + std::abs(expected[i]))),
          "CPU prediction differs");
}
auto Data(std::vector<float> const& x, std::size_t rows) {
  data::DenseAdapter adapter{x.data(), rows, 3};
  return std::shared_ptr<DMatrix>{DMatrix::Create(&adapter, NAN, 2)};
}
std::string Config(int type, int end, bool strict) {
  return "{\"type\":" + std::to_string(type) +
         ",\"iteration_begin\":0,\"iteration_end\":" + std::to_string(end) +
         ",\"training\":false,\"strict_shape\":" + (strict ? "true" : "false") +
         ",\"missing\":NaN}";
}
struct Result {
  float const* values{nullptr};
  std::vector<bst_ulong> shape;
  std::size_t Size() const {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>{});
  }
  common::Span<float const> Span() const { return {values, Size()}; }
  std::vector<float> Copy() const {
    if (!Size()) return {};
    return {values, values + Size()};
  }
};
Result Predict(Learner* learner, std::shared_ptr<DMatrix>* data, int type, int end, bool legacy,
               bool strict) {
  Result result;
  if (legacy) {
    int masks[]{0, 1, 4, 12, 16, 24, 2};
    bst_ulong length = 0;
    API(XGBoosterPredict(learner, data, masks[type], end, 0, &length, &result.values));
    result.shape = {length};
  } else {
    bst_ulong const* shape = nullptr;
    bst_ulong dims = 0;
    auto config = Config(type, end, strict);
    API(XGBoosterPredictFromDMatrix(learner, data, config.c_str(), &shape, &dims, &result.values));
    result.shape.assign(shape, shape + dims);
  }
  Check(learner->GetThreadLocal().prediction_entry.predictions.Size() == result.Size(),
        "Shape exceeds prediction storage");
  return result;
}
struct Snapshot {
  void* handle{nullptr};
  std::vector<float> expected;
  Snapshot(Result const& result) : expected{result.Copy()} {
    auto before = metal::GetStatistics();
    API(XGMetalOutputCreate(result.values, result.shape.data(), result.shape.size(), &handle));
    auto after = metal::GetStatistics();
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Snapshot read host mirror");
    Check(after.output_snapshot_host_copy_bytes == before.output_snapshot_host_copy_bytes,
          "Snapshot copied through host");
    Check(after.output_snapshot_shared_input_bytes - before.output_snapshot_shared_input_bytes ==
              expected.size() * sizeof(float),
          "Snapshot did not bind shared result");
  }
  ~Snapshot() {
    if (handle) XGMetalOutputFree(handle);
  }
  void Verify() const {
    std::vector<float> values(expected.size() + 2, -971.f);
    API(XGMetalOutputRead(handle, values.data() + 1, expected.size()));
    Check(values.front() == -971.f && values.back() == -971.f, "Readback guards");
    Check(expected.empty() || !std::memcmp(values.data() + 1, expected.data(), expected.size() * 4),
          "Snapshot changed after cache/model mutation");
  }
};
void Inspect(Learner* learner, Result const& result, bool resident) {
  auto const& storage = learner->GetThreadLocal().prediction_entry.predictions;
  Check(storage.Size() == result.Size(), "TLS result size");
  if (resident) {
    Check(storage.Device().IsMPS() && storage.DeviceCanRead() && !storage.HostCanRead(),
          "Public result acquired a host mirror");
    Check(result.values == (storage.Empty() ? nullptr : storage.ConstDevicePointer()),
          "Public pointer does not refer to shared Metal contents");
  } else {
    Check(storage.HostCanRead() && !storage.DeviceCanRead(), "Host fallback was uploaded");
    Check(result.values == (storage.Empty() ? nullptr : storage.ConstHostSpan().data()),
          "Host fallback pointer");
  }
  Check(result.Size() != 0 || result.values == nullptr, "Empty result must be null");
  auto copy = result.Copy();
  bst_ulong features = 0;
  API(XGBoosterGetNumFeature(learner, &features));
  Check(features == 3, "Feature metadata");
  Compare(result.Span(), copy);  // Unrelated API queries do not invalidate prediction storage.
}
void Run(std::string const& kind, bool margin) {
  current = kind;
  bool vector = kind == "vector", linear = kind == "linear";
  bool multi = kind == "multi:softmax" || kind == "multi:softprob";
  unsigned groups = vector || multi || kind == "reg:expectileerror" ? 3 : 1;
  std::vector<float> train(65 * 3);
  for (std::size_t i = 0; i < train.size(); ++i) train[i] = float(int(i % 17) - 8) / 8;
  auto dtrain = Data(train, 65);
  dtrain->Info().labels.Reshape(65, vector ? 3 : 1);
  auto& y = dtrain->Info().labels.Data()->HostVector();
  for (std::size_t i = 0; i < y.size(); ++i)
    y[i] = multi                       ? float(i % 3)
           : kind == "binary:logistic" ? float(i % 2)
                                       : train[i % train.size()];
  std::unique_ptr<Learner> cpu{Learner::Create({dtrain})}, gpu{Learner::Create({})};
  Args params{{"device", "cpu"}, {"nthread", "2"}, {"base_score", "0.5"}, {"max_depth", "2"}};
  if (vector) params.emplace_back("multi_strategy", "multi_output_tree");
  if (multi) params.emplace_back("num_class", "3");
  if (kind == "reg:expectileerror") params.emplace_back("expectile_alpha", "[0.1,0.5,0.9]");
  if (kind == "dart") params.emplace_back("rate_drop", "0.4");
  if (linear) {
    params.emplace_back("booster", "gblinear");
    params.emplace_back("updater", "coord_descent");
  }
  if (multi || kind == "binary:logistic" || kind == "reg:expectileerror")
    params.emplace_back("objective", kind);
  cpu->SetParams(params);
  for (int i = 0; i < 3; ++i) cpu->UpdateOneIter(i, dtrain);
  Json model{Object{}};
  cpu->SaveModel(&model);
  gpu->LoadModel(model);
  gpu->SetParams({{"device", "mps"}, {"nthread", "2"}});
  gpu->Configure();
  std::vector<std::unique_ptr<Snapshot>> snapshots;
  // Empty results follow nonempty results, exercising retained allocation capacity.
  for (std::size_t rows : {33, 0, 1}) {
    std::vector<float> x(std::max<std::size_t>(rows * 3, 1));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = i % 11 ? train[i % train.size()] : NAN;
    auto data = Data(x, rows);
    if (margin) {
      auto& m = data->Info().base_margin_;
      m.Reshape(rows, groups);
      m.SetDevice(DeviceOrd::MPS());
      m.Data()->DevicePointer();
      m.Data()->Fill(.25f);
    }
    for (int type : {0, 1, 2, 3, 4, 5, 6}) {
      // The upstream linear interaction producer has a different size from the
      // common shaped API; it is not a CUDA producer. Exercise linear raw and
      // contribution host fallbacks without reading that inconsistent shape.
      if ((vector && (type == 3 || type == 5)) || (linear && type >= 4)) continue;
      bool resident = !linear && type != 3 && type != 5;
      // Linear's identity value transform explicitly selects Metal; its raw and
      // attribution producers remain host code, just as in upstream CUDA builds.
      resident = resident || (linear && type == 0);
      int end = linear ? 0 : 2;
      for (bool legacy : {false, true}) {
        current = kind + "/" + std::to_string(margin) + "/" + std::to_string(rows) + "/" +
                  std::to_string(type) + "/" + std::to_string(legacy);
        auto expected_result = Predict(cpu.get(), &data, type, end, legacy, !(vector && type == 6));
        auto expected = expected_result.Copy();
        auto shape = expected_result.shape;
        auto before = metal::GetStatistics();
        auto result = Predict(gpu.get(), &data, type, end, legacy, !(vector && type == 6));
        if (resident && !linear)
          Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
                "Resident prediction made a host copy");
        Check(shape == result.shape, "CPU result shape");
        Inspect(gpu.get(), result, resident);
        Compare(result.Span(), expected);
        snapshots.push_back(std::make_unique<Snapshot>(result));
        ++cases;
      }
    }
    if (linear) continue;
    for (bool native : {false, true}) {
      std::vector<bst_ulong> input_shape{rows, 3};
      auto source = metal::MakeOutput(x.data(), input_shape);
      std::ostringstream desc;
      if (native) {
        desc << "{\"device_type\":8,\"device_id\":0,\"buffer\":"
             << reinterpret_cast<std::uintptr_t>(source->View().data)
             << ",\"byte_offset\":0,\"shape\":[" << rows
             << ",3],\"strides\":[3,1],\"dtype_code\":2,\"dtype_bits\":32,\"dtype_lanes\":1}";
      } else {
        desc << "{\"data\":[" << reinterpret_cast<std::uintptr_t>(x.data()) << ",false],\"shape\":["
             << rows << ",3],\"typestr\":\"<f4\",\"version\":3}";
      }
      std::shared_ptr<DMatrix> proxy = std::make_shared<data::DMatrixProxy>();
      proxy->Info().base_margin_.Reshape(rows, margin ? groups : 0);
      proxy->Info().base_margin_.Data()->Copy(*data->Info().base_margin_.Data());
      for (int type : {0, 1}) {
        current = kind + "/inplace/" + std::to_string(rows) + "/" + std::to_string(native) + "/" +
                  std::to_string(type);
        auto expected = Predict(cpu.get(), &data, type, 2, false, true).Copy();
        auto config = Config(type, 2, true);
        Result result;
        bst_ulong const* shape = nullptr;
        bst_ulong dims = 0;
        auto before = metal::GetStatistics();
        auto fn = native ? XGBoosterPredictFromMetal : XGBoosterPredictFromDense;
        API(fn(gpu.get(), desc.str().c_str(), config.c_str(), &proxy, &shape, &dims,
               &result.values));
        result.shape.assign(shape, shape + dims);
        Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
              "Inplace result made a host copy");
        Inspect(gpu.get(), result, true);
        Compare(result.Span(), expected);
        snapshots.push_back(std::make_unique<Snapshot>(result));
        ++cases;
      }
    }
  }
  gpu.reset();
  cpu.reset();
  for (auto const& snapshot : snapshots) snapshot->Verify();
  std::printf("Checked %s margins=%d: %zu cases / %zu checks\n", kind.c_str(), margin, cases,
              checks);
  std::fflush(stdout);
}
int main() {
  try {
    for (auto kind : {"scalar", "binary:logistic", "multi:softprob", "multi:softmax", "vector",
                      "dart", "reg:expectileerror", "linear"})
      for (bool margin : {false, true}) Run(kind, margin);
    auto stats = metal::GetStatistics();
    Check(stats.output_live_buffers == 0 && stats.output_live_exports == 0 &&
              stats.vector_live_buffers == 0 && stats.vector_capacity_bytes == 0,
          "Output/vector lifetime leak");
    std::printf("PASS %zu public prediction result cases / %zu checks\n", cases, checks);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
