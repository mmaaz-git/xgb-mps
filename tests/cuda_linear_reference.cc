// Test-only host updater derived from updater_gpu_coordinate.cu. No Metal calls.
// CUDA uses CUB's architecture-dependent FP32 tree; this oracle uses an adjacent
// pairwise FP32 tree, so comparisons allow the existing public numerical tolerance.
// Host selectors/coordinate formulas are shared upstream code, as in CUDA.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "../xgboost/src/c_api/c_api_utils.h"
#include "../xgboost/src/linear/coordinate_common.h"
#include "xgboost/json.h"
#include "xgboost/linear_updater.h"

namespace xgboost::linear {
class CudaSourceReference : public LinearUpdater {
  LinearTrainParam param_;
  CoordinateParam coordinate_;
  std::unique_ptr<FeatureSelector> selector_;

 public:
  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
    coordinate_.UpdateAllowUnknown(args);
    selector_.reset(FeatureSelector::Create(param_.feature_selector));
  }
  void LoadConfig(Json const& in) override {
    FromJson(in["linear_train_param"], &param_);
    FromJson(in["coordinate_param"], &coordinate_);
    selector_.reset(FeatureSelector::Create(param_.feature_selector));
  }
  void SaveConfig(Json* out) const override {
    *out = Object{};
    (*out)["linear_train_param"] = ToJson(param_);
    (*out)["coordinate_param"] = ToJson(coordinate_);
  }
  void Update(linalg::Matrix<GradientPair>* input, DMatrix* matrix, gbm::GBLinearModel* model,
              double sum_weight) override {
    CHECK(matrix->SingleColBlock());
    param_.DenormalizePenalties(sum_weight);
    auto const& original = input->Data()->ConstHostVector();
    auto residual = original;
    auto batches = matrix->GetBatches<CSCPage>(ctx_);
    auto const& batch = *batches.begin();
    auto page = batch.GetView();
    auto groups = model->learner_model_param->num_output_group;
    auto rows = matrix->Info().num_row_;
    auto gradient = [&](int feature, unsigned group) {
      std::vector<GradientPair> terms;
      if (feature < 0) {
        for (std::size_t row = 0; row < rows; ++row)
          terms.push_back(residual[row * groups + group]);
      } else {
        for (auto entry : page[feature]) {
          auto pair = residual[entry.index * groups + group];
          terms.emplace_back(pair.GetGrad() * entry.fvalue,
                             (pair.GetHess() * entry.fvalue) * entry.fvalue);
        }
      }
      // Every term has FP32 products and every addition has FP32 rounding.
      for (std::size_t n = terms.size(); n > 1; n = (n + 1) / 2) {
        for (std::size_t i = 0; i < n; i += 2)
          terms[i / 2] = terms[i] + (i + 1 < n ? terms[i + 1] : GradientPair{});
      }
      return terms.empty() ? GradientPair{} : GradientPair{} + terms.front();
    };
    auto update = [&](int feature, unsigned group, float delta) {
      if (feature < 0) {
        if (delta == 0) return;
        for (std::size_t row = 0; row < rows; ++row) {
          auto& pair = residual[row * groups + group];
          pair += GradientPair{pair.GetHess() * delta, 0};
        }
      } else {
        // Defined for distinct row IDs, as CUDA's concurrent column writes require.
        for (auto entry : page[feature]) {
          auto& pair = residual[entry.index * groups + group];
          pair += GradientPair{(pair.GetHess() * delta) * entry.fvalue, 0};
        }
      }
    };
    for (unsigned group = 0; group < groups; ++group) {
      auto sum = gradient(-1, group);
      auto delta = static_cast<float>(param_.learning_rate *
                                      CoordinateDeltaBias(sum.GetGrad(), sum.GetHess()));
      model->Bias()[group] += delta;
      update(-1, group, delta);
    }
    // CUDA passes the unmodified input to both selector methods, even after bias.
    selector_->Setup(ctx_, *model, original, matrix, param_.reg_alpha_denorm,
                     param_.reg_lambda_denorm, coordinate_.top_k);
    for (unsigned group = 0; group < groups; ++group) {
      for (unsigned i = 0; i < model->learner_model_param->num_feature; ++i) {
        int feature = selector_->NextFeature(ctx_, i, *model, group, original, matrix,
                                             param_.reg_alpha_denorm, param_.reg_lambda_denorm);
        if (feature < 0) break;
        auto sum = gradient(feature, group);
        float& weight = (*model)[feature][group];
        auto delta =
            static_cast<float>(param_.learning_rate *
                               CoordinateDelta(sum.GetGrad(), sum.GetHess(), weight,
                                               param_.reg_alpha_denorm, param_.reg_lambda_denorm));
        weight += delta;
        update(feature, group, delta);
      }
    }
  }
};
XGBOOST_REGISTER_LINEAR_UPDATER(CudaSourceReference, "test_cuda_source_reference")
    .describe("Test-only host CUDA-source oracle; not an NVIDIA execution comparison")
    .set_body([] { return new CudaSourceReference; });
}  // namespace xgboost::linear

// Test the production updater's input ownership directly. Python's boost API
// packs user arrays first, so checking the Python arrays would miss a copy-back.
extern "C" int TestCudaLinearInputSnapshot(DMatrixHandle handle, char const* selector,
                                           unsigned targets) {
  using namespace xgboost;
  try {
    auto matrix = CastDMatrixHandle(handle);
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "1"}});
    std::size_t shape[]{1};
    LearnerModelParam param(matrix->Info().num_col_,
                            linalg::Vector<float>{{0.f}, shape, DeviceOrd::CPU()}, targets, 1,
                            MultiStrategy::kOneOutputPerTree);
    gbm::GBLinearModel model(&param);
    model.LazyInitModel();
    std::size_t gradient_shape[]{static_cast<std::size_t>(matrix->Info().num_row_), targets};
    linalg::Matrix<GradientPair> input(gradient_shape, DeviceOrd::CPU());
    auto& values = input.Data()->HostVector();
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = GradientPair(float(int(i % 13) - 7) / 4, i % 3 ? 1.f : -.25f);
    auto original = values;
    auto updater = std::unique_ptr<LinearUpdater>(LinearUpdater::Create("coord_descent", &ctx));
    updater->Configure({{"eta", "0.5"}, {"feature_selector", selector}});
    updater->Update(&input, matrix.get(), &model, matrix->Info().num_row_);
    auto const& final = input.Data()->ConstHostVector();
    CHECK(std::any_of(model.weight.begin(), model.weight.end(), [](float v) { return v != 0; }));
    return std::memcmp(original.data(), final.data(), final.size() * sizeof(GradientPair)) == 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "Input snapshot check: %s\n", e.what());
    return -1;
  }
}
