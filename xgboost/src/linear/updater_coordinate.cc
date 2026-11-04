/**
 * Copyright 2018-2025, XGBoost Contributors
 * \author Rory Mitchell
 */

#include <xgboost/linear_updater.h>
#include "./param.h"
#include "../common/timer.h"
#include "coordinate_common.h"
#include "xgboost/json.h"
#if defined(XGBOOST_USE_METAL)
#include "../common/metal_linear.h"
#endif

namespace xgboost::linear {

DMLC_REGISTER_PARAMETER(CoordinateParam);
DMLC_REGISTRY_FILE_TAG(updater_coordinate);

// training parameter
/**
 * \class CoordinateUpdater
 *
 * \brief Coordinate descent algorithm that updates one feature per iteration
 */

class CoordinateUpdater : public LinearUpdater {
 public:
  // set training parameter
  void Configure(Args const& args) override {
    const std::vector<std::pair<std::string, std::string> > rest {
      tparam_.UpdateAllowUnknown(args)
    };
    cparam_.UpdateAllowUnknown(rest);
    selector_.reset(FeatureSelector::Create(tparam_.feature_selector));
    monitor_.Init("CoordinateUpdater");
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("linear_train_param"), &tparam_);
    FromJson(config.at("coordinate_param"), &cparam_);
  }
  void SaveConfig(Json *p_out) const override {
    LOG(DEBUG) << "Save config for CPU updater.";
    auto &out = *p_out;
    out["linear_train_param"] = ToJson(tparam_);
    out["coordinate_param"] = ToJson(cparam_);
  }

  void Update(linalg::Matrix<GradientPair> *in_gpair, DMatrix *p_fmat, gbm::GBLinearModel *model,
              double sum_instance_weight) override {
    auto gpair = in_gpair->Data();
    tparam_.DenormalizePenalties(sum_instance_weight);
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      UpdateMetal(in_gpair, p_fmat, model);
      return;
    }
#endif
    auto ngroup = model->learner_model_param->num_output_group;
    // update bias
    for (decltype(ngroup) group_idx = 0; group_idx < ngroup; ++group_idx) {
      auto grad = GetBiasGradientParallel(group_idx, ngroup, gpair->ConstHostVector(), p_fmat,
                                          ctx_->Threads());
      auto dbias =
          static_cast<float>(tparam_.learning_rate * CoordinateDeltaBias(grad.first, grad.second));
      model->Bias()[group_idx] += dbias;
      UpdateBiasResidualParallel(ctx_, group_idx, ngroup, dbias, &gpair->HostVector(), p_fmat);
    }
    // prepare for updating the weights
    selector_->Setup(ctx_, *model, gpair->ConstHostVector(), p_fmat, tparam_.reg_alpha_denorm,
                     tparam_.reg_lambda_denorm, cparam_.top_k);
    // update weights
    for (decltype(ngroup) group_idx = 0; group_idx < ngroup; ++group_idx) {
      for (unsigned i = 0U; i < model->learner_model_param->num_feature; i++) {
        int fidx =
            selector_->NextFeature(ctx_, i, *model, group_idx, gpair->ConstHostVector(), p_fmat,
                                   tparam_.reg_alpha_denorm, tparam_.reg_lambda_denorm);
        if (fidx < 0) break;
        this->UpdateFeature(fidx, group_idx, &gpair->HostVector(), p_fmat, model);
      }
    }
    monitor_.Stop("UpdateFeature");
  }

  void UpdateFeature(int fidx, int group_idx, std::vector<GradientPair> *in_gpair, DMatrix *p_fmat,
                     gbm::GBLinearModel *model) {
    const int ngroup = model->learner_model_param->num_output_group;
    bst_float &w = (*model)[fidx][group_idx];
    auto gradient = GetGradientParallel(ctx_, group_idx, ngroup, fidx,
                                        *in_gpair, p_fmat);
    auto dw = static_cast<float>(
        tparam_.learning_rate *
        CoordinateDelta(gradient.first, gradient.second, w, tparam_.reg_alpha_denorm,
                        tparam_.reg_lambda_denorm));
    w += dw;
    UpdateResidualParallel(ctx_, fidx, group_idx, ngroup, dw, in_gpair, p_fmat);
  }

 private:
#if defined(XGBOOST_USE_METAL)
  void UpdateMetal(linalg::Matrix<GradientPair>* in_gpair, DMatrix* data,
                   gbm::GBLinearModel* model) {
    CHECK(data->SingleColBlock()) << "Metal coordinate descent requires a single CSC block.";
    CHECK(in_gpair->HostView().CContiguous());
    auto groups = model->learner_model_param->num_output_group;
    auto rows = static_cast<std::size_t>(data->Info().num_row_);
    auto batches = data->GetBatches<CSCPage>(ctx_);
    auto const& page = *batches.begin();
    auto state = common::metal::MakeLinearState(rows, groups, page.offset.ConstHostSpan(),
                                                page.data.ConstHostSpan(),
                                                in_gpair->Data()->ConstHostSpan());
    auto host = in_gpair->Data()->HostSpan();
    for (bst_target_t group = 0; group < groups; ++group) {
      auto sum = state->Gradient(-1, group);
      auto delta = static_cast<float>(tparam_.learning_rate *
                                      CoordinateDeltaBias(sum.GetGrad(), sum.GetHess()));
      model->Bias()[group] += delta;
      state->UpdateResidual(-1, group, delta);
    }
    // Host feature selectors are also used by CUDA. Unlike a stale snapshot,
    // this view includes completed GPU bias/residual changes as the CPU expects.
    state->CopyGradients(host);
    selector_->Setup(ctx_, *model, in_gpair->Data()->ConstHostVector(), data,
                     tparam_.reg_alpha_denorm, tparam_.reg_lambda_denorm, cparam_.top_k);
    for (bst_target_t group = 0; group < groups; ++group) {
      for (bst_feature_t i = 0; i < model->learner_model_param->num_feature; ++i) {
        if (tparam_.feature_selector == kGreedy) state->CopyGradients(host);
        auto feature =
            selector_->NextFeature(ctx_, i, *model, group, in_gpair->Data()->ConstHostVector(),
                                   data, tparam_.reg_alpha_denorm, tparam_.reg_lambda_denorm);
        if (feature < 0) break;
        auto sum = state->Gradient(feature, group);
        auto& weight = (*model)[feature][group];
        auto delta = static_cast<float>(tparam_.learning_rate *
                                        CoordinateDelta(sum.GetGrad(), sum.GetHess(), weight,
                                                        tparam_.reg_alpha_denorm,
                                                        tparam_.reg_lambda_denorm));
        weight += delta;
        state->UpdateResidual(feature, group, delta);
      }
    }
    state->CopyGradients(host);
  }
#endif

  CoordinateParam cparam_;
  // training parameter
  LinearTrainParam tparam_;
  std::unique_ptr<FeatureSelector> selector_;
  common::Monitor monitor_;
};

XGBOOST_REGISTER_LINEAR_UPDATER(CoordinateUpdater, "coord_descent")
    .describe("Update linear model according to coordinate descent algorithm.")
    .set_body([]() { return new CoordinateUpdater(); });
}  // namespace xgboost::linear
