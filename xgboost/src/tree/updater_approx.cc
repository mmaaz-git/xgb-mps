/**
 * Copyright 2021-2026, XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#include <algorithm>  // for max, transform, fill_n
#include <cstddef>    // for size_t
#include <map>        // for map
#include <memory>     // for allocator, unique_ptr, make_shared, make_unique
#include <utility>    // for move
#include <vector>     // for vector

#include "../collective/aggregator.h"        // for GlobalSum
#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/hist_util.h"             // for HistogramCuts
#include "../common/random.h"                // for ColumnSampler
#include "../common/timer.h"                 // for Monitor
#if defined(XGBOOST_USE_METAL)
#include "../common/metal_gradient.h"
#include "../common/metal_reduction.h"
#endif
#include "../data/gradient_index.h"          // for GHistIndexMatrix
#include "common_row_partitioner.h"          // for CommonRowPartitioner
#include "dmlc/registry.h"                   // for DMLC_REGISTRY_FILE_TAG
#include "driver.h"                          // for Driver
#include "hist/evaluate_splits.h"            // for HistEvaluator
#include "hist/expand_entry.h"               // for CPUExpandEntry
#include "hist/hist_param.h"                 // for HistMakerTrainParam
#include "hist/histogram.h"                  // for MultiHistogramBuilder
#include "hist/sampler.h"                    // for Sampler
#include "param.h"                           // for GradStats, TrainParam
#include "xgboost/base.h"                    // for Args, GradientPair, bst_node_t, bst_bin_t
#include "xgboost/context.h"                 // for Context
#include "xgboost/data.h"                    // for DMatrix, BatchSet, BatchIterator, MetaInfo
#include "xgboost/gradient.h"                // for GradientContainer
#include "xgboost/host_device_vector.h"      // for HostDeviceVector
#include "xgboost/json.h"                    // for Object, Json, FromJson, ToJson, get
#include "xgboost/linalg.h"                  // for Matrix, MakeTensorView, Empty, MatrixView
#include "xgboost/logging.h"                 // for LogCheck_EQ, CHECK_EQ, CHECK
#include "xgboost/span.h"                    // for Span
#include "xgboost/task.h"                    // for ObjInfo
#include "xgboost/tree_model.h"              // for RegTree, RTreeNodeStat
#include "xgboost/tree_updater.h"            // for TreeUpdater, TreeUpdaterReg, XGBOOST_REGISTE...

namespace xgboost::tree {

DMLC_REGISTRY_FILE_TAG(updater_approx);

namespace {
// Return the BatchParam used by DMatrix.
auto BatchSpec(TrainParam const &p, common::Span<float> hess, ObjInfo const task) {
  return BatchParam{p.max_bin, hess, !task.const_hess};
}

auto BatchSpec(TrainParam const &p, common::Span<float> hess) {
  return BatchParam{p.max_bin, hess, false};
}
}  // anonymous namespace

class GlobalApproxBuilder {
#if defined(XGBOOST_USE_METAL)
  std::shared_ptr<common::metal::QuantizedGradients const> quantized_;
#endif

 protected:
  TrainParam const *param_;
  HistMakerTrainParam const *hist_param_{nullptr};
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  HistEvaluator evaluator_;
  MultiHistogramBuilder histogram_builder_;
  Context const *ctx_;
  ObjInfo const *const task_;

  std::vector<CommonRowPartitioner> partitioner_;
  // Pointer to last updated tree, used for update prediction cache.
  RegTree *p_last_tree_{nullptr};
  common::Monitor *monitor_;
  size_t n_batches_{0};
  // Cache for histogram cuts.
  common::HistogramCuts feature_values_{0};

 public:
#if defined(XGBOOST_USE_METAL)
  void SetQuantized(std::shared_ptr<common::metal::QuantizedGradients const> gradients) {
    quantized_ = std::move(gradients);
  }
#endif

  void InitData(DMatrix* p_fmat, RegTree const* p_tree, common::Span<float> hess, bool regenerate) {
    monitor_->Start(__func__);

    n_batches_ = 0;
    bst_bin_t n_total_bins = 0;
    partitioner_.clear();
    // Generating the GHistIndexMatrix is quite slow, is there a way to speed it up?
    for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(
             ctx_, regenerate ? BatchSpec(*param_, hess, *task_) : BatchSpec(*param_, hess))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
        feature_values_ = page.cut;
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      partitioner_.emplace_back(this->ctx_, page.Size(), page.base_rowid);
      n_batches_++;
    }

    histogram_builder_.Reset(ctx_, n_total_bins, p_tree->NumTargets(), BatchSpec(*param_, hess),
                             collective::IsDistributed(), hist_param_);
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) histogram_builder_.BindQuantized(quantized_);
#endif
    monitor_->Stop(__func__);
  }

  CPUExpandEntry InitRoot(DMatrix *p_fmat, std::vector<GradientPair> const &gpair,
                          common::Span<float> hess, RegTree *p_tree) {
    monitor_->Start(__func__);
    CPUExpandEntry best;
    best.nid = RegTree::kRoot;
    best.depth = 0;
    GradStats root_sum;
    GradientPairInt64 quantized_root;
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      GradientPairPrecise sum;
      quantized_->Sum(linalg::MakeVec(&sum, 1), {&quantized_root, 1});
      root_sum = GradStats{sum};
    } else
#endif
    {
      for (auto const& g : gpair) root_sum.Add(g);
    }
    auto rc =
        collective::GlobalSum(ctx_, linalg::MakeVec(reinterpret_cast<double *>(&root_sum), 2));
    collective::SafeColl(rc);

    std::vector<CPUExpandEntry> nodes{best};
    this->histogram_builder_.BuildRootHist(p_fmat, p_tree->HostScView(), partitioner_,
                                           linalg::MakeTensorView(ctx_, gpair, gpair.size(), 1),
                                           best, BatchSpec(*param_, hess));

    auto weight = evaluator_.InitRoot(root_sum, &quantized_root);
    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_->learning_rate * weight);

    auto const &histograms = histogram_builder_.Histogram(0);
    auto ft = p_fmat->Info().feature_types.ConstHostSpan();
    evaluator_.EvaluateSplits(histograms, feature_values_, ft, &nodes);
    monitor_->Stop(__func__);

    return nodes.front();
  }

  void BuildHistogram(DMatrix *p_fmat, RegTree *p_tree,
                      std::vector<CPUExpandEntry> const &valid_candidates,
                      std::vector<GradientPair> const &gpair, common::Span<float> hess) {
    monitor_->Start(__func__);
    this->histogram_builder_.BuildHistLeftRight(
        ctx_, p_fmat, p_tree->HostScView(), partitioner_, valid_candidates,
        linalg::MakeTensorView(ctx_, gpair, gpair.size(), 1), BatchSpec(*param_, hess));
    monitor_->Stop(__func__);
  }

  void LeafPartition(RegTree const &tree, common::Span<float const> hess,
                     std::vector<bst_node_t> *p_out_position) {
    monitor_->Start(__func__);
    p_out_position->resize(hess.size());
    for (auto const &part : partitioner_) {
      part.LeafPartition(ctx_, tree.HostScView(), hess,
                         common::Span{p_out_position->data(), p_out_position->size()});
    }
    monitor_->Stop(__func__);
  }

 public:
  explicit GlobalApproxBuilder(TrainParam const *param, HistMakerTrainParam const *hist_param,
                               MetaInfo const &info, Context const *ctx,
                               std::shared_ptr<common::ColumnSampler> column_sampler,
                               ObjInfo const *task, common::Monitor *monitor)
      : param_{param},
        hist_param_{hist_param},
        col_sampler_{std::move(column_sampler)},
        evaluator_{ctx, param_, info, col_sampler_},
        ctx_{ctx},
        task_{task},
        monitor_{monitor} {}

  void UpdateTree(DMatrix* p_fmat, std::vector<GradientPair> const& gpair, common::Span<float> hess,
                  RegTree* p_tree, HostDeviceVector<bst_node_t>* p_out_position,
                  bool regenerate = true) {
    CHECK(!p_tree->IsMultiTarget()) << "approx" << MTNotImplemented();
    p_last_tree_ = p_tree;
    this->InitData(p_fmat, p_tree, hess, regenerate);

    Driver<CPUExpandEntry> driver(*param_);
    auto &tree = *p_tree;
    driver.Push({this->InitRoot(p_fmat, gpair, hess, p_tree)});
    auto expand_set = driver.Pop();

    /**
     * Note for update position
     * Root:
     *   Not applied: No need to update position as initialization has got all the rows ordered.
     *   Applied: Update position is run on applied nodes so the rows are partitioned.
     * Non-root:
     *   Not applied: That node is root of the subtree, same rule as root.
     *   Applied: Ditto
     */

    while (!expand_set.empty()) {
      // candidates that can be further splited.
      std::vector<CPUExpandEntry> valid_candidates;
      // candidates that can be applied.
      std::vector<CPUExpandEntry> applied;
      for (auto const &candidate : expand_set) {
        evaluator_.ApplyTreeSplit(candidate, p_tree);
        applied.push_back(candidate);
        if (driver.IsChildValid(candidate)) {
          valid_candidates.emplace_back(candidate);
        }
      }

      monitor_->Start("UpdatePosition");
      size_t page_id = 0;
      for (auto const &page :
           p_fmat->GetBatches<GHistIndexMatrix>(ctx_, BatchSpec(*param_, hess))) {
        partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree->HostScView());
        page_id++;
      }
      monitor_->Stop("UpdatePosition");

      std::vector<CPUExpandEntry> best_splits;
      if (!valid_candidates.empty()) {
        this->BuildHistogram(p_fmat, p_tree, valid_candidates, gpair, hess);
        for (auto const &candidate : valid_candidates) {
          int left_child_nidx = tree[candidate.nid].LeftChild();
          int right_child_nidx = tree[candidate.nid].RightChild();
          CPUExpandEntry l_best{left_child_nidx, tree.GetDepth(left_child_nidx)};
          CPUExpandEntry r_best{right_child_nidx, tree.GetDepth(right_child_nidx)};
          best_splits.push_back(l_best);
          best_splits.push_back(r_best);
        }
        auto const &histograms = histogram_builder_.Histogram(0);
        auto ft = p_fmat->Info().feature_types.ConstHostSpan();
        monitor_->Start("EvaluateSplits");
        evaluator_.EvaluateSplits(histograms, feature_values_, ft, &best_splits);
        monitor_->Stop("EvaluateSplits");
      }
      driver.Push(best_splits.begin(), best_splits.end());
      expand_set = driver.Pop();
    }

    auto &h_position = p_out_position->HostVector();
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      h_position.resize(gpair.size());
      auto gradients = linalg::MakeTensorView(ctx_, gpair, gpair.size(), 1);
      for (auto const& part : partitioner_) {
        part.LeafPartition(ctx_, tree.HostScView(), gradients, common::Span{h_position});
      }
    } else
#endif
    {
      this->LeafPartition(tree, hess, &h_position);
    }
  }
};

/**
 * \brief Implementation for the approx tree method.  It constructs quantile for every
 *        iteration.
 */
class GlobalApproxUpdater : public TreeUpdater {
  common::Monitor monitor_;
  // specializations for different histogram precision.
  std::unique_ptr<GlobalApproxBuilder> pimpl_;
  // pointer to the last DMatrix, used for update prediction cache.
  DMatrix *cached_{nullptr};
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  ObjInfo const *task_;
  HistMakerTrainParam hist_param_;

 public:
  explicit GlobalApproxUpdater(Context const *ctx, ObjInfo const *task)
      : TreeUpdater(ctx), column_sampler_{std::make_shared<common::ColumnSampler>()}, task_{task} {
    monitor_.Init(__func__);
  }

  void Configure(Args const &args) override { hist_param_.UpdateAllowUnknown(args); }
  void LoadConfig(Json const &in) override {
    auto const &config = get<Object const>(in);
    FromJson(config.at("hist_train_param"), &hist_param_);
  }
  void SaveConfig(Json *p_out) const override {
    auto &out = *p_out;
    out["hist_train_param"] = ToJson(hist_param_);
  }

  void InitData(TrainParam const &param, linalg::Matrix<GradientPair> const *gpair,
                linalg::Matrix<GradientPair> *sampled) {
    *sampled = linalg::Empty<GradientPair>(ctx_, gpair->Size(), 1);
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      CHECK_EQ(gpair->Shape(1), 1);
      common::metal::CopyGradients(gpair->HostView(), sampled->HostView());
    } else
#endif
    {
      auto in = gpair->HostView().Values();
      std::copy(in.data(), in.data() + in.size(), sampled->HostView().Values().data());
    }
    cpu_impl::Sampler sampler{param};
    sampler.Sample(ctx_, sampled->HostView());
  }

  [[nodiscard]] char const *Name() const override { return "grow_histmaker"; }

  void Update(TrainParam const *param, GradientContainer *in_gpair, DMatrix *m,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree *> &trees) override {
    CHECK(hist_param_.GetInitialised());
    auto gpair = in_gpair->FullGradOnly();
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      // Match GPUGlobalApproxMaker::Update: cuts use the original Hessians, before
      // any sampling, and are regenerated at most once per updater invocation.
      std::vector<float> hess(gpair->Size());
      common::metal::ExtractHessians(gpair->HostView(),
                                     linalg::MakeTensorView(ctx_, hess, hess.size(), 1));
      for (auto const& page :
           m->GetBatches<GHistIndexMatrix>(ctx_, BatchSpec(*param, hess, *task_))) {
        (void)page;  // Consume all external-memory pages before resetting the iterator.
      }
      cached_ = m;
      for (std::size_t t = 0; t < trees.size(); ++t) {
        // GPUHistMakerDevice::Reset samples and initializes column sampling for
        // EACH tree, in that order. CPU approx shares both across a forest.
        // Weighted sketch and gradient-based sampling still need CUDA alignment.
        auto sampled = linalg::Empty<GradientPair>(ctx_, gpair->Size(), 1);
        common::metal::CopyGradients(gpair->HostView(), sampled.HostView());
        cpu_impl::Sampler sampler{*param};
        sampler.Sample(ctx_, sampled.HostView());
        pimpl_ = std::make_unique<GlobalApproxBuilder>(param, &hist_param_, m->Info(), ctx_,
                                                       column_sampler_, task_, &monitor_);
        pimpl_->SetQuantized(sampler.Quantized());
        pimpl_->UpdateTree(m, sampled.Data()->ConstHostVector(), hess, trees[t], &out_position[t],
                           false);
        hist_param_.CheckTreesSynchronized(ctx_, trees[t]);
      }
      return;
    }
#endif
    pimpl_ = std::make_unique<GlobalApproxBuilder>(param, &hist_param_, m->Info(), ctx_,
                                                   column_sampler_, task_, &monitor_);
    linalg::Matrix<GradientPair> h_gpair;
    // CPU approx intentionally sketches the sampled Hessians.
    InitData(*param, gpair, &h_gpair);
    std::vector<float> hess(h_gpair.Size());
    auto const &s_gpair = h_gpair.Data()->ConstHostVector();
    std::transform(s_gpair.begin(), s_gpair.end(), hess.begin(),
                   [](auto g) { return g.GetHess(); });

    cached_ = m;

    std::size_t t_idx = 0;
    for (auto p_tree : trees) {
      this->pimpl_->UpdateTree(m, s_gpair, hess, p_tree, &out_position[t_idx]);
      hist_param_.CheckTreesSynchronized(ctx_, p_tree);
      ++t_idx;
    }
  }
};

DMLC_REGISTRY_FILE_TAG(grow_histmaker);

XGBOOST_REGISTER_TREE_UPDATER(GlobalHistMaker, "grow_histmaker")
    .describe(
        "Tree constructor that uses approximate histogram construction "
        "for each node.")
    .set_body([](Context const *ctx, ObjInfo const *task) {
      return new GlobalApproxUpdater(ctx, task);
    });
}  // namespace xgboost::tree
