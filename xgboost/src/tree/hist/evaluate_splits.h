/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#define XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_

#include <algorithm>  // for copy
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <memory>     // for shared_ptr
#include <numeric>    // for accumulate
#include <utility>    // for move
#include <vector>     // for vector

#include "../../common/categorical.h"  // for CatBitField
#include "../../common/hist_util.h"    // for GHistRow, HistogramCuts
#include "../../common/linalg_op.h"    // for cbegin, cend, begin
#include "../../common/random.h"       // for ColumnSampler
#include "../constraints.h"            // for FeatureInteractionConstraintHost
#include "../param.h"                  // for TrainParam
#include "../split_evaluator.h"        // for TreeEvaluator
#include "../tree_view.h"              // for MultiTargetTreeView
#include "expand_entry.h"              // for MultiExpandEntry
#include "hist_cache.h"                // for BoundedHistCollection
#include "xgboost/base.h"              // for bst_node_t, bst_target_t, bst_feature_t
#include "xgboost/context.h"           // for COntext
#include "xgboost/linalg.h"            // for Constants, Vector

#if defined(XGBOOST_USE_METAL)
#include "../../common/metal_split.h"
#endif

namespace xgboost::tree {
class HistEvaluator {
 private:
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain{0.0f};
#if defined(XGBOOST_USE_METAL)
    // Persist winners by node ID: loss-guide entries can outlive split scratch.
    GradientPairInt64 quantized, quantized_left, quantized_right;
#endif
  };

 private:
  Context const *ctx_;
  TrainParam const *param_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  TreeEvaluator tree_evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  std::vector<NodeEntry> snode_;
#if defined(XGBOOST_USE_METAL)
  std::unique_ptr<common::metal::SplitWorkspace> metal_workspace_;
#endif

  // if sum of statistics for non-missing values in the node
  // is equal to sum of statistics for all values:
  // then - there are no missing values
  // else - there are missing values
  bool static SplitContainsMissingValues(const GradStats e, const NodeEntry &snode) {
    if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
      return false;
    } else {
      return true;
    }
  }

  /**
   * \brief Use learned direction with one-hot split. Other implementations (LGB) create a
   *        pseudo-category for missing value but here we just do a complete scan to avoid
   *        making specialized histogram bin.
   */
  void EnumerateOneHot(common::HistogramCuts const &cut, common::ConstGHistRow hist,
                       bst_feature_t fidx, bst_node_t nidx,
                       TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                       SplitEntry *p_best) const {
    const std::vector<uint32_t> &cut_ptr = cut.Ptrs();
    const std::vector<bst_float> &cut_val = cut.Values();

    bst_bin_t ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
    bst_bin_t iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
    bst_bin_t n_bins = iend - ibegin;

    GradStats left_sum;
    GradStats right_sum;
    // best split so far
    SplitEntry best;
    best.is_cat = false;  // marker for whether it's updated or not.

    auto f_hist = hist.subspan(cut_ptr[fidx], n_bins);
    auto feature_sum = GradStats{
        std::accumulate(f_hist.data(), f_hist.data() + f_hist.size(), GradientPairPrecise{})};
    GradStats missing;
    auto const &parent = snode_[nidx];
    missing.SetSubstract(parent.stats, feature_sum);

    for (bst_bin_t i = ibegin; i != iend; i += 1) {
      auto split_pt = cut_val[i];

      // missing on left (treat missing as other categories)
      right_sum = GradStats{hist[i]};
      left_sum.SetSubstract(parent.stats, right_sum);
      auto missing_left_chg = static_cast<float>(
          evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum}, GradStats{right_sum}) -
          parent.root_gain);
      best.Update(missing_left_chg, fidx, split_pt, true, true, left_sum, right_sum);

      // missing on right (treat missing as chosen category)
      right_sum.Add(missing);
      left_sum.SetSubstract(parent.stats, right_sum);
      auto missing_right_chg = static_cast<float>(
          evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum}, GradStats{right_sum}) -
          parent.root_gain);
      best.Update(missing_right_chg, fidx, split_pt, false, true, left_sum, right_sum);
    }

    if (best.is_cat) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins + 1);
      best.cat_bits.resize(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      cat_bits.Set(best.split_value);
    }

    p_best->Update(best);
  }

  /**
   * @brief Enumerate with partition-based splits.
   *
   * The implementation is different from LightGBM. Firstly we don't have a
   * pseudo-cateogry for missing value, instead of we make 2 complete scans over the
   * histogram. Secondly, both scan directions generate splits in the same
   * order. Following table depicts the scan process, square bracket means the gradient in
   * missing values is resided on that partition:
   *
   *   | Forward  | Backward |
   *   |----------+----------|
   *   | [BCDE] A | E [ABCD] |
   *   | [CDE] AB | DE [ABC] |
   *   | [DE] ABC | CDE [AB] |
   *   | [E] ABCD | BCDE [A] |
   */
  template <int d_step>
  void EnumeratePart(common::HistogramCuts const &cut, common::Span<size_t const> sorted_idx,
                     common::ConstGHistRow hist, bst_feature_t fidx, bst_node_t nidx,
                     TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                     SplitEntry *p_best) {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");

    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();
    auto const &parent = snode_[nidx];

    bst_bin_t f_begin = cut_ptr[fidx];
    bst_bin_t f_end = cut_ptr[fidx + 1];
    bst_bin_t n_bins_feature{f_end - f_begin};
    auto n_bins = std::min(param_->max_cat_threshold, n_bins_feature);

    // statistics on both sides of split
    GradStats left_sum;
    GradStats right_sum;
    // best split so far
    SplitEntry best;

    auto f_hist = hist.subspan(f_begin, n_bins_feature);
    bst_bin_t it_begin, it_end;
    if (d_step > 0) {
      it_begin = f_begin;
      it_end = it_begin + n_bins - 1;
    } else {
      it_begin = f_end - 1;
      it_end = it_begin - n_bins + 1;
    }

    bst_bin_t best_thresh{-1};
    for (bst_bin_t i = it_begin; i != it_end; i += d_step) {
      auto j = i - f_begin;  // index local to current feature
      if (d_step == 1) {
        right_sum.Add(f_hist[sorted_idx[j]].GetGrad(), f_hist[sorted_idx[j]].GetHess());
        left_sum.SetSubstract(parent.stats, right_sum);  // missing on left
      } else {
        left_sum.Add(f_hist[sorted_idx[j]].GetGrad(), f_hist[sorted_idx[j]].GetHess());
        right_sum.SetSubstract(parent.stats, left_sum);  // missing on right
      }
      auto loss_chg =
          evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum}, GradStats{right_sum}) -
          parent.root_gain;
      // We don't have a numeric split point, nan here is a dummy split.
      if (best.Update(loss_chg, fidx, std::numeric_limits<float>::quiet_NaN(), d_step == 1, true,
                      left_sum, right_sum)) {
        best_thresh = i;
      }
    }

    if (best_thresh != -1) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins_feature);
      best.cat_bits = decltype(best.cat_bits)(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      bst_bin_t partition = d_step == 1 ? (best_thresh - it_begin + 1) : (best_thresh - f_begin);
      CHECK_GT(partition, 0);
      std::for_each(sorted_idx.begin(), sorted_idx.begin() + partition, [&](std::size_t c) {
        auto cat = cut_val[c + f_begin];
        cat_bits.Set(cat);
      });
    }

    p_best->Update(best);
  }

  // Enumerate/Scan the split values of specific feature
  // Returns the sum of gradients corresponding to the data points that contains
  // a non-missing value for the particular feature fid.
  template <int d_step>
  GradStats EnumerateSplit(common::HistogramCuts const &cut, common::ConstGHistRow hist,
                           bst_feature_t fidx, bst_node_t nidx,
                           TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                           SplitEntry *p_best) const {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");

    // aliases
    const std::vector<uint32_t> &cut_ptr = cut.Ptrs();
    const std::vector<bst_float> &cut_val = cut.Values();
    auto const &parent = snode_[nidx];

    // statistics on both sides of split
    GradStats left_sum;
    GradStats right_sum;
    // best split so far
    SplitEntry best;

    // bin boundaries
    CHECK_LE(cut_ptr[fidx], static_cast<uint32_t>(std::numeric_limits<bst_bin_t>::max()));
    CHECK_LE(cut_ptr[fidx + 1], static_cast<uint32_t>(std::numeric_limits<bst_bin_t>::max()));
    // ibegin, iend: smallest/largest cut points for feature fid use int to allow for
    // value -1
    bst_bin_t ibegin, iend;
    if (d_step > 0) {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
      iend = static_cast<bst_bin_t>(cut_ptr.at(fidx + 1));
    } else {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx + 1]) - 1;
      iend = static_cast<bst_bin_t>(cut_ptr[fidx]) - 1;
    }

    for (bst_bin_t i = ibegin; i != iend; i += d_step) {
      // start working
      // try to find a split
      left_sum.Add(hist[i].GetGrad(), hist[i].GetHess());
      right_sum.SetSubstract(parent.stats, left_sum);
      bst_float loss_chg;
      bst_float split_pt;
      if (d_step > 0) {
        // forward enumeration: split at right bound of each bin
        loss_chg =
            static_cast<float>(evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum},
                                                       GradStats{right_sum}) -
                               parent.root_gain);
        split_pt = cut_val[i];  // not used for partition based
        best.Update(loss_chg, fidx, split_pt, d_step == -1, false, left_sum, right_sum);
      } else {
        // backward enumeration: split at left bound of each bin
        loss_chg =
            static_cast<float>(evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{right_sum},
                                                       GradStats{left_sum}) -
                               parent.root_gain);
        split_pt = common::HistogramCuts::NumericBinLowerBound(cut_ptr, cut_val, fidx, i);
        best.Update(loss_chg, fidx, split_pt, d_step == -1, false, right_sum, left_sum);
      }
    }

    p_best->Update(best);
    return left_sum;
  }

 public:
  void EvaluateSplits(const BoundedHistCollection &hist, common::HistogramCuts const &cut,
                      common::Span<FeatureType const> feature_types,
                      std::vector<CPUExpandEntry> *p_entries) {
    auto n_threads = ctx_->Threads();
    auto &entries = *p_entries;
    // All nodes are on the same level, so we can store the shared ptr.
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(entries.size());
    for (size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      features[nidx_in_set] = column_sampler_->GetFeatureSet(ctx_, entries[nidx_in_set].depth);
    }
    CHECK(!features.empty());
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      if (!metal_workspace_) metal_workspace_ = common::metal::MakeSplitWorkspace();
      auto evaluator = tree_evaluator_.GetEvaluator();
      std::vector<common::metal::QuantizedSplitTask> tasks;
      std::size_t bins = 0;
      std::vector<common::metal::HistogramInput> inputs{{hist.MetalBuffer(), {}}};
      std::vector<std::pair<std::size_t, bst_feature_t>> mapping;
      auto const &ptr = cut.Ptrs();
      for (std::size_t i = 0; i < entries.size(); ++i) {
        auto nid = entries[i].nid;
        for (auto f : features[i]->ConstHostSpan()) {
          if (!interaction_constraints_.Query(nid, f)) continue;
          auto n = ptr[f + 1] - ptr[f];
          if (!n) continue;
          CHECK_LE(bins + n, std::numeric_limits<std::uint32_t>::max());
          bool categorical = common::IsCat(feature_types, f);
          std::uint32_t kind =
              categorical ? (common::UseOneHot(n, param_->max_cat_to_onehot) ? 1 : 2) : 0;
          tasks.push_back({static_cast<std::uint32_t>(bins), static_cast<std::uint32_t>(bins + n),
                           kind, static_cast<std::uint32_t>(param_->max_cat_threshold),
                           snode_[nid].quantized,
                           evaluator.has_constraint ? evaluator.lower[nid]
                                                    : -std::numeric_limits<float>::infinity(),
                           evaluator.has_constraint ? evaluator.upper[nid]
                                                    : std::numeric_limits<float>::infinity(),
                           evaluator.has_constraint ? evaluator.constraints[f] : 0});
          inputs[0].segments.push_back(
              {hist.Offset(nid) + ptr[f], static_cast<std::uint32_t>(bins), n, 1, 0});
          bins += n;
          mapping.emplace_back(i, f);
        }
      }
      common::metal::SplitParameters parameters{param_->reg_lambda, param_->reg_alpha,
                                                param_->max_delta_step, param_->min_child_weight};
      std::vector<common::metal::QuantizedSplitResult> results;
      std::vector<std::uint32_t> order;
      common::metal::EvaluateQuantizedSplits(inputs, tasks, parameters, &results, &order,
                                             metal_workspace_.get());
      // CUDA reduces raw child gains in sampled-feature order, then subtracts
      // the parent gain once. Subtracting first can create artificial FP32 ties.
      std::vector<std::size_t> winners(entries.size(), tasks.size());
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        if (results[i].bin < 0) continue;
        auto& winner = winners[mapping[i].first];
        if (winner == tasks.size() || results[i].gain > results[winner].gain) winner = i;
      }
      for (std::size_t entry = 0; entry < entries.size(); ++entry) {
        auto i = winners[entry];
        if (i == tasks.size()) continue;
        auto const& r = results[i];
        auto f = mapping[i].second;
        auto nid = entries[entry].nid;
        auto const &task = tasks[i];
        float split;
        if (task.kind == 2)
          split = std::numeric_limits<float>::quiet_NaN();
        else if (task.kind == 0)
          split = r.bin == 0 ? -std::numeric_limits<float>::infinity()
                             : cut.Values()[ptr[f] + r.bin - 1];
        else
          split = cut.Values()[ptr[f] + r.bin];
        SplitEntry candidate;
        candidate.Update(r.gain - snode_[nid].root_gain, f, split, r.default_left, task.kind != 0,
                         GradStats{r.left_sum}, GradStats{r.right_sum});
        if (task.kind) {
          candidate.cat_bits.resize(common::CatBitField::ComputeStorageSize(cut.MaxCategory() + 1),
                                    0);
          common::CatBitField bits{candidate.cat_bits};
          if (task.kind == 1)
            bits.Set(split);
          else
            for (std::uint32_t j = 0; j < r.partition; ++j)
              bits.Set(cut.Values()[ptr[f] + order[task.begin + j]]);
        }
        if (entries[entry].split.Update(candidate)) {
          snode_[nid].quantized_left = r.left;
          snode_[nid].quantized_right = r.right;
        }
      }
      return;
    }
#endif

    const size_t grain_size = std::max<size_t>(1, features.front()->Size() / n_threads);
    common::BlockedSpace2d space(
        entries.size(), [&](size_t nidx_in_set) { return features[nidx_in_set]->Size(); },
        grain_size);

    std::vector<CPUExpandEntry> tloc_candidates(n_threads * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      for (decltype(n_threads) j = 0; j < n_threads; ++j) {
        tloc_candidates[i * n_threads + j] = entries[i];
      }
    }
    auto evaluator = tree_evaluator_.GetEvaluator();
    auto const &cut_ptrs = cut.Ptrs();

    common::ParallelFor2d(space, n_threads, [&](std::size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[n_threads * nidx_in_set + tidx];
      auto best = &entry->split;
      auto nidx = entry->nid;
      auto histogram = hist[nidx];
      auto features_set = features[nidx_in_set]->ConstHostSpan();
      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        bool is_cat = common::IsCat(feature_types, fidx);
        if (!interaction_constraints_.Query(nidx, fidx)) {
          continue;
        }
        if (is_cat) {
          auto n_bins = cut_ptrs.at(fidx + 1) - cut_ptrs[fidx];
          if (common::UseOneHot(n_bins, param_->max_cat_to_onehot)) {
            this->EnumerateOneHot(cut, histogram, fidx, nidx, evaluator, best);
          } else {
            std::vector<size_t> sorted_idx(n_bins);
            std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
            auto feat_hist = histogram.subspan(cut_ptrs[fidx], n_bins);
            // Sort the histogram to get contiguous partitions.
            std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                             [&](std::size_t l, std::size_t r) {
                               return evaluator.CalcWeightCat(*param_, feat_hist[l]) <
                                      evaluator.CalcWeightCat(*param_, feat_hist[r]);
                             });
            this->EnumeratePart<+1>(cut, sorted_idx, histogram, fidx, nidx, evaluator, best);
            this->EnumeratePart<-1>(cut, sorted_idx, histogram, fidx, nidx, evaluator, best);
          }
        } else {
          auto grad_stats = EnumerateSplit<+1>(cut, histogram, fidx, nidx, evaluator, best);
          if (SplitContainsMissingValues(grad_stats, snode_[nidx])) {
            EnumerateSplit<-1>(cut, histogram, fidx, nidx, evaluator, best);
          }
        }
      }
    });

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      for (auto tidx = 0; tidx < n_threads; ++tidx) {
        entries[nidx_in_set].split.Update(tloc_candidates[n_threads * nidx_in_set + tidx].split);
      }
    }
  }

  // Add splits to tree, handles all statistic
  void ApplyTreeSplit(CPUExpandEntry const &candidate, RegTree *p_tree) {
    auto evaluator = tree_evaluator_.GetEvaluator();
    RegTree &tree = *p_tree;

    GradStats parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
#if defined(XGBOOST_USE_METAL)
    // CUDA converts (integer left + integer right) only once for parent weight
    // and cover. Adding the two rounded views can lose cancellation bits.
    if (ctx_->IsMPS()) parent_sum = snode_[candidate.nid].stats;
#endif
    auto base_weight = evaluator.CalcWeight(candidate.nid, *param_, GradStats{parent_sum});
    auto left_weight =
        evaluator.CalcWeight(candidate.nid, *param_, GradStats{candidate.split.left_sum});
    auto right_weight =
        evaluator.CalcWeight(candidate.nid, *param_, GradStats{candidate.split.right_sum});

    if (candidate.split.is_cat) {
      tree.ExpandCategorical(
          candidate.nid, candidate.split.SplitIndex(), candidate.split.cat_bits,
          candidate.split.DefaultLeft(), base_weight, left_weight * param_->learning_rate,
          right_weight * param_->learning_rate, candidate.split.loss_chg, parent_sum.GetHess(),
          candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());
    } else {
      tree.ExpandNode(candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
                      candidate.split.DefaultLeft(), base_weight,
                      left_weight * param_->learning_rate, right_weight * param_->learning_rate,
                      candidate.split.loss_chg, parent_sum.GetHess(),
                      candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());
    }

    // Set up child constraints
    auto left_child = tree[candidate.nid].LeftChild();
    auto right_child = tree[candidate.nid].RightChild();
    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child,
                             tree[candidate.nid].SplitIndex(), left_weight, right_weight);
    evaluator = tree_evaluator_.GetEvaluator();

    snode_.resize(tree.Size());
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      snode_[left_child].quantized = snode_[candidate.nid].quantized_left;
      snode_[right_child].quantized = snode_[candidate.nid].quantized_right;
    }
#endif
    snode_.at(left_child).stats = candidate.split.left_sum;
    snode_.at(left_child).root_gain =
        evaluator.CalcGain(candidate.nid, *param_, GradStats{candidate.split.left_sum});
    snode_.at(right_child).stats = candidate.split.right_sum;
    snode_.at(right_child).root_gain =
        evaluator.CalcGain(candidate.nid, *param_, GradStats{candidate.split.right_sum});

    interaction_constraints_.Split(candidate.nid, tree[candidate.nid].SplitIndex(), left_child,
                                   right_child);
  }

  [[nodiscard]] auto Evaluator() const { return tree_evaluator_.GetEvaluator(); }
  [[nodiscard]] auto const &Stats() const { return snode_; }

  float InitRoot(GradStats const& root_sum, GradientPairInt64 const* quantized = nullptr) {
    snode_.resize(1);
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      CHECK(quantized);
      snode_[0].quantized = *quantized;
    }
#endif
    auto root_evaluator = tree_evaluator_.GetEvaluator();

    snode_[0].stats = GradStats{root_sum.GetGrad(), root_sum.GetHess()};
    snode_[0].root_gain =
        root_evaluator.CalcGain(RegTree::kRoot, *param_, GradStats{snode_[0].stats});
    auto weight = root_evaluator.CalcWeight(RegTree::kRoot, *param_, GradStats{snode_[0].stats});
    return weight;
  }

 public:
  // The column sampler must be constructed by caller since we need to preserve the rng
  // for the entire training session.
  explicit HistEvaluator(Context const *ctx, TrainParam const *param, MetaInfo const &info,
                         std::shared_ptr<common::ColumnSampler> sampler)
      : ctx_{ctx},
        param_{param},
        column_sampler_{std::move(sampler)},
        tree_evaluator_{*param, static_cast<bst_feature_t>(info.num_col_), DeviceOrd::CPU(), 1u} {
    interaction_constraints_.Configure(*param, info.num_col_);
    column_sampler_->Init(ctx, info.num_col_, info.feature_weights, param_->colsample_bynode,
                          param_->colsample_bylevel, param_->colsample_bytree);
  }
};

class HistMultiEvaluator {
  std::vector<double> gain_;
  linalg::Matrix<GradientPairPrecise> stats_;
  TrainParam const *param_;
  TreeEvaluator tree_evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  Context const *ctx_;

#if defined(XGBOOST_USE_METAL)
  std::unique_ptr<common::metal::SplitWorkspace> metal_workspace_;
  // Exact node sums and selected child sums persist by node ID across queues.
  std::vector<GradientPairInt64> metal_node_sums_, metal_split_sums_;
#endif

 private:
  template <bst_bin_t d_step>
  bool EnumerateSplit(common::HistogramCuts const &cut, bst_feature_t fidx,
                      common::Span<common::ConstGHistRow> hist,
                      linalg::VectorView<GradientPairPrecise const> parent_sum, double parent_gain,
                      bst_node_t nidx, TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                      SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) const {
    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();

    auto sum = linalg::Empty<GradientPairPrecise>(ctx_, 2, hist.size());
    auto left_sum = sum.Slice(0, linalg::All());
    auto right_sum = sum.Slice(1, linalg::All());

    bst_bin_t ibegin, iend;
    if (d_step > 0) {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
      iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
    } else {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx + 1]) - 1;
      iend = static_cast<bst_bin_t>(cut_ptr[fidx]) - 1;
    }
    auto n_targets = hist.size();

    for (bst_bin_t i = ibegin; i != iend; i += d_step) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto t_hist = hist[t];
        auto t_p = parent_sum(t);
        left_sum(t) += t_hist[i];
        right_sum(t) = t_p - left_sum(t);
      }

      if (d_step > 0) {
        auto split_pt = cut_val[i];
        auto loss_chg =
            evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
        p_best->Update(loss_chg, fidx, split_pt, d_step == -1, false, left_sum, right_sum);
      } else {
        auto split_pt = common::HistogramCuts::NumericBinLowerBound(cut_ptr, cut_val, fidx, i);
        auto loss_chg =
            evaluator.CalcSplitGain(*param_, nidx, fidx, right_sum, left_sum) - parent_gain;
        p_best->Update(loss_chg, fidx, split_pt, d_step == -1, false, right_sum, left_sum);
      }
    }
    // return true if there's missing. Doesn't handle floating-point error well.
    if (d_step == +1) {
      return !std::equal(linalg::cbegin(left_sum), linalg::cend(left_sum),
                         linalg::cbegin(parent_sum));
    }
    return false;
  }

  void EnumerateOneHot(common::HistogramCuts const &cut, bst_feature_t fidx,
                       common::Span<common::ConstGHistRow> hist,
                       linalg::VectorView<GradientPairPrecise const> parent_sum, double parent_gain,
                       bst_node_t nidx, TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                       SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) const {
    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();

    bst_bin_t ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
    bst_bin_t iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
    bst_bin_t n_bins = iend - ibegin;
    auto n_targets = hist.size();

    auto sum = linalg::Empty<GradientPairPrecise>(ctx_, 2, n_targets);
    auto left_sum = sum.Slice(0, linalg::All());
    auto right_sum = sum.Slice(1, linalg::All());

    // Per-target missing gradient: parent_sum - sum_of_all_bins.
    auto missing_storage = linalg::Empty<GradientPairPrecise>(ctx_, n_targets);
    auto missing = missing_storage.HostView();
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto f_hist = hist[t].subspan(cut_ptr[fidx], n_bins);
      GradientPairPrecise feature_sum{};
      for (bst_bin_t b = 0; b < n_bins; ++b) {
        feature_sum += f_hist[b];
      }
      missing(t) = parent_sum(t) - feature_sum;
    }

    SplitEntryContainer<std::vector<GradientPairPrecise>> best;
    best.is_cat = false;

    for (bst_bin_t i = ibegin; i != iend; ++i) {
      auto split_pt = cut_val[i];

      // Missing on left (missing grouped with other categories).
      for (bst_target_t t = 0; t < n_targets; ++t) {
        right_sum(t) = GradientPairPrecise{hist[t][i]};
        left_sum(t) = parent_sum(t) - right_sum(t);
      }
      auto missing_left_gain =
          evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
      best.Update(missing_left_gain, fidx, split_pt, true, true, left_sum, right_sum);

      // Missing on right (missing grouped with chosen category).
      for (bst_target_t t = 0; t < n_targets; ++t) {
        right_sum(t) = GradientPairPrecise{hist[t][i]} + missing(t);  // NOLINT
        left_sum(t) = parent_sum(t) - right_sum(t);
      }
      auto missing_right_gain =
          evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
      best.Update(missing_right_gain, fidx, split_pt, false, true, left_sum, right_sum);
    }

    if (best.is_cat) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins + 1);
      best.cat_bits.resize(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      cat_bits.Set(best.split_value);
    }

    p_best->Update(best);
  }

  template <bst_bin_t d_step>
  void EnumeratePart(common::HistogramCuts const &cut, common::Span<size_t const> sorted_idx,
                     common::Span<common::ConstGHistRow> hist, bst_feature_t fidx, bst_node_t nidx,
                     TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                     SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");
    auto n_targets = hist.size();

    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();
    auto parent_sum = stats_.Slice(nidx, linalg::All());
    auto parent_gain = gain_[nidx];

    bst_bin_t f_begin = cut_ptr[fidx];
    bst_bin_t f_end = cut_ptr[fidx + 1];
    bst_bin_t n_bins_feature{f_end - f_begin};
    auto n_bins = std::min(param_->max_cat_threshold, n_bins_feature);

    bst_bin_t it_begin, it_end;
    if (d_step > 0) {
      it_begin = f_begin;
      it_end = it_begin + n_bins - 1;
    } else {
      it_begin = f_end - 1;
      it_end = it_begin - n_bins + 1;
    }

    auto sum = linalg::Constant(ctx_, GradientPairPrecise{}, 2, n_targets);
    auto left_sum = sum.Slice(0, linalg::All());
    auto right_sum = sum.Slice(1, linalg::All());

    SplitEntryContainer<std::vector<GradientPairPrecise>> best;
    bst_bin_t best_thresh{-1};
    for (bst_bin_t i = it_begin; i != it_end; i += d_step) {
      auto j = i - f_begin;  // index local to current feature
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto f_hist = hist[t].subspan(f_begin, n_bins_feature);
        if (d_step == 1) {
          right_sum(t) += f_hist[sorted_idx[j]];
          left_sum(t) = parent_sum(t) - right_sum(t);  // missing on left
        } else {
          left_sum(t) += f_hist[sorted_idx[j]];
          right_sum(t) = parent_sum(t) - left_sum(t);  // missing on right
        }
      }

      auto loss_chg =
          evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
      // We don't have a numeric split point, nan here is a dummy split.
      if (best.Update(loss_chg, fidx, std::numeric_limits<float>::quiet_NaN(), d_step == 1, true,
                      left_sum, right_sum)) {
        best_thresh = i;
      }
    }

    if (best_thresh != -1) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins_feature);
      best.cat_bits = decltype(best.cat_bits)(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      bst_bin_t partition = d_step == 1 ? (best_thresh - it_begin + 1) : (best_thresh - f_begin);
      CHECK_GT(partition, 0);
      std::for_each(sorted_idx.begin(), sorted_idx.begin() + partition, [&](std::size_t c) {
        auto cat = cut_val[c + f_begin];
        cat_bits.Set(cat);
      });
    }

    p_best->Update(best);
  }

 public:
  void EvaluateSplits(common::Span<const BoundedHistCollection *> hist,
                      common::HistogramCuts const &cut,
                      common::Span<FeatureType const> feature_types,
                      std::vector<MultiExpandEntry> *p_entries) {
    auto &entries = *p_entries;
    CHECK_EQ(stats_.Shape(1), hist.size());
    CHECK(!hist.empty());
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(entries.size());

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      features[nidx_in_set] = column_sampler_->GetFeatureSet(ctx_, entries[nidx_in_set].depth);
    }
    CHECK(!features.empty());
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      if (!metal_workspace_) metal_workspace_ = common::metal::MakeSplitWorkspace();
      auto evaluator = tree_evaluator_.GetEvaluator();
      auto targets = hist.size();
      std::vector<common::metal::QuantizedMultiSplitTask> tasks;
      std::vector<GradientPairInt64> parents;
      std::size_t bins = 0;
      std::vector<common::metal::HistogramInput> inputs;
      for (auto target : hist) inputs.push_back({target->MetalBuffer(), {}});
      std::vector<common::metal::SplitBounds> bounds;
      std::vector<std::pair<std::size_t, bst_feature_t>> mapping;
      auto const &ptr = cut.Ptrs();
      for (std::size_t i = 0; i < entries.size(); ++i) {
        auto nid = entries[i].nid;
        for (auto f : features[i]->ConstHostSpan()) {
          if (!interaction_constraints_.Query(nid, f)) continue;
          auto n = ptr[f + 1] - ptr[f];
          if (!n) continue;
          CHECK_LE(bins + n * targets, std::numeric_limits<std::uint32_t>::max());
          CHECK_LE(parents.size() + targets, std::numeric_limits<std::uint32_t>::max());
          bool categorical = common::IsCat(feature_types, f);
          std::uint32_t kind =
              categorical ? (common::UseOneHot(n, param_->max_cat_to_onehot) ? 1 : 2) : 0;
          tasks.push_back(
              {static_cast<std::uint32_t>(bins), static_cast<std::uint32_t>(bins + n * targets),
               kind, static_cast<std::uint32_t>(param_->max_cat_threshold),
               static_cast<std::uint32_t>(parents.size()), static_cast<std::uint32_t>(targets),
               evaluator.has_constraint ? evaluator.constraints[f] : 0,
               static_cast<std::uint32_t>(evaluator.has_constraint)});
          for (std::size_t t = 0; t < targets; ++t) {
            parents.push_back(metal_node_sums_.at(nid * targets + t));
            bounds.push_back({evaluator.has_constraint ? evaluator.lower[nid * targets + t]
                                                       : -std::numeric_limits<float>::infinity(),
                              evaluator.has_constraint ? evaluator.upper[nid * targets + t]
                                                       : std::numeric_limits<float>::infinity()});
          }
          for (std::size_t t = 0; t < targets; ++t)
            inputs[t].segments.push_back({hist[t]->Offset(nid) + ptr[f],
                                          static_cast<std::uint32_t>(bins + t), n,
                                          static_cast<std::uint32_t>(targets), 0});
          bins += n * targets;
          mapping.emplace_back(i, f);
        }
      }
      common::metal::SplitParameters parameters{param_->reg_lambda, param_->reg_alpha,
                                                param_->max_delta_step, param_->min_child_weight};
      std::vector<common::metal::QuantizedMultiSplitResult> results;
      std::vector<GradientPairInt64> integer_children;
      std::vector<GradientPairPrecise> children;
      std::vector<std::uint32_t> order;
      common::metal::EvaluateQuantizedMultiSplits(inputs, tasks, parents, bounds, parameters,
                                                  &results, &integer_children, &children, &order,
                                                  metal_workspace_.get());
      // CUDA compares rounded chunk winners between features, with lower
      // feature ID winning ties, then subtracts the binary64 parent gain.
      std::vector<std::size_t> winners(entries.size(), tasks.size());
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        if (results[i].bin < 0) continue;
        auto& winner = winners[mapping[i].first];
        if (winner == tasks.size() || results[i].gain > results[winner].gain ||
            (results[i].gain == results[winner].gain && mapping[i].second < mapping[winner].second))
          winner = i;
      }
      for (std::size_t entry = 0; entry < entries.size(); ++entry) {
        auto i = winners[entry];
        if (i == tasks.size()) continue;
        auto const& r = results[i];
        auto f = mapping[i].second;
        auto nid = entries[entry].nid;
        auto const &task = tasks[i];
        float split;
        if (task.kind == 2)
          split = std::numeric_limits<float>::quiet_NaN();
        else if (task.kind == 0 && r.default_left)
          split = common::HistogramCuts::NumericBinLowerBound(ptr, cut.Values(), f, ptr[f] + r.bin);
        else
          split = cut.Values()[ptr[f] + r.bin];
        SplitEntryContainer<std::vector<GradientPairPrecise>> candidate;
        auto left = linalg::MakeVec(children.data() + 2 * task.parent_begin, targets);
        auto right = linalg::MakeVec(children.data() + 2 * task.parent_begin + targets, targets);
        auto parent_gain = evaluator.CalcGain(nid, *param_, stats_.Slice(nid, linalg::All()));
        candidate.Update(static_cast<float>(double(r.gain) - parent_gain), f, split, r.default_left,
                         task.kind != 0, left, right);
        if (task.kind) {
          candidate.cat_bits.resize(common::CatBitField::ComputeStorageSize(cut.MaxCategory() + 1),
                                    0);
          common::CatBitField bits{candidate.cat_bits};
          if (task.kind == 1)
            bits.Set(split);
          else
            for (std::uint32_t j = 0; j < r.partition; ++j)
              bits.Set(cut.Values()[ptr[f] + order[task.begin + j]]);
        }
        if (entries[entry].split.Update(candidate)) {
          auto offset = std::size_t(nid) * 2 * targets;
          if (metal_split_sums_.size() < offset + 2 * targets)
            metal_split_sums_.resize(offset + 2 * targets);
          std::copy_n(integer_children.data() + 2 * task.parent_begin, 2 * targets,
                      metal_split_sums_.data() + offset);
        }
      }
      return;
    }
#endif

    std::int32_t n_threads = ctx_->Threads();
    std::size_t const grain_size = std::max<std::size_t>(1, features.front()->Size() / n_threads);
    common::BlockedSpace2d space{
        entries.size(), [&](std::size_t nidx_in_set) { return features[nidx_in_set]->Size(); },
        grain_size};

    std::vector<MultiExpandEntry> tloc_candidates(n_threads * entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
      for (std::int32_t j = 0; j < n_threads; ++j) {
        tloc_candidates[i * n_threads + j] = entries[i];
      }
    }

    auto evaluator = tree_evaluator_.GetEvaluator();
    common::ParallelFor2d(space, n_threads, [&](std::size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[n_threads * nidx_in_set + tidx];
      auto best = &entry->split;

      auto parent_sum = stats_.Slice(entry->nid, linalg::All());
      auto base_weight = linalg::Zeros<float>(ctx_, parent_sum.Shape());
      auto h_bw = base_weight.HostView();
      evaluator.CalcWeightCat(*param_, parent_sum, h_bw);

      std::vector<common::ConstGHistRow> node_hist;
      for (auto t_hist : hist) {
        node_hist.emplace_back((*t_hist)[entry->nid]);
      }
      auto features_set = features[nidx_in_set]->ConstHostSpan();
      auto n_targets = hist.size();

      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        if (!interaction_constraints_.Query(entry->nid, fidx)) {
          continue;
        }
        auto parent_gain = gain_[entry->nid];
        bool is_cat = common::IsCat(feature_types, fidx);
        // Numeric split
        if (!is_cat) {
          bool missing = this->EnumerateSplit<+1>(cut, fidx, node_hist, parent_sum, parent_gain,
                                                  entry->nid, evaluator, best);
          if (missing) {
            this->EnumerateSplit<-1>(cut, fidx, node_hist, parent_sum, parent_gain, entry->nid,
                                     evaluator, best);
          }
          continue;
        }

        auto const &cut_ptr = cut.Ptrs();
        auto n_bins = cut_ptr.at(fidx + 1) - cut_ptr[fidx];
        // One hot split
        if (common::UseOneHot(n_bins, param_->max_cat_to_onehot)) {
          this->EnumerateOneHot(cut, fidx, node_hist, parent_sum, parent_gain, entry->nid,
                                evaluator, best);
          continue;
        }

        // Partition split
        std::vector<size_t> sorted_idx(n_bins);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0ul);
        linalg::Vector<GradientPairPrecise> grads({n_targets}, this->ctx_->Device());
        auto h_grads = grads.HostView();
        std::vector<float> child_w(n_targets, .0f);

        // Sort by s_c = w_p^T w_c
        // Project onto the parent's update direction to compare scores l_s < r_s
        // Since we care only about the ordering, no need to divide the norm.
        std::vector<double> scores(n_bins);
        for (std::size_t bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
          for (decltype(n_targets) t_idx = 0; t_idx < n_targets; ++t_idx) {
            auto f_hist = node_hist[t_idx].subspan(cut_ptr[fidx], n_bins);
            h_grads(t_idx) = f_hist[bin_idx];
          }
          evaluator.CalcWeightCat(*param_, h_grads,
                                  linalg::MakeVec(child_w.data(), child_w.size()));
          double sc = .0;
          for (decltype(n_targets) t_idx = 0; t_idx < n_targets; ++t_idx) {
            sc += h_bw(t_idx) * child_w[t_idx];
          }
          scores[bin_idx] = sc;
        }

        std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                         [&](std::size_t l, std::size_t r) { return scores[l] < scores[r]; });

        this->EnumeratePart<+1>(cut, sorted_idx, node_hist, fidx, entry->nid, evaluator, best);
        this->EnumeratePart<-1>(cut, sorted_idx, node_hist, fidx, entry->nid, evaluator, best);
      }
    });

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      for (auto tidx = 0; tidx < n_threads; ++tidx) {
        entries[nidx_in_set].split.Update(tloc_candidates[n_threads * nidx_in_set + tidx].split);
      }
    }
  }

  linalg::Vector<float> InitRoot(linalg::VectorView<GradientPairPrecise const> root_sum,
                                 common::Span<GradientPairInt64 const> quantized = {}) {
    auto n_targets = root_sum.Size();
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      CHECK_EQ(quantized.size(), n_targets);
      metal_node_sums_.assign(quantized.begin(), quantized.end());
      metal_split_sums_.clear();
    }
#endif
    stats_ = linalg::Constant(ctx_, GradientPairPrecise{}, 1, n_targets);
    gain_.resize(1);

    linalg::Vector<float> weight({n_targets}, ctx_->Device());
    auto evaluator = tree_evaluator_.GetEvaluator();
    evaluator.CalcWeight(RegTree::kRoot, *param_, root_sum, weight.HostView());
    auto root_gain = evaluator.CalcGainGivenWeight(*param_, root_sum, weight.HostView());
    gain_.front() = root_gain;

    auto h_stats = stats_.HostView();
    std::copy(linalg::cbegin(root_sum), linalg::cend(root_sum), linalg::begin(h_stats));

    return weight;
  }

  void ApplyTreeSplit(MultiExpandEntry const &candidate, RegTree *p_tree) {
    // Use the split gradient's number of targets for intermediate weights
    // This may differ from p_tree->NumTargets() when using reduced gradient
    auto n_split_targets = candidate.split.left_sum.size();
    auto parent_sum = stats_.Slice(candidate.nid, linalg::All());

    auto evaluator = tree_evaluator_.GetEvaluator();
    auto weight = linalg::Empty<float>(ctx_, 3, n_split_targets);
    auto base_weight = weight.Slice(0, linalg::All());
    evaluator.CalcWeight(candidate.nid, *param_, parent_sum, base_weight);

    auto left_weight = weight.Slice(1, linalg::All());
    auto left_sum =
        linalg::MakeVec(candidate.split.left_sum.data(), candidate.split.left_sum.size());

    auto right_weight = weight.Slice(2, linalg::All());
    auto right_sum =
        linalg::MakeVec(candidate.split.right_sum.data(), candidate.split.right_sum.size());
    evaluator.CalcSplitWeights(*param_, candidate.nid, candidate.split.SplitIndex(), left_sum,
                               right_sum, left_weight, right_weight);

    // Compute the loss_chg and sum hessians for parent and children
    float loss_chg = candidate.split.loss_chg;
    // Sum hessians across all targets for each child
    double left_sum_hess = 0.0, right_sum_hess = 0.0;
    for (std::size_t t = 0; t < candidate.split.left_sum.size(); ++t) {
      left_sum_hess += candidate.split.left_sum[t].GetHess();
      right_sum_hess += candidate.split.right_sum[t].GetHess();
    }

    common::Span<CatWordT const> cat_bits;
    if (candidate.split.is_cat) {
      cat_bits = candidate.split.cat_bits;
    }
    auto as_span = [](linalg::VectorView<float const> weight) {
      // sliced weight might have larger span as the underlying data
      return weight.Values().subspan(0, weight.Size());
    };
    ExpandBatch batch{param_->learning_rate};
    batch.Push(candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
               candidate.split.DefaultLeft(), as_span(base_weight), as_span(left_weight),
               as_span(right_weight), loss_chg, left_sum_hess, right_sum_hess, cat_bits);
    p_tree->Expand(ctx_, batch);

    CHECK(p_tree->IsMultiTarget());
    auto left_child = p_tree->LeftChild(candidate.nid);
    CHECK_GT(left_child, candidate.nid);
    auto right_child = p_tree->RightChild(candidate.nid);
    CHECK_GT(right_child, candidate.nid);

    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child, candidate.split.SplitIndex(),
                             left_weight, right_weight);
    evaluator = tree_evaluator_.GetEvaluator();
    interaction_constraints_.Split(candidate.nid, candidate.split.SplitIndex(), left_child,
                                   right_child);

    std::size_t n_nodes = p_tree->Size();
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      metal_node_sums_.resize(n_nodes * n_split_targets);
      auto offset = std::size_t(candidate.nid) * 2 * n_split_targets;
      CHECK_LE(offset + 2 * n_split_targets, metal_split_sums_.size());
      std::copy_n(metal_split_sums_.data() + offset, n_split_targets,
                  metal_node_sums_.data() + left_child * n_split_targets);
      std::copy_n(metal_split_sums_.data() + offset + n_split_targets, n_split_targets,
                  metal_node_sums_.data() + right_child * n_split_targets);
    }
#endif
    gain_.resize(n_nodes);
    gain_[left_child] = evaluator.CalcGainGivenWeight(*param_, left_sum, left_weight);
    gain_[right_child] = evaluator.CalcGainGivenWeight(*param_, right_sum, right_weight);

    if (n_nodes >= stats_.Shape(0)) {
      stats_.Reshape(n_nodes * 2, stats_.Shape(1));
    }
    CHECK_EQ(stats_.Shape(1), n_split_targets);
    auto left_sum_stat = stats_.Slice(left_child, linalg::All());
    std::copy(candidate.split.left_sum.cbegin(), candidate.split.left_sum.cend(),
              linalg::begin(left_sum_stat));
    auto right_sum_stat = stats_.Slice(right_child, linalg::All());
    std::copy(candidate.split.right_sum.cbegin(), candidate.split.right_sum.cend(),
              linalg::begin(right_sum_stat));
  }

  explicit HistMultiEvaluator(Context const *ctx, MetaInfo const &info, TrainParam const *param,
                              bst_target_t n_targets,
                              std::shared_ptr<common::ColumnSampler> sampler)
      : param_{param},
        tree_evaluator_{*param, static_cast<bst_feature_t>(info.num_col_), DeviceOrd::CPU(),
                        n_targets},
        column_sampler_{std::move(sampler)},
        ctx_{ctx} {
    interaction_constraints_.Configure(*param, info.num_col_);
    column_sampler_->Init(ctx, info.num_col_, info.feature_weights, param_->colsample_bynode,
                          param_->colsample_bylevel, param_->colsample_bytree);
  }

  [[nodiscard]] auto Evaluator() const { return tree_evaluator_.GetEvaluator(); }
};

}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
