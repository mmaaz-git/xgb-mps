// CUDA leaf_sum.cu oracle: original value scales, sampled FLOAT input, integer
// leaf/page accumulation, conversion only for the completed sum and weight.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_leaf.h"
#include "../xgboost/src/common/metal_training.h"
#include "../xgboost/src/tree/param.h"
#include "cuda_gradient_reference.h"
#include "integer_leaf_contract_io.h"
using namespace xgboost;
namespace metal = common::metal;
namespace lc = integer_leaf_contracts;
std::unique_ptr<lc::Exporter> report;
std::vector<float> Flatten(std::vector<GradientPair> const& pairs) {
  std::vector<float> values;values.reserve(2*pairs.size());
  for(auto p:pairs){values.push_back(p.GetGrad());values.push_back(p.GetHess());}
  return values;
}
std::size_t cases = 0, pairs = 0, fields = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Same(GradientPairPrecise a, GradientPairPrecise b) {
  if (a.GetGrad() != b.GetGrad() || a.GetHess() != b.GetHess()) {
    std::fprintf(stderr, "actual (%a,%a) expected (%a,%a)\n", a.GetGrad(), a.GetHess(), b.GetGrad(),
                 b.GetHess());
    throw std::runtime_error("Integer leaf sum mismatch");
  }
  fields += 2;
}
struct Oracle {
  std::size_t targets;
  std::vector<cuda_reference::Scale> scales;
  std::vector<std::uint64_t> sums;
  Oracle(linalg::MatrixView<GradientPair const> original, std::size_t leaves)
      : targets{original.Shape(1)}, scales(targets), sums(leaves * targets * 2) {
    for (std::size_t t = 0; t < targets; ++t) {
      double positive[2]{}, negative[2]{};
      for (std::size_t r = 0; r < original.Shape(0); ++r) {
        float values[]{original(r, t).GetGrad(), original(r, t).GetHess()};
        for (int k = 0; k < 2; ++k)
          if (values[k] >= 0)
            positive[k] += values[k];
          else
            negative[k] -= values[k];
      }
      scales[t] = {cuda_reference::Exponent(positive[0], negative[0], original.Shape(0)),
                   cuda_reference::Exponent(positive[1], negative[1], original.Shape(0))};
    }
  }
  void Add(std::size_t leaf, std::size_t target, GradientPair pair, bool count=true) {
    auto scale = scales[target];
    double g = std::ldexp(double(pair.GetGrad()), scale.g),
           h = std::ldexp(double(pair.GetHess()), scale.h);
    Check(g >= -0x1p63 && g < 0x1p63 && h >= -0x1p63 && h < 0x1p63,
          "Host fixture outside CUDA conversion domain");
    auto qg = static_cast<std::int64_t>(g), qh = static_cast<std::int64_t>(h);
    if (pair.GetHess() > 0 && qh == 0) qh = 1;
    auto at = 2 * (leaf * targets + target);
    sums[at] += static_cast<std::uint64_t>(qg);
    sums[at + 1] += static_cast<std::uint64_t>(qh);
    if(count)++pairs;
  }
  GradientPairPrecise Sum(std::size_t leaf, std::size_t target) const {
    auto at = 2 * (leaf * targets + target);
    std::int64_t g, h;
    std::memcpy(&g, &sums[at], 8);
    std::memcpy(&h, &sums[at + 1], 8);
    return {std::ldexp(double(g), -scales[target].g), std::ldexp(double(h), -scales[target].h)};
  }
  void Verify(metal::LeafSums const& actual) const {
    auto values = actual.Data();
    Check(values.size() * 2 == sums.size(), "Sum shape changed");
    for (std::size_t i = 0; i < values.size(); ++i) Same(values[i], Sum(i / targets, i % targets));
  }
  void Observe(metal::LeafSums const& actual,lc::Observation* out) const {
    auto values=actual.Data();
    for(std::size_t i=0;i<values.size();++i) {
      auto ref=Sum(i/targets,i%targets);
      out->sums.push_back(lc::Bits<std::uint64_t>(values[i].GetGrad()));
      out->sums.push_back(lc::Bits<std::uint64_t>(values[i].GetHess()));
      out->reference_sums.push_back(lc::Bits<std::uint64_t>(ref.GetGrad()));
      out->reference_sums.push_back(lc::Bits<std::uint64_t>(ref.GetHess()));
    }
    // Reference-only: LeafSums exposes no actual integer buffer.
    out->integers.insert(out->integers.end(),sums.begin(),sums.end());
  }
  void Weights(metal::LeafSums const& actual,lc::Observation* out=nullptr) const {
    for (int kind = 0; kind < 5; ++kind) {
      tree::TrainParam param;
      param.reg_lambda = kind == 0 ? 0.f : 1.25f;
      param.reg_alpha = kind == 2 ? .75f : 0;
      param.max_delta_step = kind == 3 ? .125f : 0;
      param.min_child_weight = 1e20f;
      std::vector<float> output(sums.size() / 2 + 2, -913);
      std::vector<metal::SplitBounds> bounds;
      if (kind == 4) bounds.resize(output.size() - 2, {.125f, .25f});
      actual.Weights(
          {param.reg_lambda, param.reg_alpha, param.max_delta_step, param.min_child_weight}, bounds,
          .3f, {output.data() + 1, output.size() - 2});
      if(out)out->reference_weights.push_back(lc::Bits<std::uint32_t>(-913.f));
      for (std::size_t i = 0; i < output.size() - 2; ++i) {
        float expected = tree::CalcWeight(param, Sum(i / targets, i % targets));
        if (kind == 4) expected = std::min(std::max(expected, .125f), .25f);
        expected *= .3f;
        Check(!std::memcmp(&output[i + 1], &expected, 4), "Integer-derived leaf weight differs");
        if(out)out->reference_weights.push_back(lc::Bits<std::uint32_t>(expected));
      }
      Check(output.front() == -913 && output.back() == -913, "Weight guards changed");
      if(out) {
        out->reference_weights.push_back(lc::Bits<std::uint32_t>(-913.f));
        for(float value:output)out->weights.push_back(lc::Bits<std::uint32_t>(value));
      }
    }
  }
};
void Run(std::size_t rows, std::size_t targets, unsigned layout, unsigned mode) {
  current = "rows=" + std::to_string(rows) + " targets=" + std::to_string(targets) +
            " layout=" + std::to_string(layout) + " mode=" + std::to_string(mode);
  std::size_t n = rows + 11, s0 = layout == 1 ? 1 : layout == 2 ? targets * 2 + 3 : targets;
  std::size_t s1 = layout == 1 ? n + 3 : layout == 2 ? 2 : 1;
  if (layout == 3) s0 = 0;
  if (layout == 4) {
    s0 = 1;
    s1 = 0;
  }
  auto extent = (n - 1) * s0 + (targets - 1) * s1 + 1;
  std::vector<GradientPair> original(extent + 8, {981, -237});
  linalg::MatrixView<GradientPair> view{
      {original.data() + 4, extent}, {n, targets}, {s0, s1}, DeviceOrd::CPU()};
  std::mt19937 random(317);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t t = 0; t < targets; ++t) {
      float g = float(int(random() % 2047) - 1023) / 16, h = float(random() % 511) / 8;
      if (mode == 1) {
        g = std::ldexp(g, -140);
        h = std::ldexp(h, -140);
      }
      if (mode == 2) {
        g = std::ldexp(g, 110);
        h = std::ldexp(h, 110);
      }
      if (mode == 3) {
        g = r == 0 ? 0x1p60f : float(int(r % 7) - 3);
        h = r == 0 ? 0x1p65f : 0x1p-100f;
      }
      view(r, t) = {g, h};
    }
  auto saved = original;
  Oracle reference(view, 7);
  auto before = metal::GetStatistics();
  auto quantizer = metal::MakeGradientQuantizer(view);
  Check(quantizer->Rows() == n && quantizer->Targets() == targets, "Quantizer shape changed");
  Check(metal::GetStatistics().gradient_bound_dispatches - before.gradient_bound_dispatches ==
            targets,
        "Missing original bounds");
  Check(metal::GetStatistics().quantization_dispatches == before.quantization_dispatches,
        "Scale-only factory allocated/quantized original rows");
  Check(!std::memcmp(original.data(), saved.data(), original.size() * sizeof(GradientPair)),
        "Quantizer mutated original values");
  auto output = metal::MakeLeafSums(7, targets, quantizer);
  quantizer.reset();  // Leaf state must retain the original scales.
  std::vector<GradientPair> sampled = original;
  linalg::MatrixView<GradientPair> values{
      {sampled.data() + 4, extent}, {n, targets}, {s0, s1}, DeviceOrd::CPU()};
  float factors[]{0, 1, 1.25f, 1.5f};
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t t = 0; t < targets; ++t) {
      float factor = factors[r % 4];
      values(r, t) = {cuda_reference::Multiply(view(r, t).GetGrad(), factor),
                      cuda_reference::Multiply(view(r, t).GetHess(), factor)};
    }
  auto saved_sampled = sampled;
  lc::Input input{{unsigned(cases),0,unsigned(rows),unsigned(targets),layout,mode,0,0},{},{}};
  lc::Observation observed;
  if(report){input.original=Flatten(saved);input.sampled=Flatten(saved_sampled);}
  std::fill(original.begin(), original.end(), GradientPair{919, 313});
  std::size_t boundaries[]{0, rows / 3, rows * 2 / 3, rows};
  for (unsigned page = 0; page < 3; ++page) {
    auto begin = boundaries[page], end = boundaries[page + 1];
    auto indices = metal::MakeDeviceRows();
    indices->Reset(end - begin, begin + 5);
    std::vector<metal::LeafSumSegment> segments;
    for (unsigned leaf = 0; leaf < 7; ++leaf) {
      auto slot = (leaf * 3 + 2) % 7;
      auto lo = std::max(begin, rows * std::min(leaf, 5u) / 5),
           hi = std::min(end, rows * std::min(leaf + 1, 5u) / 5);
      if (hi <= lo) {
        segments.push_back({0, 0, slot, 0});
        continue;
      }
      segments.push_back({lo - begin, hi - begin, slot, 0});
      for (auto row = lo; row < hi; ++row)
        for (std::size_t t = 0; t < targets; ++t) reference.Add(slot, t, values(row + 5, t));
    }
    std::reverse(segments.begin(), segments.end());
    auto previous = metal::GetStatistics();
    indices->AddLeafGradients(segments, values, output.get());
    auto after = metal::GetStatistics();
    Check(after.leaf_integer_sum_dispatches - previous.leaf_integer_sum_dispatches ==
              bool(end - begin),
          "Missing integer page sum");
    Check(after.gradient_bound_dispatches == previous.gradient_bound_dispatches,
          "Value bounds recomputed after sampling");
    reference.Verify(*output);
    if(report)reference.Observe(*output,&observed);
  }
  Check(!std::memcmp(sampled.data(), saved_sampled.data(), sampled.size() * sizeof(GradientPair)),
        "Leaf sum modified values/guards");
  reference.Weights(*output,report?&observed:nullptr);
  if(report)report->Add(input,observed);
  ++cases;
  if (cases % 30 == 0 || rows > 1000000) {
    std::printf("Checked %zu cases / %zu pairs / %zu exact fields\n", cases, pairs, fields);
    std::fflush(stdout);
  }
}
void Cancellation() {
  for (int exponent : {-80, 0, 54, 100})
    for (bool paged : {false, true}) {
      current =
          "cancellation exponent=" + std::to_string(exponent) + " paged=" + std::to_string(paged);
      float big = std::ldexp(1.f, exponent), small = std::ldexp(1.f, exponent - 54);
      std::vector<GradientPair> input{{big, 0}, {small, 1}, {small, 1}, {-big, 0}};
      auto view = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span{input}, 4, 1);
      lc::Input exported{{unsigned(cases),1,4,1,0,unsigned(paged),std::uint32_t(exponent),0},{},{}};
      lc::Observation observed;std::unique_ptr<Oracle> reference;
      if(report){exported.original=Flatten(input);exported.sampled=exported.original;reference=std::make_unique<Oracle>(view,2);}
      auto output = metal::MakeLeafSums(2, 1, metal::MakeGradientQuantizer(view));
      if (paged) {
        auto indices = metal::MakeDeviceRows();
        indices->Reset(3, 0);
        indices->AddLeafGradients(std::vector<metal::LeafSumSegment>{{0, 3, 1, 0}}, view,
                                  output.get());
        Same(output->Data()[1], {double(big), 2});  // Rounded view loses both small terms.
        if(report){for(unsigned r=0;r<3;++r)reference->Add(1,0,input[r],false);reference->Observe(*output,&observed);}
        indices->Reset(1, 3);
        indices->AddLeafGradients(std::vector<metal::LeafSumSegment>{{0, 1, 1, 0}}, view,
                                  output.get());
        if(report)reference->Add(1,0,input[3],false);
      } else {
        auto indices = metal::MakeDeviceRows();
        indices->Reset(4, 0);
        indices->AddLeafGradients(std::vector<metal::LeafSumSegment>{{0, 4, 1, 0}}, view,
                                  output.get());
        if(report)for(auto value:input)reference->Add(1,0,value,false);
      }
      Same(output->Data()[0], {});
      Same(output->Data()[1], {2. * small, 2});
      if(report){reference->Observe(*output,&observed);report->Add(exported,observed);}
      ++cases;
    }
}
void CurvatureAndRange() {
  current = "curvature/original-scale/range";
  std::vector<GradientPair> original{{0, 0x1p65f}, {-1, 0x1p-149f}, {-1, 0x1p-100f}, {0, 0}};
  auto view = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span{original}, 4, 1);
  lc::Input exported{{250,2,4,1,0,0,0,0},{},{}};lc::Observation observed;
  std::unique_ptr<Oracle> reference;
  if(report){exported.original=Flatten(original);reference=std::make_unique<Oracle>(view,1);}
  auto output = metal::MakeLeafSums(1, 1, metal::MakeGradientQuantizer(view));
  original[0] = {};  // Large original Hessian is rejected. Scale must remain 16.
  original[1] *= 1.5f;
  original[2] *= 1.5f;
  auto rows = metal::MakeDeviceRows();
  rows->Reset(4, 0);
  rows->AddLeafGradients(std::vector<metal::LeafSumSegment>{{0, 4, 0, 0}}, view, output.get());
  Same(output->Data()[0], {-3, 32});
  if(report) {
    exported.sampled=Flatten(original);for(auto value:original)reference->Add(0,0,value,false);
    reference->Observe(*output,&observed);report->Add(exported,observed);
  }
  unsigned range_id=251;
  for (float value : {-4.f, std::nextafter(4.f, 0.f), 4.f, INFINITY, NAN}) {
    std::vector<GradientPair> source{{1, 1}};
    auto source_view = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span{source}, 1, 1);
    lc::Input range_input{{range_id++,3,1,1,0,0,lc::Bits<std::uint32_t>(value),unsigned(!std::isfinite(value)||value>=4)},{},{}};
    lc::Observation range_observed;std::unique_ptr<Oracle> range_reference;
    if(report){range_input.original=Flatten(source);range_reference=std::make_unique<Oracle>(source_view,1);}
    auto sums = metal::MakeLeafSums(1, 1, metal::MakeGradientQuantizer(source_view));
    source[0] = {value, 1};
    rows->Reset(1, 0);
    bool failed = false;
    try {
      rows->AddLeafGradients(std::vector<metal::LeafSumSegment>{{0, 1, 0, 0}}, source_view,
                             sums.get());
    } catch (dmlc::Error const& e) {
      failed = std::string(e.what()).find("original quantizer's int64 range") != std::string::npos;
    }
    Check(failed == (!std::isfinite(value) || value >= 4), "CUDA conversion domain mismatch");
    if (!failed) Same(sums->Data()[0], {value, 1});
    if(report) {
      range_input.sampled=Flatten(source);range_observed.status=failed;
      if(!failed){range_reference->Add(0,0,source[0],false);range_reference->Observe(*sums,&range_observed);}
      report->Add(range_input,range_observed);
    }
  }
  std::puts("PASS cancellation, original scales, positive curvature and conversion-domain cases");
}
int main(int argc,char** argv) {
  try {
    report=lc::ReportOption(argc,argv);
    for (std::size_t n : {0, 1, 31, 257, 4097, 65539})
      for (std::size_t t : {1, 3})
        for (unsigned layout = 0; layout < 5; ++layout)
          for (unsigned mode = 0; mode < 4; ++mode) Run(n, t, layout, mode);
    Run(1000003, 3, 0, 0);
    Run(1000003, 3, 1, 0);
    Cancellation();
    CurvatureAndRange();
    if(report)report->Finish();
    std::printf("PASS %zu integer leaf cases / %zu source pairs / %zu exact sum fields\n", cases,
                pairs, fields);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), e.what());
    return 1;
  }
}
