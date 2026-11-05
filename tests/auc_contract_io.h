// Exact interchange for every existing metal_auc_api case, in execution order.
// CUDA's public binary triples and multi-output scalar/metadata are distinct
// observations. Source-copy diagnostics never stand in for linked-library calls.
#pragma once
#include <algorithm>
#include <memory>
#include "ranking_rounding_io.h"

namespace auc_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases = 528, kInput = 0, kMetal = 1, kCuda = 2;
inline constexpr char kMagic[8] = {'X','G','B','A','U','C','1','\0'};
enum Family : unsigned { kEdge, kNormalize, kMetadata, kMalformed, kWide, kMain, kExtreme };
struct Spec {
  std::uint32_t id, family, mode, layout, rows, targets, multiclass, roc, weights, summaries;
};
inline auto Identity(Spec s) {
  return std::array<std::uint32_t, 10>{s.id, s.family, s.mode, s.layout, s.rows, s.targets,
                                      s.multiclass, s.roc, s.weights, s.summaries};
}
inline std::vector<Spec> Specs() {
  std::vector<Spec> out;
  auto add = [&](unsigned family, unsigned mode, unsigned layout, unsigned rows,
                 unsigned targets, unsigned multiclass, unsigned roc, unsigned weights,
                 unsigned summaries = 0) {
    out.push_back({unsigned(out.size()), family, mode, layout, rows, targets, multiclass,
                   roc, weights, summaries});
  };
  for (unsigned i = 0; i < 4; ++i) add(kEdge, i, 0, i < 2 ? 1 : 2, 1, 0, i % 2 == 0, i < 2 ? 0 : 2);
  unsigned mode = 0;
  for (unsigned n : {3u,6u,3u,3u,3u,3u,3u,3u,6u,3u,3u,3u})
    add(kNormalize, mode++, 0, 0, n / 3, 0, 0, 0, n);
  for (unsigned n : {1u,2u,31u,257u,65535u,65536u,65537u,262145u})
    add(kNormalize, mode++, 0, 0, n, 0, 0, 0, 3 * n);
  for (unsigned multiclass : {0u, 1u}) for (unsigned roc : {0u, 1u})
    for (unsigned rows : {0u, 4u}) add(kMetadata, 0, 0, rows, 3, multiclass, roc, rows);
  add(kNormalize, mode++, 0, 0, 0, 0, 0, 0, 0);
  add(kNormalize, mode++, 0, 0, 65536, 0, 0, 0, 3 * 65536);
  add(kMalformed, 0, 0, 0, 0, 0, 0, 0, 2);
  for (unsigned targets : {65535u,65536u,65537u}) for (unsigned roc : {1u,0u})
    add(kWide, 0, 0, 2, targets, 0, roc, 0);
  for (unsigned rows : {0u,1u,2u,31u,255u,256u,257u,4097u,65537u})
    for (unsigned targets : {1u,3u}) for (unsigned layout = 0; layout < 3; ++layout)
      for (unsigned m = 0; m < 9; ++m)
        add(kMain, m, layout, rows, targets, targets == 3 && m % 2, 2, m >= 2 ? rows : 0);
  add(kExtreme, 0, 0, 2, 1, 0, 0, 2);
  Require(out.size() == kCases && mode == 22, "Incomplete AUC control matrix");
  return out;
}
inline unsigned LabelColumns(Spec s) { return s.multiclass ? 1 : s.targets; }
inline std::array<std::size_t, 2> Strides(Spec s, unsigned columns) {
  if (s.family != kMain) return {columns, 1};
  return {s.layout == 1 ? 1 : columns * (s.layout == 2 ? 2 : 1) + (s.layout == 2 ? 3 : 0),
          s.layout == 1 ? std::max(s.rows, 1u) : s.layout == 2 ? 2u : 1u};
}
inline std::size_t Storage(Spec s, unsigned columns) {
  auto stride = Strides(s, columns);
  return s.family == kMain ? s.rows * stride[0] + columns * stride[1] + 1
                           : std::size_t{s.rows} * columns;
}
inline bool IsROC(Spec s, unsigned call) { return s.roc == 2 ? call == 0 : s.roc != 0; }
inline bool IsMulti(Spec s) { return s.targets > 1 || s.multiclass; }
inline bool Trace(Spec s, unsigned call) { return s.family == kMain && s.mode == 4 && !IsROC(s, call); }
inline unsigned Calls(Spec s) { return s.family == kMain ? 2 : 1; }
struct Input {
  Spec spec;
  std::vector<float> predictions, labels, weights;
  std::vector<double> summaries;
};
// Presence is encoded by vector length. An empty skipped producer is distinct
// from an observed zero-valued vector; a scalar NaN is always one stored double.
struct Observation {
  unsigned status{0};  // 0 actual result, 1 observed source guard, 2 no matching API shape.
  std::vector<double> curves, source_curves, metadata, final, cpu, linked, guards;
  std::vector<double> terms, prefixes, totals;
  std::vector<std::uint64_t> order;
};
inline void Validate(Input const& in, Spec s) {
  Require(Identity(in.spec) == Identity(s) &&
              in.predictions.size() == Storage(s, s.targets) &&
              in.labels.size() == Storage(s, LabelColumns(s)) && in.weights.size() == s.weights &&
              in.summaries.size() == s.summaries, "Wrong AUC identity/order/input shape");
}
inline void ValidateMetal(Spec s, unsigned call, Observation const& out) {
  bool curve = s.family == kEdge || s.family == kWide || s.family == kMain || s.family == kExtreme;
  bool metadata = s.family == kMetadata || (curve && IsMulti(s));
  bool final = s.family == kNormalize || s.family == kWide || metadata;
  bool traced = Trace(s, call) && s.rows;
  Require(out.status == unsigned(s.family == kMalformed), "Wrong Metal AUC guard status");
  Require(out.curves.size() == (curve ? 3 * s.targets : 0) &&
              out.metadata.size() == (metadata ? 3 * s.targets : 0) &&
              out.final.size() == unsigned(final) &&
              out.cpu.size() == unsigned(s.family == kWide || s.family == kExtreme) &&
              out.guards.size() == (s.family == kMetadata ? 4u : 0u) &&
              out.terms.size() == (traced ? 2 * s.rows * s.targets : 0) &&
              out.prefixes.size() == out.terms.size() &&
              out.totals.size() == (traced ? 2 * s.targets : 0) &&
              out.source_curves.empty() && out.linked.empty() && out.order.empty(),
          "Incomplete or unexpected Metal AUC observations");
}
inline void ValidateCuda(Spec s, unsigned call, Observation const& out) {
  bool malformed = s.family == kMalformed;
  bool normalize = s.family == kNormalize;
  bool multi = !normalize && IsMulti(s);
  bool roc = IsROC(s, call);
  bool guard = !malformed && !normalize && !multi && roc && !s.rows;
  bool binary = !malformed && !normalize && !multi && !guard;
  bool totals = !malformed && !normalize && !roc;
  Require(out.status == (malformed ? 2u : guard ? 1u : 0u), "Wrong CUDA AUC guard status");
  Require(out.totals.size() == (totals ? 2 * s.targets : 0), "Missing original CUDA PR totals");
  bool traced = Trace(s, call) && s.rows &&
                (multi || (out.totals[0] > 0 && out.totals[1] > 0));
  Require(out.curves.size() == (binary ? 3u : 0u) &&
              out.source_curves.size() == (multi && s.rows ? 3 * s.targets : 0) &&
              out.metadata.size() == (multi || normalize ? 3 * s.targets : 0) &&
              out.final.size() == unsigned(multi || normalize) &&
              out.linked.size() == (binary ? 3u : multi || normalize ? 1u : 0u) &&
              out.cpu.empty() && out.guards.empty() &&
              out.terms.size() == (traced ? 2 * s.rows * s.targets : 0) &&
              out.prefixes.size() == out.terms.size() &&
              out.order.size() == (traced ? s.rows * s.targets : 0),
          "Incomplete or unexpected CUDA AUC observations");
}
class Writer {
  std::ofstream stream_;
  unsigned kind_, index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path, unsigned kind) : kind_{kind} {
    Require(kind <= kCuda && !std::filesystem::exists(path),
            "Unknown backend or existing AUC report");
    stream_.exceptions(std::ios::badbit | std::ios::failbit);
    stream_.open(path, std::ios::binary);
    stream_.write(kMagic, 8); Write(stream_, kind); Write(stream_, kCases);
  }
  void Add(Input const& in, std::vector<Observation> const& calls = {}) {
    Require(index_ < kCases, "Too many AUC records"); Validate(in, specs_[index_]);
    Require(calls.size() == (kind_ ? Calls(in.spec) : 0), "Missing AUC calls");
    for (auto word : Identity(in.spec)) Write(stream_, word);
    Values(stream_, in.predictions); Values(stream_, in.labels); Values(stream_, in.weights);
    Values(stream_, in.summaries);
    for (unsigned i = 0; i < calls.size(); ++i) {
      auto const& out = calls[i];
      if (kind_ == kMetal) ValidateMetal(in.spec, i, out);
      else ValidateCuda(in.spec, i, out);
      Write(stream_, out.status);
      for (auto values : {&out.curves, &out.source_curves, &out.metadata, &out.final, &out.cpu,
                         &out.linked, &out.guards, &out.terms, &out.prefixes, &out.totals}) {
        Write(stream_, std::uint64_t(values->size())); Values(stream_, *values);
      }
      Write(stream_, std::uint64_t(out.order.size())); Values(stream_, out.order);
    }
    ++index_; stream_.flush();
  }
  void Finish() {
    Require(index_ == kCases, "Incomplete AUC corpus");
    stream_.write("AUCE", 4); Write(stream_, index_); stream_.flush();
  }
};
class InputReader {
  std::ifstream stream_;
  unsigned index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::badbit | std::ios::failbit); stream_.open(path, std::ios::binary);
    char magic[8]; stream_.read(magic, 8);
    Require(!std::memcmp(magic, kMagic, 8) && Read<unsigned>(stream_) == kInput &&
                Read<unsigned>(stream_) == kCases, "Requires complete input-only AUC corpus");
  }
  Input Next() {
    Require(index_ < kCases, "Too many AUC inputs"); auto s = specs_[index_];
    for (auto word : Identity(s)) Require(Read<unsigned>(stream_) == word, "Wrong AUC identity/order");
    Input in{s, Values<float>(stream_, Storage(s, s.targets)),
             Values<float>(stream_, Storage(s, LabelColumns(s))), Values<float>(stream_, s.weights),
             Values<double>(stream_, s.summaries)};
    Validate(in, s); ++index_; return in;
  }
  void Finish() {
    Require(index_ == kCases, "Unconsumed AUC inputs"); char end[4]; stream_.read(end, 4);
    Require(!std::memcmp(end, "AUCE", 4) && Read<unsigned>(stream_) == kCases &&
                stream_.peek() == std::char_traits<char>::eof(), "Incomplete/trailing AUC inputs");
  }
};
class Exporter {
  unsigned index_{0};
  std::vector<Spec> specs_{Specs()};
  Writer inputs_, metal_;
 public:
  explicit Exporter(std::string const& prefix) : inputs_{prefix + ".inputs.bin", kInput},
                                                metal_{prefix + ".metal.bin", kMetal} {}
  Spec NextSpec() const { Require(index_ < kCases, "Too many AUC exports"); return specs_[index_]; }
  void Add(Input const& in, std::vector<Observation> const& out) {
    inputs_.Add(in); metal_.Add(in, out); ++index_;
  }
  void Finish() { inputs_.Finish(); metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc, char** argv) {
  Require(argc == 1 || (argc == 3 && std::string(argv[1]) == "--report-prefix"),
          "Usage: metal_auc_api_test [--report-prefix NEW_PREFIX]");
  return argc == 1 ? nullptr : std::make_unique<Exporter>(argv[2]);
}
}  // namespace auc_contracts
