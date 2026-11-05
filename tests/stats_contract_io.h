// Exact input bits for the complete existing statistics API corpus. Reference
// values are stored only in Metal observations, never in CUDA input files.
#pragma once
#include <memory>
#include "ranking_rounding_io.h"

namespace stats_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
using Words = std::vector<std::uint64_t>;
constexpr unsigned kCases = 122, kInput = 0, kMetal = 1, kCuda = 2;
inline constexpr char kMagic[8] = {'X','G','B','S','T','A','1','\0'};
struct Spec {
  std::uint32_t id, kind, rows, columns, mode, values, weights, outputs, scalar;
};
inline std::array<std::uint32_t, 9> Identity(Spec s) {
  return {s.id,s.kind,s.rows,s.columns,s.mode,s.values,s.weights,s.outputs,s.scalar};
}
inline std::vector<Spec> Specs() {
  std::vector<Spec> result;
  for (unsigned rows : {0u,1u,2u,31u,256u,257u,4097u})
    for (unsigned columns : {1u,3u}) for (unsigned mode = 0; mode < 4; ++mode) {
      auto weighted = mode && rows && mode != 2;
      result.push_back({static_cast<unsigned>(result.size()),0,rows,columns,mode,rows*columns,
                        mode ? rows : 0,2*columns+2+(weighted ? columns : 0),Bits<std::uint32_t>(1.f)});
    }
  for (unsigned rows : {0u,1u,2u,31u,256u,257u,4097u,65537u,1000003u})
    for (unsigned mode = 0; mode < 7; ++mode)
      result.push_back({static_cast<unsigned>(result.size()),1,rows,1,mode,rows,0,2,Bits<std::uint32_t>(1.f)});
  for (unsigned mode : {0u,1u})
    result.push_back({static_cast<unsigned>(result.size()),2,2,1,mode,2,0,2,Bits<std::uint32_t>(1.f)});
  result.push_back({static_cast<unsigned>(result.size()),3,11,1,0,0,0,1,Bits<std::uint32_t>(3.5f)});
  Require(result.size() == kCases, "Incomplete statistics matrix");
  return result;
}
struct Input { Spec spec; std::vector<float> values, weights; };
struct Observation { Words actual, reference; double compensated{0}, magnitude{0}; };
inline void Validate(Input const& in, Spec expected) {
  Require(Identity(in.spec) == Identity(expected) && in.values.size() == expected.values &&
          in.weights.size() == expected.weights, "Wrong statistics identity/order/input shape");
}
inline void CheckOutputs(Spec s, Words const& words) {
  Require(words.size() == s.outputs, "Wrong statistics output shape");
  if (s.kind == 0) {
    for (auto word : words) Require(word <= UINT32_MAX, "Nonzero upper FP32/status bits");
    Require(words.back() <= 2, "Unknown weighted-mean status");
  }
}
class Writer {
  std::ofstream out_;
  unsigned kind_, index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path, unsigned kind) : kind_{kind} {
    Require(kind <= kCuda && !std::filesystem::exists(path), "Unknown statistics backend or existing report");
    out_.exceptions(std::ios::badbit | std::ios::failbit); out_.open(path, std::ios::binary);
    out_.write(kMagic, sizeof(kMagic)); Write(out_, std::uint32_t{kind}); Write(out_, std::uint32_t{kCases});
  }
  void Add(Input const& in, Observation const& observed = {}) {
    Require(index_ < kCases, "Too many statistics cases"); Validate(in, specs_[index_]);
    if (kind_) CheckOutputs(in.spec, observed.actual);
    else Require(observed.actual.empty(), "Input-only file contains observations");
    if (kind_ == kMetal) CheckOutputs(in.spec, observed.reference);
    else Require(observed.reference.empty(), "References belong only in Metal reports");
    for (auto word : Identity(in.spec)) Write(out_, word);
    Values(out_, in.values); Values(out_, in.weights);
    for (auto word : observed.actual) Write(out_, word);
    if (kind_ == kMetal) {
      for (auto word : observed.reference) Write(out_, word);
      Write(out_, Bits<std::uint64_t>(observed.compensated));
      Write(out_, Bits<std::uint64_t>(observed.magnitude));
    }
    ++index_; out_.flush();
  }
  void Finish() {
    Require(index_ == kCases, "Incomplete statistics corpus");
    out_.write("STAE", 4); Write(out_, index_); out_.flush();
  }
};
class InputReader {
  std::ifstream in_;
  unsigned index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  explicit InputReader(std::string const& path) {
    in_.exceptions(std::ios::badbit | std::ios::failbit); in_.open(path, std::ios::binary);
    char magic[8]; in_.read(magic, sizeof(magic));
    Require(!std::memcmp(magic, kMagic, sizeof(magic)), "Unknown statistics interchange version");
    Require(Read<std::uint32_t>(in_) == kInput && Read<std::uint32_t>(in_) == kCases,
            "CUDA requires complete input-only statistics records");
  }
  Input Next() {
    Require(index_ < kCases, "Too many statistics inputs"); Input input{specs_[index_],{}, {}};
    for (auto expected : Identity(input.spec))
      Require(Read<std::uint32_t>(in_) == expected, "Wrong statistics identity/order/shape");
    input.values = Values<float>(in_, input.spec.values); input.weights = Values<float>(in_, input.spec.weights);
    ++index_; return input;
  }
  void Finish() {
    Require(index_ == kCases, "Unconsumed statistics inputs");
    char footer[4]; in_.read(footer, sizeof(footer));
    Require(!std::memcmp(footer, "STAE", 4) && Read<std::uint32_t>(in_) == kCases,
            "Missing statistics completion marker");
    Require(in_.peek() == std::char_traits<char>::eof(), "Trailing statistics bytes");
  }
};
class Exporter {
  unsigned index_{0};
  std::vector<Spec> specs_{Specs()};
  Writer inputs_, metal_;
 public:
  explicit Exporter(std::string const& prefix)
      : inputs_{prefix+".inputs.bin", kInput}, metal_{prefix+".metal.bin", kMetal} {}
  void Add(unsigned kind, unsigned rows, unsigned columns, unsigned mode,
           std::vector<float> const& values, std::vector<float> const& weights, Observation const& out) {
    Require(index_ < kCases, "Too many native statistics exports");
    auto s = specs_[index_];
    Require(s.kind == kind && s.rows == rows && s.columns == columns && s.mode == mode,
            "Native statistics parameters differ from matrix");
    Input in{s,values,weights}; inputs_.Add(in); metal_.Add(in,out); ++index_;
  }
  void Finish() { inputs_.Finish(); metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc, char** argv) {
  Require(argc == 1 || (argc == 3 && std::string(argv[1]) == "--report-prefix"),
          "Usage: native-test [--report-prefix new-prefix]");
  if (argc == 1) return {};
  std::string prefix = argv[2];
  Require(!std::filesystem::exists(prefix+".inputs.bin") && !std::filesystem::exists(prefix+".metal.bin"),
          "Use a new statistics report prefix");
  return std::make_unique<Exporter>(prefix);
}
}  // namespace stats_contracts
