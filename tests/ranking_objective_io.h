// Lossless inputs and public objective outputs for the complete existing ranking
// gradient and unbiased suites. Reuse the ranking codec's scalar primitives.
#pragma once
#include <algorithm>
#include <numeric>
#include "ranking_rounding_io.h"

namespace ranking_objective {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kGradient = 0, kUnbiased = 1;
inline constexpr char kMagic[8] = {'X','G','B','R','O','B','1','\0'};
struct Spec {
  std::uint32_t sequence, iteration, objective, mean, pairs, normalized, score,
      exponential, unbiased, batched, mode, seed, rows, groups, weights, positions;
  double bias_norm;
  std::vector<std::uint32_t> group_ptr;
};
inline std::array<std::uint32_t, 16> Identity(Spec const& s) {
  return {s.sequence,s.iteration,s.objective,s.mean,s.pairs,s.normalized,s.score,
          s.exponential,s.unbiased,s.batched,s.mode,s.seed,s.rows,s.groups,s.weights,s.positions};
}
inline std::vector<Spec> Specs(unsigned suite) {
  Require(suite <= kUnbiased, "Unknown ranking objective suite");
  std::vector<Spec> out;
  auto add = [&](Spec s, std::vector<std::uint32_t> const& sizes) {
    s.group_ptr = {0};
    for (auto n : sizes) s.group_ptr.push_back(s.group_ptr.back()+n);
    s.rows = s.group_ptr.back(); s.groups = sizes.size();
    s.weights = s.weights ? s.groups : 0;
    s.positions = !s.unbiased ? 0 : s.mean ? std::min(32u, *std::max_element(sizes.begin(), sizes.end())) : s.pairs;
    out.push_back(std::move(s));
  };
  if (suite == kGradient) {
    for (unsigned mean : {0u,1u}) {
      for (unsigned objective = 0; objective < 3; ++objective)
        for (auto sizes : {std::vector<std::uint32_t>{1}, {7,31,33}, {255,256,257}, {4097,31,1}})
          for (unsigned k : {1u,8u,32u})
            for (unsigned norm : {0u,1u}) for (unsigned score : {0u,1u})
              for (unsigned mode = 0; mode < 5; ++mode) {
                auto index = static_cast<std::uint32_t>(out.size());
                add({index,0,objective,mean,k,norm,score,mode!=3,0,0,mode,
                     17891+index,0,0,mode>=2,0,1.0,{}}, sizes);
              }
      auto index = static_cast<std::uint32_t>(out.size());
      add({index,0,0,mean,32,1,1,1,0,1,0,17891+index,0,0,0,0,1.0,{}},
          std::vector<std::uint32_t>(2560,128));
    }
    Require(out.size() == 1442, "Incomplete gradient case matrix");
  } else {
    for (unsigned mean : {0u,1u}) for (unsigned iteration = 0; iteration < 2; ++iteration)
      add({mean,iteration,0,mean,8,1,1,1,1,1,0,12345+iteration,0,0,0,0,1.0,{}},
          std::vector<std::uint32_t>(4096,128));
    unsigned sequence = 2, calls = 0;
    for (unsigned objective = 0; objective < 3; ++objective)
      for (unsigned mean : {0u,1u}) for (unsigned k : {1u,8u,32u})
        for (unsigned norm : {0u,1u}) for (unsigned score : {0u,1u})
          for (double bias : {0.,1.,2.5}) for (unsigned mode = 0; mode < 4; ++mode) {
            for (unsigned iteration = 0; iteration < 3; ++iteration) {
              add({sequence,iteration,objective,mean,k,norm,score,1,1,0,mode,
                   8912+calls,0,0,mode==3,0,bias,{}}, {1,7,31,33,257});
              ++calls;
            }
            ++sequence;
          }
    Require(out.size() == 2596, "Incomplete unbiased case matrix");
  }
  return out;
}
struct Input {
  Spec spec;
  std::vector<float> predictions, labels, weights;
};
struct Observation {
  std::vector<float> gradients, positive, negative;
  std::uint32_t next_rng;
};
inline bool SameBits(std::vector<float> const& a, std::vector<float> const& b) {
  return a.size() == b.size() && (a.empty() || !std::memcmp(a.data(), b.data(), a.size()*sizeof(float)));
}
inline void Validate(Input const& in, Spec const& s) {
  Require(Identity(in.spec) == Identity(s) &&
          Bits<std::uint64_t>(in.spec.bias_norm) == Bits<std::uint64_t>(s.bias_norm),
          "Wrong ranking objective identity/order/parameters");
  Require(in.spec.group_ptr == s.group_ptr && in.predictions.size() == s.rows &&
          in.labels.size() == s.rows && in.weights.size() == s.weights,
          "Wrong ranking objective input shape/groups");
}
inline void CheckContinuation(Input const& in, Input const& previous) {
  if (in.spec.iteration) {
    Require(in.spec.sequence == previous.spec.sequence &&
            in.spec.iteration == previous.spec.iteration+1 &&
            SameBits(in.labels, previous.labels) && SameBits(in.weights, previous.weights),
            "Ranking sequence metadata changed between objective calls");
  }
}
class Writer {
  std::ofstream stream_;
  unsigned kind_, suite_, index_{0};
  std::vector<Spec> specs_;
  Input previous_{};
 public:
  Writer(std::string const& path, unsigned kind, unsigned suite)
      : kind_{kind}, suite_{suite}, specs_{Specs(suite)} {
    Require(kind <= 2 && !std::filesystem::exists(path), "Unknown backend or existing objective report");
    stream_.exceptions(std::ios::failbit | std::ios::badbit); stream_.open(path, std::ios::binary);
    stream_.write(kMagic, sizeof(kMagic)); Write(stream_, std::uint32_t{kind});
    Write(stream_, std::uint32_t{suite}); Write(stream_, static_cast<std::uint32_t>(specs_.size()));
  }
  void Add(Input const& in, Observation const& observed = {}) {
    Require(index_ < specs_.size(), "Too many objective cases");
    Validate(in, specs_[index_]); CheckContinuation(in, previous_);
    auto s = in.spec;
    Require(observed.gradients.size() == (kind_ ? 2*s.rows : 0) &&
            observed.positive.size() == (kind_ ? s.positions : 0) &&
            observed.negative.size() == (kind_ ? s.positions : 0), "Wrong objective observation shape");
    for (auto value : Identity(s)) Write(stream_, value);
    Write(stream_, Bits<std::uint64_t>(s.bias_norm));
    for (auto value : s.group_ptr) Write(stream_, value);
    Values(stream_, in.predictions); Values(stream_, in.labels); Values(stream_, in.weights);
    if (kind_) {
      Values(stream_, observed.gradients); Values(stream_, observed.positive); Values(stream_, observed.negative);
      Write(stream_, observed.next_rng);
    }
    previous_ = in; ++index_; stream_.flush();
  }
  void Finish() {
    Require(index_ == specs_.size(), "Incomplete objective corpus");
    stream_.write("ROBE", 4); Write(stream_, std::uint32_t{suite_}); Write(stream_, index_); stream_.flush();
  }
};
class InputReader {
  std::ifstream stream_;
  unsigned suite_, index_{0};
  std::vector<Spec> specs_;
  Input previous_{};
 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::failbit | std::ios::badbit); stream_.open(path, std::ios::binary);
    char magic[8]; stream_.read(magic, sizeof(magic));
    Require(!std::memcmp(magic, kMagic, sizeof(magic)), "Unknown objective interchange version");
    Require(Read<std::uint32_t>(stream_) == 0, "CUDA requires input-only objective records");
    suite_ = Read<std::uint32_t>(stream_); specs_ = Specs(suite_);
    Require(Read<std::uint32_t>(stream_) == specs_.size(), "Wrong objective corpus size");
  }
  unsigned Suite() const { return suite_; }
  std::size_t Size() const { return specs_.size(); }
  Input Next() {
    Require(index_ < specs_.size(), "Too many input cases");
    Input in; in.spec = specs_[index_];
    for (auto expected : Identity(in.spec))
      Require(Read<std::uint32_t>(stream_) == expected, "Wrong objective identity/order/parameters");
    Require(Read<std::uint64_t>(stream_) == Bits<std::uint64_t>(in.spec.bias_norm), "Wrong bias norm");
    for (auto expected : in.spec.group_ptr)
      Require(Read<std::uint32_t>(stream_) == expected, "Wrong query boundaries");
    in.predictions = Values<float>(stream_, in.spec.rows); in.labels = Values<float>(stream_, in.spec.rows);
    in.weights = Values<float>(stream_, in.spec.weights);
    Validate(in, specs_[index_]); CheckContinuation(in, previous_); previous_ = in; ++index_;
    return in;
  }
  void Finish() {
    Require(index_ == specs_.size(), "Unconsumed objective inputs");
    char footer[4]; stream_.read(footer, sizeof(footer));
    Require(!std::memcmp(footer, "ROBE", 4) && Read<std::uint32_t>(stream_) == suite_ &&
            Read<std::uint32_t>(stream_) == index_, "Missing objective completion marker");
    Require(stream_.peek() == std::char_traits<char>::eof(), "Trailing objective bytes");
  }
};
}  // namespace ranking_objective
