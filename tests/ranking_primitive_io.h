// Lossless interchange for the complete existing ranking rounding boundary test.
// Serialize explicit fields, never native struct padding.
#pragma once
#include <cstddef>
#include "ranking_rounding_io.h"

namespace ranking_primitive {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Write;
constexpr std::uint32_t kCases = 162851, kInput = 0, kMetal = 1, kCuda = 2;
inline constexpr char kMagic[8] = {'X','G','B','R','P','R','1','\0'};
struct Query {
  std::uint32_t maximum, rows, value, padding;
  double cost, minimum, term;
};
struct Result {
  std::uint32_t factor, positive, negative, delta;
  double cost_factor, cost_positive, cost_negative, cost_delta;
};
static_assert(sizeof(Query) == 40 && offsetof(Query, cost) == 16);
static_assert(sizeof(Result) == 48 && offsetof(Result, cost_factor) == 16);

inline void Save(std::string const& path, unsigned kind, std::vector<Query> const& queries,
                 Result const* observations = nullptr) {
  Require(kind <= kCuda && queries.size() == kCases && bool(observations) == (kind != kInput),
          "Invalid ranking primitive backend/count/observations");
  Require(!std::filesystem::exists(path), "Primitive report already exists");
  std::ofstream out; out.exceptions(std::ios::failbit | std::ios::badbit);
  out.open(path, std::ios::binary);
  out.write(kMagic, sizeof(kMagic)); Write(out, std::uint32_t{kind}); Write(out, kCases);
  for (auto const& q : queries) {
    for (auto value : {q.maximum, q.rows, q.value}) Write(out, value);
    for (auto value : {q.cost, q.minimum, q.term}) Write(out, Bits<std::uint64_t>(value));
  }
  if (observations) for (std::size_t i = 0; i < queries.size(); ++i) {
    auto const& r = observations[i];
    for (auto value : {r.factor, r.positive, r.negative, r.delta}) Write(out, value);
    for (auto value : {r.cost_factor, r.cost_positive, r.cost_negative, r.cost_delta})
      Write(out, Bits<std::uint64_t>(value));
  }
  out.write("RPRE", 4); Write(out, kCases); out.flush();
}
inline std::vector<Query> LoadInputs(std::string const& path) {
  std::ifstream in; in.exceptions(std::ios::failbit | std::ios::badbit);
  in.open(path, std::ios::binary);
  char magic[8]; in.read(magic, sizeof(magic));
  Require(!std::memcmp(magic, kMagic, sizeof(magic)), "Unknown primitive interchange version");
  Require(Read<std::uint32_t>(in) == kInput && Read<std::uint32_t>(in) == kCases,
          "CUDA requires the complete input-only primitive corpus");
  std::vector<Query> queries(kCases);
  for (auto& q : queries) {
    q.maximum = Read<std::uint32_t>(in); q.rows = Read<std::uint32_t>(in);
    q.value = Read<std::uint32_t>(in);
    q.cost = Bits<double>(Read<std::uint64_t>(in)); q.minimum = Bits<double>(Read<std::uint64_t>(in));
    q.term = Bits<double>(Read<std::uint64_t>(in));
  }
  char footer[4]; in.read(footer, sizeof(footer));
  Require(!std::memcmp(footer, "RPRE", 4) && Read<std::uint32_t>(in) == kCases,
          "Missing primitive completion marker");
  Require(in.peek() == std::char_traits<char>::eof(), "Trailing primitive bytes");
  return queries;
}
}  // namespace ranking_primitive
