// Exact input/output interchange for the existing 64 ranking-rounding cases.
// Repetitions use fresh CUDA state to observe the source's shared-factor race.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ranking_rounding {
constexpr std::uint32_t kCases = 64, kInputs = 0, kMetal = 1, kCuda = 2;
inline constexpr char kMagic[8] = {'X','G','B','R','N','D','1','\0'};
inline void Require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
template <typename To, typename From> To Bits(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result; std::memcpy(&result, &value, sizeof(result)); return result;
}
template <typename Word> void Write(std::ostream& stream, Word value) {
  char bytes[sizeof(Word)];
  for (std::size_t i = 0; i < sizeof(Word); ++i) bytes[i] = (value >> (8*i)) & 255;
  stream.write(bytes, sizeof(bytes));
}
template <typename Word> Word Read(std::istream& stream) {
  unsigned char bytes[sizeof(Word)];
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  Word value = 0;
  for (std::size_t i = 0; i < sizeof(Word); ++i) value |= Word{bytes[i]} << (8*i);
  return value;
}
template <typename T> using Word = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
template <typename T> void Values(std::ostream& stream, std::vector<T> const& values) {
  for (auto value : values) Write(stream, Bits<Word<T>>(value));
}
template <typename T> std::vector<T> Values(std::istream& stream, std::size_t count) {
  std::vector<T> values(count);
  for (auto& value : values) value = Bits<T>(Read<Word<T>>(stream));
  return values;
}
struct Shape { std::uint32_t objective, mean, normalized, largest, mode, rows, positions; };
inline Shape Layout(std::uint32_t index) {
  Require(index < kCases, "Too many ranking cases");
  auto largest = index/4%2 ? 65537u : 257u, mean = index/16%2;
  return {index/32, mean, index/8%2, largest, index%4, 129+largest, mean ? 32u : 8u};
}
inline std::uint32_t Repetitions(std::uint32_t kind, Shape shape) {
  return kind == kInputs ? 0 : kind == kCuda && shape.largest == 257 ? 32 : 1;
}
struct Input {
  Shape shape;
  std::vector<float> predictions, labels, weights;
  std::vector<double> positive, negative;
};
struct Group {
  float normalization, gradient_rounding, hessian_rounding, weight;
  double cost_rounding;
};
struct Observation {
  double global_cost_rounding, weight_norm;
  std::vector<Group> groups;
  std::vector<float> gradients;
  std::vector<double> prefix, positive, negative;
};
inline std::array<std::uint32_t, 7> Identity(Shape s) {
  return {s.objective,s.mean,s.normalized,s.largest,s.mode,s.rows,s.positions};
}
inline void Validate(Input const& in, std::uint32_t index) {
  auto s = Layout(index);
  Require(Identity(in.shape) == Identity(s), "Wrong ranking case identity/order");
  Require(in.predictions.size() == s.rows && in.labels.size() == s.rows &&
          in.weights.size() == 5 && in.positive.size() == s.positions &&
          in.negative.size() == s.positions, "Wrong ranking input shape");
}
class Writer {
  std::ofstream stream_;
  std::uint32_t kind_, cases_{0}, repetitions_{0}, written_{0};
  Shape shape_{};
 public:
  Writer(std::string const& path, std::uint32_t kind) : kind_{kind} {
    Require(kind <= kCuda, "Unknown ranking backend");
    Require(!std::filesystem::exists(path), "Ranking output already exists");
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    stream_.open(path, std::ios::binary);
    stream_.write(kMagic, sizeof(kMagic)); Write(stream_, kind); Write(stream_, kCases);
  }
  void Case(Input const& in) {
    Require(written_ == repetitions_, "Incomplete ranking repetitions");
    Validate(in, cases_); shape_ = in.shape;
    repetitions_ = Repetitions(kind_, shape_); written_ = 0;
    for (auto word : Identity(shape_)) Write(stream_, word);
    Write(stream_, repetitions_);
    Values(stream_, in.predictions); Values(stream_, in.labels); Values(stream_, in.weights);
    Values(stream_, in.positive); Values(stream_, in.negative);
    ++cases_;
  }
  void Observe(Observation const& out) {
    Require(cases_ && written_ < repetitions_, "Unexpected ranking observation");
    Require(out.groups.size() == 5 && out.gradients.size() == 2*shape_.rows &&
            out.prefix.size() == (shape_.objective ? 2*shape_.rows : 0) &&
            out.positive.size() == shape_.positions && out.negative.size() == shape_.positions,
            "Wrong ranking observation shape");
    Write(stream_, written_);
    Write(stream_, Bits<std::uint64_t>(out.global_cost_rounding));
    Write(stream_, Bits<std::uint64_t>(out.weight_norm));
    for (auto g : out.groups) {
      for (float value : {g.normalization,g.gradient_rounding,g.hessian_rounding,g.weight})
        Write(stream_, Bits<std::uint32_t>(value));
      Write(stream_, Bits<std::uint64_t>(g.cost_rounding));
    }
    Values(stream_, out.gradients); Values(stream_, out.prefix);
    Values(stream_, out.positive); Values(stream_, out.negative);
    ++written_; stream_.flush();
  }
  void Finish() {
    Require(cases_ == kCases && written_ == repetitions_, "Incomplete ranking corpus");
    stream_.write("RNDE", 4); Write(stream_, kCases); stream_.flush();
  }
};
class InputReader {
  std::ifstream stream_;
  std::uint32_t index_{0};
 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    stream_.open(path, std::ios::binary);
    char magic[8]; stream_.read(magic, sizeof(magic));
    Require(!std::memcmp(magic, kMagic, sizeof(magic)), "Unknown ranking format");
    Require(Read<std::uint32_t>(stream_) == kInputs && Read<std::uint32_t>(stream_) == kCases,
            "CUDA requires the complete input-only ranking corpus");
  }
  Input Next() {
    Input in; in.shape = Layout(index_);
    for (auto expected : Identity(in.shape))
      Require(Read<std::uint32_t>(stream_) == expected, "Invalid ranking identity/order/shape");
    Require(Read<std::uint32_t>(stream_) == 0, "Input file contains observations");
    in.predictions = Values<float>(stream_, in.shape.rows);
    in.labels = Values<float>(stream_, in.shape.rows); in.weights = Values<float>(stream_, 5);
    in.positive = Values<double>(stream_, in.shape.positions);
    in.negative = Values<double>(stream_, in.shape.positions);
    Validate(in, index_++); return in;
  }
  void Finish() {
    Require(index_ == kCases, "Unconsumed ranking inputs");
    char footer[4]; stream_.read(footer, sizeof(footer));
    Require(!std::memcmp(footer, "RNDE", 4) && Read<std::uint32_t>(stream_) == kCases,
            "Missing ranking completion marker");
    Require(stream_.peek() == std::char_traits<char>::eof(), "Trailing ranking bytes");
  }
};
static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
              std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
}  // namespace ranking_rounding
