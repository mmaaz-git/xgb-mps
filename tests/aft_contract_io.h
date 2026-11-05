// Lossless little-endian interchange for the existing 96-case AFT corpus.
// CUDA reads an input-only file; Metal observations never feed GPU computation.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aft_contracts {
constexpr std::uint32_t kCases = 96, kInputs = 0, kMetal = 1, kCuda = 2;
inline constexpr char kMagic[8] = {'X', 'G', 'B', 'A', 'F', 'T', '1', '\0'};
static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
              std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
inline void Require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename To, typename From> To Bits(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
struct Input {
  std::uint32_t distribution, mode;
  float scale;
  std::vector<float> lower, upper, prediction, weights;
};
struct Observation {
  double gradient, hessian, likelihood;
  float weighted_gradient, weighted_hessian;
};
static_assert(sizeof(Observation) == 32);

inline void Validate(Input const& input, std::uint32_t index) {
  Require(index < kCases, "Too many AFT cases");
  auto within = index % 32;
  float scales[]{.1f, .5f, 1.f, 2.f, 10.f};
  float guards[]{.1f, 1.f, 0.f, -1.f, 1e-40f, INFINITY, NAN};
  auto mode = within < 25 ? within % 5 : 5;
  float scale = within < 25 ? scales[within / 5] : guards[within - 25];
  std::size_t rows = within < 25 ? 4097 : 448;
  Require(input.distribution == index / 32 && input.mode == mode &&
              (Bits<std::uint32_t>(input.scale) == Bits<std::uint32_t>(scale) ||
               (std::isnan(input.scale) && std::isnan(scale))), "Wrong AFT case identity/order");
  Require(input.lower.size() == rows && input.upper.size() == rows &&
              input.prediction.size() == rows && input.weights.size() == (mode ? rows : 0),
          "Wrong AFT input shape");
}

class Writer {
  std::ofstream stream_;
  std::uint32_t kind_, count_{0};
  template <typename T> void Word(T word) {
    char bytes[sizeof(T)];
    for (std::size_t i = 0; i < sizeof(T); ++i) bytes[i] = (word >> (8*i)) & 255;
    stream_.write(bytes, sizeof(bytes));
  }
  void Floats(std::vector<float> const& values) {
    for (float value : values) Word(Bits<std::uint32_t>(value));
  }

 public:
  Writer(std::string const& path, std::uint32_t kind) : kind_{kind} {
    Require(kind <= kCuda, "Unknown AFT file kind");
    Require(!std::ifstream{path, std::ios::binary}.good(), "AFT output already exists");
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    stream_.open(path, std::ios::binary);
    stream_.write(kMagic, sizeof(kMagic));
    Word(kind);
    Word(kCases);
  }
  void Add(Input const& input, std::vector<Observation> const& observations = {}) {
    Validate(input, count_);
    Require(observations.size() == (kind_ == kInputs ? 0 : input.lower.size()),
            "Wrong AFT observation shape");
    Word(input.distribution);
    Word(input.mode);
    Word(Bits<std::uint32_t>(input.scale));
    Word(static_cast<std::uint32_t>(input.lower.size()));
    Word(static_cast<std::uint32_t>(input.weights.size()));
    Floats(input.lower); Floats(input.upper); Floats(input.prediction); Floats(input.weights);
    for (auto const& o : observations) {
      Word(Bits<std::uint64_t>(o.gradient));
      Word(Bits<std::uint64_t>(o.hessian));
      Word(Bits<std::uint64_t>(o.likelihood));
      Word(Bits<std::uint32_t>(o.weighted_gradient));
      Word(Bits<std::uint32_t>(o.weighted_hessian));
    }
    ++count_;
    stream_.flush();
  }
  void Finish() {
    Require(count_ == kCases, "Incomplete AFT corpus");
    stream_.write("AFTE", 4);
    Word(count_);
    stream_.flush();
  }
};

class InputReader {
  std::ifstream stream_;
  std::uint32_t count_{0};
  std::uint32_t Word() {
    unsigned char bytes[4];
    stream_.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    return std::uint32_t{bytes[0]} | (std::uint32_t{bytes[1]} << 8) |
           (std::uint32_t{bytes[2]} << 16) | (std::uint32_t{bytes[3]} << 24);
  }
  std::vector<float> Floats(std::size_t size) {
    std::vector<float> values(size);
    for (auto& value : values) value = Bits<float>(Word());
    return values;
  }

 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    stream_.open(path, std::ios::binary);
    char magic[8];
    stream_.read(magic, sizeof(magic));
    Require(!std::memcmp(magic, kMagic, sizeof(magic)), "Unknown AFT interchange version");
    Require(Word() == kInputs && Word() == kCases, "CUDA requires the complete input-only AFT file");
  }
  Input Next() {
    Input input{};
    input.distribution = Word(); input.mode = Word(); input.scale = Bits<float>(Word());
    auto rows = Word(), weights = Word();
    Require(rows <= 4097 && weights <= rows, "Invalid AFT allocation size");
    input.lower = Floats(rows); input.upper = Floats(rows);
    input.prediction = Floats(rows); input.weights = Floats(weights);
    Validate(input, count_++);
    return input;
  }
  void Finish() {
    Require(count_ == kCases, "AFT input was not fully consumed");
    char footer[4];
    stream_.read(footer, sizeof(footer));
    Require(!std::memcmp(footer, "AFTE", sizeof(footer)) && Word() == kCases,
            "Missing AFT completion marker");
    Require(stream_.peek() == std::char_traits<char>::eof(), "Trailing AFT input bytes");
  }
};
}  // namespace aft_contracts
