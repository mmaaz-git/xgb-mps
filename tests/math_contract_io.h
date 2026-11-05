// Lossless interchange for the existing primitive-math corpora. All cells are
// explicit little-endian uint64 words; FP32 fields occupy their low 32 bits.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace math_contracts {
using Words = std::vector<std::uint64_t>;
constexpr std::uint32_t kInput = 0, kMetal = 1, kCuda = 2, kAll = 15;
inline constexpr char kMagic[8] = {'X', 'G', 'B', 'M', 'A', 'T', '1', '\0'};
struct Shape { std::uint32_t rows, inputs, outputs; };
inline constexpr Shape kShapes[] = {{100196, 2, 6}, {100007, 1, 1},
                                    {80060, 1, 3}, {500400, 2, 2}};
static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
              std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
inline void Require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
template <typename To, typename From> To Bits(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result; std::memcpy(&result, &value, sizeof(result)); return result;
}
inline unsigned Next(std::uint32_t mask, std::uint32_t seen) {
  for (unsigned id = 0; id < 4; ++id)
    if ((mask & (1u << id)) && !(seen & (1u << id))) return id;
  throw std::runtime_error("Too many math groups");
}
inline void CheckWords(unsigned id, Words const& words, bool output) {
  auto shape = kShapes[id];
  auto width = output ? shape.outputs : shape.inputs;
  Require(words.size() == std::size_t{shape.rows}*width, "Wrong math field count");
  for (std::size_t i = 0; i < words.size(); ++i) {
    bool fp32 = id == 3 || (id == 1 && !output) || (id == 0 && output && i%width >= 4);
    Require(!fp32 || words[i] <= UINT32_MAX, "Nonzero upper bits in FP32 math cell");
  }
}
template <typename Word> inline void Write(std::ostream& stream, Word value) {
  char bytes[sizeof(Word)];
  for (std::size_t i = 0; i < sizeof(Word); ++i) bytes[i] = (value >> (8*i)) & 255;
  stream.write(bytes, sizeof(bytes));
}
template <typename Word> inline Word Read(std::istream& stream) {
  unsigned char bytes[sizeof(Word)];
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  Word result = 0;
  for (std::size_t i = 0; i < sizeof(Word); ++i) result |= Word{bytes[i]} << (8*i);
  return result;
}

class Writer {
  std::ofstream stream_;
  std::uint32_t kind_, mask_, seen_{0};
 public:
  Writer(std::string const& path, std::uint32_t kind, std::uint32_t mask)
      : kind_{kind}, mask_{mask} {
    Require(kind <= kCuda && mask && mask <= kAll, "Invalid math report kind/mask");
    Require(!std::filesystem::exists(path), "Math report already exists");
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    stream_.open(path, std::ios::binary);
    stream_.write(kMagic, sizeof(kMagic)); Write(stream_, kind); Write(stream_, mask);
  }
  void Add(unsigned id, Words const& input, Words const& output = {}) {
    Require(id == Next(mask_, seen_), "Wrong math group order");
    CheckWords(id, input, false);
    if (kind_ == kInput) Require(output.empty(), "Input-only file contains observations");
    else CheckWords(id, output, true);
    auto shape = kShapes[id];
    for (std::uint32_t word : {id, shape.rows, shape.inputs, shape.outputs}) Write(stream_, word);
    for (auto word : input) Write(stream_, word);
    for (auto word : output) Write(stream_, word);
    seen_ |= 1u << id;
    stream_.flush();
  }
  void Finish() {
    Require(seen_ == mask_, "Incomplete math report");
    stream_.write("MEND", 4); Write(stream_, seen_); stream_.flush();
  }
};

class InputReader {
  std::ifstream stream_;
  std::uint32_t seen_{0};
 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    stream_.open(path, std::ios::binary);
    char magic[8]; stream_.read(magic, sizeof(magic));
    Require(!std::memcmp(magic, kMagic, sizeof(magic)), "Unknown math interchange version");
    Require(Read<std::uint32_t>(stream_) == kInput && Read<std::uint32_t>(stream_) == kAll,
            "CUDA requires the complete input-only math corpus");
  }
  Words Next(unsigned id) {
    Require(id == math_contracts::Next(kAll, seen_), "Wrong math input group order");
    auto shape = kShapes[id];
    Require(Read<std::uint32_t>(stream_) == id && Read<std::uint32_t>(stream_) == shape.rows &&
            Read<std::uint32_t>(stream_) == shape.inputs && Read<std::uint32_t>(stream_) == shape.outputs,
            "Invalid math input identity/shape");
    Words words(std::size_t{shape.rows}*shape.inputs);
    for (auto& word : words) word = Read<std::uint64_t>(stream_);
    CheckWords(id, words, false); seen_ |= 1u << id;
    return words;
  }
  void Finish() {
    Require(seen_ == kAll, "Unconsumed math input groups");
    char footer[4]; stream_.read(footer, sizeof(footer));
    Require(!std::memcmp(footer, "MEND", 4) && Read<std::uint32_t>(stream_) == kAll,
            "Missing math completion marker");
    Require(stream_.peek() == std::char_traits<char>::eof(), "Trailing math input bytes");
  }
};

inline std::unique_ptr<Writer> ReportOption(int argc, char** argv, std::uint32_t mask) {
  Require(argc == 2 || (argc == 4 && std::string(argv[2]) == "--report"),
          "Pass shader path [--report new-file]");
  if (argc == 2) return {};
  return std::make_unique<Writer>(argv[3], kMetal, mask);
}
}  // namespace math_contracts
