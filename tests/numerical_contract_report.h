// Lossless observations shared by the Metal and actual-CUDA fixture runners.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace numerical_contracts {
using Words = std::vector<std::uint64_t>;
using Fields = std::map<std::string, Words>;
inline std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
inline Words FloatWords(float const* values, std::size_t size) {
  Words result;
  for (std::size_t i = 0; i < size; ++i) result.push_back(FloatBits(values[i]));
  return result;
}
inline Words FloatWords(std::vector<float> const& values) {
  return FloatWords(values.data(), values.size());
}
template <typename Entry> Words EntryWords(Entry const* entries, std::size_t size) {
  Words result;
  for (std::size_t i = 0; i < size; ++i) {
    auto e = entries[i];
    for (float value : {e.rmin, e.rmax, e.wmin, e.value}) result.push_back(FloatBits(value));
  }
  return result;
}

class Report {
  std::ofstream output_;
  std::size_t cases_{0};
  void String(std::string const& text) {
    static char const hex[] = "0123456789abcdef";
    output_ << '"';
    for (unsigned char c : text) {
      if (c == '"' || c == '\\') output_ << '\\' << c;
      else if (c < 32) output_ << "\\u00" << hex[c >> 4] << hex[c & 15];
      else output_ << c;
    }
    output_ << '"';
  }
  void Arrays(Fields const& fields) {
    output_ << '{';
    bool comma = false;
    for (auto const& field : fields) {
      if (comma) output_ << ',';
      comma = true;
      String(field.first);
      output_ << ":[";
      for (std::size_t i = 0; i < field.second.size(); ++i) {
        if (i) output_ << ',';
        output_ << field.second[i];
      }
      output_ << ']';
    }
    output_ << '}';
  }

 public:
  explicit Report(std::string const& path = "") {
    if (path.empty()) return;
    if (std::ifstream{path}.good()) throw std::runtime_error("Report already exists: " + path);
    output_.exceptions(std::ios::badbit | std::ios::failbit);
    output_.open(path);
  }
  void Metadata(std::map<std::string, std::string> const& fields) {
    if (!output_.is_open()) return;
    output_ << "{\"kind\":\"metadata\",\"schema_version\":1";
    for (auto const& field : fields) {
      output_ << ',';
      String(field.first);
      output_ << ':';
      String(field.second);
    }
    output_ << "}\n";
  }
  void Case(std::string const& id, unsigned rows, int exponent, unsigned iteration,
            Fields const& input, Fields const& output) {
    ++cases_;
    if (!output_.is_open()) return;
    output_ << "{\"kind\":\"case\",\"id\":";
    String(id);
    output_ << ",\"rows\":" << rows << ",\"exponent\":" << exponent
            << ",\"iteration\":" << iteration << ",\"input\":";
    Arrays(input);
    output_ << ",\"output\":";
    Arrays(output);
    output_ << "}\n";
    output_.flush();
  }
  void Finish() {
    if (!output_.is_open()) return;
    output_ << "{\"kind\":\"summary\",\"status\":\"completed\",\"cases\":" << cases_ << "}\n";
    output_.flush();
  }
};
}  // namespace numerical_contracts
