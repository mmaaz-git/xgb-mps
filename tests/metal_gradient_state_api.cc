// Gradient layout copies and Hessian extraction must preserve every input bit.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, copied = 0, extracted = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
std::uint32_t Pattern(std::size_t i) {
  std::uint32_t special[] = {0,          0x80000000, 1,          0x80000001,
                             0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
                             0x7f800000, 0xff800000, 0x7fc12345, 0xffa54321};
  return i % 23 < 12 ? special[i % 23] : std::uint32_t(i * 2654435761u + 127);
}
template <typename T>
void Fill(std::vector<T>* buffer) {
  auto bytes = reinterpret_cast<unsigned char*>(buffer->data());
  for (std::size_t i = 0; i < buffer->size() * sizeof(T) / 4; ++i) {
    auto bits = Pattern(i);
    std::memcpy(bytes + i * 4, &bits, 4);
  }
}
template <typename T>
void Same(std::vector<T> const& actual, std::vector<T> const& expected) {
  Check(actual.size() == expected.size(), "Different storage size");
  Check(actual.empty() || !std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)),
        "Storage bits, padding or guards differ");
}
struct Layout {
  std::size_t a, b, size;
};
Layout Strides(std::size_t rows, std::size_t columns, unsigned kind) {
  auto a = columns, b = std::size_t{1};
  if (kind == 1) {
    a = 1;
    b = rows + 3;
  }
  if (kind == 2) {
    a = columns * 2 + 3;
    b = 2;
  }
  if (kind == 3) a = 0;
  if (kind == 4) {
    a = 1;
    b = 0;
  }
  return {a, b, rows && columns ? (rows - 1) * a + (columns - 1) * b + 1 : 0};
}
void Run(std::size_t rows, std::size_t columns, unsigned in_kind, unsigned out_kind) {
  current = "rows=" + std::to_string(rows) + " columns=" + std::to_string(columns) +
            " layouts=" + std::to_string(in_kind) + "/" + std::to_string(out_kind);
  auto in = Strides(rows, columns, in_kind), out = Strides(rows, columns, out_kind);
  std::vector<GradientPair> source(in.size + 8), output(out.size + 8);
  std::vector<float> hessians(out.size + 8);
  Fill(&source);
  Fill(&output);
  Fill(&hessians);
  auto saved = source, expected = output;
  auto expected_h = hessians;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < columns; ++c) {
      auto src = 4 + r * in.a + c * in.b, dst = 4 + r * out.a + c * out.b;
      std::memcpy(&expected[dst], &source[src], sizeof(GradientPair));
      std::memcpy(&expected_h[dst], reinterpret_cast<unsigned char const*>(&source[src]) + 4, 4);
    }
  }
  linalg::MatrixView<GradientPair const> input{
      {source.data() + 4, in.size}, {rows, columns}, {in.a, in.b}, DeviceOrd::CPU()};
  linalg::MatrixView<GradientPair> destination{
      {output.data() + 4, out.size}, {rows, columns}, {out.a, out.b}, DeviceOrd::CPU()};
  linalg::MatrixView<float> hessian{
      {hessians.data() + 4, out.size}, {rows, columns}, {out.a, out.b}, DeviceOrd::CPU()};
  auto before = metal::GetStatistics();
  metal::CopyGradients(input, destination);
  metal::ExtractHessians(input, hessian);
  auto after = metal::GetStatistics();
  Check(after.gradient_copy_dispatches - before.gradient_copy_dispatches == bool(rows * columns),
        "Copy dispatch count differs");
  Check(
      after.hessian_extract_dispatches - before.hessian_extract_dispatches == bool(rows * columns),
      "Extraction dispatch count differs");
  auto bytes = after.gradient_state_shared_input_bytes - before.gradient_state_shared_input_bytes +
               after.gradient_state_shared_output_bytes -
               before.gradient_state_shared_output_bytes + after.gradient_state_staging_bytes -
               before.gradient_state_staging_bytes;
  Check(bytes == (rows * columns ? 2 * in.size * 8 + out.size * 12 : 0),
        "Memory accounting differs");
  Same(source, saved);
  Same(output, expected);
  Same(hessians, expected_h);
  copied += rows * columns;
  extracted += rows * columns;
  ++cases;
}
void Aliases() {
  for (bool transpose : {false, true}) {
    current = "aliased copy";
    std::vector<GradientPair> data(24);
    Fill(&data);
    auto saved = data, expected = data;
    auto base = transpose ? 4u : 5u;
    auto a = transpose ? 1u : 3u, b = transpose ? 2u : 1u;
    for (std::size_t r = 0; r < 2; ++r)
      for (std::size_t c = 0; c < 3; ++c) expected[base + r * a + c * b] = saved[4 + r * 3 + c];
    auto before = metal::GetStatistics();
    metal::CopyGradients({{data.data() + 4, 6}, {2, 3}, {3, 1}, DeviceOrd::CPU()},
                         {{data.data() + base, 6}, {2u, 3u}, {a, b}, DeviceOrd::CPU()});
    Check(metal::GetStatistics().gradient_state_staging_bytes > before.gradient_state_staging_bytes,
          "Aliased output was not staged");
    Same(data, expected);
    ++cases;
  }
  current = "aliased Hessians";
  std::vector<GradientPair> data(24);
  Fill(&data);
  auto saved = data, expected = data;
  for (std::size_t i = 0; i < 6; ++i)
    std::memcpy(reinterpret_cast<unsigned char*>(expected.data() + 4) + i * 4,
                reinterpret_cast<unsigned char const*>(saved.data() + 4 + i) + 4, 4);
  metal::ExtractHessians(
      {{data.data() + 4, 6}, {2, 3}, {3, 1}, DeviceOrd::CPU()},
      {{reinterpret_cast<float*>(data.data() + 4), 6}, {2, 3}, {3, 1}, DeviceOrd::CPU()});
  Same(data, expected);
  ++cases;
}
void ReadOnly() {
  current = "read-only source";
  auto page = std::size_t(getpagesize());
  void* memory = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  Check(memory != MAP_FAILED, "mmap failed");
  std::vector<GradientPair> original(31), output(31);
  Fill(&original);
  std::memcpy(memory, original.data(), original.size() * sizeof(GradientPair));
  Check(!mprotect(memory, page, PROT_READ), "mprotect failed");
  auto before = metal::GetStatistics();
  metal::CopyGradients(
      {{static_cast<GradientPair const*>(memory), 31}, {31, 1}, {1, 1}, DeviceOrd::CPU()},
      {output, {31, 1}, {1, 1}, DeviceOrd::CPU()});
  auto after = metal::GetStatistics();
  Check(after.gradient_state_shared_input_bytes == before.gradient_state_shared_input_bytes,
        "Read-only memory incorrectly borrowed");
  Check(after.gradient_state_staging_bytes - before.gradient_state_staging_bytes >= 31 * 8,
        "Read-only input not staged");
  Same(output, original);
  Check(!munmap(memory, page), "munmap failed");
  ++cases;
}
void Invalid() {
  current = "invalid views";
  std::vector<GradientPair> data(32), output(32);
  Fill(&data);
  Fill(&output);
  auto saved = output;
  linalg::MatrixView<GradientPair const> input{data, {2, 3}, {3, 1}, DeviceOrd::CPU()};
  auto before = metal::GetStatistics();
  for (auto strides : {std::array<std::size_t, 2>{0, 1},
                       {1, 0},
                       {2, 1},
                       {1, 1},
                       {100, 1},
                       {std::numeric_limits<std::size_t>::max(), 2}}) {
    bool failed = false;
    try {
      metal::CopyGradients(
          input,
          {output, {std::size_t{2}, std::size_t{3}}, {strides[0], strides[1]}, DeviceOrd::CPU()});
    } catch (std::exception const&) {
      failed = true;
    }
    Check(failed, "Invalid destination accepted");
  }
  Check(metal::GetStatistics().gradient_copy_dispatches == before.gradient_copy_dispatches,
        "Invalid output dispatched");
  Same(output, saved);
}
int main() {
  try {
    for (std::size_t rows : {0u, 1u, 31u, 257u, 4097u, 65539u}) {
      for (std::size_t targets : {1u, 3u, 17u})
        for (unsigned src = 0; src < 5; ++src)
          for (unsigned dst = 0; dst < 3; ++dst) Run(rows, targets, src, dst);
      std::printf("Checked %zu state cases / %zu copied pairs / %zu extracted Hessians\n", cases,
                  copied, extracted);
      std::fflush(stdout);
    }
    Run(1000003, 3, 2, 1);
    Run(31, 0, 0, 0);
    Aliases();
    ReadOnly();
    Invalid();
    std::printf(
        "PASS %zu gradient-state cases / %zu copied pairs / %zu extracted Hessians; "
        "all bits including NaN payloads, layouts, padding, aliases, read-only inputs and errors\n",
        cases, copied, extracted);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
