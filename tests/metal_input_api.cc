// Raw input decoding against independent typed values and quantized bin bounds.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_input.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/gradient_index.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, errors = 0;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
std::uint32_t Bits(float v) {
  std::uint32_t result;
  std::memcpy(&result, &v, 4);
  return result;
}
float Float(std::uint32_t v) {
  float result;
  std::memcpy(&result, &v, 4);
  return result;
}
void Verify(std::unique_ptr<metal::InputBatch> batch, std::vector<float> const& expected) {
  auto before = metal::GetStatistics();
  std::vector<float> result(expected.size() + 2, -919.f);
  batch->Readback({result.data() + 1, expected.size()});
  Check(result.front() == -919 && result.back() == -919, "Readback guards");
  Check(before.input_debug_readback_bytes + expected.size() * 4 ==
            metal::GetStatistics().input_debug_readback_bytes,
        "Readback accounting");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (Bits(result[i + 1]) != Bits(expected[i]) &&
        !(std::isnan(expected[i]) && std::isnan(result[i + 1]))) {
      std::fprintf(stderr, "case %zu element %zu: %08x != %08x\n", cases, i, Bits(result[i + 1]),
                   Bits(expected[i]));
      throw std::runtime_error("Input differs from independent typed reference");
    }
  }
  fields += expected.size();
  ++cases;
}
std::string Interface(void const* ptr, std::string type, std::vector<std::size_t> shape,
                      std::vector<std::int64_t> strides, std::string mask = "null") {
  std::ostringstream s;
  s << "{\"version\":3,\"typestr\":\"" << type << "\",\"data\":["
    << reinterpret_cast<std::uintptr_t>(ptr) << ",false],\"shape\":[";
  for (std::size_t i = 0; i < shape.size(); ++i) s << (i ? "," : "") << shape[i];
  s << "],\"strides\":[";
  for (std::size_t i = 0; i < strides.size(); ++i) s << (i ? "," : "") << strides[i];
  s << "],\"mask\":" << mask << "}";
  return s.str();
}
template <class T>
void Typed(std::string type, std::mt19937_64* rng) {
  constexpr std::size_t rows = 16387, cols = 7;
  std::vector<T> data(rows * cols);
  for (auto& v : data) {
    auto bits = (*rng)();
    std::memcpy(&v, &bits, sizeof(T));
  }
  data[0] = std::numeric_limits<T>::lowest();
  data[1] = std::numeric_limits<T>::max();
  data[2] = 0;
  data[3] = 1;
  data[4] = std::numeric_limits<T>::denorm_min();
  data[5] = std::numeric_limits<T>::min();
  if constexpr (std::is_floating_point_v<T>) {
    data[6] = -T(0);
    data[7] = std::numeric_limits<T>::infinity();
    data[8] = std::numeric_limits<T>::quiet_NaN();
    // Adjacent doubles around a float rounding tie, underflow, overflow.
    data[9] = T(1.0 + std::ldexp(1.0, -24));
    data[10] = std::nextafter(data[9], T(2));
    data[11] = T(std::ldexp(1.0, -150));
    data[12] = std::nextafter(data[11], T(1));
  }
  for (unsigned layout = 0; layout < 5; ++layout) {
    std::int64_t rs = cols, cs = 1, start = 0;
    if (layout == 1) {
      rs = -rs;
      cs = -cs;
      start = data.size() - 1;
    }
    if (layout == 2) {
      rs = 1;
      cs = rows;
    }
    if (layout == 3) {
      rs = 0;
      cs = 0;
      start = 1;
    }
    if (layout == 4) {
      rs = 0;
      cs = -1;
      start = cols - 1;
    }
    auto json = Interface(data.data() + start, type, {rows, cols},
                          {rs * std::int64_t(sizeof(T)), cs * std::int64_t(sizeof(T))});
    data::ArrayAdapter adapter{StringView{json}};
    auto begin = layout == 1 ? 3u : 0u;
    auto n = rows - begin;
    std::vector<float> expected(n * (cols + 1), kNaN);
    for (std::size_t r = 0; r < n; ++r)
      for (std::size_t f = 0; f < cols; ++f)
        expected[r * (cols + 1) + f] =
            static_cast<float>(data[start + (std::int64_t(r) + begin) * rs + f * cs]);
    Verify(metal::PrepareInput(adapter, kNaN, begin, n, cols + 1, {}), expected);
  }
  // Non-unit signed row strides and validity masks, including unaligned bit slices.
  std::vector<std::uint8_t> mask((rows + 7) / 8, 0);
  for (std::size_t r = 0; r < rows; ++r)
    if (r % 5) mask[r / 8] |= 1u << (r % 8);
  auto validity = Interface(mask.data(), "|t1", {rows}, {1});
  auto col0 = Interface(data.data(), type, {rows}, {2 * sizeof(T)}, validity);
  auto col1 =
      Interface(data.data() + data.size() - 1, type, {rows}, {-2 * std::int64_t(sizeof(T))});
  auto columns = "[" + col0 + "," + col1 + "]";
  data::ColumnarAdapter adapter{StringView{columns}};
  for (std::size_t begin = 0; begin < 8; ++begin) {
    std::vector<float> expected(513 * 3, kNaN);
    for (std::size_t r = 0; r < 513; ++r) {
      if ((r + begin) % 5) expected[r * 3] = float(data[(r + begin) * 2]);
      expected[r * 3 + 1] = float(data[data.size() - 1 - (r + begin) * 2]);
    }
    Verify(metal::PrepareInput(adapter, kNaN, begin, 513, 3, {}), expected);
  }
  std::printf("typed %s complete: %zu fields\n", type.c_str(), fields);
}
void DenseAndCategories() {
  std::vector<float> values{kNaN,
                            0,
                            -0.f,
                            .75f,
                            -.75f,
                            1.75f,
                            -1.f,
                            2.f,
                            3.f,
                            Float(1),
                            Float(0x80000001),
                            Float(0x7f800000),
                            Float(0xff800000),
                            Float(0x4f000000),
                            Float(0xcf000001)};
  std::vector<std::int32_t> segments{0, 3}, mapping{2, 0, 1};
  data::DenseAdapter adapter{values.data(), values.size(), 1};
  auto expected = values;
  for (std::size_t i = 1; i < 6; ++i) expected[i] = i == 5 ? 0.f : 2.f;
  expected[7] = 1.f;
  expected[9] = expected[10] = 2.f;
  auto before = metal::GetStatistics();
  Verify(metal::PrepareInput(adapter, kNaN, 0, values.size(), 1, {segments, mapping}), expected);
  Check(metal::GetStatistics().input_recode_dispatches == before.input_recode_dispatches + 1,
        "GPU row mapping counter");
  for (auto missing : {0.f, -0.f, .75f, Float(1), Float(0x80000001)}) {
    auto ref = values;
    for (auto& v : ref)
      if (v == missing || std::isnan(v)) v = kNaN;
    Verify(metal::PrepareInput(adapter, missing, 0, values.size(), 1, {}), ref);
  }
  data::DenseAdapter empty{nullptr, 0, 3};
  before = metal::GetStatistics();
  Verify(metal::PrepareInput(empty, kNaN, 0, 0, 3, {}), {});
  Check(before.input_prepare_dispatches == metal::GetStatistics().input_prepare_dispatches,
        "Empty input must not dispatch");
}
void Sparse() {
  std::vector<bst_idx_t> offsets{0, 2, 2, 5, 6};
  std::vector<Entry> entries{{2, 8.f}, {0, 4.f}, {3, 2.f}, {1, -0.f}, {1, 3.f}, {0, kNaN}};
  HostSparsePageView page{offsets, entries};
  Verify(metal::PrepareInput(page, 1, 3, 4, {}),
         {kNaN, kNaN, kNaN, kNaN, kNaN, 3.f, kNaN, 2.f, kNaN, kNaN, kNaN, kNaN});
  // All three CSR arrays have distinct physical strides; first selected offset is nonzero.
  std::vector<std::uint32_t> ptr{0, 919, 1, 919, 1, 919, 4, 919, 5, 919};
  std::vector<std::int16_t> idx{2, 919, 0, 919, 3, 919, 1, 919, 2, 919};
  std::vector<double> val{7, 919, 919, 8, 919, 919, -99, 919, 919, -0., 919, 919, 9, 919, 919};
  auto a = Interface(ptr.data(), "<u4", {5}, {8});
  auto b = Interface(idx.data(), "<i2", {5}, {4});
  auto c = Interface(val.data(), "<f8", {5}, {24});
  data::CSRArrayAdapter adapter{StringView{a}, StringView{b}, StringView{c}, 4};
  Verify(metal::PrepareInput(adapter, -99, 1, 3, 4, {}),
         {kNaN, kNaN, kNaN, kNaN, 8.f, -0.f, kNaN, kNaN, kNaN, kNaN, 9.f, kNaN});
  entries[2].index = 4;
  try {
    metal::PrepareInput(page, 1, 3, 4, {});
    throw std::runtime_error("Expected index error");
  } catch (dmlc::Error const&) {
    ++errors;
  }
  entries[2].index = 3;
  offsets[2] = 7;
  try {
    metal::PrepareInput(page, 0, 4, 4, {});
    throw std::runtime_error("Expected row error");
  } catch (dmlc::Error const&) {
    ++errors;
  }
}
template <class T>
void Bins(bool dense) {
  GHistIndexMatrix page;
  page.SetDense(dense);
  page.base_rowid = 1234;
  page.cut.cut_ptrs_.HostVector() = {0, 3, 6, 9};
  page.cut.cut_values_.HostVector() = {-2, 0, 3, 0, 1, 2, 10, 20, 30};
  std::vector<std::size_t> offsets{0, 3, 6, 9};
  std::vector<T> bins{0, 1, 2, 2, 0, 1, 1, 2, 0};
  if (!dense)
    for (unsigned i = 0; i < bins.size(); ++i) bins[i] += (i % 3) * 3;
  page.row_ptr = common::MakeFixedVecWithMalloc(offsets.size(), std::size_t{0});
  std::copy(offsets.begin(), offsets.end(), page.row_ptr.begin());
  page.index =
      common::Index{{reinterpret_cast<std::uint8_t*>(bins.data()), bins.size() * sizeof(T)},
                    static_cast<common::BinTypeSize>(sizeof(T))};
  if (dense) page.index.SetBinOffset(page.cut.Ptrs());
  std::vector<FeatureType> types{FeatureType::kNumerical, FeatureType::kCategorical,
                                 FeatureType::kNumerical};
  std::vector<std::int32_t> segments{0, 0, 3, 3}, mapping{2, 0, 1};
  // No column transpose exists: decoding must read row-wise bins directly.
  Verify(metal::PrepareInput(page, types, 1, 2, 3, {segments, mapping}),
         {0, 2, 10, -2, 1, -std::numeric_limits<float>::infinity()});
  if (!dense) {
    page.row_ptr[2] = 3;
    page.row_ptr[3] = 5;
    bins[3] = 4;
    bins[4] = 8;
    Verify(metal::PrepareInput(page, types, 1, 2, 3, {segments, mapping}),
           {kNaN, kNaN, kNaN, kNaN, 0, 20});
  }
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    std::mt19937_64 rng{9271};
    Typed<float>("<f4", &rng);
    Typed<double>("<f8", &rng);
    Typed<std::int8_t>("|i1", &rng);
    Typed<std::uint8_t>("|u1", &rng);
    Typed<std::int16_t>("<i2", &rng);
    Typed<std::uint16_t>("<u2", &rng);
    Typed<std::int32_t>("<i4", &rng);
    Typed<std::uint32_t>("<u4", &rng);
    Typed<std::int64_t>("<i8", &rng);
    Typed<std::uint64_t>("<u8", &rng);
    DenseAndCategories();
    Sparse();
    Bins<std::uint8_t>(true);
    Bins<std::uint16_t>(true);
    Bins<std::uint32_t>(true);
    Bins<std::uint32_t>(false);
    std::printf("PASS: %zu input cases, %zu exact fields, %zu expected errors\n", cases, fields,
                errors);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
