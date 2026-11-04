// CUDA adapter windows -> resident sketches/bins, independently checked at each push.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_adapter.h"
#include "../xgboost/src/common/metal_cuda_sketch.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/gradient_index.h"
#include "cuda_sketch_reference.h"
using namespace xgboost;
namespace metal = common::metal;
namespace ref = cuda_sketch_reference;
std::size_t cases = 0, fields = 0, windows = 0;
void Check(bool ok, char const* reason) {
  if (!ok) throw std::runtime_error(reason);
}
void Same(float a, float b) {
  ++fields;
  Check((std::isnan(a) && std::isnan(b)) || !std::memcmp(&a, &b, 4), "Float bits differ");
}
void Verify(metal::CudaSketchSummary const& actual, ref::Sketch const& expected) {
  std::size_t offset = 0;
  for (std::size_t f = 0; f < expected.columns.size(); ++f) {
    Check(actual.ptr[f] == offset, "Sketch column pointer");
    for (auto e : expected.columns[f]) {
      Check(offset < actual.entries.size(), "Sketch missing entry");
      auto a = actual.entries[offset++];
      Same(a.rmin, e.rmin);
      Same(a.rmax, e.rmax);
      Same(a.wmin, e.wmin);
      Same(a.value, e.value);
    }
  }
  Check(actual.ptr.back() == offset && actual.entries.size() == offset, "Sketch count");
}
template <class T>
void Case(std::size_t rows, std::size_t window, bool weighted, bool cats, bool missing) {
  constexpr std::size_t cols = 7;
  std::vector<T> data(rows * cols);
  std::vector<float> weights(weighted ? rows : 0);
  std::vector<FeatureType> types(cols, FeatureType::kNumerical);
  if (cats) types[2] = FeatureType::kCategorical;
  std::vector<bool> categorical(cols, false);
  categorical[2] = cats;
  std::vector<ref::Raw> all;
  std::vector<bst_idx_t> slots, ptr{0};
  for (std::size_t r = 0; r < rows; ++r) {
    auto w = float(r % 7) / 4;
    if (weighted) weights[r] = w;
    for (std::size_t f = 0; f < cols; ++f) {
      auto v =
          (cats && f == 2) ? float((r * 3) % 7) : float(int((r * 193 + f * 17) % 997) - 498) / 8;
      if (missing && (r % 17 == 0 || (r + f) % 5 == 0)) v = -99;
      data[r * cols + f] = v;
      if (v != -99) {
        all.push_back({std::uint32_t(f), v, w});
        slots.push_back(r * cols + f);
      }
    }
    ptr.push_back(all.size());
  }
  std::ostringstream json;
  json << "{\"version\":3,\"typestr\":\"<f" << sizeof(T) << "\",\"data\":["
       << reinterpret_cast<std::uintptr_t>(data.data()) << ",false],\"shape\":[" << rows << ","
       << cols << "],\"strides\":null}";
  auto str = json.str();
  data::ArrayAdapter adapter{StringView{str}};
  auto gpu = metal::MakeCudaSketch(cols, 8, types, weights);
  ref::Sketch oracle(cols, 8, categorical);
  std::vector<std::unique_ptr<metal::AdapterInput>> inputs;
  for (std::size_t begin = 0; begin < data.size(); begin += window) {
    auto count = std::min(window, data.size() - begin), end = begin + count;
    auto input = metal::PrepareAdapter(adapter, -99, begin, count);
    auto lo = std::lower_bound(slots.begin(), slots.end(), begin) - slots.begin();
    auto hi = std::lower_bound(slots.begin(), slots.end(), end) - slots.begin();
    auto first = begin / cols, last = (end + cols - 1) / cols;
    Check(input->BaseRow() == first && input->Rows() == last - first, "Partial row geometry");
    Check(input->NonMissing() == hi - lo, "Selected nonmissing count");
    Check(input->SketchRows() == (count + cols - 1) / cols, "CUDA approximate window rows");
    bst_idx_t stride = 0;
    for (auto r = first; r < last; ++r) stride = std::max(stride, ptr[r + 1] - ptr[r]);
    Check(input->RowStride() == stride, "Full row max despite partial boundary");
    SparsePage page;
    input->Readback(&page);
    Check(page.Size() == last - first && page.data.Size() == hi - lo, "Readback shape");
    auto view = page.GetView();
    std::size_t offset = 0;
    for (auto r = first; r < last; ++r) {
      Check(view.offset[r - first] == offset, "Readback row pointer");
      for (auto k = lo; k < hi; ++k)
        if (slots[k] / cols == r) {
          auto actual = view.data[offset++];
          Check(actual.index == all[k].feature, "Source feature order");
          Same(actual.fvalue, all[k].value);
        }
    }
    Check(view.offset.back() == offset, "Final readback pointer");
    auto before = metal::GetStatistics();
    gpu->Push(*input);
    Check(
        metal::GetStatistics().adapter_debug_readback_bytes == before.adapter_debug_readback_bytes,
        "Sketch readback");
    oracle.Push({all.begin() + lo, all.begin() + hi}, input->SketchRows(), weighted);
    Verify(gpu->Read(), oracle);
    inputs.push_back(std::move(input));
    ++windows;
  }
  auto cuts = metal::MakeCudaCuts(gpu->Read(), cols, 8, types);
  MetaInfo info;
  info.num_row_ = rows;
  info.num_col_ = cols;
  info.num_nonzero_ = all.size();
  GHistIndexMatrix index{info, std::move(cuts), 8};
  Context ctx;
  index.ResizeIndex(&ctx, all.size(), info.IsDense());
  if (info.IsDense()) index.index.SetBinOffset(index.cut.Ptrs());
  bst_idx_t offset = 0;
  for (auto const& input : inputs) {
    input->WriteIndex(types, &index, 0, offset);
    offset += input->NonMissing();
  }
  Check(std::equal(ptr.begin(), ptr.end(), index.row_ptr.begin()), "GPU global row offsets");
  for (std::size_t i = 0; i < all.size(); ++i) {
    auto v = all[i];
    auto a = index.cut.Ptrs()[v.feature], b = index.cut.Ptrs()[v.feature + 1];
    auto const& c = index.cut.Values();
    auto found =
        (categorical[v.feature] ? std::lower_bound(c.begin() + a, c.begin() + b, v.value)
                                : std::upper_bound(c.begin() + a, c.begin() + b, v.value)) -
        c.begin();
    auto expected = std::min<std::size_t>(b - 1, found);
    Check(index.index[i] == expected, "GPU bin assignment");
    ++fields;
  }
  ++cases;
}
void RecodeBeforeMissing() {
  std::vector<float> codes{0, 1, 2, 0, 2, 1, 0};
  std::ostringstream s;
  s << "[{\"version\":3,\"typestr\":\"<f4\",\"data\":["
    << reinterpret_cast<std::uintptr_t>(codes.data()) << ",false],\"shape\":[7],\"strides\":null}]";
  auto str = s.str();
  data::ColumnarAdapter adapter{StringView{str}};
  std::vector<std::int32_t> segments{0, 3}, mapping{2, 0, 1};
  auto input = metal::PrepareAdapter(adapter, 2, 0, 7, {segments, mapping});
  SparsePage result;
  input->Readback(&result);
  Check(result.offset.ConstHostVector() == std::vector<bst_idx_t>({0, 0, 1, 2, 2, 3, 4, 4}),
        "Training recode/filter order");
  std::vector<float> expected{0, 1, 1, 0};
  for (std::size_t i = 0; i < expected.size(); ++i)
    Same(result.data.ConstHostVector()[i].fvalue, expected[i]);
  ++cases;
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    for (bool weighted : {false, true})
      for (bool cats : {false, true})
        for (bool missing : {false, true}) {
          for (auto window : {1u, 2u, 7u, 13u, 31u})
            Case<float>(19, window, weighted, cats, missing);
          for (auto window : {256u, 1025u, 65536u})
            Case<double>(1027, window, weighted, cats, missing);
          std::printf("checked %zu cases / %zu windows / %zu fields\n", cases, windows, fields);
        }
    RecodeBeforeMissing();
    std::printf("PASS %zu adapter cases / %zu windows / %zu exact fields\n", cases, windows,
                fields);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
