// CUDA access-state rules, exact GPU payloads and independent metadata row oracles.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_metadata.h"
#include "../xgboost/src/common/metal_vector.h"
#include "../xgboost/src/data/metainfo.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"
#include "xgboost/tree_model.h"

namespace metal = xgboost::common::metal;
using xgboost::DeviceOrd;
using xgboost::HostDeviceVector;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++fields;
  if (!ok) throw std::runtime_error(message);
}
template <typename F>
void Reject(F&& fn) {
  bool failed = false;
  try {
    fn();
  } catch (std::exception const&) {
    failed = true;
  }
  Check(failed, "Invalid vector operation accepted");
  ++rejected;
}
template <typename T>
void Same(HostDeviceVector<T> const& actual, std::vector<T> const& expected) {
  Check(actual.Size() == expected.size(), "Vector size");
  auto const& host = actual.ConstHostVector();
  for (std::size_t i = 0; i < host.size(); ++i)
    Check(std::memcmp(&host[i], &expected[i], sizeof(T)) == 0, "Exact vector payload");
}
template <typename T>
void State(std::size_t count) {
  auto baseline = metal::GetStatistics();
  {
    std::vector<T> expected(count);
    for (std::size_t i = 0; i < count; ++i) expected[i] = static_cast<T>(i % 97);
    HostDeviceVector<T> v{expected};
    Check(v.Device().IsCPU() && v.HostCanWrite() && !v.DeviceCanRead(), "Initial CPU state");
    v.SetDevice(DeviceOrd::MPS());
    Check(v.Device().IsMPS() && v.HostCanWrite() && !v.DeviceCanRead(), "Lazy device selection");
    auto before = metal::GetStatistics();
    auto pointer = v.ConstDevicePointer();
    Check(v.DeviceCanRead() && v.HostCanRead() && !v.HostCanWrite(), "Shared read state");
    Check(metal::GetStatistics().vector_upload_bytes - before.vector_upload_bytes ==
              count * sizeof(T),
          "First device read uploads once");
    Check(pointer == v.ConstDevicePointer(), "Repeated device read retains allocation");
    Check(metal::GetStatistics().vector_upload_bytes - before.vector_upload_bytes ==
              count * sizeof(T),
          "Repeated device read does not upload");
    auto const& stale = v.ConstHostVector();
    v.Fill(T{7});
    Check(v.DeviceCanWrite() && !v.HostCanRead(), "GPU fill invalidates host view");
    Check(stale == expected, "GPU fill must not mutate the old host mirror");
    before = metal::GetStatistics();
    expected.assign(count, T{7});
    Same(v, expected);
    Check(v.HostCanRead() && v.DeviceCanRead() && !v.HostCanWrite(),
          "Host read retains GPU validity");
    Check(metal::GetStatistics().vector_readback_bytes - before.vector_readback_bytes ==
              count * sizeof(T),
          "Explicit host readback count");
    Same(v, expected);
    Check(metal::GetStatistics().vector_readback_bytes - before.vector_readback_bytes ==
              count * sizeof(T),
          "Repeated host read has no copy");

    // Mutable host access can resize the vector without calling its Resize method.
    v.HostVector().push_back(T{3});
    expected.push_back(T{3});
    Check(v.HostCanWrite() && !v.DeviceCanRead(), "Mutable host access invalidates device");
    v.DevicePointer();
    Check(!v.HostCanRead() && v.DeviceCanWrite(), "Mutable device access invalidates host");
    before = metal::GetStatistics();
    v.Resize(count + 19, T{5});
    expected.resize(count + 19, T{5});
    Check(v.DeviceCanWrite() && !v.HostCanRead(), "Device resize retains GPU authority");
    Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
          "Device resize must not read back");
    HostDeviceVector<T> copy(v.Size(), T{}, DeviceOrd::MPS());
    copy.Copy(v);
    Check(copy.DeviceCanWrite() && !copy.HostCanRead(), "Device copy retains GPU authority");
    Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
          "Device copy must not read back");
    v.Fill(T{9});
    Same(copy, expected);
    Check(copy.ConstDevicePointer() != v.ConstDevicePointer(), "Copy has independent storage");
    auto moved = std::move(copy);
    Check(copy.Empty(), "Moved-from vector is empty");
    Same(moved, expected);
    copy.Resize(3, T{2});
    Same(copy, std::vector<T>(3, T{2}));
    moved.DevicePointer();
    moved.Resize(2);
    moved.Resize(11, T{4});
    expected.resize(2);
    expected.resize(11, T{4});
    Same(moved, expected);
    moved.SetDevice(DeviceOrd::CPU());
    Check(moved.HostCanWrite() && !moved.DeviceCanRead() && moved.Device().IsCPU(), "CPU switch");
    Check(moved.DevicePointer() == nullptr && moved.DeviceSpan().empty(),
          "CPU build-compatible access");
    Same(moved, expected);
    moved.SetDevice(DeviceOrd::MPS());
    moved.DevicePointer();
    moved.Copy(expected);
    Same(moved, expected);
    moved.Copy(moved);
    Same(moved, expected);
  }
  auto after = metal::GetStatistics();
  Check(after.vector_live_buffers == baseline.vector_live_buffers, "Vector allocation leak");
  Check(after.vector_capacity_bytes == baseline.vector_capacity_bytes, "Vector capacity leak");
  ++cases;
}
void Extend() {
  for (int left = 0; left < 3; ++left) {
    for (int right = 0; right < 3; ++right) {
      HostDeviceVector<float> a(std::vector<float>{1, 2, 3}), b(std::vector<float>{4, 5});
      auto access = [](HostDeviceVector<float>* v, int mode) {
        if (!mode) return;
        v->SetDevice(DeviceOrd::MPS());
        if (mode == 1)
          v->ConstDevicePointer();
        else
          v->DevicePointer();
      };
      access(&a, left);
      access(&b, right);
      a.Extend(b);
      Same(a, std::vector<float>{1, 2, 3, 4, 5});
      Same(b, std::vector<float>{4, 5});
      if (a.Device().IsMPS()) a.DevicePointer();
      a.Extend(a);
      Same(a, std::vector<float>{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
      HostDeviceVector<float> empty;
      a.Extend(empty);
      Check(a.Size() == 10, "Empty extend");
      ++cases;
    }
  }
}
void CopyStates() {
  std::uint32_t bits[] = {0,           0x80000000u, 1,           0x80000001u, 0x7fc12345u,
                          0xffc54321u, 0x7f800000u, 0xff800000u, 0x3f800000u};
  std::vector<float> expected(std::size(bits));
  std::memcpy(expected.data(), bits, sizeof(bits));
  auto state = [](HostDeviceVector<float>* v, int mode) {
    if (!mode) return;
    v->SetDevice(DeviceOrd::MPS());
    if (mode == 2) v->ConstDevicePointer();
    if (mode == 3) v->DevicePointer();
  };
  for (int left = 0; left < 4; ++left) {
    for (int right = 0; right < 4; ++right) {
      HostDeviceVector<float> a(expected.size(), -1), b(expected);
      state(&a, left);
      state(&b, right);
      a.Copy(b);
      Check(a.Device() == b.Device(), "Copy adopts source device like CUDA");
      Same(a, expected);
      a.Fill(7);
      Same(b, expected);
      ++cases;
    }
  }
  HostDeviceVector<float> list({1, -2, 3}, DeviceOrd::MPS());
  list.Copy({4, 5, -6});
  Same(list, std::vector<float>{4, 5, -6});
  HostDeviceVector<float> assigned(1, 9);
  assigned = std::move(list);
  Same(assigned, std::vector<float>{4, 5, -6});
  Check(list.Empty(), "Move assignment source is reusable");
  list.Resize(2, 8);
  Same(list, std::vector<float>{8, 8});
  assigned.DevicePointer();
  Reject([&] { assigned.Resize(std::numeric_limits<std::size_t>::max()); });
  Reject([&] { assigned.SetDevice({DeviceOrd::kMetal, 1}); });
  Same(assigned, std::vector<float>{4, 5, -6});
  ++cases;
}
template <typename T>
void Typed(T value) {
  std::vector<T> expected(31, value);
  HostDeviceVector<T> a(expected, DeviceOrd::MPS()), b(expected.size(), T{}, DeviceOrd::MPS());
  b.Copy(a);
  a.Fill(value);
  Same(a, expected);
  Same(b, expected);
  ++cases;
}
void PatternBytes() {
  for (std::size_t width : {1u, 2u, 3u, 4u, 8u, 12u, 16u, 24u, 32u, 48u}) {
    std::vector<std::byte> pattern(width);
    for (std::size_t i = 0; i < width; ++i) pattern[i] = std::byte((i * 37 + 139) % 256);
    auto buffer = metal::MakeVectorBuffer();
    auto size = width * 259;
    buffer->Resize(size);
    buffer->Fill(pattern, 0, size);
    std::vector<std::byte> host(size + 2, std::byte{0xa5});
    buffer->Readback(host.data() + 1, size);
    Check(host.front() == std::byte{0xa5} && host.back() == std::byte{0xa5},
          "Byte readback guards");
    for (std::size_t i = 0; i < size; ++i)
      Check(host[i + 1] == pattern[i % width], "POD byte fill");
    auto expected = std::vector<std::byte>(host.begin() + 1, host.end() - 1);
    std::memmove(expected.data() + 1, expected.data(), size - 1);
    buffer->Copy(*buffer, 0, 1, size - 1);
    buffer->Readback(host.data() + 1, size);
    Check(std::equal(expected.begin(), expected.end(), host.begin() + 1), "Overlapping GPU copy");
    auto same_pointer = buffer->Data();
    buffer->Resize(size / 2);
    buffer->Resize(size);
    Check(buffer->Data() == same_pointer, "Shrink/regrow reuses capacity");
    // No default initialization is promised for a GPU Resize without a value.
    buffer->Readback(host.data() + 1, size);
    Check(std::equal(expected.begin(), expected.end(), host.begin() + 1),
          "Uninitialized regrow retained bytes");
    Reject([&] { buffer->Fill(pattern, 0, size + width); });
    Reject([&] { buffer->Readback(host.data(), size + 1); });
    ++cases;
  }
}
void Metadata() {
  xgboost::Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "2"}});
  for (std::size_t targets : {1ul, 3ul, 17ul}) {
    for (std::size_t count : {0ul, 1ul, 35ul, 257ul}) {
      xgboost::MetaInfo info;
      info.num_row_ = 257;
      info.num_col_ = 3;
      std::vector<float> values(info.num_row_ * targets);
      for (std::size_t i = 0; i < values.size(); ++i) values[i] = (int(i % 137) - 68) / 16.f;
      info.labels.SetDevice(DeviceOrd::MPS());
      info.labels.Reshape(info.num_row_, targets);
      info.labels.Data()->Copy(values);
      info.base_margin_.SetDevice(DeviceOrd::MPS());
      info.base_margin_.Reshape(info.num_row_, targets);
      info.base_margin_.Data()->Copy(values);
      std::vector<float> weights(info.num_row_);
      std::iota(weights.begin(), weights.end(), 1.f);
      for (auto p : {&info.weights_, &info.labels_lower_bound_, &info.labels_upper_bound_}) {
        p->SetDevice(DeviceOrd::MPS());
        p->Resize(weights.size());
        p->Copy(weights);
      }
      std::vector<xgboost::bst_idx_t> rows(count);
      for (std::size_t i = 0; i < count; ++i) rows[i] = (i % 23) * 7;
      HostDeviceVector<xgboost::bst_idx_t> indices(rows, DeviceOrd::MPS());
      auto before = metal::GetStatistics();
      auto sliced = info.Slice(&ctx, indices.ConstDeviceSpan(), count * 2);
      Check(!sliced.labels.Data()->HostCanRead() && sliced.labels.Device().IsMPS(),
            "Slice GPU authority");
      auto after = metal::GetStatistics();
      Check(after.vector_readback_bytes == before.vector_readback_bytes,
            "Slice avoids host readback");
      Check(after.metadata_gather_dispatches - before.metadata_gather_dispatches == (count ? 5 : 0),
            "Every metadata field gathered on GPU");
      std::vector<float> expected, expected_weight;
      for (auto r : rows) {
        expected.insert(expected.end(), values.begin() + r * targets,
                        values.begin() + (r + 1) * targets);
        expected_weight.push_back(weights[r]);
      }
      Same(*sliced.labels.Data(), expected);
      Same(*sliced.base_margin_.Data(), expected);
      Same(sliced.weights_, expected_weight);
      Same(sliced.labels_lower_bound_, expected_weight);
      Same(sliced.labels_upper_bound_, expected_weight);
      Check(sliced.num_row_ == count && sliced.num_col_ == 3 && sliced.num_nonzero_ == count * 2,
            "Sliced dimensions");
      before = metal::GetStatistics();
      auto copy = info.Copy();
      copy.Extend(info, true, true);
      Check(copy.labels.Device().IsMPS() && !copy.labels.Data()->HostCanRead(),
            "Stack retains GPU storage");
      Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
            "Metadata Copy/Extend avoids readback");
      auto doubled = values;
      doubled.insert(doubled.end(), values.begin(), values.end());
      Same(*copy.labels.Data(), doubled);
      Check(copy.labels.Shape(0) == 514 && copy.labels.Shape(1) == targets, "Stack shape");
      if (count) {
        rows.back() = 257;
        Reject([&] { info.Slice(&ctx, rows, 0); });
      }
      ++cases;
    }
  }
}
void FeatureTypes() {
  using FT = xgboost::FeatureType;
  for (std::size_t n : {0ul, 1ul, 257ul}) {
    HostDeviceVector<FT> a(n, FT::kNumerical), b(n, FT::kNumerical, DeviceOrd::MPS());
    auto before = metal::GetStatistics();
    xgboost::data::CheckFeatureTypes(a, b);
    Check(metal::GetStatistics().metadata_equal_dispatches - before.metadata_equal_dispatches ==
              bool(n),
          "Feature type equality is GPU work");
    if (n) {
      b.HostVector().back() = FT::kCategorical;
      b.DevicePointer();
      Reject([&] { xgboost::data::CheckFeatureTypes(a, b); });
    }
    ++cases;
  }
}
int main() {
  try {
    for (auto size : {0ul, 1ul, 17ul, 1025ul, 65537ul}) {
      State<float>(size);
      State<double>(size);
      State<std::int32_t>(size);
      State<std::uint64_t>(size);
      State<std::uint8_t>(size);
    }
    Extend();
    CopyStates();
    Typed(xgboost::GradientPair{1.25f, 3.5f});
    Typed(xgboost::GradientPairPrecise{1.25, 3.5});
    Typed(xgboost::GradientPairInt64{123456789012345, -9753123456789});
    Typed(xgboost::Entry{53, -3.125f});
    Typed(xgboost::RegTree::Node{1, 2, 0, 7, -1.25f, true});
    Typed(xgboost::RegTree::CategoricalSplitMatrix::Segment{57, 31});
    Typed(xgboost::RTreeNodeStat{1.25f, 17.5f, -3.f});
    PatternBytes();
    Metadata();
    FeatureTypes();
    auto stats = metal::GetStatistics();
    Check(stats.vector_live_buffers == 0 && stats.vector_capacity_bytes == 0, "Final vector leaks");
    std::printf("PASS %zu vector/metadata cases / %zu exact fields / %zu rejected calls\n", cases,
                fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
