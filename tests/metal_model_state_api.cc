// CUDA model-buffer contracts: raw copies and one IEEE FP32 multiply per child.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_model.h"
#include "xgboost/context.h"
#include "xgboost/json.h"
#include "xgboost/tree_model.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t copy_cases = 0, scale_cases = 0, scaled = 0, tree_cases = 0;
void Check(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
float Float(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}
std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}
std::uint32_t Pattern(std::size_t i) {
  std::uint32_t special[] = {0,          0x80000000, 1,          0x80000001,
                             0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
                             0x7f800000, 0xff800000, 0x7fc12345, 0xffa54321};
  return i % 23 < 12 ? special[i % 23] : std::uint32_t(i * 2654435761u + 127);
}
void Copies(std::size_t count, metal::ModelBufferKind kind, bool unaligned, bool alias) {
  auto prefix = unaligned ? 9ul : 8ul;
  auto words = count * 4 + 7;
  std::vector<unsigned char> output(prefix + words * 4 + 8, 0xa5);
  std::vector<std::uint32_t> source(count * 3 + 1);
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = Pattern(i);
  for (std::size_t i = 0; i < words; ++i) {
    auto bits = Pattern(i + 17);
    std::memcpy(output.data() + prefix + i * 4, &bits, 4);
  }
  auto before = output, expected = output;
  auto data = output.data() + prefix;
  auto src = alias ? data : reinterpret_cast<unsigned char*>(source.data());
  // Unsorted disjoint destinations; aliased inputs can overlap destinations of
  // other copies, so all copies must read the original snapshot.
  std::vector<metal::ModelCopy> copies{{src + count * 4, count * 3 + 3, count},
                                       {src, 1, count},
                                       {nullptr, words, 0},
                                       {src + count * 8, count + 2, count}};
  auto expected_src =
      alias ? before.data() + prefix : reinterpret_cast<unsigned char*>(source.data());
  for (auto const& copy : copies) {
    if (copy.count) {
      auto offset = static_cast<unsigned char const*>(copy.source) - src;
      std::memcpy(expected.data() + prefix + copy.destination * 4, expected_src + offset,
                  copy.count * 4);
    }
  }
  auto stats = metal::GetStatistics();
  metal::CopyModelBuffer(data, words, copies, kind);
  auto after = metal::GetStatistics();
  Check(output == expected, "Copy changed payload bits, holes or guards");
  Check(after.model_copy_words - stats.model_copy_words == count * 3, "Copy word counter");
  Check(after.model_weight_copy_dispatches - stats.model_weight_copy_dispatches ==
            std::uint64_t(count && kind == metal::ModelBufferKind::kWeights),
        "Weight copy counter");
  Check(after.model_category_copy_dispatches - stats.model_category_copy_dispatches ==
            std::uint64_t(count && kind == metal::ModelBufferKind::kCategories),
        "Category counter");
  ++copy_cases;
}
void ReadOnlySource() {
  auto bytes = static_cast<std::size_t>(getpagesize());
  auto source = static_cast<std::uint32_t*>(
      mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
  Check(source != MAP_FAILED, "mmap failed");
  for (std::size_t i = 0; i < bytes / 4; ++i) source[i] = Pattern(i);
  Check(mprotect(source, bytes, PROT_READ) == 0, "mprotect failed");
  std::vector<std::uint32_t> output(bytes / 4, 0);
  metal::ModelCopy copy{source, 0, output.size()};
  metal::CopyModelBuffer(output.data(), output.size(), {&copy, 1},
                         metal::ModelBufferKind::kWeights);
  Check(std::memcmp(output.data(), source, bytes) == 0, "Read-only source fallback");
  Check(munmap(source, bytes) == 0, "munmap failed");
  ++copy_cases;
}
void Scale(std::size_t count, std::uint32_t eta_bits) {
  std::mt19937 random{static_cast<unsigned>(count)};
  std::vector<float> values(count + 16, Float(0xffa12345));
  for (std::size_t i = 0; i < count; ++i) {
    // Random bits exercise every exponent, including normal/subnormal borders.
    values[i + 8] = Float(i % 2 ? random() : Pattern(i / 2));
  }
  auto before = values;
  auto stats = metal::GetStatistics();
  metal::ScaleModelWeights({values.data() + 8, count}, Float(eta_bits));
  auto after = metal::GetStatistics();
  Check(after.model_scale_dispatches - stats.model_scale_dispatches == std::uint64_t(count != 0),
        "Scale dispatch counter");
  Check(std::memcmp(values.data(), before.data(), 8 * 4) == 0 &&
            std::memcmp(values.data() + count + 8, before.data() + count + 8, 8 * 4) == 0,
        "Scale changed guards");
  for (std::size_t i = 0; i < count; ++i) {
    // Independent host FP32 multiply, with contraction disabled. NaN payloads
    // are unspecified for arithmetic (unlike the bit-preserving copy contract).
    volatile float weight = before[i + 8], eta = Float(eta_bits);
    float expected = weight * eta;
    auto actual = values[i + 8];
    if (!(std::isnan(expected) ? std::isnan(actual) : Bits(actual) == Bits(expected))) {
      std::fprintf(stderr, "scale n=%zu i=%zu weight=%08x eta=%08x got=%08x want=%08x\n", count, i,
                   Bits(weight), eta_bits, Bits(actual), Bits(expected));
      throw std::runtime_error("FP32 product mismatch");
    }
  }
  ++scale_cases;
  scaled += count;
}
template <typename F>
void Reject(F fn) {
  bool threw = false;
  try {
    fn();
  } catch (std::exception const&) {
    threw = true;
  }
  Check(threw, "Invalid model operation accepted");
}
void Invalid() {
  std::uint32_t output[16]{}, source[16]{};
  auto run = [&](std::vector<metal::ModelCopy> copies) {
    metal::CopyModelBuffer(output, 16, copies, metal::ModelBufferKind::kWeights);
  };
  auto before = metal::GetStatistics();
  Reject([&] { run({{source, 16, 1}}); });
  Reject([&] { run({{source, 17, 0}}); });
  Reject([&] { run({{nullptr, 0, 1}}); });
  Reject([&] { run({{source, 8, 4}, {source, 7, 2}}); });
  Reject([&] { metal::CopyModelBuffer(nullptr, 1, {}, metal::ModelBufferKind::kWeights); });
  Reject([&] { metal::CopyModelBuffer(output, 16, {}, static_cast<metal::ModelBufferKind>(2)); });
  Reject([&] {
    metal::CopyModelBuffer(output, std::numeric_limits<std::size_t>::max(), {},
                           metal::ModelBufferKind::kWeights);
  });
  metal::CopyModelBuffer(nullptr, 0, {}, metal::ModelBufferKind::kCategories);
  metal::ScaleModelWeights({}, .25f);
  auto after = metal::GetStatistics();
  Check(before.model_weight_copy_dispatches == after.model_weight_copy_dispatches &&
            before.model_category_copy_dispatches == after.model_category_copy_dispatches &&
            before.model_scale_dispatches == after.model_scale_dispatches,
        "Invalid or empty operation dispatched work");
}
void ResidentAliases(std::size_t count) {
  std::vector<std::uint32_t> expected(count * 3);
  for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = Pattern(i);
  HostDeviceVector<std::uint32_t> buffer(expected, DeviceOrd::MPS());
  auto data = buffer.DeviceSpan();
  auto original = expected;
  std::copy_n(original.begin() + count, count, expected.begin());
  std::copy_n(original.begin(), count, expected.begin() + count);
  std::vector<metal::ModelCopy> copies{{data.data() + count, 0, count},
                                       {data.data(), count, count}};
  auto before = metal::GetStatistics();
  metal::CopyModelBuffer(data.data(), data.size(), copies, metal::ModelBufferKind::kWeights);
  auto after = metal::GetStatistics();
  Check(buffer.DeviceCanWrite() && !buffer.HostCanRead(), "Alias lost device authority");
  Check(after.model_host_copy_bytes == before.model_host_copy_bytes &&
            after.vector_upload_bytes == before.vector_upload_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes,
        "Resident aliases copied payload through CPU memory");
  Check(after.model_snapshot_bytes - before.model_snapshot_bytes == 2 * count * 4,
        "Resident alias snapshots missing");
  Check(buffer.ConstHostVector() == expected, "Resident alias snapshot mismatch");
  ++copy_cases;
}
void Tree(std::size_t targets, float eta, bool resident = false) {
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "1"}});
  RegTree tree(targets, 3);
  std::vector<float> base(targets), left(targets), right(targets);
  for (std::size_t i = 0; i < targets; ++i) {
    base[i] = float(i + 1) * -.25f;
    left[i] = float(i + 1) * .375f;
    right[i] = float(i + 1) * -.625f;
  }
  HostDeviceVector<float> gpu_base(base, DeviceOrd::MPS()), gpu_left(left, DeviceOrd::MPS()),
      gpu_right(right, DeviceOrd::MPS());
  auto root_before = metal::GetStatistics();
  tree.SetRoot(
      linalg::MakeVec(resident ? DeviceOrd::MPS() : DeviceOrd::CPU(),
                      resident ? gpu_base.ConstDeviceSpan() : common::Span<float const>{base}),
      100.f);
  if (resident) {
    auto after = metal::GetStatistics();
    Check(after.vector_upload_bytes == root_before.vector_upload_bytes &&
              after.vector_readback_bytes == root_before.vector_readback_bytes &&
              after.model_host_copy_bytes == root_before.model_host_copy_bytes,
          "Native root copied weights through CPU memory");
  }
  std::vector<float> expected = base;
  std::vector<std::uint32_t> cats{0x7fc12345, 0xffa54321, 0x80000000, 0xffffffff};
  HostDeviceVector<std::uint32_t> gpu_cats(cats, DeviceOrd::MPS());
  std::vector<std::uint32_t> expected_cats;
  auto stats = metal::GetStatistics();
  for (auto nodes : {std::vector<bst_node_t>{0}, {2, 1}, {6, 3}}) {
    auto old_nodes = expected.size() / targets;
    expected.resize((old_nodes + nodes.size() * 2) * targets);
    tree::ExpandBatch batch{eta};
    // After the first expansion, source categories alias the old model. The
    // following growth must snapshot them before HostDeviceVector reallocates.
    auto source_cats =
        expected_cats.empty()
            ? (resident ? gpu_cats.ConstDeviceSpan() : common::Span<std::uint32_t const>{cats})
            : tree.GetCategoriesMatrix(resident ? DeviceOrd::MPS() : DeviceOrd::CPU())
                  .categories.subspan(0, 4);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      batch.Push(nodes[i], i % 3, .5f, i % 2,
                 resident ? gpu_base.ConstDeviceSpan() : common::Span<float const>{base},
                 resident ? gpu_left.ConstDeviceSpan() : common::Span<float const>{left},
                 resident ? gpu_right.ConstDeviceSpan() : common::Span<float const>{right}, 5.f,
                 20., 30., source_cats);
      std::copy(base.begin(), base.end(), expected.begin() + nodes[i] * targets);
      for (std::size_t t = 0; t < targets; ++t) {
        expected[(old_nodes + i * 2) * targets + t] = left[t] * eta;
        expected[(old_nodes + i * 2 + 1) * targets + t] = right[t] * eta;
      }
      expected_cats.insert(expected_cats.end(), cats.begin(), cats.end());
    }
    auto before_expand = metal::GetStatistics();
    tree.Expand(&ctx, batch);
    if (resident) {
      auto after_expand = metal::GetStatistics();
      Check(after_expand.vector_upload_bytes == before_expand.vector_upload_bytes &&
                after_expand.vector_readback_bytes == before_expand.vector_readback_bytes &&
                after_expand.model_host_copy_bytes == before_expand.model_host_copy_bytes,
            "Native model expansion copied payload through CPU memory");
      // Validate after the entire growth chain, so CPU inspection cannot make
      // the next expansion accidentally succeed using a fresh host mirror.
      if (nodes != std::vector<bst_node_t>{6, 3}) continue;
      Check(after_expand.model_snapshot_bytes > stats.model_snapshot_bytes,
            "Category growth did not snapshot GPU aliases");
    }
    // Serialization requires finalized leaves; keep the growing tree partial.
    std::unique_ptr<RegTree> finalized{tree.Copy()};
    finalized->GetMultiTargetTree()->SetLeaves();
    Json model{Object{}};
    finalized->SaveModel(&model);
    auto const& actual = get<F32Array>(model["base_weights"]);
    Check(actual.size() == expected.size(), "Expanded weight size");
    for (std::size_t i = 0; i < expected.size(); ++i)
      Check(Bits(actual[i]) == Bits(expected[i]), "Parent/preserved/new child weight mismatch");
    auto actual_cats = tree.GetCategoriesMatrix(DeviceOrd::CPU()).categories;
    Check(actual_cats.size() == expected_cats.size() &&
              std::memcmp(actual_cats.data(), expected_cats.data(), actual_cats.size_bytes()) == 0,
          "Aliased category storage lost during model growth");
    RegTree loaded;
    loaded.LoadModel(model);
    Json roundtrip{Object{}};
    loaded.SaveModel(&roundtrip);
    Check(Json::Dump(model) == Json::Dump(roundtrip), "Tree serialization mismatch");
  }
  auto after = metal::GetStatistics();
  Check(after.model_weight_copy_dispatches - stats.model_weight_copy_dispatches == 3 &&
            after.model_category_copy_dispatches - stats.model_category_copy_dispatches == 3 &&
            after.model_scale_dispatches - stats.model_scale_dispatches == 3,
        "Tree expansion did not dispatch copies and scales");
  ++tree_cases;
}
int main() {
  try {
    Check(metal::IsAvailable(), "Metal unavailable");
    for (auto n : {0ul, 1ul, 3ul, 31ul, 32ul, 33ul, 255ul, 256ul, 257ul, 4097ul})
      for (auto kind : {metal::ModelBufferKind::kWeights, metal::ModelBufferKind::kCategories})
        for (bool unaligned : {false, true})
          for (bool alias : {false, true}) Copies(n, kind, unaligned, alias);
    ReadOnlySource();
    for (auto n : {1ul, 33ul, 257ul, 4097ul}) ResidentAliases(n);
    std::puts("Model bit copies, aliases, staging and guards passed");
    for (auto n : {0ul, 1ul, 33ul, 257ul, 65537ul})
      for (auto eta : {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x00800000u, 0x3e99999au,
                       0x3f000000u, 0x3f800000u, 0xbf800000u, 0x3f800001u, 0x4b800000u, 0x7f7fffffu,
                       0x7f800000u, 0xff800000u, 0x7fc12345u})
        Scale(n, eta);
    Invalid();
    std::puts("Model FP32 products, guards and invalid-input rejection passed");
    for (auto targets : {2ul, 3ul, 17ul, 129ul})
      for (float eta : {0.f, .3f, 1.f, Float(0x00800001)})
        for (bool resident : {false, true}) Tree(targets, eta, resident);
    std::printf("PASS: %zu copy cases, %zu scale cases / %zu products, %zu growing trees\n",
                copy_cases, scale_cases, scaled, tree_cases);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
