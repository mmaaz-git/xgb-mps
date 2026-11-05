// Actual federated RPCs over Metal storage, with independent arithmetic references.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../xgboost/src/collective/allgather.h"
#include "../xgboost/src/collective/allreduce.h"
#include "../xgboost/src/common/metal_collective.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_vector.h"
#include "xgboost/c_api.h"

using namespace xgboost;
namespace coll = collective;
namespace metal = common::metal;
using H = ArrayInterfaceHandler;
using Op = coll::Op;
using Bytes = common::Span<std::int8_t>;
std::size_t cases = 0, fields = 0, rejected = 0;
std::string current;
void Check(bool ok, char const* why) {
  if (!ok) throw std::runtime_error(current + ": " + why);
}
void Api(int code) {
  if (code) throw std::runtime_error(XGBGetLastError());
}
struct Buffer {
  std::unique_ptr<metal::VectorBuffer> owner{metal::MakeVectorBuffer()};
  std::size_t size;
  static constexpr std::size_t guard = 17;
  explicit Buffer(std::size_t n) : size{n} {
    owner->Resize(n + guard * 2);
    std::byte sentinel{0xcd};
    owner->Fill({&sentinel, 1}, 0, n + guard * 2);
  }
  Bytes View() { return {static_cast<std::int8_t*>(owner->Data()) + guard, size}; }
  template <typename T>
  void Set(std::vector<T> const& v) {
    Check(v.size() * sizeof(T) == size, "Input size");
    owner->Import(v.data(), guard, size);
  }
  template <typename T>
  std::vector<T> Read() {
    std::vector<std::int8_t> bytes(size + guard * 2);
    owner->Readback(bytes.data(), bytes.size());
    for (std::size_t i = 0; i < guard; ++i)
      Check(std::uint8_t(bytes[i]) == 0xcd && std::uint8_t(bytes[guard + size + i]) == 0xcd,
            "Guard overwritten");
    std::vector<T> result(size / sizeof(T));
    if (size) std::memcpy(result.data(), bytes.data() + guard, size);
    return result;
  }
};
void Resident(metal::Statistics const& before, metal::CollectiveStatistics const& collective) {
  auto after = metal::GetStatistics();
  auto now = metal::GetCollectiveStatistics();
  Check(before.vector_readback_bytes == after.vector_readback_bytes &&
            before.vector_upload_bytes == after.vector_upload_bytes,
        "RPC created a vector mirror/upload");
  Check(now.staging_bytes == collective.staging_bytes, "GPU copy staged resident payload");
  Check(now.reduce_dispatches == collective.reduce_dispatches,
        "Federated server reduction unexpectedly ran locally");
}
template <typename T>
T InputValue(std::size_t i, int rank, bool bits) {
  if constexpr (std::is_integral_v<T>) {
    if (bits) return T((std::uint64_t{0xfedcba9876543210} >> (i % 13)) ^ (rank * 37u));
    return T((i + rank * 3) % 11);
  } else {
    return T((int(i % 17) - 8 + rank) * .25);
  }
}
template <typename T>
T Combine(T a, T b, Op op) {
  if (op == Op::kSum) return T(a + b);
  if (op == Op::kMin) return std::min(a, b);
  if (op == Op::kMax) return std::max(a, b);
  if constexpr (std::is_integral_v<T>) {
    if (op == Op::kBitwiseAND) return T(a & b);
    if (op == Op::kBitwiseOR) return T(a | b);
    if (op == Op::kBitwiseXOR) return T(a ^ b);
  }
  throw std::runtime_error("Invalid reference operation");
}
template <typename T>
void Numeric(Context const& ctx, H::Type type, int rank, int world, bool large = false) {
  auto& group = *coll::GlobalCommGroup();
  auto cpu = ctx.MakeCPU();
  auto backend = group.Backend(DeviceOrd::MPS());
  auto const& comm = group.Ctx(&ctx, DeviceOrd::MPS());
  std::vector<Op> ops{Op::kSum, Op::kMin, Op::kMax};
  if constexpr (std::is_integral_v<T>)
    ops.insert(ops.end(), {Op::kBitwiseAND, Op::kBitwiseOR, Op::kBitwiseXOR});
  auto lengths = large ? std::vector<std::size_t>{(1u << 20) + 3}
                       : std::vector<std::size_t>{0, 1, 7, 257, 4099};
  for (auto op : ops)
    for (auto n : lengths) {
      current = "numeric/" + std::to_string(type) + "/" + std::to_string(int(op)) + "/" +
                std::to_string(n);
      bool bits = op == Op::kBitwiseAND || op == Op::kBitwiseOR || op == Op::kBitwiseXOR;
      std::vector<T> local(n), expected(n);
      for (std::size_t i = 0; i < n; ++i) {
        local[i] = InputValue<T>(i, rank, bits);
        expected[i] = InputValue<T>(i, 0, bits);
        for (int r = 1; r < world; ++r)
          expected[i] = Combine(expected[i], InputValue<T>(i, r, bits), op);
      }
      Buffer input(n * sizeof(T)), consumed(n * sizeof(T));
      input.Set(local);
      auto before = metal::GetStatistics();
      auto collective = metal::GetCollectiveStatistics();
      coll::SafeColl(backend->Allreduce(&ctx, comm, input.View(), type, op));
      // A GPU consumer must see the RPC's shared-memory writes immediately.
      metal::CollectiveCopy(input.View(), consumed.View());
      Resident(before, collective);
      Check((metal::GetCollectiveStatistics().copy_dispatches > collective.copy_dispatches) ==
                (n != 0),
            "Missing GPU consumer");
      coll::SafeColl(coll::Allreduce(&cpu, linalg::MakeVec(local.data(), local.size()), op));
      Check(local == expected && input.Read<T>() == expected && consumed.Read<T>() == expected,
            "Server/CPU/Metal reduction disagrees with independent reference");
      fields += n * 3;
      ++cases;
    }
  std::printf("PASS world%d rank%d type%d: %zu cases / %zu fields\n", world, rank, int(type), cases,
              fields);
  std::fflush(stdout);
}
std::vector<std::int8_t> Pattern(std::size_t n, int rank) {
  std::vector<std::int8_t> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = std::int8_t(i * 71 + rank * 37);
  return out;
}
void Transport(Context const& ctx, int rank, int world) {
  auto& group = *coll::GlobalCommGroup();
  auto backend = group.Backend(DeviceOrd::MPS());
  auto const& comm = group.Ctx(&ctx, DeviceOrd::MPS());
  for (std::size_t n : {0ul, 1ul, 31ul, 4099ul, (1ul << 20) + 3}) {
    current = "transport/" + std::to_string(n);
    for (int root = 0; root < world; ++root) {
      Buffer data(n), consumed(n);
      data.Set(Pattern(n, rank));
      auto state = metal::GetStatistics();
      auto before = metal::GetCollectiveStatistics();
      coll::SafeColl(backend->Broadcast(&ctx, comm, data.View(), root));
      metal::CollectiveCopy(data.View(), consumed.View());
      Resident(state, before);
      Check(consumed.Read<std::int8_t>() == Pattern(n, root), "Broadcast bytes");
      Check(data.Read<std::int8_t>() == Pattern(n, root), "Broadcast guards/data");
      fields += n * 2;
      ++cases;
    }
    Buffer data(n * world);
    std::vector<std::int8_t> local(n * world), expected;
    auto segment = Pattern(n, rank);
    std::copy(segment.begin(), segment.end(), local.begin() + rank * n);
    for (int r = 0; r < world; ++r) {
      auto peer = Pattern(n, r);
      expected.insert(expected.end(), peer.begin(), peer.end());
    }
    data.Set(local);
    auto state = metal::GetStatistics();
    auto before = metal::GetCollectiveStatistics();
    coll::SafeColl(backend->Allgather(&ctx, comm, data.View()));
    Resident(state, before);
    Check(data.Read<std::int8_t>() == expected, "Allgather bytes");
    fields += expected.size();
    ++cases;
  }
  for (int layout = 0; layout < 4; ++layout)
    for (auto algo : {coll::AllgatherVAlgo::kRing, coll::AllgatherVAlgo::kBcast}) {
      current = "allgatherv/" + std::to_string(layout);
      std::vector<std::int64_t> sizes(world), offsets(world + 1, -1), expected_offsets{0};
      std::vector<std::int8_t> expected;
      for (int r = 0; r < world; ++r) {
        sizes[r] = layout == 0 || (layout == 1 && r == 0) || (layout == 2 && r != 0) ? 0 : 257 + r;
        auto peer = Pattern(sizes[r], r);
        expected.insert(expected.end(), peer.begin(), peer.end());
        expected_offsets.push_back(expected.size());
      }
      Buffer input(sizes[rank]), output(expected.size());
      input.Set(Pattern(sizes[rank], rank));
      auto state = metal::GetStatistics();
      auto before = metal::GetCollectiveStatistics();
      coll::SafeColl(
          backend->AllgatherV(&ctx, comm, input.View(), sizes, offsets, output.View(), algo));
      Resident(state, before);
      Check(output.Read<std::int8_t>() == expected && offsets == expected_offsets,
            "AllgatherV output");
      Check(input.Read<std::int8_t>() == Pattern(sizes[rank], rank), "AllgatherV input mutated");
      fields += expected.size() + offsets.size() + sizes[rank];
      ++cases;
    }
}
void Invalid(Context const& ctx, int world) {
  current = "invalid";
  auto& group = *coll::GlobalCommGroup();
  auto backend = group.Backend(DeviceOrd::MPS());
  auto const& comm = group.Ctx(&ctx, DeviceOrd::MPS());
  Buffer data(32);
  auto reject = [&](coll::Result rc) {
    Check(!rc.OK(), "Invalid call accepted");
    ++rejected;
  };
  for (auto type : {H::kF2, H::kF16, H::kI2, H::kU2})
    reject(backend->Allreduce(&ctx, comm, data.View(), type, Op::kSum));
  reject(backend->Allreduce(&ctx, comm, data.View().first(3), H::kF4, Op::kSum));
  reject(backend->Allreduce(&ctx, comm, data.View(), H::kF4, Op::kBitwiseOR));
  reject(backend->Allreduce(&ctx, comm, data.View(), H::kI4, static_cast<Op>(99)));
  reject(backend->Broadcast(&ctx, comm, data.View(), world));
  if (world > 1) reject(backend->Allgather(&ctx, comm, data.View().first(1)));
  std::vector<std::int64_t> sizes(world), offsets(world + 1);
  reject(backend->AllgatherV(&ctx, comm, {}, sizes, offsets, {},
                             static_cast<coll::AllgatherVAlgo>(99)));
  sizes[0] = -1;
  reject(backend->AllgatherV(&ctx, comm, {}, sizes, offsets, {}, coll::AllgatherVAlgo::kRing));
  // Rejected local calls must not consume RPC sequence numbers.
  std::int32_t one = 1;
  coll::SafeColl(coll::Allreduce(&ctx, &one, Op::kSum));
  Check(one == world, "CPU/Metal sequence after rejection");
}
int main(int argc, char** argv) {
  try {
    Check(argc == 2, "Communicator JSON required");
    Api(XGCommunicatorInit(argv[1]));
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    auto& group = *coll::GlobalCommGroup();
    int rank = group.Rank(), world = group.World();
    Check(group.Ctx(&ctx, DeviceOrd::MPS()).IsFederated(), "Real federated communicator required");
    Check(group.Backend(DeviceOrd::CPU()) != group.Backend(DeviceOrd::MPS()),
          "Missing Metal adapter");
    Numeric<std::int8_t>(ctx, H::kI1, rank, world);
    Numeric<std::uint8_t>(ctx, H::kU1, rank, world);
    Numeric<std::int32_t>(ctx, H::kI4, rank, world);
    Numeric<std::uint32_t>(ctx, H::kU4, rank, world);
    Numeric<std::int64_t>(ctx, H::kI8, rank, world);
    Numeric<std::uint64_t>(ctx, H::kU8, rank, world);
    Numeric<float>(ctx, H::kF4, rank, world);
    Numeric<double>(ctx, H::kF8, rank, world);
    Numeric<float>(ctx, H::kF4, rank, world, true);
    Transport(ctx, rank, world);
    Invalid(ctx, world);
    Check(metal::GetStatistics().vector_live_buffers == 0, "Leaked Metal buffers");
    Api(XGCommunicatorFinalize());
    std::printf("PASS world%d rank%d: %zu federated cases / %zu fields / %zu rejected\n", world,
                rank, cases, fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), e.what());
    return 1;
  }
}
