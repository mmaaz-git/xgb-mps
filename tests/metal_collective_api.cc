// Real multi-process transport and independent scalar arithmetic references.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../xgboost/src/collective/allgather.h"
#include "../xgboost/src/collective/allreduce.h"
#include "../xgboost/src/collective/broadcast.h"
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
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Api(int code) {
  if (code) throw std::runtime_error(XGBGetLastError());
}
void Success(coll::Result rc) { coll::SafeColl(rc); }
struct Buffer {
  std::unique_ptr<metal::VectorBuffer> storage{metal::MakeVectorBuffer()};
  std::size_t size;
  static constexpr std::size_t guard = 17;
  explicit Buffer(std::size_t n) : size{n} {
    storage->Resize(n + guard * 2);
    std::byte sentinel{0xcd};
    storage->Fill({&sentinel, 1}, 0, n + guard * 2);
  }
  Bytes View() { return {static_cast<std::int8_t*>(storage->Data()) + guard, size}; }
  template <typename T>
  void Set(std::vector<T> const& v) {
    Check(v.size() * sizeof(T) == size, "Input size");
    storage->Import(v.data(), guard, size);
  }
  template <typename T>
  std::vector<T> Read() {
    std::vector<std::int8_t> bytes(size + guard * 2);
    storage->Readback(bytes.data(), bytes.size());
    for (std::size_t i = 0; i < guard; ++i)
      Check(std::uint8_t(bytes[i]) == 0xcd && std::uint8_t(bytes[guard + size + i]) == 0xcd,
            "Output guard overwritten");
    std::vector<T> result(size / sizeof(T));
    if (size) std::memcpy(result.data(), bytes.data() + guard, size);
    return result;
  }
};
template <typename T>
constexpr bool Floating = std::is_floating_point_v<T> || std::is_same_v<T, _Float16>;
template <typename T>
T Combine(T a, T b, Op op) {
  if constexpr (Floating<T>) {
    if (op == Op::kSum) return T(a + b);
    // NVIDIA documents number-preferred min/max for any single NaN.
    // ARM fminnm can propagate a signaling NaN; handle that policy explicitly
    // instead of treating the local instruction as the CUDA reference.
    if (std::isnan(double(a))) return b;
    if (std::isnan(double(b))) return a;
    if (a == 0 && b == 0) {
      bool negative = op == Op::kMin ? std::signbit(double(a)) || std::signbit(double(b))
                                     : std::signbit(double(a)) && std::signbit(double(b));
      return T(negative ? -0. : 0.);
    }
    return T(op == Op::kMin ? std::fmin(double(a), double(b)) : std::fmax(double(a), double(b)));
  } else {
    if (op == Op::kSum) {
      using U = std::make_unsigned_t<T>;
      U bits = U(a) + U(b);
      T value;
      std::memcpy(&value, &bits, sizeof(T));
      return value;
    }
    return op == Op::kMin ? std::min(a, b) : std::max(a, b);
  }
}
template <typename T>
void Equal(T a, T b) {
  ++fields;
  if constexpr (Floating<T>) {
    if (std::isnan(double(a)) && std::isnan(double(b))) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &a, sizeof(T));
      auto quiet = std::uint64_t{1} << (sizeof(T) == 2 ? 9 : sizeof(T) == 4 ? 22 : 51);
      Check(bits & quiet, "Reduction emitted signaling NaN");
      return;
    }
  }
  if (std::memcmp(&a, &b, sizeof(T))) {
    std::fprintf(stderr, "%s got %.17g expected %.17g\n", current.c_str(), double(a), double(b));
    Check(false, "Scalar reference differs (bits)");
  }
}
template <typename T>
std::vector<T> Values(std::size_t n, int rank, bool extreme) {
  std::vector<T> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    if constexpr (Floating<T>) {
      if (!extreme) {
        result[i] = T((int(i % 17) - 8 + rank) * .25);
        continue;
      }
      T denorm;
      std::uint64_t one = 1;
      std::memcpy(&denorm, &one, sizeof(T));
      T inf = T(std::numeric_limits<double>::infinity());
      T nan = T(std::numeric_limits<double>::quiet_NaN());
      T vals[]{T(0.), T(-0.), denorm, T(-denorm), T(1.), T(-1.), T(.25), inf, T(-inf), nan};
      result[i] = vals[(i + rank * 3) % 10];
    } else {
      if (!extreme) {
        result[i] = T(i * 71 + rank * 37);
        continue;
      }
      T vals[]{T(0), T(1), T(-1), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()};
      result[i] = vals[(i + rank) % 5];
    }
  }
  return result;
}
template <typename T>
void Numeric(Context const& ctx, H::Type type, int rank, int world) {
  std::printf("numeric type%d world%d rank%d\n", int(type), world, rank);
  std::fflush(stdout);
  auto& cg = *coll::GlobalCommGroup();
  for (auto op : {Op::kSum, Op::kMin, Op::kMax})
    for (bool extreme : {false, true})
      for (std::size_t n : {0, 1, 2, 3, 17, 257, 4097}) {
        current = "numeric/" + std::to_string(type) + "/" + std::to_string(int(op)) + "/" +
                  std::to_string(n) + "/" + std::to_string(extreme);
        Buffer data(n * sizeof(T));
        data.Set(Values<T>(n, rank, extreme));
        auto before = metal::GetCollectiveStatistics();
        Success(cg.Backend(DeviceOrd::MPS())
                    ->Allreduce(&ctx, cg.Ctx(&ctx, DeviceOrd::MPS()), data.View(), type, op));
        auto after = metal::GetCollectiveStatistics();
        Check(after.staging_bytes == before.staging_bytes, "Collective host staging");
        Check((after.reduce_dispatches > before.reduce_dispatches) == (n && world > 1),
              "Missing/unexpected GPU reduction");
        auto actual = data.Read<T>();
        std::vector<std::vector<T>> inputs;
        for (int r = 0; r < world; ++r) inputs.push_back(Values<T>(n, r, extreme));
        for (std::size_t i = 0; i < n; ++i) {
          // Independent simulation of the documented ring topology, including
          // rank rotation per segment. Small arrays fold rank 0..world-1.
          int first = n < std::size_t(world) || world == 1
                          ? 0
                          : int(std::min(i / (n / world), std::size_t(world - 1)));
          T expected = inputs[first][i];
          for (int r = 1; r < world; ++r) {
            int next = n < std::size_t(world) ? r : (first + r) % world;
            expected = Combine(inputs[next][i], expected, op);
          }
          Equal(actual[i], expected);
        }
        ++cases;
      }
}
void Bitwise(Context const& ctx, int rank, int world) {
  auto& cg = *coll::GlobalCommGroup();
  for (int type = 0; type < 12; ++type)
    for (auto op : {Op::kBitwiseAND, Op::kBitwiseOR, Op::kBitwiseXOR})
      for (std::size_t n : {0, 1, 3, 257, 8197}) {
        current = "bitwise/" + std::to_string(type) + "/" + std::to_string(int(op));
        Buffer data(n);
        auto values = Values<std::uint8_t>(n, rank, false);
        data.Set(values);
        auto before = metal::GetCollectiveStatistics();
        Success(
            cg.Backend(DeviceOrd::MPS())
                ->Allreduce(&ctx, cg.Ctx(&ctx, DeviceOrd::MPS()), data.View(), H::Type(type), op));
        auto after = metal::GetCollectiveStatistics();
        Check(after.staging_bytes == before.staging_bytes, "Bitwise host staging");
        Check(after.reduce_dispatches - before.reduce_dispatches == (n && world > 1 ? 1 : 0),
              "Bitwise GPU dispatch count");
        auto result = data.Read<std::uint8_t>();
        for (std::size_t i = 0; i < n; ++i) {
          auto expected = std::uint8_t(i * 71);
          for (int r = 1; r < world; ++r) {
            auto v = std::uint8_t(i * 71 + r * 37);
            if (op == Op::kBitwiseAND) expected &= v;
            if (op == Op::kBitwiseOR) expected |= v;
            if (op == Op::kBitwiseXOR) expected ^= v;
          }
          Equal(result[i], expected);
        }
        ++cases;
      }
}
void Transfers(Context const& ctx, int rank, int world) {
  auto& cg = *coll::GlobalCommGroup();
  auto backend = cg.Backend(DeviceOrd::MPS());
  auto& comm = cg.Ctx(&ctx, DeviceOrd::MPS());
  for (std::size_t n : {0, 1, 3, 257, 65539}) {
    current = "transfers/" + std::to_string(n);
    for (int root = 0; root < world; ++root) {
      Buffer data(n);
      data.Set(Values<std::uint8_t>(n, rank, false));
      Success(backend->Broadcast(&ctx, comm, data.View(), root));
      Check(data.Read<std::uint8_t>() == Values<std::uint8_t>(n, root, false), "Broadcast data");
      fields += n;
      ++cases;
    }
    Buffer data(n * world);
    auto local = Values<std::uint8_t>(n, rank, false);
    data.storage->Import(local.data(), Buffer::guard + rank * n, n);
    Success(backend->Allgather(&ctx, comm, data.View()));
    auto result = data.Read<std::uint8_t>();
    for (int r = 0; r < world; ++r) {
      auto values = Values<std::uint8_t>(n, r, false);
      Check(std::equal(values.begin(), values.end(), result.begin() + r * n), "Allgather data");
    }
    fields += n * world;
    ++cases;
  }
  for (auto algo : {coll::AllgatherVAlgo::kRing, coll::AllgatherVAlgo::kBcast})
    for (int layout = 0; layout < 4; ++layout)
      for (bool inplace : {false, true}) {
        current = "allgatherv/" + std::to_string(int(algo)) + "/" + std::to_string(layout);
        std::vector<std::int64_t> sizes(world), offsets(world + 1, -1);
        std::size_t total = 0, own = 0;
        for (int r = 0; r < world; ++r) {
          if (r == rank) own = total;
          sizes[r] =
              layout == 0 || (layout == 1 && r == 0) || (layout == 2 && r != 0) ? 0 : 8197 + r * 13;
          total += sizes[r];
        }
        Buffer data(sizes[rank]), output(total);
        auto local = Values<std::uint8_t>(sizes[rank], rank, false);
        data.Set(local);
        auto input = data.View();
        if (inplace) {
          output.storage->Import(local.data(), Buffer::guard + own, local.size());
          input = output.View().subspan(own, local.size());
        }
        auto before = metal::GetCollectiveStatistics();
        Success(backend->AllgatherV(&ctx, comm, input, sizes, offsets, output.View(), algo));
        auto after = metal::GetCollectiveStatistics();
        Check(
            after.copy_dispatches - before.copy_dispatches == (!inplace && !local.empty() ? 1 : 0),
            "Missing/unexpected GPU local copy");
        Check(after.staging_bytes == before.staging_bytes, "AllgatherV host staging");
        auto result = output.Read<std::uint8_t>();
        std::size_t offset = 0;
        for (int r = 0; r < world; ++r) {
          Equal(offsets[r], std::int64_t(offset));
          auto values = Values<std::uint8_t>(sizes[r], r, false);
          Check(std::equal(values.begin(), values.end(), result.begin() + offset),
                "AllgatherV data");
          offset += sizes[r];
        }
        Equal(offsets.back(), std::int64_t(total));
        fields += total;
        ++cases;
      }
}
void Wrappers(Context const& ctx, int rank, int world) {
  current = "typed wrappers and ownership";
  HostDeviceVector<float> values(17, float(rank + 1), DeviceOrd::MPS());
  auto view = linalg::MakeVec(DeviceOrd::MPS(), values.DeviceSpan());
  Success(coll::Allreduce(&ctx, view, Op::kSum));
  Check(!values.HostCanRead(), "Allreduce requested host mirror");
  for (auto value : values.ConstHostVector()) Equal(value, float(world * (world + 1) / 2));
  auto cpu = ctx.MakeCPU();
  float scalar = rank + 1;
  auto before = metal::GetCollectiveStatistics();
  Success(coll::Allreduce(&cpu, linalg::MakeVec(&scalar, 1), Op::kSum));
  Equal(scalar, float(world * (world + 1) / 2));
  Check(before.reduce_dispatches == metal::GetCollectiveStatistics().reduce_dispatches,
        "CPU control scalar dispatched to Metal");
  values.HostVector().assign(rank * 3, float(rank));
  view = linalg::MakeVec(DeviceOrd::MPS(), metal::DeviceSpan(values));
  HostDeviceVector<std::int8_t> output;
  std::vector<std::int64_t> offsets;
  Success(coll::AllgatherV(&ctx, view, &offsets, &output));
  if (world > 1) {
    Check(!output.HostCanRead() && output.Device().IsMPS(), "AllgatherV created CPU output");
    auto bytes = output.ConstHostVector();
    for (int r = 0; r < world; ++r)
      for (int i = 0; i < r * 3; ++i) {
        float value;
        std::memcpy(&value, bytes.data() + offsets[r] + i * 4, 4);
        Equal(value, float(r));
      }
  }
  ++cases;
}
void Invalid(Context const& ctx, int world) {
  if (world == 1) return;
  current = "invalid contracts";
  auto& cg = *coll::GlobalCommGroup();
  auto b = cg.Backend(DeviceOrd::MPS());
  auto& comm = cg.Ctx(&ctx, DeviceOrd::MPS());
  Buffer data(16);
  std::vector<std::int64_t> sizes(world, 0), offsets(world + 1);
  auto reject = [&](coll::Result rc) {
    Check(!rc.OK(), "Invalid call accepted");
    ++rejected;
  };
  for (auto type : {H::kI2, H::kU2, H::kF16})
    reject(b->Allreduce(&ctx, comm, data.View(), type, Op::kSum));
  reject(b->Allreduce(&ctx, comm, data.View().first(3), H::kF4, Op::kSum));
  reject(b->Allreduce(&ctx, comm, data.View(), H::kU1, Op(99)));
  reject(b->Broadcast(&ctx, comm, data.View(), -1));
  reject(b->Broadcast(&ctx, comm, data.View(), world));
  reject(b->Allgather(&ctx, comm, data.View().first(1)));
  reject(b->AllgatherV(&ctx, comm, {}, sizes, offsets, {}, coll::AllgatherVAlgo(99)));
  sizes[0] = -1;
  reject(b->AllgatherV(&ctx, comm, {}, sizes, offsets, {}, coll::AllgatherVAlgo::kRing));
  sizes[0] = std::numeric_limits<std::int64_t>::max();
  sizes[1] = 1;
  reject(b->AllgatherV(&ctx, comm, {}, sizes, offsets, {}, coll::AllgatherVAlgo::kRing));
}

// Exhaustive half bit patterns and independently generated full-exponent float/
// double/integer operands. Native host arithmetic is the oracle, not the shader.
template <typename T>
void RawArithmetic(H::Type type, std::size_t n) {
  current = "raw arithmetic/" + std::to_string(type);
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  std::vector<T> inputs(n * 4), initial(n);
  auto random = [&] {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto bits = sizeof(T) == 2 && i < n ? i : random();
    std::memcpy(&inputs[i], &bits, sizeof(T));
  }
  for (auto& value : initial) {
    auto bits = random();
    std::memcpy(&value, &bits, sizeof(T));
  }
  if constexpr (Floating<T>) {
    auto sign = std::uint64_t{1} << (8 * sizeof(T) - 1);
    auto inf = sizeof(T) == 2 ? 0x7c00ull : sizeof(T) == 4 ? 0x7f800000ull : 0x7ff0000000000000ull;
    for (int r = 0; r < 4; ++r) {
      std::uint64_t bits[]{inf | 1, r % 2 ? sign : 0, inf | (r % 2 ? sign : 0), sign, 0, inf - 1};
      for (std::size_t i = 0; i < 6; ++i) {
        std::memcpy(&inputs[r * n + n - 6 + i], &bits[i], sizeof(T));
        std::memcpy(&initial[n - 6 + i], &bits[i], sizeof(T));
      }
    }
  }
  Buffer in(inputs.size() * sizeof(T)), out(n * sizeof(T));
  in.Set(inputs);
  for (auto op : {Op::kSum, Op::kMin, Op::kMax})
    for (bool initialize : {false, true}) {
      out.Set(initial);
      auto before = metal::GetCollectiveStatistics();
      metal::CollectiveReduce(in.View(), out.View(), type, op, 4, initialize);
      auto after = metal::GetCollectiveStatistics();
      Check(after.reduce_dispatches == before.reduce_dispatches + 1, "Raw kernel dispatch");
      Check(after.staging_bytes == before.staging_bytes, "Raw kernel staged");
      auto actual = out.Read<T>();
      for (std::size_t i = 0; i < n; ++i) {
        T expected = initialize ? inputs[i] : initial[i];
        for (int r = initialize ? 1 : 0; r < 4; ++r)
          expected = Combine(expected, inputs[r * n + i], op);
        if (std::memcmp(&actual[i], &expected, sizeof(T)) &&
            !(Floating<T> && std::isnan(double(actual[i])) && std::isnan(double(expected)))) {
          std::fprintf(stderr, "raw type%d op%d init%d index%zu initial%.17g inputs", int(type),
                       int(op), int(initialize), i, double(initial[i]));
          for (int r = 0; r < 4; ++r) std::fprintf(stderr, " %.17g", double(inputs[r * n + i]));
          std::fprintf(stderr, "\n");
        }
        Equal(actual[i], expected);
      }
      ++cases;
    }
}
void CopyOverlap() {
  current = "overlapping GPU copy";
  for (bool forward : {false, true}) {
    Buffer data(8197);
    auto expected = Values<std::uint8_t>(8197, 7, false);
    data.Set(expected);
    auto from = forward ? 0 : 3, to = forward ? 3 : 0;
    auto before = metal::GetCollectiveStatistics();
    metal::CollectiveCopy(data.View().subspan(from, 8194), data.View().subspan(to, 8194));
    auto after = metal::GetCollectiveStatistics();
    Check(after.copy_dispatches == before.copy_dispatches + 1 &&
              after.staging_bytes == before.staging_bytes,
          "Overlap did not snapshot on GPU");
    std::memmove(expected.data() + to, expected.data() + from, 8194);
    Check(data.Read<std::uint8_t>() == expected, "Overlapping copy differs");
    fields += expected.size();
    ++cases;
  }
}
void CPUEmptyGather(Context const& ctx, int rank, int world) {
  current = "CPU empty-segment regression";
  auto cpu = ctx.MakeCPU();
  auto& cg = *coll::GlobalCommGroup();
  for (bool all_empty : {false, true}) {
    std::vector<std::int64_t> sizes(world), offsets(world + 1);
    std::size_t total = 0;
    for (int r = 0; r < world; ++r) {
      sizes[r] = all_empty ? 0 : r * 17;
      total += sizes[r];
    }
    auto local = Values<std::int8_t>(sizes[rank], rank, false);
    std::vector<std::int8_t> output(total);
    Success(cg.Backend(DeviceOrd::CPU())
                ->AllgatherV(&cpu, cg.Ctx(&cpu, DeviceOrd::CPU()), local, sizes, offsets, output,
                             coll::AllgatherVAlgo::kRing));
    for (int r = 0; r < world; ++r) {
      auto expected = Values<std::int8_t>(sizes[r], r, false);
      Check(std::equal(expected.begin(), expected.end(), output.begin() + offsets[r]),
            "CPU gather data");
    }
    fields += total;
    ++cases;
  }
}

void Disconnect(Context const& ctx, int rank) {
  current = "disconnected peer";
  {
    Buffer data(1 << 20);
    auto values = Values<std::int8_t>(1 << 20, rank, false);
    data.Set(values);
    auto cpu = ctx.MakeCPU();
    int ready = 1;
    Success(coll::Allreduce(&cpu, linalg::MakeVec(&ready, 1), Op::kSum));
    if (rank == 1) {
      std::puts("Deliberately disconnecting rank1 after initialization");
      std::fflush(stdout);
      std::_Exit(0);
    }
    auto& cg = *coll::GlobalCommGroup();
    auto rc = cg.Backend(DeviceOrd::MPS())
                  ->Allreduce(&ctx, cg.Ctx(&ctx, DeviceOrd::MPS()), data.View(), H::kI8, Op::kSum);
    Check(!rc.OK(), "Disconnected peer returned success");
    std::puts("PASS Metal collective propagated disconnected-peer failure");
    std::fflush(stdout);
  }
  Check(metal::GetStatistics().vector_live_buffers == 0, "Failure leaked vector buffers");
  Api(XGCommunicatorFinalize());
}
int main(int argc, char** argv) {
  try {
    Check(argc == 2 || argc == 3, "Communicator JSON required");
    Api(XGCommunicatorInit(argv[1]));
    int rank = XGCommunicatorGetRank(), world = XGCommunicatorGetWorldSize();
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    if (argc == 3) {
      Check(world == 2 && std::string(argv[2]) == "--disconnect", "Failure mode");
      Disconnect(ctx, rank);
      return 0;
    }
    Numeric<_Float16>(ctx, H::kF2, rank, world);
    Numeric<float>(ctx, H::kF4, rank, world);
    Numeric<double>(ctx, H::kF8, rank, world);
    Numeric<std::int8_t>(ctx, H::kI1, rank, world);
    Numeric<std::uint8_t>(ctx, H::kU1, rank, world);
    Numeric<std::int32_t>(ctx, H::kI4, rank, world);
    Numeric<std::uint32_t>(ctx, H::kU4, rank, world);
    Numeric<std::int64_t>(ctx, H::kI8, rank, world);
    Numeric<std::uint64_t>(ctx, H::kU8, rank, world);
    if (world == 1) {
      RawArithmetic<_Float16>(H::kF2, 65542);
      RawArithmetic<float>(H::kF4, 16387);
      RawArithmetic<double>(H::kF8, 16387);
      RawArithmetic<std::int8_t>(H::kI1, 8197);
      RawArithmetic<std::uint8_t>(H::kU1, 8197);
      RawArithmetic<std::int32_t>(H::kI4, 8197);
      RawArithmetic<std::uint32_t>(H::kU4, 8197);
      RawArithmetic<std::int64_t>(H::kI8, 8197);
      RawArithmetic<std::uint64_t>(H::kU8, 8197);
      CopyOverlap();
    }
    CPUEmptyGather(ctx, rank, world);
    Bitwise(ctx, rank, world);
    Transfers(ctx, rank, world);
    Wrappers(ctx, rank, world);
    Invalid(ctx, world);
    Check(metal::GetStatistics().vector_live_buffers == 0, "Leaked GPU vector buffers");
    Api(XGCommunicatorFinalize());
    std::printf("PASS world%d rank%d: %zu collective cases / %zu fields / %zu rejected calls\n",
                world, rank, cases, fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
