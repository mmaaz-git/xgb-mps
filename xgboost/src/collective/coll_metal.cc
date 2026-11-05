/** Copyright 2026, XGBoost Contributors */
#if defined(XGBOOST_USE_METAL)
#include <limits>
#include <memory>
#include <utility>

#include "../common/metal_collective.h"
#include "../common/metal_vector.h"
#include "allgather.h"
#include "broadcast.h"
#include "coll.h"

namespace xgboost::collective {
namespace {
namespace metal = common::metal;
using Bytes = common::Span<std::int8_t>;
using Type = ArrayInterfaceHandler;

Bytes Buffer(metal::VectorBuffer* buffer, std::size_t bytes) {
  buffer->Resize(bytes);
  return {static_cast<std::int8_t*>(buffer->Data()), bytes};
}

std::size_t ElementBytes(Type::Type type) {
  // Same numeric type contract as GetNCCLType in coll.cu. Bitwise reductions
  // bypass this mapping and operate on bytes, including floating-point payloads.
  switch (type) {
    case Type::kF2:
      return 2;
    case Type::kF4:
    case Type::kI4:
    case Type::kU4:
      return 4;
    case Type::kF8:
    case Type::kI8:
    case Type::kU8:
      return 8;
    case Type::kI1:
    case Type::kU1:
      return 1;
    default:
      return 0;
  }
}

// Apple shared allocations are addressable by the socket transport. GPU
// producers complete synchronously; Block completes network receives before
// any kernel reads them. No CPU arithmetic or host vector mirror is used.
class MetalColl final : public Coll {
 public:
  Result Allreduce(Context const* ctx, Comm const& comm, Bytes data, Type::Type type,
                   Op op) override {
    if (!comm.IsDistributed()) return Success();
    CHECK(ctx);
    auto bitwise = op == Op::kBitwiseAND || op == Op::kBitwiseOR || op == Op::kBitwiseXOR;
    if (!bitwise && op != Op::kSum && op != Op::kMax && op != Op::kMin)
      return Fail("Invalid Metal reduction operation.");
    auto width = bitwise ? 1 : ElementBytes(type);
    if (!width || data.size() % width)
      return Fail("Invalid type or byte count for Metal collective reduction.");
    // A tracker-connected communicator can be distributed with just one rank.
    // The reduce-scatter ring's final gather uses offset 1, invalid for that case.
    if (data.empty() || comm.World() == 1) return Success();
    auto world = comm.World(), rank = comm.Rank();
    auto scratch = metal::MakeVectorBuffer();
    auto count = data.size() / width;
    if (bitwise || count < static_cast<std::size_t>(world)) {
      if (data.size() > std::numeric_limits<std::size_t>::max() / world)
        return Fail("Metal allreduce buffer size overflow.");
      auto gathered = Buffer(scratch.get(), data.size() * world);
      metal::CollectiveCopy(data, gathered.subspan(rank * data.size(), data.size()));
      auto rc = RingAllgather(comm, gathered);
      if (!rc.OK()) return rc;
      metal::CollectiveReduce(gathered, data, bitwise ? Type::kU1 : type, op, world, true);
      return Success();
    }
    auto segment = count / world * width;
    auto received = Buffer(scratch.get(), data.size() - (world - 1) * segment);
    auto next = comm.Chan(BootstrapNext(rank, world));
    auto prev = comm.Chan(BootstrapPrev(rank, world));
    auto part = [&](std::int32_t r) {
      auto offset = r * segment;
      return data.subspan(offset, r == world - 1 ? data.size() - offset : segment);
    };
    for (std::int32_t r = 0; r < world - 1; ++r) {
      auto send = part((rank + world - r) % world);
      auto out = part((rank + world - r - 1) % world);
      auto input = received.first(out.size());
      auto rc = next->SendAll(send) << [&] {
        return prev->RecvAll(input);
      };
      rc = std::move(rc) + comm.Block();  // drain before scratch can be destroyed
      if (!rc.OK()) return Fail("Metal reduce-scatter failed.", std::move(rc));
      metal::CollectiveReduce(input, out, type, op, 1, false);
    }
    return cpu_impl::RingAllgather(comm, data, segment, 1, prev, next);
  }

  Result Broadcast(Context const*, Comm const& comm, Bytes data, std::int32_t root) override {
    if (root < 0 || root >= comm.World()) return Fail("Invalid Metal broadcast root.");
    if (!comm.IsDistributed() || data.empty()) return Success();
    return cpu_impl::Broadcast(comm, data, root);
  }

  Result Allgather(Context const*, Comm const& comm, Bytes data) override {
    if (data.size() % comm.World()) return Fail("Invalid Metal allgather byte count.");
    if (!comm.IsDistributed() || data.empty()) return Success();
    return RingAllgather(comm, data);
  }

  Result AllgatherV(Context const*, Comm const& comm, common::Span<std::int8_t const> data,
                    common::Span<std::int64_t const> sizes, common::Span<std::int64_t> offsets,
                    Bytes recv, AllgatherVAlgo algo) override {
    if (algo != AllgatherVAlgo::kRing && algo != AllgatherVAlgo::kBcast)
      return Fail("Unknown algorithm for Metal allgather-v.");
    if (sizes.size() != static_cast<std::size_t>(comm.World()) ||
        offsets.size() != sizes.size() + 1)
      return Fail("Invalid Metal allgather-v metadata size.");
    std::int64_t total = 0;
    for (auto size : sizes) {
      if (size < 0 || size > std::numeric_limits<std::int64_t>::max() - total)
        return Fail("Invalid Metal allgather-v payload size.");
      total += size;
    }
    if (static_cast<std::size_t>(total) != recv.size() ||
        static_cast<std::size_t>(sizes[comm.Rank()]) != data.size())
      return Fail("Metal allgather-v payload size mismatch.");
    detail::AllgatherVOffset(sizes, offsets);
    metal::CollectiveCopy(data, recv.subspan(offsets[comm.Rank()], data.size()));
    if (!total || !comm.IsDistributed()) return Success();
    if (algo == AllgatherVAlgo::kRing) return detail::RingAllgatherV(comm, sizes, offsets, recv);
    // cpu_impl::Broadcast rejects empty payloads, while CUDA accepts them.
    for (std::int32_t r = 0; r < comm.World(); ++r) {
      if (!sizes[r]) continue;
      auto rc = cpu_impl::Broadcast(comm, recv.subspan(offsets[r], sizes[r]), r);
      if (!rc.OK()) return rc;
    }
    return Success();
  }
};
}  // namespace
Coll* Coll::MakeMPSVar() { return new MetalColl{}; }
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_METAL)
