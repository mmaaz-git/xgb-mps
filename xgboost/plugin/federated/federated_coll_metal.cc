/** Copyright 2026, XGBoost Contributors */
#include <limits>
#include <memory>
#include <utility>

#include "../../src/collective/allgather.h"
#include "federated_coll.h"

namespace xgboost::collective {
namespace {
using Bytes = common::Span<std::int8_t>;
using Type = ArrayInterfaceHandler;

std::size_t ElementBytes(Type::Type type) {
  // The federated server's AllreduceFunctor supports fewer types than NCCL.
  switch (type) {
    case Type::kI1:
    case Type::kU1:
      return 1;
    case Type::kI4:
    case Type::kU4:
    case Type::kF4:
      return 4;
    case Type::kI8:
    case Type::kU8:
    case Type::kF8:
      return 8;
    default:
      return 0;
  }
}

// CUDAFederatedColl stages device payloads through the host RPC backend. Metal
// shared allocations are already CPU-addressable, and producers finish before
// entry. RPC serialization/receipt can access them directly without an HDV host
// mirror. The caller retains each allocation until the synchronous RPC returns.
// Keep the same CPU backend owner so CPU/MPS calls share its sequence number.
class MetalFederatedColl final : public Coll {
  std::shared_ptr<FederatedColl> impl_;

 public:
  explicit MetalFederatedColl(std::shared_ptr<FederatedColl> impl) : impl_{std::move(impl)} {
    CHECK(impl_);
  }

  Result Allreduce(Context const* ctx, Comm const& comm, Bytes data, Type::Type type,
                   Op op) override {
    auto width = ElementBytes(type);
    if (!width || data.size() % width) return Fail("Invalid federated Metal reduction type/size.");
    auto bitwise = op == Op::kBitwiseAND || op == Op::kBitwiseOR || op == Op::kBitwiseXOR;
    if (!bitwise && op != Op::kSum && op != Op::kMin && op != Op::kMax)
      return Fail("Invalid federated Metal reduction operation.");
    if (bitwise && (type == Type::kF4 || type == Type::kF8))
      return Fail("Federated floating-point reductions do not support bitwise operations.");
    return impl_->Allreduce(ctx, comm, data, type, op);
  }

  Result Broadcast(Context const* ctx, Comm const& comm, Bytes data, std::int32_t root) override {
    if (root < 0 || root >= comm.World()) return Fail("Invalid federated Metal broadcast root.");
    return impl_->Broadcast(ctx, comm, data, root);
  }

  Result Allgather(Context const* ctx, Comm const& comm, Bytes data) override {
    if (data.size() % comm.World()) return Fail("Invalid federated Metal allgather byte count.");
    return impl_->Allgather(ctx, comm, data);
  }

  Result AllgatherV(Context const* ctx, Comm const& comm, common::Span<std::int8_t const> data,
                    common::Span<std::int64_t const> sizes, common::Span<std::int64_t> offsets,
                    Bytes recv, AllgatherVAlgo algo) override {
    if (algo != AllgatherVAlgo::kRing && algo != AllgatherVAlgo::kBcast)
      return Fail("Unknown algorithm for federated Metal allgather-v.");
    if (sizes.size() != static_cast<std::size_t>(comm.World()) ||
        offsets.size() != sizes.size() + 1)
      return Fail("Invalid federated Metal allgather-v metadata size.");
    std::int64_t total = 0;
    for (auto size : sizes) {
      if (size < 0 || size > std::numeric_limits<std::int64_t>::max() - total)
        return Fail("Invalid federated Metal allgather-v payload size.");
      total += size;
    }
    if (static_cast<std::size_t>(total) != recv.size() ||
        static_cast<std::size_t>(sizes[comm.Rank()]) != data.size())
      return Fail("Federated Metal allgather-v payload size mismatch.");
    detail::AllgatherVOffset(sizes, offsets);
    return impl_->AllgatherV(ctx, comm, data, sizes, offsets, recv, algo);
  }
};
}  // namespace

Coll* FederatedColl::MakeMPSVar() {
  return new MetalFederatedColl{std::dynamic_pointer_cast<FederatedColl>(this->shared_from_this())};
}
}  // namespace xgboost::collective
