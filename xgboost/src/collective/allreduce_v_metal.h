/** Copyright 2026, XGBoost Contributors */
#pragma once

#if defined(XGBOOST_USE_METAL)
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../common/metal_collective.h"
#include "../common/metal_vector.h"
#include "allgather.h"
#include "broadcast.h"
#include "comm_group.h"
#include "topo.h"

namespace xgboost::collective::metal_impl {
using Payload = std::unique_ptr<common::metal::VectorBuffer>;
struct AllreduceVScratch {
  Payload payload{common::metal::MakeVectorBuffer()};
  Payload next{common::metal::MakeVectorBuffer()};

  void Reserve(std::size_t bytes) {
    for (auto* buffer : {payload.get(), next.get()}) {
      buffer->Resize(bytes);
      buffer->Resize(0);
    }
  }
};

// Counterpart of gpu_detail::AllreduceVHostFallback: gather every payload, then
// merge peers in ascending rank order on rank zero. Federated communicators do
// not have peer channels. This order deliberately differs from the NCCL tree.
// Shared Metal memory lets the RPC backend serialize GPU payloads directly;
// all payload snapshots and reduction callbacks remain GPU operations.
template <typename Fn>
Result AllreduceVFederated(Context const* ctx, CommGroup const& group, Payload* data,
                           AllreduceVScratch* scratch, Fn&& redop) {
  auto bytes = [](common::metal::VectorBuffer const& buffer) {
    return common::Span<std::int8_t>{static_cast<std::int8_t*>(buffer.Data()), buffer.SizeBytes()};
  };
  if ((*data)->SizeBytes() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    return Fail("Federated Metal AllreduceV payload size overflow.");
  std::vector<std::int64_t> offsets;
  HostDeviceVector<std::int8_t> gathered;
  auto input = bytes(**data);
  auto rc = AllgatherV(ctx, group, linalg::MakeTensorView(DeviceOrd::MPS(), input, input.size()),
                       &offsets, &gathered);
  if (!rc.OK()) return Fail("Federated Metal AllreduceV allgather failed.", std::move(rc));

  constexpr std::int32_t kRoot = 0;
  if (group.Rank() == kRoot) {
    auto all = gathered.ConstDeviceSpan();
    auto copy = [&](std::int32_t peer, common::metal::VectorBuffer* out) {
      auto segment = all.subspan(offsets[peer], offsets[peer + 1] - offsets[peer]);
      out->Resize(segment.size());
      common::metal::CollectiveCopy(segment, bytes(*out));
    };
    copy(kRoot, data->get());
    for (std::int32_t peer = 1; peer < group.World(); ++peer) {
      copy(peer, scratch->payload.get());
      redop(**data, *scratch->payload, scratch->next.get());
      std::swap(*data, scratch->next);
    }
  }

  if ((*data)->SizeBytes() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    return Fail("Federated Metal AllreduceV result size overflow.");
  std::int64_t size = group.Rank() == kRoot ? static_cast<std::int64_t>((*data)->SizeBytes()) : 0;
  rc = Broadcast(ctx, group, linalg::MakeVec(&size, 1), kRoot);
  if (!rc.OK()) return Fail("Federated Metal AllreduceV size broadcast failed.", std::move(rc));
  if (size < 0) return Fail("Invalid federated Metal AllreduceV result size.");
  if (group.Rank() != kRoot) (*data)->Resize(static_cast<std::size_t>(size));
  if (size) {
    auto result = bytes(**data);
    rc = Broadcast(ctx, group, linalg::MakeTensorView(DeviceOrd::MPS(), result, result.size()),
                   kRoot);
    if (!rc.OK())
      return Fail("Federated Metal AllreduceV payload broadcast failed.", std::move(rc));
  }
  return Success();
}

// Byte payload counterpart of gpu_impl::AllreduceV in allreduce_v.cuh. The
// callback must finish its GPU work before returning, resize/write `out`, and
// leave the input owners alive. TCP accesses the owned shared allocations
// directly. Payload lengths are host control metadata, as in the CUDA routine.
template <typename Fn>
Result AllreduceV(Context const* ctx, CommGroup const& group, Payload* data,
                  AllreduceVScratch* scratch, Fn&& redop) {
  CHECK(ctx);
  CHECK(data && *data && scratch && scratch->payload && scratch->next);
  CHECK(data->get() != scratch->payload.get() && data->get() != scratch->next.get());
  if (!group.IsDistributed()) return Success();
  auto const& comm = group.Ctx(ctx, DeviceOrd::MPS());
  if (comm.IsFederated()) {
    return AllreduceVFederated(ctx, group, data, scratch, std::forward<Fn>(redop));
  }
  auto backend = group.Backend(DeviceOrd::MPS());
  auto bytes = [](common::metal::VectorBuffer const& buffer) {
    return common::Span<std::int8_t>{static_cast<std::int8_t*>(buffer.Data()), buffer.SizeBytes()};
  };
  auto send = [&](std::int32_t peer, common::metal::VectorBuffer const& payload) {
    if (payload.SizeBytes() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
      return Fail("Metal AllreduceV payload size overflow.");
    auto size = static_cast<std::int64_t>(payload.SizeBytes());
    auto header =
        common::Span<std::int8_t const>{reinterpret_cast<std::int8_t const*>(&size), sizeof(size)};
    auto channel = comm.Chan(peer);
    auto rc = channel->SendAll(header) << [&] {
      return size ? channel->SendAll(bytes(payload)) : Success();
    };
    return std::move(rc) + channel->Block();
  };
  auto receive = [&](std::int32_t peer, common::metal::VectorBuffer* payload) {
    std::int64_t size = 0;
    auto header = common::Span<std::int8_t>{reinterpret_cast<std::int8_t*>(&size), sizeof(size)};
    auto channel = comm.Chan(peer);
    auto rc = channel->RecvAll(header);
    rc = std::move(rc) + channel->Block();
    if (!rc.OK()) return rc;
    if (size < 0) return Fail("Invalid Metal AllreduceV payload size.");
    payload->Resize(static_cast<std::size_t>(size));
    if (!size) return Success();
    rc = channel->RecvAll(bytes(*payload));
    return std::move(rc) + channel->Block();
  };

  auto rank = comm.Rank(), world = comm.World();
  // Preserve CUDA's binomial-tree callback order, including non-power-of-two
  // worker counts. Bounded sketch merge/prune is not associative.
  for (std::int32_t level = 0; (std::int64_t{1} << level) < world; ++level) {
    if (rank > 0 && binomial_tree::ParentLevel(rank) == level) {
      auto rc = send(binomial_tree::Parent(rank), **data);
      if (!rc.OK()) return Fail("Metal AllreduceV send to parent failed.", std::move(rc));
      break;
    }
    if (binomial_tree::HasChild(rank, level, world)) {
      auto rc = receive(binomial_tree::Child(rank, level), scratch->payload.get());
      if (!rc.OK()) return Fail("Metal AllreduceV receive from child failed.", std::move(rc));
      redop(**data, *scratch->payload, scratch->next.get());
      std::swap(*data, scratch->next);
    }
  }

  if ((*data)->SizeBytes() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    return Fail("Metal AllreduceV result size overflow.");
  std::int64_t size = rank == 0 ? static_cast<std::int64_t>((*data)->SizeBytes()) : 0;
  auto rc = cpu_impl::Broadcast(comm, {reinterpret_cast<std::int8_t*>(&size), sizeof(size)}, 0);
  if (!rc.OK()) return Fail("Metal AllreduceV size broadcast failed.", std::move(rc));
  if (size < 0) return Fail("Invalid Metal AllreduceV result size.");
  (*data)->Resize(static_cast<std::size_t>(size));
  if (!size) return Success();
  return backend->Broadcast(ctx, comm, bytes(**data), 0);
}

template <typename Fn>
Result AllreduceV(Context const* ctx, Payload* data, AllreduceVScratch* scratch, Fn&& redop) {
  return AllreduceV(ctx, *GlobalCommGroup(), data, scratch, std::forward<Fn>(redop));
}
}  // namespace xgboost::collective::metal_impl
#endif  // defined(XGBOOST_USE_METAL)
