/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include "xgboost/host_device_vector.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Select storage explicitly while the execution context's generic consumers
// still default to CPU. Const input access preserves a valid device copy.
template <typename T>
auto ConstDeviceSpan(HostDeviceVector<T> const& values) {
  values.SetDevice(DeviceOrd::MPS());
  return values.ConstDeviceSpan();
}
template <typename T>
auto DeviceSpan(HostDeviceVector<T>& values) {
  values.SetDevice(DeviceOrd::MPS());
  return values.DeviceSpan();
}
// Owned shared Metal allocation. Data() is an addressable span, not an MTLBuffer
// handle. All operations complete before returning; foreign writers must finish
// before passing the allocation back. Resize without a pattern is uninitialized.
class VectorBuffer {
 public:
  virtual ~VectorBuffer() = default;
  virtual void* Data() const = 0;
  virtual std::size_t SizeBytes() const = 0;
  virtual void Resize(std::size_t bytes) = 0;
  virtual void Fill(Span<std::byte const> pattern, std::size_t offset, std::size_t bytes) = 0;
  virtual void Copy(VectorBuffer const& source, std::size_t source_offset, std::size_t offset,
                    std::size_t bytes) = 0;
  virtual void Import(void const* source, std::size_t offset, std::size_t bytes) = 0;
  virtual void Readback(void* destination, std::size_t bytes) const = 0;
};
std::unique_ptr<VectorBuffer> MakeVectorBuffer();
// Host-addressable spans may be owned Metal allocations or guarded host memory.
bool EqualBytes(Span<std::byte const> lhs, Span<std::byte const> rhs);
void GatherRows(Span<float const> input, std::size_t rows, std::size_t columns,
                Span<std::uint64_t const> indices, Span<float> output);
}  // namespace xgboost::common::metal
