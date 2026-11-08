/** Copyright 2017-2026, XGBoost Contributors
 * CUDA-derived access transitions with synchronous Metal buffer operations.
 */
#if defined(XGBOOST_USE_METAL)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "metal_vector.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"

namespace xgboost {
template <typename T>
class HostDeviceVectorImpl {
  static_assert(std::is_trivially_copyable_v<T>);
  DeviceOrd device_{DeviceOrd::CPU()};
  std::vector<T> data_h_;
  std::unique_ptr<common::metal::VectorBuffer> data_d_;
  GPUAccess access_{kNone};
  // Lazy host/device transitions mutate shared state even on const access, and
  // a DMatrix's metadata is read by every booster training on it. Serialize the
  // transitions so concurrent boosters cannot replace a buffer another is using.
  mutable std::recursive_mutex mutex_;

  static std::size_t Bytes(std::size_t size) {
    CHECK_LE(size, std::numeric_limits<std::size_t>::max() / sizeof(T));
    return size * sizeof(T);
  }
  static auto Pattern(T const& value) {
    return common::Span<std::byte const>{reinterpret_cast<std::byte const*>(&value), sizeof(T)};
  }
  void LazyResizeDevice(std::size_t size) {
    CHECK(device_.IsMPS() && device_.ordinal == 0) << "Expected Metal storage on device 0.";
    if (!data_d_) data_d_ = common::metal::MakeVectorBuffer();
    data_d_->Resize(Bytes(size));
  }
  void LazySyncDevice(GPUAccess access) {
    if (access_ >= access) return;
    if (DeviceCanRead()) {
      access_ = access;
      return;
    }
    LazyResizeDevice(data_h_.size());
    data_d_->Import(data_h_.data(), 0, Bytes(data_h_.size()));
    access_ = access;
  }
  void LazySyncHost(GPUAccess access) {
    if (access_ <= access) return;
    if (!HostCanRead()) {
      data_h_.resize(data_d_->SizeBytes() / sizeof(T));
      data_d_->Readback(data_h_.data(), Bytes(data_h_.size()));
    }
    access_ = access;
  }
  void CopyToDevice(T const* source) {
    LazyResizeDevice(Size());
    data_d_->Import(source, 0, data_d_->SizeBytes());
    access_ = kWrite;
  }

 public:
  HostDeviceVectorImpl(std::size_t size, T value, DeviceOrd device) : device_{device} {
    CHECK(device.IsCPU() || (device.IsMPS() && device.ordinal == 0));
    if (device.IsMPS()) {
      LazyResizeDevice(size);
      data_d_->Fill(Pattern(value), 0, Bytes(size));
      access_ = kWrite;
    } else {
      data_h_.resize(size, value);
    }
  }
  template <typename Initializer>
  HostDeviceVectorImpl(Initializer const& init, DeviceOrd device) : device_{device} {
    CHECK(device.IsCPU() || (device.IsMPS() && device.ordinal == 0));
    if (device.IsMPS()) {
      LazyResizeDevice(init.size());
      data_d_->Import(std::data(init), 0, Bytes(init.size()));
      access_ = kWrite;
    } else {
      data_h_ = init;
    }
  }
  HostDeviceVectorImpl(HostDeviceVectorImpl&& other) noexcept {
    std::lock_guard<std::recursive_mutex> guard{other.mutex_};
    device_ = other.device_;
    data_h_ = std::move(other.data_h_);
    data_d_ = std::move(other.data_d_);
    access_ = other.access_;
    // Keep moved-from objects reusable with a coherent empty host state.
    other.data_h_.clear();
    other.access_ = kNone;
  }
  std::size_t Size() const {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    return HostCanRead() ? data_h_.size() : data_d_ ? data_d_->SizeBytes() / sizeof(T) : 0;
  }
  DeviceOrd Device() const {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    return device_;
  }
  T* DevicePointer() {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    if (device_.IsCPU()) return nullptr;
    LazySyncDevice(kWrite);
    return static_cast<T*>(data_d_->Data());
  }
  T const* ConstDevicePointer() {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    if (device_.IsCPU()) return nullptr;
    LazySyncDevice(kRead);
    return static_cast<T const*>(data_d_->Data());
  }
  common::Span<T> DeviceSpan() {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    auto ptr = DevicePointer();
    return {ptr, device_.IsMPS() ? Size() : 0};
  }
  common::Span<T const> ConstDeviceSpan() {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    auto ptr = ConstDevicePointer();
    return {ptr, device_.IsMPS() ? Size() : 0};
  }
  std::vector<T>& HostVector() {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    LazySyncHost(kNone);
    return data_h_;
  }
  std::vector<T> const& ConstHostVector() {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    LazySyncHost(kRead);
    return data_h_;
  }
  bool HostCanRead() const {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    return access_ <= kRead;
  }
  bool HostCanWrite() const {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    return access_ == kNone;
  }
  bool DeviceCanRead() const {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    return access_ >= kRead;
  }
  bool DeviceCanWrite() const {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    return access_ == kWrite;
  }
  GPUAccess Access() const {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    return access_;
  }
  void SetDevice(DeviceOrd device) {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    CHECK(device.IsCPU() || (device.IsMPS() && device.ordinal == 0));
    if (device == device_) return;
    if (device_.IsMPS()) LazySyncHost(kNone);
    device_ = device;
    if (device.IsMPS()) LazyResizeDevice(data_h_.size());
  }
  void Fill(T value) {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), value);
    } else {
      data_d_->Fill(Pattern(value), 0, data_d_->SizeBytes());
      access_ = kWrite;
    }
  }
  void Copy(HostDeviceVectorImpl* other) {
    if (this == other) return;
    std::scoped_lock guard{mutex_, other->mutex_};
    CHECK_EQ(Size(), other->Size());
    SetDevice(other->device_);
    if (HostCanWrite() && other->HostCanWrite()) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
    } else if (other->HostCanWrite()) {
      CopyToDevice(other->data_h_.data());
    } else {
      LazyResizeDevice(Size());
      data_d_->Copy(*other->data_d_, 0, 0, data_d_->SizeBytes());
      access_ = kWrite;
    }
  }
  template <typename Initializer>
  void Copy(Initializer const& source) {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    CHECK_EQ(Size(), source.size());
    if (HostCanWrite())
      std::copy(source.begin(), source.end(), data_h_.begin());
    else
      CopyToDevice(std::data(source));
  }
  void Resize(std::size_t size) { ResizeImpl(size, nullptr); }
  void Resize(std::size_t size, T value) { ResizeImpl(size, &value); }
  void ResizeImpl(std::size_t size, T const* value) {
    std::lock_guard<std::recursive_mutex> guard{mutex_};
    auto previous = Size();
    if (size == previous) return;
    if (device_.IsMPS() && (previous == 0 || DeviceCanWrite())) {
      LazyResizeDevice(size);
      if (size > previous && value)
        data_d_->Fill(Pattern(*value), Bytes(previous), Bytes(size - previous));
      access_ = kWrite;
    } else {
      LazySyncHost(kNone);
      if (value)
        data_h_.resize(size, *value);
      else
        data_h_.resize(size);
    }
  }
  void Extend(HostDeviceVectorImpl* other) {
    CHECK_NE(this, other) << "Cannot extend a vector with itself.";
    std::scoped_lock guard{mutex_, other->mutex_};
    auto original = Size(), extra = other->Size();
    CHECK_LE(extra, std::numeric_limits<std::size_t>::max() - original);
    if (!extra) return;
    // Mixed host/native iterator batches remain supported. A device-only source
    // supplies the destination device; a host source can be imported directly.
    if (device_.IsCPU() && !other->HostCanRead()) SetDevice(other->Device());
    Resize(original + extra, T{});
    if (HostCanWrite() && other->HostCanRead()) {
      auto& source = other->HostVector();  // Match CUDA's access transition.
      std::copy_n(source.begin(), extra, data_h_.begin() + original);
    } else if (other->HostCanWrite()) {
      DevicePointer();
      data_d_->Import(other->data_h_.data(), Bytes(original), Bytes(extra));
    } else {
      CHECK_EQ(Device(), other->Device());
      other->ConstDevicePointer();
      DevicePointer();
      data_d_->Copy(*other->data_d_, 0, Bytes(original), Bytes(extra));
    }
  }
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(size, v, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(HostDeviceVector<T>&& other)
    : impl_(new HostDeviceVectorImpl<T>(std::move(*other.impl_))) {}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=(HostDeviceVector<T>&& other) {
  if (this == &other) {
    return *this;
  }

  std::unique_ptr<HostDeviceVectorImpl<T>> new_impl(
      new HostDeviceVectorImpl<T>(std::move(*other.impl_)));
  delete impl_;
  impl_ = new_impl.release();
  return *this;
}

template <typename T>
HostDeviceVector<T>::~HostDeviceVector() {
  delete impl_;
  impl_ = nullptr;
}

template <typename T>
size_t HostDeviceVector<T>::Size() const {
  return impl_->Size();
}

template <typename T>
DeviceOrd HostDeviceVector<T>::Device() const {
  return impl_->Device();
}

template <typename T>
T* HostDeviceVector<T>::DevicePointer() {
  return impl_->DevicePointer();
}

template <typename T>
const T* HostDeviceVector<T>::ConstDevicePointer() const {
  return impl_->ConstDevicePointer();
}

template <typename T>
common::Span<T> HostDeviceVector<T>::DeviceSpan() {
  return impl_->DeviceSpan();
}

template <typename T>
common::Span<const T> HostDeviceVector<T>::ConstDeviceSpan() const {
  return impl_->ConstDeviceSpan();
}

template <typename T>
void HostDeviceVector<T>::Fill(T v) {
  impl_->Fill(v);
}

template <typename T>
void HostDeviceVector<T>::Copy(const HostDeviceVector<T>& other) {
  impl_->Copy(other.impl_);
}

template <typename T>
void HostDeviceVector<T>::Copy(const std::vector<T>& other) {
  impl_->Copy(other);
}

template <typename T>
void HostDeviceVector<T>::Copy(std::initializer_list<T> other) {
  impl_->Copy(other);
}

template <typename T>
void HostDeviceVector<T>::Extend(HostDeviceVector const& other) {
  impl_->Extend(other.impl_);
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::HostVector() {
  return impl_->HostVector();
}

template <typename T>
const std::vector<T>& HostDeviceVector<T>::ConstHostVector() const {
  return impl_->ConstHostVector();
}

template <typename T>
bool HostDeviceVector<T>::HostCanRead() const {
  return impl_->HostCanRead();
}

template <typename T>
bool HostDeviceVector<T>::HostCanWrite() const {
  return impl_->HostCanWrite();
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanRead() const {
  return impl_->DeviceCanRead();
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanWrite() const {
  return impl_->DeviceCanWrite();
}

template <typename T>
GPUAccess HostDeviceVector<T>::DeviceAccess() const {
  return impl_->Access();
}

template <typename T>
void HostDeviceVector<T>::SetDevice(DeviceOrd device) const {
  impl_->SetDevice(device);
}

template <typename T>
void HostDeviceVector<T>::Resize(std::size_t new_size) {
  impl_->Resize(new_size);
}

template <typename T>
void HostDeviceVector<T>::Resize(std::size_t new_size, T v) {
  impl_->Resize(new_size, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<double>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<GradientPairPrecise>;
template class HostDeviceVector<GradientPairInt64>;
template class HostDeviceVector<std::int32_t>;  // bst_node_t
template class HostDeviceVector<std::uint8_t>;
template class HostDeviceVector<std::int8_t>;
template class HostDeviceVector<FeatureType>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<bst_idx_t>;
template class HostDeviceVector<std::uint32_t>;  // bst_feature_t
template class HostDeviceVector<RegTree::Node>;
template class HostDeviceVector<RegTree::CategoricalSplitMatrix::Segment>;
template class HostDeviceVector<RTreeNodeStat>;

#if defined(__APPLE__)
/*
 * On OSX:
 *
 * typedef unsigned int         uint32_t;
 * typedef unsigned long long   uint64_t;
 * typedef unsigned long       __darwin_size_t;
 */
template class HostDeviceVector<std::size_t>;
#endif  // defined(__APPLE__)
}  // namespace xgboost

#endif  // XGBOOST_USE_METAL
