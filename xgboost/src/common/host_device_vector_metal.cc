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
  // Const access can allocate/copy storage and change its authority. Keep those
  // transitions atomic for readers sharing initially host-resident metadata.
  // Returned views outlive the lock: callers must coordinate writes, resizing,
  // device changes and destruction with any outstanding readers.
  mutable std::mutex mutex_;
  DeviceOrd device_{DeviceOrd::CPU()};
  std::vector<T> data_h_;
  std::unique_ptr<common::metal::VectorBuffer> data_d_;
  GPUAccess access_{kNone};

  // Private helpers require mutex_ (or an unpublished object in a constructor).
  std::size_t SizeImpl() const {
    return access_ <= kRead ? data_h_.size() : data_d_ ? data_d_->SizeBytes() / sizeof(T) : 0;
  }

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
    if (access_ >= kRead) {
      access_ = access;
      return;
    }
    LazyResizeDevice(data_h_.size());
    data_d_->Import(data_h_.data(), 0, Bytes(data_h_.size()));
    access_ = access;
  }
  void LazySyncHost(GPUAccess access) {
    if (access_ <= access) return;
    if (access_ > kRead) {
      data_h_.resize(data_d_->SizeBytes() / sizeof(T));
      data_d_->Readback(data_h_.data(), Bytes(data_h_.size()));
    }
    access_ = access;
  }
  void CopyToDevice(T const* source) {
    LazyResizeDevice(SizeImpl());
    data_d_->Import(source, 0, data_d_->SizeBytes());
    access_ = kWrite;
  }
  void SetDeviceImpl(DeviceOrd device) {
    CHECK(device.IsCPU() || (device.IsMPS() && device.ordinal == 0));
    if (device == device_) return;
    if (device_.IsMPS()) LazySyncHost(kNone);
    device_ = device;
    if (device.IsMPS()) LazyResizeDevice(data_h_.size());
  }
  T* DevicePointerImpl(GPUAccess access) {
    if (device_.IsCPU()) return nullptr;
    LazySyncDevice(access);
    return static_cast<T*>(data_d_->Data());
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
    std::lock_guard<std::mutex> lock{other.mutex_};
    device_ = other.device_;
    data_h_ = std::move(other.data_h_);
    data_d_ = std::move(other.data_d_);
    access_ = other.access_;
    // Keep moved-from objects reusable with a coherent empty host state.
    other.data_h_.clear();
    other.access_ = kNone;
  }
  std::size_t Size() const {
    std::lock_guard<std::mutex> lock{mutex_};
    return SizeImpl();
  }
  DeviceOrd Device() const {
    std::lock_guard<std::mutex> lock{mutex_};
    return device_;
  }
  T* DevicePointer() {
    std::lock_guard<std::mutex> lock{mutex_};
    return DevicePointerImpl(kWrite);
  }
  T const* ConstDevicePointer() {
    std::lock_guard<std::mutex> lock{mutex_};
    return DevicePointerImpl(kRead);
  }
  common::Span<T> DeviceSpan() {
    std::lock_guard<std::mutex> lock{mutex_};
    auto ptr = DevicePointerImpl(kWrite);
    return {ptr, device_.IsMPS() ? SizeImpl() : 0};
  }
  common::Span<T const> ConstDeviceSpan() {
    std::lock_guard<std::mutex> lock{mutex_};
    auto ptr = DevicePointerImpl(kRead);
    return {ptr, device_.IsMPS() ? SizeImpl() : 0};
  }
  std::vector<T>& HostVector() {
    std::lock_guard<std::mutex> lock{mutex_};
    LazySyncHost(kNone);
    return data_h_;
  }
  std::vector<T> const& ConstHostVector() {
    std::lock_guard<std::mutex> lock{mutex_};
    LazySyncHost(kRead);
    return data_h_;
  }
  bool HostCanRead() const { return Access() <= kRead; }
  bool HostCanWrite() const { return Access() == kNone; }
  bool DeviceCanRead() const { return Access() >= kRead; }
  bool DeviceCanWrite() const { return Access() == kWrite; }
  GPUAccess Access() const {
    std::lock_guard<std::mutex> lock{mutex_};
    return access_;
  }
  void SetDevice(DeviceOrd device) {
    std::lock_guard<std::mutex> lock{mutex_};
    SetDeviceImpl(device);
  }
  void Fill(T value) {
    std::lock_guard<std::mutex> lock{mutex_};
    if (access_ == kNone) {
      std::fill(data_h_.begin(), data_h_.end(), value);
    } else {
      data_d_->Fill(Pattern(value), 0, data_d_->SizeBytes());
      access_ = kWrite;
    }
  }
  void Copy(HostDeviceVectorImpl* other) {
    if (this == other) return;
    std::scoped_lock lock{mutex_, other->mutex_};
    CHECK_EQ(SizeImpl(), other->SizeImpl());
    SetDeviceImpl(other->device_);
    if (access_ == kNone && other->access_ == kNone) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
    } else if (other->access_ == kNone) {
      CopyToDevice(other->data_h_.data());
    } else {
      LazyResizeDevice(SizeImpl());
      data_d_->Copy(*other->data_d_, 0, 0, data_d_->SizeBytes());
      access_ = kWrite;
    }
  }
  template <typename Initializer>
  void Copy(Initializer const& source) {
    std::lock_guard<std::mutex> lock{mutex_};
    CHECK_EQ(SizeImpl(), source.size());
    if (access_ == kNone)
      std::copy(source.begin(), source.end(), data_h_.begin());
    else
      CopyToDevice(std::data(source));
  }
  void Resize(std::size_t size) {
    std::lock_guard<std::mutex> lock{mutex_};
    ResizeImpl(size, nullptr);
  }
  void Resize(std::size_t size, T value) {
    std::lock_guard<std::mutex> lock{mutex_};
    ResizeImpl(size, &value);
  }
  void Extend(HostDeviceVectorImpl* other) {
    if (this == other) {
      // Self-extension duplicates the original contents and only locks once.
      std::lock_guard<std::mutex> lock{mutex_};
      ExtendImpl(other);
    } else {
      std::scoped_lock lock{mutex_, other->mutex_};
      ExtendImpl(other);
    }
  }

 private:
  void ResizeImpl(std::size_t size, T const* value) {
    auto previous = SizeImpl();
    if (size == previous) return;
    if (device_.IsMPS() && (previous == 0 || access_ == kWrite)) {
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
  void ExtendImpl(HostDeviceVectorImpl* other) {
    auto original = SizeImpl(), extra = other->SizeImpl();
    CHECK_LE(extra, std::numeric_limits<std::size_t>::max() - original);
    if (!extra) return;
    // Mixed host/native iterator batches remain supported. A device-only source
    // supplies the destination device; a host source can be imported directly.
    if (device_.IsCPU() && other->access_ > kRead) SetDeviceImpl(other->device_);
    T value{};
    ResizeImpl(original + extra, &value);
    if (access_ == kNone && other->access_ <= kRead) {
      other->LazySyncHost(kNone);  // Match CUDA's access transition.
      std::copy_n(other->data_h_.begin(), extra, data_h_.begin() + original);
    } else if (other->access_ == kNone) {
      DevicePointerImpl(kWrite);
      data_d_->Import(other->data_h_.data(), Bytes(original), Bytes(extra));
    } else {
      CHECK_EQ(device_, other->device_);
      other->DevicePointerImpl(kRead);
      DevicePointerImpl(kWrite);
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
