/** Copyright 2026, XGBoost Contributors
 * Objective-C++ helpers for synchronous views over caller-owned host memory.
 */
#pragma once
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace xgboost::common::metal {
struct BorrowedMemory {
  id<MTLBuffer> buffer{nil};
  std::size_t offset{0};
};
// Retain the allocation until completion AND release of all buffer/command
// references. Use an autorelease pool inside the allocation owner's lifetime.
// Avoid concurrent mutation. Only synchronous operations may use this helper;
// never store these views in caches.
// Apple requires the page-aligned region to fit within a single VM allocation:
// https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)
inline BorrowedMemory TryBorrowMemory(id<MTLDevice> device, void const* data, std::size_t bytes,
                                      std::size_t alignment) {
  if (!device.hasUnifiedMemory || !data || !bytes || !alignment) return {};
  auto address = reinterpret_cast<std::uintptr_t>(data);
  if (address % alignment) return {};
  auto page = static_cast<std::size_t>(getpagesize());
  auto start = address - address % page;
  auto offset = address - start;
  if (bytes > std::numeric_limits<std::size_t>::max() - offset - (page - 1)) return {};
  auto length = ((offset + bytes + page - 1) / page) * page;
  if (length > device.maxBufferLength) return {};

  mach_vm_address_t region = start;
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  auto status = mach_vm_region(mach_task_self(), &region, &region_size, VM_REGION_BASIC_INFO_64,
                               reinterpret_cast<vm_region_info_t>(&info), &count, &object);
  if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
  if (status != KERN_SUCCESS || region > start || length > region_size ||
      start - region > region_size - length ||
      (info.protection & (VM_PROT_READ | VM_PROT_WRITE)) != (VM_PROT_READ | VM_PROT_WRITE))
    return {};
  auto buffer = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(start)
                                          length:length
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
  return {buffer, buffer ? offset : 0};
}
}  // namespace xgboost::common::metal
