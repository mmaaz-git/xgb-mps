// Copyright 2026, XGBoost Contributors
#include <metal_stdlib>
using namespace metal;

struct RowEntry {
  ulong row;
  uint node;
  uint padding;
};

struct HistogramParams {
  ulong n_rows;
  ulong base_rowid;
  uint n_bins;
  uint n_features;
  uint bin_bytes;
  uint dense;
};

// An exact modular 64-bit sum using universally available 32-bit atomics.
// The two words are only read after the command buffer completes. Each low-word
// wrap contributes exactly one carry, including for negative two's-complement
// addends. No 64-bit atomic support or floating-point atomics are required.
inline void add_i64(device atomic_uint* dst, long value) {
  ulong bits = as_type<ulong>(value);
  uint low = uint(bits);
  uint previous = atomic_fetch_add_explicit(dst, low, memory_order_relaxed);
  uint carry = uint(previous > (0xffffffffu - low));
  atomic_fetch_add_explicit(dst + 1, uint(bits >> 32) + carry, memory_order_relaxed);
}

inline void add_i64(threadgroup atomic_uint* dst, long value) {
  ulong bits = as_type<ulong>(value);
  uint low = uint(bits);
  uint previous = atomic_fetch_add_explicit(dst, low, memory_order_relaxed);
  uint carry = uint(previous > (0xffffffffu - low));
  atomic_fetch_add_explicit(dst + 1, uint(bits >> 32) + carry, memory_order_relaxed);
}

kernel void xgb_histogram(
    device const uchar* index [[buffer(0)]], device const ulong* row_ptr [[buffer(1)]],
    device const uint* offsets [[buffer(2)]], device const long2* gradients [[buffer(3)]],
    device const RowEntry* rows [[buffer(4)]], device atomic_uint* hist [[buffer(5)]],
    constant HistogramParams& p [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.n_rows) return;
  RowEntry item = rows[tid];
  ulong local_row = item.row - p.base_rowid;
  long2 g = gradients[item.row];
  ulong begin = row_ptr[local_row];
  ulong end = row_ptr[local_row + 1];
  for (ulong i = begin; i < end; ++i) {
    uint bin;
    if (p.bin_bytes == 1) {
      bin = uint(index[i]);
    } else if (p.bin_bytes == 2) {
      bin = uint(reinterpret_cast<device const ushort*>(index)[i]);
    } else {
      bin = reinterpret_cast<device const uint*>(index)[i];
    }
    if (p.dense) bin += offsets[i - begin];
    ulong dst = (ulong(item.node) * p.n_bins + bin) * 4;
    add_i64(hist + dst, g.x);
    add_i64(hist + dst + 2, g.y);
  }
}

// Privatize one feature histogram per threadgroup. Eight rows per lane amortize
// clearing and merging the shared histogram. Each group handles exactly one
// node, so threads never contend across nodes or features in shared memory.
kernel void xgb_histogram_dense_shared(
    device const uchar* index [[buffer(0)]], device const ulong* row_ptr [[buffer(1)]],
    device const uint* offsets [[buffer(2)]], device const long2* gradients [[buffer(3)]],
    device const RowEntry* rows [[buffer(4)]], device atomic_uint* hist [[buffer(5)]],
    constant HistogramParams& p [[buffer(6)]], device const ulong* node_ptr [[buffer(7)]],
    threadgroup atomic_uint* local_hist [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
  uint feature = group.y;
  uint node = group.z;
  ulong begin = node_ptr[node] + ulong(group.x) * threads.x * 8;
  ulong end = min(node_ptr[node + 1], begin + threads.x * 8);
  if (begin >= end) return;  // Uniform across the threadgroup, before any barrier.
  uint feature_offset = offsets[feature];
  uint bins = offsets[feature + 1] - feature_offset;
  for (uint i = tid; i < bins * 4; i += threads.x)
    atomic_store_explicit(local_hist + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong r = begin + tid; r < end; r += threads.x) {
    ulong row = rows[r].row;
    ulong i = row_ptr[row - p.base_rowid] + feature;
    uint bin;
    if (p.bin_bytes == 1)
      bin = uint(index[i]);
    else if (p.bin_bytes == 2)
      bin = uint(reinterpret_cast<device const ushort*>(index)[i]);
    else
      bin = reinterpret_cast<device const uint*>(index)[i];
    long2 g = gradients[row];
    add_i64(local_hist + bin * 4, g.x);
    add_i64(local_hist + bin * 4 + 2, g.y);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint bin = tid; bin < bins; bin += threads.x) {
    ulong dst = (ulong(node) * p.n_bins + feature_offset + bin) * 4;
    for (uint component = 0; component < 2; ++component) {
      uint word = bin * 4 + component * 2;
      ulong low = atomic_load_explicit(local_hist + word, memory_order_relaxed);
      ulong high = atomic_load_explicit(local_hist + word + 1, memory_order_relaxed);
      long sum = as_type<long>(low | (high << 32));
      add_i64(hist + dst + component * 2, sum);
    }
  }
}
