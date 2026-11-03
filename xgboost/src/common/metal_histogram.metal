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

kernel void xgb_histogram(
    device const uchar* index [[buffer(0)]],
    device const ulong* row_ptr [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const long2* gradients [[buffer(3)]],
    device const RowEntry* rows [[buffer(4)]],
    device atomic_uint* hist [[buffer(5)]],
    constant HistogramParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
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

