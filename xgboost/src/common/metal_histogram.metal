// Copyright 2026, XGBoost Contributors
#include <metal_stdlib>
using namespace metal;

struct HistogramParams {
  ulong n_rows;
  ulong base_rowid;
  ulong page_rows;
  uint n_bins;
  uint n_features;
  uint bin_bytes;
  uint dense;
  uint n_nodes;
  uint padding;
  uint rows_per_lane;
};

// An exact modular 64-bit sum using universally available 32-bit atomics.
// The two words are only read after the command buffer completes. Each low-word
// wrap contributes exactly one carry, including for negative two's-complement
// addends. No 64-bit atomic support or floating-point atomics are required.
inline void add_i64(device atomic_uint* dst, long value) {
  ulong bits = as_type<ulong>(value);
  uint low = uint(bits);
  uint carry = 0;
  if (low) {
    uint previous = atomic_fetch_add_explicit(dst, low, memory_order_relaxed);
    carry = uint(previous > 0xffffffffu - low);
  }
  uint high = uint(bits >> 32) + carry;
  if (high) atomic_fetch_add_explicit(dst + 1, high, memory_order_relaxed);
}

inline void add_i64(threadgroup atomic_uint* dst, long value) {
  ulong bits = as_type<ulong>(value);
  uint low = uint(bits);
  uint carry = 0;
  if (low) {
    uint previous = atomic_fetch_add_explicit(dst, low, memory_order_relaxed);
    carry = uint(previous > 0xffffffffu - low);
  }
  uint high = uint(bits >> 32) + carry;
  if (high) atomic_fetch_add_explicit(dst + 1, high, memory_order_relaxed);
}

kernel void xgb_histogram(
    device const uchar* index [[buffer(0)]], device const ulong* row_ptr [[buffer(1)]],
    device const uint* offsets [[buffer(2)]], device const long2* gradients [[buffer(3)]],
    device const ulong* rows [[buffer(4)]], device atomic_uint* hist [[buffer(5)]],
    constant HistogramParams& p [[buffer(6)]], device const ulong* node_ptr [[buffer(7)]],
    device const ulong* row_begin [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.n_rows) return;
  // Upper bound handles repeated offsets from empty nodes.
  uint lo = 0, hi = p.n_nodes;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (node_ptr[mid + 1] <= tid)
      lo = mid + 1;
    else
      hi = mid;
  }
  uint node = lo;
  ulong row = rows[row_begin[node] + ulong(tid) - node_ptr[node]];
  ulong local_row = row - p.base_rowid;
  long2 g = gradients[row];
  if (g.x == 0 && g.y == 0) return;
  ulong begin = row_ptr[local_row];
  ulong end = row_ptr[local_row + 1];
  for (ulong i = begin; i < end; ++i) {
    ulong pos = p.padding ? (i - begin) * p.page_rows + local_row : i;
    uint bin;
    if (p.bin_bytes == 1) {
      bin = uint(index[pos]);
    } else if (p.bin_bytes == 2) {
      bin = uint(reinterpret_cast<device const ushort*>(index)[pos]);
    } else {
      bin = reinterpret_cast<device const uint*>(index)[pos];
    }
    if (p.dense) bin += offsets[i - begin];
    ulong dst = (ulong(node) * p.n_bins + bin) * 4;
    add_i64(hist + dst, g.x);
    add_i64(hist + dst + 2, g.y);
  }
}

// Privatize one feature histogram per threadgroup. Multiple rows per lane amortize
// clearing and merging the shared histogram. Each group handles exactly one
// node, so threads never contend across nodes or features in shared memory.
kernel void xgb_histogram_dense_shared(
    device const uchar* index [[buffer(0)]], device const ulong* row_ptr [[buffer(1)]],
    device const uint* offsets [[buffer(2)]], device const long2* gradients [[buffer(3)]],
    device const ulong* rows [[buffer(4)]], device atomic_uint* hist [[buffer(5)]],
    constant HistogramParams& p [[buffer(6)]], device const ulong* node_ptr [[buffer(7)]],
    device const ulong* row_begin [[buffer(8)]], device const ulong* block_ptr [[buffer(9)]],
    threadgroup atomic_uint* local_hist [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_width [[threads_per_simdgroup]]) {
  uint feature = group.y;
  // Only actual node blocks are scheduled. Empty and short nodes do not inherit
  // the largest node's grid, which wastes groups as tree branches become uneven.
  uint lo = 0, hi = p.n_nodes;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (block_ptr[mid + 1] <= group.x)
      lo = mid + 1;
    else
      hi = mid;
  }
  uint node = lo;
  ulong begin = node_ptr[node] + (ulong(group.x) - block_ptr[node]) * threads.x * p.rows_per_lane;
  ulong end = min(node_ptr[node + 1], begin + threads.x * p.rows_per_lane);
  if (begin >= end) return;  // Uniform across the threadgroup, before any barrier.
  uint feature_offset = offsets[feature];
  uint bins = offsets[feature + 1] - feature_offset;
  if (bins <= 2) {
    // Accumulate in registers before the SIMD reduction. Every lane reaches
    // the collective, including lanes with no rows in the partial last block.
    ulong2 first = 0, second = 0;
    for (ulong r = begin + tid; r < end; r += threads.x) {
      ulong row = rows[row_begin[node] + r - node_ptr[node]];
      ulong2 g = as_type<ulong2>(gradients[row]);
      ulong i = p.padding ? ulong(feature) * p.page_rows + (row - p.base_rowid)
                          : row_ptr[row - p.base_rowid] + feature;
      uint bin;
      if (p.bin_bytes == 1) bin = uint(index[i]);
      else if (p.bin_bytes == 2) bin = uint(reinterpret_cast<device const ushort*>(index)[i]);
      else bin = reinterpret_cast<device const uint*>(index)[i];
      if (bin == 0) first += g;
      else second += g;
    }
    for (uint delta = simd_width / 2; delta != 0; delta >>= 1) {
      first += as_type<ulong2>(simd_shuffle_down(as_type<uint4>(first), delta));
      second += as_type<ulong2>(simd_shuffle_down(as_type<uint4>(second), delta));
    }
    if (simd_lane == 0) {
      ulong dst = (ulong(node) * p.n_bins + feature_offset) * 4;
      add_i64(hist + dst, as_type<long>(first.x));
      add_i64(hist + dst + 2, as_type<long>(first.y));
      if (bins == 2) {
        add_i64(hist + dst + 4, as_type<long>(second.x));
        add_i64(hist + dst + 6, as_type<long>(second.y));
      }
    }
    return;
  }
  for (uint i = tid; i < bins * 4; i += threads.x)
    atomic_store_explicit(local_hist + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong r = begin + tid; r < end; r += threads.x) {
    ulong row = rows[row_begin[node] + r - node_ptr[node]];
    long2 g = gradients[row];
    if (g.x == 0 && g.y == 0) continue;
    ulong i = p.padding ? ulong(feature) * p.page_rows + (row - p.base_rowid)
                        : row_ptr[row - p.base_rowid] + feature;
    uint bin;
    if (p.bin_bytes == 1)
      bin = uint(index[i]);
    else if (p.bin_bytes == 2)
      bin = uint(reinterpret_cast<device const ushort*>(index)[i]);
    else
      bin = reinterpret_cast<device const uint*>(index)[i];
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

// Exact round-to-nearest, ties away from zero, for float * 2^exponent.
// Integer bit decoding avoids Metal's lack of double precision and preserves
// subnormal inputs. Host bounds guarantee the result fits within 61 bits.
inline long quantize_float(uint bits, int exponent) {
  uint encoded = (bits >> 23) & 255;
  ulong mantissa = bits & 0x7fffff;
  if (encoded) mantissa |= 0x800000;
  int shift = exponent + (encoded ? int(encoded) - 150 : -149);
  ulong magnitude = 0;
  if (mantissa != 0) {
    if (shift >= 0)
      magnitude = mantissa << shift;
    else if (shift > -64) {
      uint right = uint(-shift);
      magnitude = (mantissa >> right) + ((mantissa >> (right - 1)) & 1);
    }
  }
  long value = long(magnitude);
  return (bits >> 31) ? -value : value;
}
struct GradientScale {
  int grad_exponent, hess_exponent;
  uint valid, padding;
};
// Absolute FP32 bit patterns sort in the same order as nonnegative magnitudes,
// including subnormals. Any NaN or infinity makes the final maximum invalid.
kernel void xgb_gradient_maximum(device const uint2* input [[buffer(0)]],
                                 device atomic_uint* maximum [[buffer(1)]],
                                 constant ulong& n [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd [[simdgroup_index_in_threadgroup]]) {
  uint2 value = ulong(tid) < n ? input[tid] & 0x7fffffffu : uint2(0);
  value = uint2(simd_max(value.x), simd_max(value.y));
  threadgroup uint2 maxima[8];
  if (lane == 0) maxima[simd] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    value = lane < 8 ? maxima[lane] : uint2(0);
    value = uint2(simd_max(value.x), simd_max(value.y));
    if (lane == 0) {
      atomic_fetch_max_explicit(maximum, value.x, memory_order_relaxed);
      atomic_fetch_max_explicit(maximum + 1, value.y, memory_order_relaxed);
    }
  }
}
inline int xgb_gradient_exponent(uint magnitude, ulong n) {
  if (!magnitude || !n) return 0;
  uint encoded = magnitude >> 23;
  ulong mantissa = (magnitude & 0x7fffffu) | (encoded ? 0x800000u : 0u);
  // At most 24 + 32 bits: compute the bound exactly instead of rounding a
  // floating product or logarithm close to a power of two.
  ulong product = mantissa * n;
  int ceiling = 63 - int(clz(product)) + int(bool(product & (product - 1)));
  return 60 - ceiling - (encoded ? int(encoded) - 150 : -149);
}
kernel void xgb_gradient_scale(device const uint2& maximum [[buffer(0)]],
                               device GradientScale& scale [[buffer(1)]],
                               constant ulong& n [[buffer(2)]]) {
  bool valid = all(maximum < uint2(0x7f800000u));
  scale = {valid ? xgb_gradient_exponent(maximum.x, n) : 0,
           valid ? xgb_gradient_exponent(maximum.y, n) : 0, uint(valid), 0};
}
kernel void xgb_quantize_gradients(device const uint2* input [[buffer(0)]],
                                   device long2* output [[buffer(1)]],
                                   constant ulong& n [[buffer(2)]],
                                   device const GradientScale& scale [[buffer(3)]],
                                   uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= n) return;
  uint2 value = input[tid];
  output[tid] = scale.valid ? long2(quantize_float(value.x, scale.grad_exponent),
                                    quantize_float(value.y, scale.hess_exponent))
                            : long2(0);
}

struct ObjectiveParams {
  ulong n;
  uint targets, kind, weighted, label_targets;
  float parameter, scale_pos_weight;
};
// Metal has no log1p intrinsic. Correct the rounding of 1+x for small x.
inline float xgb_log1p(float x) {
  float u = 1.0f + x;
  if (u == 1.0f) return x;
  if (u == 0.0f || isinf(u)) return log(u);
  return log(u) * (x / (u - 1.0f));
}
inline float xgb_sigmoid(float x) { return 1.0f / (exp(min(-x, 88.7f)) + 1.0f + 1e-16f); }
inline float xgb_softplus(float x) { return x > 0.0f ? x + xgb_log1p(exp(-x)) : xgb_log1p(exp(x)); }

kernel void xgb_objective(device const float* predictions [[buffer(0)]],
                          device const float* labels [[buffer(1)]],
                          device const float* weights [[buffer(2)]],
                          device const float* alpha [[buffer(3)]],
                          device float2* output [[buffer(4)]],
                          constant ObjectiveParams& p [[buffer(5)]],
                          uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.n) return;
  ulong row = tid / p.targets;
  uint target = tid % p.targets;
  float raw = predictions[tid];
  float y = labels[row * p.label_targets + (p.label_targets == 1 ? 0 : target)];
  float w = p.weighted ? weights[row] : 1.0f;
  if (y == 1.0f) w *= p.scale_pos_weight;
  float g = 0.0f, h = 0.0f;
  switch (p.kind) {
    case 0:
      g = raw - y;
      h = 1.0f;
      break;
    case 1: {
      float prob = xgb_sigmoid(raw);
      g = prob - y;
      h = max(prob * (1.0f - prob), 1e-16f);
      break;
    }
    case 2: {
      float pred = max(raw, -0.999999f);
      float diff = xgb_log1p(pred) - xgb_log1p(y);
      g = diff / (pred + 1.0f);
      h = max((1.0f - diff) / ((pred + 1.0f) * (pred + 1.0f)), 1e-6f);
      break;
    }
    case 3: {
      float ratio = y / exp(raw);
      g = 1.0f - ratio;
      h = ratio;
      break;
    }
    case 4: {
      float z = raw - y, slope2 = p.parameter * p.parameter;
      float root = sqrt(1.0f + z * z / slope2);
      g = z / root;
      h = slope2 / ((slope2 + z * z) * root);
      break;
    }
    case 5:
      g = exp(raw) - y;
      h = exp(raw + p.parameter);
      break;
    case 6: {
      float a = 1.0f - p.parameter, b = 2.0f - p.parameter;
      g = -y * exp(a * raw) + exp(b * raw);
      h = -y * a * exp(a * raw) + b * exp(b * raw);
      break;
    }
    case 7: {
      float sign = y * 2.0f - 1.0f;
      // The Hessian floor is NOT multiplied by the sample weight upstream.
      output[tid] = raw * sign < 1.0f ? float2(-sign * w, w) : float2(0.0f, FLT_MIN);
      return;
    }
    case 8: {
      float maximum = FLT_MIN;
      for (uint k = 0; k < p.targets; ++k) maximum = max(maximum, predictions[row * p.targets + k]);
      // Compensated FP32 summation mitigates the absence of native GPU FP64.
      float sum = 0.0f, correction = 0.0f;
      for (uint k = 0; k < p.targets; ++k) {
        float value = exp(predictions[row * p.targets + k] - maximum) - correction;
        float next = sum + value;
        correction = (next - sum) - value;
        sum = next;
      }
      float prob = exp(raw - maximum) / sum;
      output[tid] = float2((prob - (y == float(target) ? 1.0f : 0.0f)) * w,
                           max(2.0f * prob * (1.0f - prob) * w, 1e-16f));
      return;
    }
    case 9: {
      float pred = predictions[row * p.targets];
      for (uint k = 0; k < p.targets; ++k) {
        if (k) pred += 1e-6f + xgb_softplus(predictions[row * p.targets + k]);
        if (k >= target) {
          float diff = pred - y;
          float a = diff >= 0.0f ? 1.0f - alpha[k] : alpha[k];
          g += a * diff * w;
          h += a * w;
        }
      }
      float scale = target == 0 ? 1.0f : xgb_sigmoid(raw);
      output[tid] = float2(scale * g, scale * scale * h);
      return;
    }
  }
  output[tid] = float2(g * w, h * w);
}

struct TransformParams {
  ulong n;
  uint targets, kind;
};
kernel void xgb_transform(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant TransformParams& p [[buffer(2)]],
                          uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.n) return;
  if (p.kind == 1)
    output[tid] = xgb_sigmoid(input[tid]);
  else if (p.kind == 2)
    output[tid] = exp(input[tid]);
  else if (p.kind == 3)
    output[tid] = input[tid] > 0.0f ? 1.0f : 0.0f;
  else {
    ulong begin = ulong(tid) * p.targets;
    if (p.kind == 7) {
      for (uint k = 0; k < p.targets; ++k) output[begin + k] = input[begin + k];
      for (uint k = 1; k < p.targets; ++k) {
        float value = output[begin + k];
        uint j = k;
        while (j && output[begin + j - 1] > value) {
          output[begin + j] = output[begin + j - 1];
          --j;
        }
        output[begin + j] = value;
      }
    } else if (p.kind == 6) {
      float value = input[begin];
      output[begin] = value;
      for (uint k = 1; k < p.targets; ++k) {
        value += 1e-6f + xgb_softplus(input[begin + k]);
        output[begin + k] = value;
      }
    } else {
      float maximum = -FLT_MAX;
      uint index = 0;
      for (uint k = 0; k < p.targets; ++k)
        if (input[begin + k] > maximum) {
          maximum = input[begin + k];
          index = k;
        }
      if (p.kind == 5) {
        output[tid] = float(index);
        return;
      }
      float sum = 0.0f, correction = 0.0f;
      for (uint k = 0; k < p.targets; ++k) {
        float value = exp(input[begin + k] - maximum) - correction;
        float next = sum + value;
        correction = (next - sum) - value;
        sum = next;
      }
      for (uint k = 0; k < p.targets; ++k)
        output[begin + k] = exp(input[begin + k] - maximum) / sum;
    }
  }
}

struct PredictionNode {
  int left, right, missing;
  uint feature;
  float split;
  uint category_begin, category_size, leaf_begin, categorical, padding;
};
struct PredictionTree {
  uint root, targets, group;
  float weight;
};
struct PredictionParams {
  ulong rows;
  uint features, groups, trees, leaf;
};
kernel void xgb_predict(
    device const float* data [[buffer(0)]], device const PredictionNode* nodes [[buffer(1)]],
    device const PredictionTree* trees [[buffer(2)]], device const uint* categories [[buffer(3)]],
    device const float* leaves [[buffer(4)]], device float* output [[buffer(5)]],
    constant PredictionParams& p [[buffer(6)]], uint row [[thread_position_in_grid]]) {
  if (ulong(row) >= p.rows) return;
  for (uint t = 0; t < p.trees; ++t) {
    PredictionTree tree = trees[t];
    uint nid = tree.root;
    while (nodes[nid].left >= 0) {
      PredictionNode node = nodes[nid];
      float value = data[ulong(row) * p.features + node.feature];
      int next;
      if (isnan(value))
        next = node.missing;
      else {
        bool left = value < node.split;
        if (node.categorical) {
          left = true;
          if (value >= 0.0f && value < 16777216.0f) {
            uint category = uint(value), word = category / 32;
            if (word < node.category_size)
              left = (categories[node.category_begin + word] & (1u << (31 - category % 32))) == 0;
          }
        }
        next = left ? node.left : node.right;
      }
      nid = tree.root + uint(next);
    }
    if (p.leaf) {
      output[ulong(row) * p.groups + t] = float(nid - tree.root);
      continue;
    }
    uint leaf = nodes[nid].leaf_begin;
    for (uint k = 0; k < tree.targets; ++k)
      output[ulong(row) * p.groups + tree.group + k] += leaves[leaf + k] * tree.weight;
  }
}

struct LeafParams {
  ulong rows, stride;
  uint targets, padding;
};
kernel void xgb_add_leaf(device const int* ids [[buffer(0)]],
                         device const float* weights [[buffer(1)]],
                         device float* output [[buffer(2)]], constant LeafParams& p [[buffer(3)]],
                         uint row [[thread_position_in_grid]]) {
  if (ulong(row) >= p.rows) return;
  int encoded = ids[row];
  uint leaf = uint(encoded >= 0 ? encoded : ~encoded);
  for (uint k = 0; k < p.targets; ++k)
    output[ulong(row) * p.stride + k] += weights[ulong(leaf) * p.targets + k];
}

struct MetricParams {
  ulong n;
  uint targets, kind, weighted, padding;
  float parameter, reserved;
};
inline float xgb_log_gamma_positive(float x) {
  // Recurrence to the Stirling region; input is y+1 for nonnegative count labels.
  if (isnan(x)) return x;
  if (isinf(x)) return INFINITY;
  if (x <= 0.0f) return NAN;
  float correction = 0.0f;
  while (x < 8.0f) {
    correction -= log(x);
    x += 1.0f;
  }
  float inv = 1.0f / x, inv2 = inv * inv;
  float series =
      inv * (1.0f / 12.0f + inv2 * (-1.0f / 360.0f + inv2 * (1.0f / 1260.0f - inv2 / 1680.0f)));
  return (x - 0.5f) * log(x) - x + 0.9189385332046727f + series + correction;
}
inline float xgb_log_gamma(float x) {
  if (isnan(x)) return x;
  if (isinf(x) || x == 0.0f) return INFINITY;
  if (x < 0.0f) {
    if (floor(x) == x) return INFINITY;
    return log(M_PI_F) - log(abs(sinpi(x))) - xgb_log_gamma_positive(1.0f - x);
  }
  return xgb_log_gamma_positive(x);
}
kernel void xgb_metric(device const float* predictions [[buffer(0)]],
                       device const float* labels [[buffer(1)]],
                       device const float* weights [[buffer(2)]],
                       device float4* partial [[buffer(3)]], constant MetricParams& p [[buffer(4)]],
                       uint tid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint group [[threadgroup_position_in_grid]],
                       uint width [[threads_per_threadgroup]]) {
  threadgroup float2 scratch[256];
  threadgroup float2 magnitude;
  float2 result = 0.0f;
  if (ulong(tid) < p.n) {
    bool multi = p.kind >= 11;
    ulong row = multi ? tid : tid / p.targets;
    float y = labels[tid], pred = predictions[tid], weight = p.weighted ? weights[row] : 1.0f;
    float loss = 0.0f;
    switch (p.kind) {
      case 0:
        loss = (y - pred) * (y - pred);
        break;
      case 1: {
        float d = xgb_log1p(y) - xgb_log1p(pred);
        loss = d * d;
        break;
      }
      case 2:
        loss = abs(y - pred);
        break;
      case 3:
        loss = abs((y - pred) / y);
        break;
      case 4:
        loss = (y == 0.0f ? 0.0f : -y * log(max(pred, 1e-16f))) +
               (y == 1.0f ? 0.0f : -(1.0f - y) * log(max(1.0f - pred, 1e-16f)));
        break;
      case 5:
        loss = pred > p.parameter ? 1.0f - y : y;
        break;
      case 6:
        pred = max(pred, 1e-16f);
        loss = xgb_log_gamma(y + 1.0f) + pred - log(pred) * y;
        break;
      case 7:
        pred += 1e-6f;
        y += 1e-6f;
        loss = log(pred / y) + y / pred - 1.0f;
        break;
      case 8:
        pred = max(pred, 1e-6f);
        loss = y / pred + log(pred);
        break;
      case 9: {
        float a = 1.0f - p.parameter, b = 2.0f - p.parameter;
        loss = -y * exp(a * log(pred)) / a + exp(b * log(pred)) / b;
        break;
      }
      case 10: {
        float d = (y - pred) / p.parameter;
        loss = p.parameter * p.parameter * (sqrt(1.0f + d * d) - 1.0f);
        break;
      }
      case 11:
        loss = -log(max(predictions[row * p.targets + uint(y)], 1e-16f));
        break;
      case 12: {
        uint best = 0;
        for (uint k = 1; k < p.targets; ++k)
          if (predictions[row * p.targets + k] > predictions[row * p.targets + best]) best = k;
        loss = best == uint(y) ? 0.0f : 1.0f;
        break;
      }
    }
    result = float2(loss * weight, weight);
  }
  // Scale each block before summing: a float block sum could overflow even
  // when upstream's double-precision reduction is finite. The host restores
  // the scale in FP64 and reduces the small array of block results.
  scratch[lane] = abs(result);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = width / 2; step; step /= 2) {
    if (lane < step) scratch[lane] = fmax(scratch[lane], scratch[lane + step]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) magnitude = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0; k < 2; ++k) {
    if (isinf(magnitude[k]))
      result[k] = isinf(result[k]) || isnan(result[k]) ? result[k] : 0.0f;
    else if (magnitude[k] > 0.0f)
      result[k] /= magnitude[k];
  }
  scratch[lane] = result;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = width / 2; step; step /= 2) {
    if (lane < step) scratch[lane] += scratch[lane + step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) partial[group] = float4(scratch[0], magnitude);
}

struct RowInitParams {
  ulong rows, base;
};
kernel void xgb_row_iota(device ulong* rows [[buffer(0)]], constant RowInitParams& p [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) < p.rows) rows[tid] = p.base + tid;
}
struct PartitionSplit {
  ulong begin, end;
  uint feature;
  int split_bin;
  uint default_left, categorical, cat_begin, cat_size, block_begin, block_end;
};
struct PartitionBlock {
  ulong begin, end;
  uint node, padding;
};
struct PartitionParams {
  ulong rows, base;
  uint bin_bytes, dense, nodes, blocks;
};
inline uint xgb_read_bin(device const uchar* index, ulong pos, uint width) {
  if (width == 1) return index[pos];
  if (width == 2) return reinterpret_cast<device const ushort*>(index)[pos];
  return reinterpret_cast<device const uint*>(index)[pos];
}
kernel void xgb_partition_classify(
    device const uchar* index [[buffer(0)]], device const ulong* row_ptr [[buffer(1)]],
    device const uint* offsets [[buffer(2)]], device const float* cuts [[buffer(3)]],
    device const ulong* rows [[buffer(4)]], device const PartitionSplit* splits [[buffer(5)]],
    device const PartitionBlock* blocks [[buffer(6)]], device const uint* categories [[buffer(7)]],
    device uint* ranks [[buffer(8)]], device uint* counts [[buffer(9)]],
    device ulong* saved_rows [[buffer(10)]], constant PartitionParams& p [[buffer(11)]],
    uint block_id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  threadgroup uint sums[8];
  PartitionBlock block = blocks[block_id];
  PartitionSplit split = splits[block.node];
  ulong pos = block.begin + lane;
  bool valid = pos < block.end, left = false;
  if (valid) {
    ulong row = rows[pos], local = row - p.base;
    ulong begin = row_ptr[local], end = row_ptr[local + 1];
    uint bin = 0xffffffffu, first = offsets[split.feature], last = offsets[split.feature + 1];
    if (p.dense)
      bin = xgb_read_bin(index, ulong(split.feature) * p.rows + local, p.bin_bytes) + first;
    else {
      ulong lo = begin, hi = end;
      while (lo < hi) {
        ulong mid = lo + (hi - lo) / 2;
        if (xgb_read_bin(index, mid, p.bin_bytes) < first)
          lo = mid + 1;
        else
          hi = mid;
      }
      if (lo < end) {
        uint candidate = xgb_read_bin(index, lo, p.bin_bytes);
        if (candidate < last) bin = candidate;
      }
    }
    left = split.default_left;
    if (bin != 0xffffffffu) {
      if (split.categorical) {
        float value = cuts[bin];
        left = true;
        if (value >= 0.0f && value < 16777216.0f) {
          uint category = uint(value), word = category / 32;
          if (word < split.cat_size)
            left = (categories[split.cat_begin + word] & (1u << (31 - category % 32))) == 0;
        }
      } else
        left = long(bin) <= long(split.split_bin);
    }
    saved_rows[pos] = row;
  }
  uint count = simd_sum(uint(left)), rank = simd_prefix_inclusive_sum(uint(left));
  if (simd_lane == 0) sums[simd_group] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = 0; g < simd_group; ++g) rank += sums[g];
  if (valid) ranks[pos] = rank | (uint(left) << 31);
  if (lane == 255) counts[block_id] = rank;
}
kernel void xgb_partition_prefix_serial(device const PartitionSplit* splits [[buffer(0)]],
                                 device const uint* counts [[buffer(1)]],
                                 device uint* prefix [[buffer(2)]],
                                 device ulong* left_counts [[buffer(3)]],
                                 constant PartitionParams& p [[buffer(4)]],
                                 uint node [[thread_position_in_grid]]) {
  if (node >= p.nodes) return;
  PartitionSplit split = splits[node];
  uint sum = 0;
  for (uint b = split.block_begin; b < split.block_end; ++b) {
    prefix[b] = sum;
    sum += counts[b];
  }
  left_counts[node] = sum;
}

// One workgroup scans each node's block counts. A serial lane previously read
// thousands of device values for large roots; SIMD scans now process 256 at once.
kernel void xgb_partition_prefix(device const PartitionSplit* splits [[buffer(0)]],
                                 device const uint* counts [[buffer(1)]],
                                 device uint* prefix [[buffer(2)]],
                                 device ulong* left_counts [[buffer(3)]],
                                 constant PartitionParams& p [[buffer(4)]],
                                 uint node [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd [[simdgroup_index_in_threadgroup]]) {
  if (node >= p.nodes) return;
  threadgroup uint sums[8], accumulated;
  if (tid == 0) accumulated = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  PartitionSplit split = splits[node];
  for (ulong begin = split.block_begin; begin < split.block_end; begin += 256) {
    ulong block = begin + tid;
    uint value = block < split.block_end ? counts[block] : 0;
    uint rank = simd_prefix_inclusive_sum(value);
    if (lane == 31) sums[simd] = rank;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < simd; ++i) rank += sums[i];
    if (block < split.block_end) prefix[block] = accumulated + rank - value;
    // All lanes must read the old carry before lane 255 updates it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 255) accumulated += rank;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) left_counts[node] = accumulated;
}
kernel void xgb_partition_scatter(
    device ulong* rows [[buffer(0)]], device const ulong* saved_rows [[buffer(1)]],
    device const PartitionSplit* splits [[buffer(2)]],
    device const PartitionBlock* blocks [[buffer(3)]], device const uint* ranks [[buffer(4)]],
    device const uint* prefix [[buffer(5)]], device const ulong* left_counts [[buffer(6)]],
    uint block_id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  PartitionBlock block = blocks[block_id];
  ulong pos = block.begin + lane;
  if (pos >= block.end) return;
  PartitionSplit split = splits[block.node];
  uint encoded = ranks[pos];
  ulong left = prefix[block_id] + (encoded & 0x7fffffffu);
  ulong dst = (encoded >> 31) ? split.begin + left - 1
                              : split.begin + left_counts[block.node] + (pos - split.begin) - left;
  rows[dst] = saved_rows[pos];
}
struct LeafRange {
  ulong begin, end;
  int node;
  uint padding;
};
struct LeafPositionParams {
  ulong rows, base, stride0, stride1;
  uint targets, leaves;
};
kernel void xgb_leaf_positions(device const ulong* rows [[buffer(0)]],
                               device const LeafRange* leaves [[buffer(1)]],
                               device const uint2* gradients [[buffer(2)]],
                               device int* positions [[buffer(3)]],
                               constant LeafPositionParams& p [[buffer(4)]],
                               uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.rows) return;
  uint lo = 0, hi = p.leaves;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (leaves[mid].end <= tid)
      lo = mid + 1;
    else
      hi = mid;
  }
  ulong row = rows[tid];
  bool invalid = true;
  for (uint t = 0; t < p.targets; ++t)
    // Metal float comparisons can flush a nonzero subnormal Hessian to zero.
    // Only the two signed-zero encodings denote a sampled-out row.
    invalid = invalid && (gradients[row * p.stride0 + t * p.stride1].y & 0x7fffffffu) == 0;
  int node = leaves[lo].node;
  positions[row - p.base] = invalid ? ~node : node;
}

struct TransposeIndexParams {
  ulong rows;
  uint features, width;
};
kernel void xgb_transpose_index(device const uchar* source [[buffer(0)]],
                                device uchar* target [[buffer(1)]],
                                constant TransposeIndexParams& p [[buffer(2)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup uint tile[32][33];
  ulong row = ulong(group.y) * 32 + tid.y;
  uint feature = group.x * 32 + tid.x;
  for (uint j = 0; j < 32; j += 8)
    if (row + j < p.rows && feature < p.features)
      tile[tid.y + j][tid.x] = xgb_read_bin(source, (row + j) * p.features + feature, p.width);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row = ulong(group.y) * 32 + tid.x;
  feature = group.x * 32 + tid.y;
  for (uint j = 0; j < 32; j += 8)
    if (row < p.rows && feature + j < p.features) {
      ulong out = ulong(feature + j) * p.rows + row;
      uint value = tile[tid.x][tid.y + j];
      if (p.width == 1)
        target[out] = uchar(value);
      else if (p.width == 2)
        reinterpret_cast<device ushort*>(target)[out] = ushort(value);
      else
        reinterpret_cast<device uint*>(target)[out] = value;
    }
}

// Software IEEE-754 binary64 for split statistics. Integer arithmetic avoids
// narrowing histograms to FP32; every operation rounds to nearest, ties to even.
// Only split scoring uses this path, not the high-volume histogram accumulation.
struct XGBDouble {
  ulong bits;
};
constant ulong kDoubleSign = 0x8000000000000000ul;
constant ulong kDoubleFrac = 0x000ffffffffffffful;
constant ulong kDoubleInf = 0x7ff0000000000000ul;
constant ulong kDoubleNaN = 0x7ff8000000000000ul;
inline ulong xgb_shift_jam(ulong value, uint shift) {
  if (!shift) return value;
  if (shift >= 64) return ulong(value != 0);
  return (value >> shift) | ulong((value & ((1ul << shift) - 1)) != 0);
}
inline XGBDouble xgb_dround(ulong sign, int exponent, ulong sig) {
  if (!sig) return {sign};
  if (exponent <= 0) {
    sig = xgb_shift_jam(sig, uint(1 - exponent));
    exponent = 0;
  }
  ulong tail = sig & 7, rounded = (sig >> 3) + ulong(tail > 4 || (tail == 4 && ((sig >> 3) & 1)));
  if (exponent == 0) {
    if (rounded & (1ul << 52)) exponent = 1;
  } else if (rounded & (1ul << 53)) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent >= 2047) return {sign | kDoubleInf};
  return {sign | (ulong(exponent) << 52) | (rounded & kDoubleFrac)};
}
inline XGBDouble xgb_dfrom_float(float value) {
  uint bits = as_type<uint>(value), e = (bits >> 23) & 255, m = bits & 0x7fffff;
  ulong sign = ulong(bits >> 31) << 63;
  if (e == 255) return {sign | kDoubleInf | (m ? (ulong(m) << 29) | 0x8000000000000ul : 0ul)};
  if (!e) {
    if (!m) return {sign};
    int shift = int(clz(m)) - 8;
    m <<= shift;
    return {sign | (ulong(897 - shift) << 52) | (ulong(m & 0x7fffff) << 29)};
  }
  return {sign | (ulong(int(e) + 896) << 52) | (ulong(m) << 29)};
}
inline float xgb_dto_float(XGBDouble value) {
  ulong bits = value.bits, m = bits & kDoubleFrac;
  int e = int((bits >> 52) & 2047);
  uint sign = uint(bits >> 63) << 31;
  if (e == 2047) return as_type<float>(sign | 0x7f800000u | (m ? 0x400000u : 0u));
  if (!e) {
    if (!m) return as_type<float>(sign);
    e = 1;
    int shift = int(clz(m)) - 11;
    m <<= shift;
    e -= shift;
  } else
    m |= 1ul << 52;
  e = e - 1023 + 127;
  ulong sig = xgb_shift_jam(m, 26);
  if (e <= 0) {
    sig = xgb_shift_jam(sig, uint(1 - e));
    e = 0;
  }
  ulong tail = sig & 7, rounded = (sig >> 3) + ulong(tail > 4 || (tail == 4 && ((sig >> 3) & 1)));
  if (e == 0) {
    if (rounded & (1ul << 23)) e = 1;
  } else if (rounded & (1ul << 24)) {
    rounded >>= 1;
    ++e;
  }
  if (e >= 255) return as_type<float>(sign | 0x7f800000u);
  return as_type<float>(sign | (uint(e) << 23) | uint(rounded & 0x7fffff));
}
inline XGBDouble xgb_dneg(XGBDouble a) { return {a.bits ^ kDoubleSign}; }
inline bool xgb_dzero(XGBDouble a) { return (a.bits & ~kDoubleSign) == 0; }
inline bool xgb_deq(XGBDouble a, XGBDouble b) {
  return a.bits == b.bits || (xgb_dzero(a) && xgb_dzero(b));
}
inline bool xgb_dlt(XGBDouble a, XGBDouble b) {
  ulong aa = a.bits & ~kDoubleSign, bb = b.bits & ~kDoubleSign;
  if (aa > kDoubleInf || bb > kDoubleInf) return false;
  if (!aa && !bb) return false;
  if ((a.bits ^ b.bits) & kDoubleSign) return bool(a.bits & kDoubleSign);
  return (a.bits & kDoubleSign) ? aa > bb : aa < bb;
}
inline XGBDouble xgb_dadd(XGBDouble a, XGBDouble b) {
  ulong aa = a.bits & ~kDoubleSign, bb = b.bits & ~kDoubleSign;
  if (aa > kDoubleInf) return {a.bits | 0x8000000000000ul};
  if (bb > kDoubleInf) return {b.bits | 0x8000000000000ul};
  if (aa == kDoubleInf || bb == kDoubleInf) {
    if (aa == kDoubleInf && bb == kDoubleInf && ((a.bits ^ b.bits) & kDoubleSign))
      return {kDoubleNaN};
    return aa == kDoubleInf ? a : b;
  }
  if (!aa && !bb) return {a.bits & b.bits & kDoubleSign};
  if (!aa) return b;
  if (!bb) return a;
  if (aa < bb) {
    XGBDouble temp = a;
    a = b;
    b = temp;
    ulong t = aa;
    aa = bb;
    bb = t;
  }
  int ea = int(aa >> 52), eb = int(bb >> 52);
  ulong ma = (aa & kDoubleFrac) | (ea ? (1ul << 52) : 0),
        mb = (bb & kDoubleFrac) | (eb ? (1ul << 52) : 0);
  ea = max(ea, 1);
  eb = max(eb, 1);
  ma <<= 3;
  mb = xgb_shift_jam(mb << 3, uint(ea - eb));
  ulong sign = a.bits & kDoubleSign, sig;
  if ((a.bits ^ b.bits) & kDoubleSign) {
    sig = ma - mb;
    if (!sig) return {0};
    int shift = int(clz(sig)) - 8;
    sig <<= shift;
    ea -= shift;
  } else {
    sig = ma + mb;
    if (sig & (1ul << 56)) {
      sig = xgb_shift_jam(sig, 1);
      ++ea;
    } else if (!(sig & (1ul << 55))) {
      int shift = int(clz(sig)) - 8;
      sig <<= shift;
      ea -= shift;
    }
  }
  return xgb_dround(sign, ea, sig);
}
inline XGBDouble xgb_dsub(XGBDouble a, XGBDouble b) { return xgb_dadd(a, xgb_dneg(b)); }
inline void xgb_dunpack(ulong bits, thread int& exponent, thread ulong& mantissa) {
  exponent = int((bits >> 52) & 2047);
  mantissa = bits & kDoubleFrac;
  if (exponent)
    mantissa |= 1ul << 52;
  else {
    exponent = 1;
    int shift = int(clz(mantissa)) - 11;
    mantissa <<= shift;
    exponent -= shift;
  }
}
inline ulong xgb_mul_high(ulong a, ulong b, thread ulong& low) {
  ulong p00 = ulong(uint(a)) * uint(b), p01 = ulong(uint(a)) * (b >> 32), p10 = (a >> 32) * uint(b),
        p11 = (a >> 32) * (b >> 32);
  ulong mid = (p00 >> 32) + uint(p01) + uint(p10);
  low = (mid << 32) | uint(p00);
  return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}
inline XGBDouble xgb_dmul(XGBDouble a, XGBDouble b) {
  ulong aa = a.bits & ~kDoubleSign, bb = b.bits & ~kDoubleSign,
        sign = (a.bits ^ b.bits) & kDoubleSign;
  if (aa > kDoubleInf || bb > kDoubleInf) return {kDoubleNaN};
  if (aa == kDoubleInf || bb == kDoubleInf) return {(!aa || !bb) ? kDoubleNaN : sign | kDoubleInf};
  if (!aa || !bb) return {sign};
  int ea, eb;
  ulong ma, mb;
  xgb_dunpack(aa, ea, ma);
  xgb_dunpack(bb, eb, mb);
  ulong low, high = xgb_mul_high(ma, mb, low);
  int exponent = ea + eb - 1023;
  uint shift = 49;
  if (high & (1ul << 41)) {
    shift = 50;
    ++exponent;
  }
  ulong sig = (high << (64 - shift)) | (low >> shift);
  if (low & ((1ul << shift) - 1)) sig |= 1;
  return xgb_dround(sign, exponent, sig);
}
inline XGBDouble xgb_ddiv(XGBDouble a, XGBDouble b) {
  ulong aa = a.bits & ~kDoubleSign, bb = b.bits & ~kDoubleSign,
        sign = (a.bits ^ b.bits) & kDoubleSign;
  if (aa > kDoubleInf || bb > kDoubleInf || (!aa && !bb) || (aa == kDoubleInf && bb == kDoubleInf))
    return {kDoubleNaN};
  if (aa == kDoubleInf || !bb) return {sign | kDoubleInf};
  if (bb == kDoubleInf || !aa) return {sign};
  int ea, eb;
  ulong ma, mb;
  xgb_dunpack(aa, ea, ma);
  xgb_dunpack(bb, eb, mb);
  int exponent = ea - eb + 1023;
  if (ma < mb) {
    ma <<= 1;
    --exponent;
  }
  ulong remainder = ma - mb, sig = 1;
  // Generate ten quotient bits at a time. FP32 estimates the small quotient;
  // exact integer residual checks correct its at-most-one-unit rounding error.
  // The widened remainder and product fit in 63 bits (53-bit divisor + 10).
  for (uint done = 0; done < 55; done += 10) {
    uint shift = min(10u, 55 - done);
    ulong numerator = remainder << shift;
    ulong quotient = ulong((float(remainder) / float(mb)) * float(1u << shift));
    ulong product = quotient * mb;
    if (product > numerator) {
      --quotient;
      product -= mb;
    }
    if (numerator - product >= mb) {
      ++quotient;
      product += mb;
    }
    remainder = numerator - product;
    sig = (sig << shift) | quotient;
  }
  if (remainder) sig |= 1;
  return xgb_dround(sign, exponent, sig);
}

struct SplitTask {
  uint begin, end, kind, limit;
  ulong parent_grad, parent_hess;
  float root_gain, lower, upper;
  int monotone;
};
struct SplitParameters {
  float reg_lambda, reg_alpha, max_delta_step, min_child_weight;
};
struct SplitResult {
  ulong lg, lh, rg, rh;
  float gain;
  int bin;
  uint missing, partition;
};
struct DoublePair {
  XGBDouble g, h;
};
inline DoublePair xgb_pair_read(ulong2 bits) { return {{bits.x}, {bits.y}}; }
inline DoublePair xgb_pair_add(DoublePair a, DoublePair b) {
  return {xgb_dadd(a.g, b.g), xgb_dadd(a.h, b.h)};
}
inline DoublePair xgb_pair_sub(DoublePair a, DoublePair b) {
  return {xgb_dsub(a.g, b.g), xgb_dsub(a.h, b.h)};
}
inline float xgb_split_weight(DoublePair sum, SplitParameters p, float lower, float upper) {
  if (!xgb_dlt({0}, sum.h)) return 0.0f;
  XGBDouble alpha = xgb_dfrom_float(p.reg_alpha), threshold{0};
  if (xgb_dlt(alpha, sum.g))
    threshold = xgb_dsub(sum.g, alpha);
  else if (xgb_dlt(sum.g, xgb_dneg(alpha)))
    threshold = xgb_dadd(sum.g, alpha);
  XGBDouble weight = xgb_ddiv(xgb_dneg(threshold), xgb_dadd(sum.h, xgb_dfrom_float(p.reg_lambda)));
  if (p.max_delta_step != 0.0f) {
    XGBDouble delta = xgb_dfrom_float(p.max_delta_step);
    if (xgb_dlt(delta, {weight.bits & ~kDoubleSign}))
      weight = {delta.bits | (weight.bits & kDoubleSign)};
  }
  float value = xgb_dto_float(weight);
  if (value < lower)
    value = lower;
  else if (value > upper)
    value = upper;
  return value;
}
inline float xgb_split_gain_weight(DoublePair sum, float weight, SplitParameters p) {
  if (!xgb_dlt({0}, sum.h)) return 0.0f;
  XGBDouble two = xgb_dfrom_float(2.0f), w = xgb_dfrom_float(weight);
  // The upstream square of the float leaf weight rounds to FP32 before the
  // enclosing double expression. Preserve that rounding boundary explicitly.
  XGBDouble square = xgb_dfrom_float(xgb_dto_float(xgb_dmul(w, w)));
  XGBDouble a = xgb_dmul(xgb_dmul(two, sum.g), w);
  XGBDouble b = xgb_dmul(xgb_dadd(sum.h, xgb_dfrom_float(p.reg_lambda)), square);
  XGBDouble c = xgb_dmul(xgb_dmul(two, xgb_dfrom_float(p.reg_alpha)), {w.bits & ~kDoubleSign});
  return xgb_dto_float(xgb_dneg(xgb_dadd(xgb_dadd(a, b), c)));
}
inline void xgb_consider_split(DoublePair left, DoublePair right, SplitTask task, SplitParameters p,
                               int bin, uint missing, uint partition, thread SplitResult& best) {
  XGBDouble minimum = xgb_dfrom_float(p.min_child_weight);
  if (!xgb_dlt({0}, left.h) || !xgb_dlt({0}, right.h) || xgb_dlt(left.h, minimum) ||
      xgb_dlt(right.h, minimum))
    return;
  float wl = xgb_split_weight(left, p, task.lower, task.upper),
        wr = xgb_split_weight(right, p, task.lower, task.upper);
  if ((task.monotone > 0 && !(wl <= wr)) || (task.monotone < 0 && !(wl >= wr))) return;
  float gain =
      (xgb_split_gain_weight(left, wl, p) + xgb_split_gain_weight(right, wr, p)) - task.root_gain;
  if (isfinite(gain) && gain > best.gain)
    best = {left.g.bits, left.h.bits, right.g.bits, right.h.bits, gain, bin, missing, partition};
}
// Outward FP32 bounds used only to reject candidates whose upper bound is
// below a valid candidate's lower bound. Survivors use the unchanged binary64
// evaluator, including float weight/square rounding and CPU tie order.
#ifndef XGB_METAL_SPLIT_FILTER
#define XGB_METAL_SPLIT_FILTER 1
#endif
inline float xgb_round_down(float v) {
  uint bits = as_type<uint>(v), mag = bits & 0x7fffffffu;
  if (mag >= 0x7f800000u) return v;
  if (mag < 0x00800000u) return as_type<float>(0x80800000u);
  return as_type<float>(bits & 0x80000000u ? bits + 1 : bits - 1);
}
inline float xgb_round_up(float v) {
  uint bits = as_type<uint>(v), mag = bits & 0x7fffffffu;
  if (mag >= 0x7f800000u) return v;
  if (mag < 0x00800000u) return as_type<float>(0x00800000u);
  return as_type<float>(bits & 0x80000000u ? bits - 1 : bits + 1);
}
inline float2 xgb_unconstrained_gain_bound(DoublePair sum, float lambda) {
  float g = abs(xgb_dto_float(sum.g)), h = xgb_dto_float(sum.h);
  if (!(h >= 1.0e-8f && h <= 1.0e8f && lambda >= 0.0f && lambda <= 1.0e8f))
    return float2(-INFINITY, INFINITY);
  if (xgb_dzero(sum.g)) return float2(0);
  // This restricted range keeps every square, denominator, quotient and float
  // weight square normal and finite. Other values receive an unbounded interval.
  if (!(g >= 1.0e-8f && g <= 1.0e8f && h >= 1.0e-8f && h <= 1.0e8f &&
        lambda >= 0.0f && lambda <= 1.0e8f)) return float2(-INFINITY, INFINITY);
  float gl = xgb_round_down(g), gh = xgb_round_up(g);
  float al = xgb_round_down(xgb_round_down(h) + lambda);
  float ah = xgb_round_up(xgb_round_up(h) + lambda);
  float lo = xgb_round_down(xgb_round_down(gl * gl) / ah);
  float hi = xgb_round_up(xgb_round_up(gh * gh) / al);
  // For w = round_float(-g/(h+lambda)), the CPU gain differs from g*g/(h+lambda)
  // by < 4*FLT_EPSILON relatively in this range. 32 eps also covers binary64
  // operation rounding and the outward interval arithmetic above.
  constexpr float margin = 0x1p-18f;  // 32 * 2^-23
  return float2(xgb_round_down(lo * (1.0f - margin)),
                xgb_round_up(hi * (1.0f + margin)));
}
inline float2 xgb_split_gain_bound(DoublePair left, DoublePair right, SplitTask task,
                                   SplitParameters p) {
  XGBDouble minimum = xgb_dfrom_float(p.min_child_weight);
  if (!xgb_dlt({0}, left.h) || !xgb_dlt({0}, right.h) || xgb_dlt(left.h, minimum) ||
      xgb_dlt(right.h, minimum)) return float2(-INFINITY);
  float2 l = xgb_unconstrained_gain_bound(left, p.reg_lambda);
  float2 r = xgb_unconstrained_gain_bound(right, p.reg_lambda);
  return float2(xgb_round_down(xgb_round_down(l.x + r.x) - task.root_gain),
                xgb_round_up(xgb_round_up(l.y + r.y) - task.root_gain));
}

// If a complete feature sum fits in 53 integer bits at its finest input unit,
// every possible prefix is exactly representable as binary64. In that case an
// integer parallel scan is equivalent to the CPU's sequential double additions.
inline ulong xgb_simd_prefix64(ulong value) {
  uint low = simd_prefix_inclusive_sum(uint(value));
  uint carry = simd_prefix_inclusive_sum(uint(low < uint(value)));
  uint high = simd_prefix_inclusive_sum(uint(value >> 32)) + carry;
  return (ulong(high) << 32) | low;
}
inline int2 xgb_exact_span(ulong bits) {
  ulong magnitude = bits & ~kDoubleSign;
  if (!magnitude) return int2(4096, -4096);
  if (magnitude >= kDoubleInf) return int2(-4096, 4096);
  int exponent;
  ulong mantissa;
  xgb_dunpack(magnitude, exponent, mantissa);
  return int2(exponent - 1075 + int(ctz(mantissa)), exponent - 1023);
}
inline ulong xgb_exact_integer(ulong bits, int unit) {
  if (!(bits & ~kDoubleSign)) return 0;
  int exponent;
  ulong mantissa;
  xgb_dunpack(bits & ~kDoubleSign, exponent, mantissa);
  int shift = exponent - 1075 - unit;
  ulong value = shift >= 0 ? mantissa << uint(shift) : mantissa >> uint(-shift);
  return bits & kDoubleSign ? 0ul - value : value;
}
inline ulong xgb_exact_double(ulong value, int unit) {
  bool negative = bool(value & kDoubleSign);
  ulong magnitude = negative ? 0ul - value : value;
  if (!magnitude) return 0;
  int high = 63 - int(clz(magnitude));
  return xgb_dround(negative ? kDoubleSign : 0, high + unit + 1023, magnitude << uint(55 - high))
      .bits;
}

// One threadgroup owns a feature. Preserve CPU scan order for the prefix sums,
// then evaluate independent candidate gains across the entire threadgroup.
kernel void xgb_evaluate_splits(
    device const ulong2* hist [[buffer(0)]], device const SplitTask* tasks [[buffer(1)]],
    constant SplitParameters& p [[buffer(2)]], device SplitResult* results [[buffer(3)]],
    device uint* order [[buffer(4)]], device float* weights [[buffer(5)]],
    device ulong4* scans [[buffer(6)]], uint task_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  SplitTask task = tasks[task_id];
  uint n = task.end - task.begin;
  DoublePair parent{{task.parent_grad}, {task.parent_hess}};
  threadgroup uint missing;
  threadgroup ulong2 missing_sum;
  threadgroup float gains[8];
  threadgroup uint ranks[8];
  if (task.kind == 2) {
    for (uint i = tid; i < n; i += width) {
      order[task.begin + i] = i;
      weights[task.begin + i] =
          xgb_split_weight(xgb_pair_read(hist[task.begin + i]), p, -INFINITY, INFINITY);
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  threadgroup int2 units[8], tops[8];
  threadgroup ulong2 chunk_totals[8], carry;
  threadgroup atomic_uint exact;
  if (tid == 0 && task.kind == 2) {
    // Stable ordering is observable when categories have identical weights.
    for (uint i = 1; i < n; ++i) {
      uint value = order[task.begin + i], j = i;
      float key = weights[task.begin + value];
      while (j && key < weights[task.begin + order[task.begin + j - 1]]) {
        order[task.begin + j] = order[task.begin + j - 1];
        --j;
      }
      order[task.begin + j] = value;
    }
  }
  int2 unit(4096), top(-4096);
  for (uint i = tid; i < n; i += width) {
    ulong2 value = hist[task.begin + i];
    int2 g = xgb_exact_span(value.x), h = xgb_exact_span(value.y);
    unit = min(unit, int2(g.x, h.x));
    top = max(top, int2(g.y, h.y));
  }
  int2 minimum(simd_min(unit.x), simd_min(unit.y)), maximum(simd_max(top.x), simd_max(top.y));
  if (lane == 0) {
    units[simd] = minimum;
    tops[simd] = maximum;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0) {
    int2 lo(4096), hi(-4096);
    for (uint j = 0; j < width / 32; ++j) {
      lo = min(lo, units[j]);
      hi = max(hi, tops[j]);
    }
    uint count_bits = n > 1 ? 32 - clz(n - 1) : 0;
    atomic_store_explicit(
        &exact, uint(all(hi - lo + int(count_bits) <= 52) && all(hi + int(count_bits) < 1024)),
        memory_order_relaxed);
    units[0] = select(lo, int2(0), lo == int2(4096));
    carry = ulong2(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (atomic_load_explicit(&exact, memory_order_relaxed)) {
    int2 u = units[0];
    for (uint base = 0; base < n; base += width) {
      uint i = base + tid;
      ulong2 value(0);
      if (i < n) {
        uint bin = task.kind == 2 ? order[task.begin + i] : i;
        ulong2 bits = hist[task.begin + bin];
        value = ulong2(xgb_exact_integer(bits.x, u.x), xgb_exact_integer(bits.y, u.y));
      }
      ulong2 prefix(xgb_simd_prefix64(value.x), xgb_simd_prefix64(value.y));
      if (lane == 31) chunk_totals[simd] = prefix;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix += carry;
      for (uint j = 0; j < simd; ++j) prefix += chunk_totals[j];
      if (i < n)
        scans[task.begin + i].xy =
            ulong2(xgb_exact_double(prefix.x, u.x), xgb_exact_double(prefix.y, u.y));
      // Every lane must read the old carry before the last lane updates it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == width - 1) carry = prefix;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    DoublePair total =
        xgb_pair_read(ulong2(xgb_exact_double(carry.x, u.x), xgb_exact_double(carry.y, u.y)));
    if (tid == 0) {
      missing = (!xgb_deq(total.g, parent.g) || !xgb_deq(total.h, parent.h));
      DoublePair absent = xgb_pair_sub(parent, total);
      missing_sum = ulong2(absent.g.bits, absent.h.bits);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if ((task.kind == 0 && missing) || task.kind == 2)
      for (uint i = tid; i < n; i += width) {
        DoublePair preceding =
            i ? xgb_pair_read(scans[task.begin + i - 1].xy) : DoublePair{{0}, {0}};
        DoublePair suffix = xgb_pair_sub(total, preceding);
        scans[task.begin + i].zw = ulong2(suffix.g.bits, suffix.h.bits);
      }
  } else if (tid == 0) {
    // Extreme exponent spreads retain the sequential, rounded binary64 scan.
    DoublePair forward{{0}, {0}}, backward{{0}, {0}};
    for (uint i = 0; i < n; ++i) {
      uint bin = task.kind == 2 ? order[task.begin + i] : i;
      forward = xgb_pair_add(forward, xgb_pair_read(hist[task.begin + bin]));
      scans[task.begin + i].xy = ulong2(forward.g.bits, forward.h.bits);
    }
    missing = (!xgb_deq(forward.g, parent.g) || !xgb_deq(forward.h, parent.h));
    DoublePair absent = xgb_pair_sub(parent, forward);
    missing_sum = ulong2(absent.g.bits, absent.h.bits);
    if ((task.kind == 0 && missing) || task.kind == 2) {
      for (int i = int(n) - 1; i >= 0; --i) {
        uint bin = task.kind == 2 ? order[task.begin + i] : uint(i);
        backward = xgb_pair_add(backward, xgb_pair_read(hist[task.begin + bin]));
        scans[task.begin + i].zw = ulong2(backward.g.bits, backward.h.bits);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  uint root_bits = as_type<uint>(task.root_gain) & 0x7fffffffu;
  bool screen = XGB_METAL_SPLIT_FILTER && task.kind == 0 &&
                task.monotone == 0 && task.lower == -INFINITY && task.upper == INFINITY &&
                (as_type<uint>(p.reg_alpha) & 0x7fffffffu) == 0 &&
                (as_type<uint>(p.max_delta_step) & 0x7fffffffu) == 0 &&
                isfinite(task.root_gain) && (!root_bits || root_bits >= 0x00800000u);
  threadgroup float cutoff;
  if (screen) {
    float lower = 0.0f;  // The original evaluator accepts only positive gains.
    for (uint i = tid; i < n; i += width) {
      DoublePair sum = xgb_pair_read(scans[task.begin + i].xy);
      float2 bound = xgb_split_gain_bound(sum, xgb_pair_sub(parent, sum), task, p);
      lower = max(lower, bound.x);
      if (missing) {
        DoublePair right = xgb_pair_read(scans[task.begin + i].zw);
        float2 backward = xgb_split_gain_bound(xgb_pair_sub(parent, right), right, task, p);
        lower = max(lower, backward.x);
      } else {
        // Numeric features without missing rows do not need the suffix scan.
        // Category-only scratch may be a dummy allocation in the real runtime.
        scans[task.begin + i].z = ulong(as_type<uint>(bound.y));
      }
    }
    float best_lower = simd_max(lower);
    if (lane == 0) gains[simd] = best_lower;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (tid == 0) {
      cutoff = 0.0f;
      for (uint j = 0; j < width / 32; ++j) cutoff = max(cutoff, gains[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  SplitResult best{0, 0, 0, 0, 0.0f, -1, 0, 0};
  if (task.kind == 0) {
    for (uint i = tid; i < n; i += width) {
      DoublePair sum = xgb_pair_read(scans[task.begin + i].xy);
      DoublePair right = xgb_pair_sub(parent, sum);
      if (screen) {
        float upper = missing ? xgb_split_gain_bound(sum, right, task, p).y
                              : as_type<float>(uint(scans[task.begin + i].z));
        if (upper < cutoff) continue;
      }
      xgb_consider_split(sum, right, task, p, int(i), 0, 0, best);
    }
    if (missing)
      for (int i = int(n) - 1 - int(tid); i >= 0; i -= int(width)) {
        DoublePair sum = xgb_pair_read(scans[task.begin + i].zw);
        DoublePair left = xgb_pair_sub(parent, sum);
        if (screen && xgb_split_gain_bound(left, sum, task, p).y < cutoff) continue;
        xgb_consider_split(left, sum, task, p, i, 1, 0, best);
      }
  } else if (task.kind == 1) {
    DoublePair absent = xgb_pair_read(missing_sum);
    for (uint i = tid; i < n; i += width) {
      DoublePair right = xgb_pair_read(hist[task.begin + i]);
      xgb_consider_split(xgb_pair_sub(parent, right), right, task, p, int(i), 1, 0, best);
      right = xgb_pair_add(right, absent);
      xgb_consider_split(xgb_pair_sub(parent, right), right, task, p, int(i), 0, 0, best);
    }
  } else {
    uint limit = min(n, task.limit);
    for (uint i = tid; i + 1 < limit; i += width) {
      DoublePair sum = xgb_pair_read(scans[task.begin + i].xy);
      xgb_consider_split(xgb_pair_sub(parent, sum), sum, task, p, int(i), 1, i + 1, best);
    }
    for (int i = int(n) - 1 - int(tid); i > int(n - limit); i -= int(width)) {
      DoublePair sum = xgb_pair_read(scans[task.begin + i].zw);
      xgb_consider_split(sum, xgb_pair_sub(parent, sum), task, p, i, 0, uint(i), best);
    }
  }
  // Gain ties retain the original CPU enumeration order, even across lanes.
  uint rank = 0xffffffffu;
  if (best.bin >= 0) {
    if (task.kind == 0)
      rank = best.missing ? 2 * n - 1 - uint(best.bin) : uint(best.bin);
    else if (task.kind == 1)
      rank = 2 * uint(best.bin) + (best.missing ? 0 : 1);
    else
      rank = best.missing ? uint(best.bin) : 2 * n - 1 - uint(best.bin);
  }
  float gain = simd_max(best.gain);
  uint winner = simd_min(best.gain == gain ? rank : 0xffffffffu);
  if (lane == 0) {
    gains[simd] = gain;
    ranks[simd] = winner;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    float g = lane < width / 32 ? gains[lane] : 0.0f;
    uint r = lane < width / 32 ? ranks[lane] : 0xffffffffu;
    float top = simd_max(g);
    uint first = simd_min(g == top ? r : 0xffffffffu);
    if (lane == 0) {
      gains[0] = top;
      ranks[0] = first;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((best.bin >= 0 && best.gain == gains[0] && rank == ranks[0]) ||
      (ranks[0] == 0xffffffffu && tid == 0))
    results[task_id] = best;
}

struct MultiSplitTask {
  uint begin, end, kind, limit, parent_begin, targets;
  ulong root_gain;
  int monotone;
  uint constrained;
};
inline XGBDouble xgb_weight_precise(DoublePair sum, SplitParameters p) {
  if (!xgb_dlt({0}, sum.h)) return {0};
  XGBDouble alpha = xgb_dfrom_float(p.reg_alpha), threshold{0};
  if (xgb_dlt(alpha, sum.g))
    threshold = xgb_dsub(sum.g, alpha);
  else if (xgb_dlt(sum.g, xgb_dneg(alpha)))
    threshold = xgb_dadd(sum.g, alpha);
  XGBDouble w = xgb_ddiv(xgb_dneg(threshold), xgb_dadd(sum.h, xgb_dfrom_float(p.reg_lambda)));
  XGBDouble delta = xgb_dfrom_float(p.max_delta_step);
  if (p.max_delta_step && xgb_dlt(delta, {w.bits & ~kDoubleSign}))
    w = {delta.bits | (w.bits & kDoubleSign)};
  return w;
}
inline XGBDouble xgb_gain_precise(DoublePair sum, XGBDouble w, XGBDouble square,
                                  SplitParameters p) {
  XGBDouble two = xgb_dfrom_float(2.0f);
  XGBDouble first = xgb_dmul(xgb_dmul(two, sum.g), w);
  XGBDouble second = xgb_dmul(xgb_dadd(sum.h, xgb_dfrom_float(p.reg_lambda)), square);
  XGBDouble penalty =
      xgb_dmul(xgb_dmul(two, xgb_dfrom_float(p.reg_alpha)), {w.bits & ~kDoubleSign});
  return xgb_dneg(xgb_dadd(xgb_dadd(first, second), penalty));
}
inline XGBDouble xgb_unconstrained_gain(DoublePair sum, SplitParameters p) {
  if (!xgb_dlt({0}, sum.h)) return {0};
  if (p.max_delta_step == 0.0f) {
    XGBDouble g = sum.g, alpha = xgb_dfrom_float(p.reg_alpha);
    if (p.reg_alpha) {
      if (xgb_dlt(alpha, g))
        g = xgb_dsub(g, alpha);
      else if (xgb_dlt(g, xgb_dneg(alpha)))
        g = xgb_dadd(g, alpha);
      else
        g = {0};
    }
    return xgb_ddiv(xgb_dmul(g, g), xgb_dadd(sum.h, xgb_dfrom_float(p.reg_lambda)));
  }
  XGBDouble w = xgb_weight_precise(sum, p);
  return xgb_gain_precise(sum, w, xgb_dmul(w, w), p);
}
inline XGBDouble xgb_vector_gain_weight(DoublePair sum, float weight, SplitParameters p) {
  XGBDouble w = xgb_dfrom_float(weight);
  XGBDouble square = xgb_dfrom_float(xgb_dto_float(xgb_dmul(w, w)));
  return xgb_gain_precise(sum, w, square, p);
}
inline void xgb_multi_sums(MultiSplitTask task, uint bin, uint missing, uint target,
                           device const ulong2* hist, device const ulong4* scans,
                           device const ulong2* parents, thread DoublePair& left,
                           thread DoublePair& right) {
  uint idx = task.begin + bin * task.targets + target;
  DoublePair parent = xgb_pair_read(parents[task.parent_begin + target]);
  if (task.kind == 0) {
    if (missing) {
      right = xgb_pair_read(scans[idx].zw);
      left = xgb_pair_sub(parent, right);
    } else {
      left = xgb_pair_read(scans[idx].xy);
      right = xgb_pair_sub(parent, left);
    }
  } else if (task.kind == 1) {
    right = xgb_pair_read(hist[idx]);
    if (!missing) {
      DoublePair total = xgb_pair_read(scans[task.end - task.targets + target].xy);
      right = xgb_pair_add(right, xgb_pair_sub(parent, total));
    }
    left = xgb_pair_sub(parent, right);
  } else {
    if (missing) {
      right = xgb_pair_read(scans[idx].xy);
      left = xgb_pair_sub(parent, right);
    } else {
      left = xgb_pair_read(scans[idx].zw);
      right = xgb_pair_sub(parent, left);
    }
  }
}
inline float xgb_multi_gain(MultiSplitTask task, SplitParameters p, uint bin, uint missing,
                            device const ulong2* hist, device const ulong4* scans,
                            device const ulong2* parents, device const float2* bounds) {
  XGBDouble gain{0}, lh{0}, rh{0};
  for (uint t = 0; t < task.targets; ++t) {
    DoublePair left, right;
    xgb_multi_sums(task, bin, missing, t, hist, scans, parents, left, right);
    lh = xgb_dadd(lh, left.h);
    rh = xgb_dadd(rh, right.h);
    if (!task.constrained) {
      gain = xgb_dadd(gain, xgb_unconstrained_gain(left, p));
      gain = xgb_dadd(gain, xgb_unconstrained_gain(right, p));
    }
  }
  XGBDouble count = xgb_dfrom_float(float(task.targets)),
            minimum = xgb_dfrom_float(p.min_child_weight);
  lh = xgb_ddiv(lh, count);
  rh = xgb_ddiv(rh, count);
  if (!xgb_dlt({0}, lh) || !xgb_dlt({0}, rh) || xgb_dlt(lh, minimum) || xgb_dlt(rh, minimum))
    return -INFINITY;
  if (task.constrained)
    for (uint t = 0; t < task.targets; ++t) {
      DoublePair left, right;
      xgb_multi_sums(task, bin, missing, t, hist, scans, parents, left, right);
      float2 b = bounds[task.parent_begin + t];
      float wl = xgb_split_weight(left, p, b.x, b.y), wr = xgb_split_weight(right, p, b.x, b.y);
      if ((task.monotone > 0 && !(wl <= wr)) || (task.monotone < 0 && !(wl >= wr))) {
        SplitParameters pooled = p;
        pooled.reg_alpha = 2.0f * p.reg_alpha;
        pooled.reg_lambda = 2.0f * p.reg_lambda;
        wl = xgb_split_weight(xgb_pair_add(left, right), pooled, b.x, b.y);
        wr = wl;
      }
      gain = xgb_dadd(gain, xgb_vector_gain_weight(left, wl, p));
      gain = xgb_dadd(gain, xgb_vector_gain_weight(right, wr, p));
    }
  return xgb_dto_float(xgb_dsub(gain, {task.root_gain}));
}
kernel void xgb_evaluate_multi_splits(
    device const ulong2* hist [[buffer(0)]], device const MultiSplitTask* tasks [[buffer(1)]],
    constant SplitParameters& p [[buffer(2)]], device SplitResult* results [[buffer(3)]],
    device uint* order [[buffer(4)]], device ulong* scores [[buffer(5)]],
    device ulong4* scans [[buffer(6)]], device const ulong2* parents [[buffer(7)]],
    device const float2* bounds [[buffer(8)]], device ulong2* children [[buffer(9)]],
    uint task_id [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  MultiSplitTask task = tasks[task_id];
  uint n = (task.end - task.begin) / task.targets;
  threadgroup atomic_uint missing;
  threadgroup float gains[8];
  threadgroup uint ranks[8];
  if (tid == 0) atomic_store_explicit(&missing, 0, memory_order_relaxed);
  if (task.kind == 2)
    for (uint i = tid; i < n; i += width) {
      order[task.begin + i] = i;
      XGBDouble score{0};
      for (uint t = 0; t < task.targets; ++t) {
        float parent =
            xgb_split_weight(xgb_pair_read(parents[task.parent_begin + t]), p, -INFINITY, INFINITY);
        float child = xgb_split_weight(xgb_pair_read(hist[task.begin + i * task.targets + t]), p,
                                       -INFINITY, INFINITY);
        // CPU category scores sum products rounded to FP32 into a double sum.
        XGBDouble product = xgb_dfrom_float(
            xgb_dto_float(xgb_dmul(xgb_dfrom_float(parent), xgb_dfrom_float(child))));
        score = xgb_dadd(score, product);
      }
      scores[task.begin + i] = score.bits;
    }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0 && task.kind == 2) {
    for (uint i = 1; i < n; ++i) {
      uint value = order[task.begin + i], j = i;
      XGBDouble key{scores[task.begin + value]};
      while (j && xgb_dlt(key, {scores[task.begin + order[task.begin + j - 1]]})) {
        order[task.begin + j] = order[task.begin + j - 1];
        --j;
      }
      order[task.begin + j] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  // Independent target scans retain each target's exact sequential sum order.
  for (uint t = tid; t < task.targets; t += width) {
    DoublePair forward{{0}, {0}}, backward{{0}, {0}},
        parent = xgb_pair_read(parents[task.parent_begin + t]);
    for (uint i = 0; i < n; ++i) {
      uint bin = task.kind == 2 ? order[task.begin + i] : i;
      forward = xgb_pair_add(forward, xgb_pair_read(hist[task.begin + bin * task.targets + t]));
      scans[task.begin + i * task.targets + t].xy = ulong2(forward.g.bits, forward.h.bits);
    }
    bool absent = !xgb_deq(forward.g, parent.g) || !xgb_deq(forward.h, parent.h);
    if (absent) atomic_store_explicit(&missing, 1, memory_order_relaxed);
    // Another target may have missing data even if this one sums to the parent.
    if (task.kind != 1)
      for (int i = int(n) - 1; i >= 0; --i) {
        uint bin = task.kind == 2 ? order[task.begin + i] : uint(i);
        backward = xgb_pair_add(backward, xgb_pair_read(hist[task.begin + bin * task.targets + t]));
        scans[task.begin + i * task.targets + t].zw = ulong2(backward.g.bits, backward.h.bits);
      }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  float best = 0.0f;
  uint rank = 0xffffffffu;
  uint limit = min(n, task.limit);
  // Candidate rank is exactly the upstream enumeration order.
  for (uint r = tid; r < 2 * n; r += width) {
    uint bin, default_left;
    if (task.kind == 0) {
      if (r >= n && !atomic_load_explicit(&missing, memory_order_relaxed)) continue;
      bin = r < n ? r : 2 * n - 1 - r;
      default_left = r >= n;
    } else if (task.kind == 1) {
      bin = r / 2;
      default_left = !(r % 2);
    } else {
      bin = r < n ? r : 2 * n - 1 - r;
      default_left = r < n;
      if ((r < n && bin + 1 >= limit) || (r >= n && bin <= n - limit)) continue;
    }
    float gain = xgb_multi_gain(task, p, bin, default_left, hist, scans, parents, bounds);
    if (isfinite(gain) && gain > best) {
      best = gain;
      rank = r;
    }
  }
  float gain = simd_max(best);
  uint winner = simd_min(best == gain ? rank : 0xffffffffu);
  if (lane == 0) {
    gains[simd] = gain;
    ranks[simd] = winner;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    float g = lane < width / 32 ? gains[lane] : 0.0f;
    uint r = lane < width / 32 ? ranks[lane] : 0xffffffffu;
    float top = simd_max(g);
    uint first = simd_min(g == top ? r : 0xffffffffu);
    if (lane == 0) {
      gains[0] = top;
      ranks[0] = first;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint r = ranks[0];
    SplitResult result{0, 0, 0, 0, gains[0], -1, 0, 0};
    if (r != 0xffffffffu) {
      uint bin = task.kind == 1 ? r / 2 : (r < n ? r : 2 * n - 1 - r);
      uint default_left = task.kind == 0 ? r >= n : (task.kind == 1 ? !(r % 2) : r < n);
      result.bin = int(bin);
      result.missing = default_left;
      result.partition = task.kind == 2 ? bin + default_left : 0;
      for (uint t = 0; t < task.targets; ++t) {
        DoublePair left, right;
        xgb_multi_sums(task, bin, default_left, t, hist, scans, parents, left, right);
        children[2 * task.parent_begin + t] = ulong2(left.g.bits, left.h.bits);
        children[2 * task.parent_begin + task.targets + t] = ulong2(right.g.bits, right.h.bits);
      }
    }
    results[task_id] = result;
  }
}

// Convert a quantized signed integer to binary64 at a power-of-two scale.
// The histogram scale keeps these values in the normal binary64 exponent range.
inline ulong xgb_scaled_integer(ulong value, int unit) {
  bool negative = bool(value & kDoubleSign);
  ulong magnitude = negative ? 0ul - value : value;
  if (!magnitude) return 0;
  int high = 63 - int(clz(magnitude));
  ulong sig = high > 55 ? xgb_shift_jam(magnitude, uint(high - 55)) : magnitude << uint(55 - high);
  return xgb_dround(negative ? kDoubleSign : 0, high + unit + 1023, sig).bits;
}
struct DequantizeParams {
  ulong count;
  uint bins, add;
};
kernel void xgb_histogram_dequantize(device const ulong2* quantized [[buffer(0)]],
                                     device ulong2* output [[buffer(1)]],
                                     device const ulong* offsets [[buffer(2)]],
                                     constant DequantizeParams& p [[buffer(3)]],
                                     device const GradientScale& scale [[buffer(4)]],
                                     uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.count) return;
  ulong2 raw = quantized[tid];
  ulong2 value(xgb_scaled_integer(raw.x, -scale.grad_exponent),
               xgb_scaled_integer(raw.y, -scale.hess_exponent));
  ulong out = offsets[tid / p.bins] + tid % p.bins;
  if (p.add) {
    ulong2 prior = output[out];
    value = ulong2(xgb_dadd({prior.x}, {value.x}).bits, xgb_dadd({prior.y}, {value.y}).bits);
  }
  output[out] = value;
}
struct HistogramSubtraction {
  ulong parent, sibling, output, bins;
};
kernel void xgb_histogram_subtract(device ulong2* histograms [[buffer(0)]],
                                   device const HistogramSubtraction* tasks [[buffer(1)]],
                                   uint2 tid [[thread_position_in_grid]]) {
  HistogramSubtraction task = tasks[tid.y];
  if (ulong(tid.x) >= task.bins) return;
  ulong2 parent = histograms[task.parent + tid.x], sibling = histograms[task.sibling + tid.x];
  histograms[task.output + tid.x] =
      ulong2(xgb_dsub({parent.x}, {sibling.x}).bits, xgb_dsub({parent.y}, {sibling.y}).bits);
}
struct HistogramGather {
  ulong source;
  uint output, bins, stride, padding;
};
kernel void xgb_histogram_gather(device const ulong2* source [[buffer(0)]],
                                 device ulong2* output [[buffer(1)]],
                                 device const HistogramGather* segments [[buffer(2)]],
                                 uint segment [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint width [[threads_per_threadgroup]]) {
  HistogramGather s = segments[segment];
  for (uint i = tid; i < s.bins; i += width)
    output[ulong(s.output) + ulong(i) * s.stride] = source[s.source + i];
}

struct AdaptiveParams {
  ulong rows;
  uint targets, label_targets, weighted, kind, blocks, padding;
};
inline XGBDouble xgb_shuffle_down_double(XGBDouble value, uint delta) {
  uint low = simd_shuffle_down(uint(value.bits), delta),
       high = simd_shuffle_down(uint(value.bits >> 32), delta);
  return {(ulong(high) << 32) | low};
}
// A checked integer path accelerates sums without sacrificing binary64 results.
// With at most 256 inputs, this bound proves every subset sum is exact in FP64.
inline ulong2 xgb_reduce_double_pair(XGBDouble a, XGBDouble b, threadgroup int4* bounds,
                                     threadgroup ulong2* sums, uint lane, uint simd) {
  int2 ga = xgb_exact_span(a.bits), gb = xgb_exact_span(b.bits);
  int4 extent(simd_min(ga.x), simd_min(gb.x), simd_max(ga.y), simd_max(gb.y));
  if (lane == 0) bounds[simd] = extent;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0 && lane == 0) {
    int2 lo(4096), hi(-4096);
    for (uint i = 0; i < 8; ++i) {
      lo = min(lo, bounds[i].xy);
      hi = max(hi, bounds[i].zw);
    }
    bool exact = all(hi - lo + 8 <= 52) && all(hi + 8 < 1024);
    lo = select(lo, int2(0), lo == int2(4096));
    bounds[0] = int4(lo, int(exact), 0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong2 result(0);
  if (bounds[0].z) {
    int2 unit = bounds[0].xy;
    ulong2 integer(xgb_exact_integer(a.bits, unit.x), xgb_exact_integer(b.bits, unit.y));
    ulong2 prefix(xgb_simd_prefix64(integer.x), xgb_simd_prefix64(integer.y));
    if (lane == 31) sums[simd] = prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
      ulong2 sum(0);
      for (uint i = 0; i < 8; ++i) sum += sums[i];
      result = ulong2(xgb_exact_double(sum.x, unit.x), xgb_exact_double(sum.y, unit.y));
    }
  } else {
    for (uint offset = 16; offset; offset >>= 1) {
      XGBDouble x = xgb_shuffle_down_double(a, offset), y = xgb_shuffle_down_double(b, offset);
      if (lane + offset < 32) {
        a = xgb_dadd(a, x);
        b = xgb_dadd(b, y);
      }
    }
    if (lane == 0) sums[simd] = ulong2(a.bits, b.bits);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
      a = {lane < 8 ? sums[lane].x : 0};
      b = {lane < 8 ? sums[lane].y : 0};
      for (uint offset = 4; offset; offset >>= 1) {
        XGBDouble x = xgb_shuffle_down_double(a, offset), y = xgb_shuffle_down_double(b, offset);
        if (lane + offset < 8) {
          a = xgb_dadd(a, x);
          b = xgb_dadd(b, y);
        }
      }
      if (lane == 0) result = ulong2(a.bits, b.bits);
    }
  }
  return result;
}

struct GradientSumParams {
  ulong rows, stride0, stride1;
  uint targets, blocks;
};
kernel void xgb_gradient_sum_reduce(device const uint2* gradients [[buffer(0)]],
                                    device ulong2* partial [[buffer(1)]],
                                    constant GradientSumParams& p [[buffer(2)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong row = ulong(group.x) * 256 + tid;
  uint2 value = row < p.rows ? gradients[row * p.stride0 + ulong(group.y) * p.stride1] : uint2(0);
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total =
      xgb_reduce_double_pair(xgb_dfrom_float(as_type<float>(value.x)),
                             xgb_dfrom_float(as_type<float>(value.y)), bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[ulong(group.y) * p.blocks + group.x] = total;
}
kernel void xgb_gradient_sum_finish(device const ulong2* partial [[buffer(0)]],
                                    device ulong2* output [[buffer(1)]],
                                    constant GradientSumParams& p [[buffer(2)]],
                                    uint target [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]]) {
  XGBDouble g{0}, h{0};
  for (uint block = tid; block < p.blocks; block += 256) {
    ulong2 value = partial[ulong(target) * p.blocks + block];
    g = xgb_dadd(g, {value.x});
    h = xgb_dadd(h, {value.y});
  }
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total = xgb_reduce_double_pair(g, h, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) output[target] = total;
}
kernel void xgb_stump_weight(device const ulong2* sums [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
  ulong2 value = sums[tid];
  XGBDouble h{value.y}, epsilon = xgb_dfrom_float(1.0e-6f);
  if (xgb_dlt(h, epsilon)) h = epsilon;
  output[tid] = xgb_dto_float(xgb_ddiv(xgb_dneg({value.x}), h));
}
// Preserve the host's left-to-right order for the first feature's histogram.
// This small root reduction must agree with split scans' missing-value checks.
kernel void xgb_histogram_sum(device const ulong2* histogram [[buffer(0)]],
                              device ulong2* output [[buffer(1)]],
                              constant ulong2& range [[buffer(2)]]) {
  XGBDouble g{0}, h{0};
  for (ulong bin = range.x; bin < range.x + range.y; ++bin) {
    ulong2 value = histogram[bin];
    g = xgb_dadd(g, {value.x});
    h = xgb_dadd(h, {value.y});
  }
  output[0] = ulong2(g.bits, h.bits);
}

kernel void xgb_residual_scale_reduce(
    device const float* predictions [[buffer(0)]], device const float* labels [[buffer(1)]],
    device const float* weights [[buffer(2)]], device ulong2* partial [[buffer(3)]],
    constant AdaptiveParams& p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong row = ulong(group.x) * 256 + tid;
  uint target = group.y;
  XGBDouble root{0}, weight{0};
  if (row < p.rows) {
    float residual = predictions[row * p.targets + target] -
                     labels[row * p.label_targets + (p.label_targets == 1 ? 0 : target)];
    float w = p.weighted ? weights[row] : 1.0f;
    root = xgb_dfrom_float(w * sqrt(abs(residual)));
    weight = xgb_dfrom_float(w);
  }
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total = xgb_reduce_double_pair(root, weight, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[ulong(group.x) * p.targets + target] = total;
}
kernel void xgb_residual_scale_finish(device const ulong2* partial [[buffer(0)]],
                                      device float* scales [[buffer(1)]],
                                      constant AdaptiveParams& p [[buffer(2)]],
                                      uint target [[thread_position_in_grid]]) {
  XGBDouble root{0}, weight{0};
  for (uint block = 0; block < p.blocks; ++block) {
    ulong2 value = partial[ulong(block) * p.targets + target];
    root = xgb_dadd(root, {value.x});
    weight = xgb_dadd(weight, {value.y});
  }
  // common::CloseTo uses the double literal 1e-6, rather than an FP32 threshold.
  if (xgb_dlt({weight.bits & ~kDoubleSign}, {0x3eb0c6f7a0b5ed8dul}))
    scales[target] = 0.0f;
  else {
    XGBDouble mean = xgb_ddiv(root, weight);
    scales[target] = xgb_dto_float(xgb_dmul(mean, mean));
  }
}
inline float xgb_hypot(float x, float y) {
  float hi = max(abs(x), abs(y)), lo = min(abs(x), abs(y));
  if (hi == 0.0f || isinf(hi)) return hi;
  float ratio = lo / hi;
  return hi * sqrt(1.0f + ratio * ratio);
}
kernel void xgb_adaptive_objective(
    device const float* predictions [[buffer(0)]], device const float* labels [[buffer(1)]],
    device const float* weights [[buffer(2)]], device const float* scales [[buffer(3)]],
    device const float* alpha [[buffer(4)]], device float2* output [[buffer(5)]],
    constant AdaptiveParams& p [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  ulong row = tid / p.targets;
  uint target = tid % p.targets;
  float residual =
      predictions[tid] - labels[row * p.label_targets + (p.label_targets == 1 ? 0 : target)];
  float scale = scales[target], w = p.weighted ? weights[row] : 1.0f;
  if (p.kind == 0) {
    float norm = xgb_hypot(scale, residual), curvature = norm > 0.0f ? scale / norm : 1.0f;
    output[tid] = float2(w * residual * curvature, w * curvature);
  } else {
    if (!(scale > 0.0f) || w == 0.0f) {
      output[tid] = float2(0);
      return;
    }
    float x = residual / (0.04f * scale), t = tanh(x);
    float ratio = x == 0.0f ? 1.0f : t / x;
    ratio = max(ratio, 3e-4f);
    float gradient = 0.5f * scale * (t + 1.0f - 2.0f * alpha[target]);
    float hessian = (0.5f / 0.04f) * ratio;
    output[tid] = float2(w * gradient, w * hessian);
  }
}

// Metal arithmetic can flush FP32 subnormals. Preserve CPU rounding only for
// the rare underflow cases; ordinary metric values retain native FP32 math.
inline float xgb_fmul_preserve(float a, float b) {
  float value = a * b;
  uint aa = as_type<uint>(a) & 0x7fffffffu, bb = as_type<uint>(b) & 0x7fffffffu;
  if ((aa && aa < 0x800000u) || (bb && bb < 0x800000u) ||
      (!(as_type<uint>(value) & 0x7fffffffu) && aa && bb))
    return xgb_dto_float(xgb_dmul(xgb_dfrom_float(a), xgb_dfrom_float(b)));
  return value;
}
inline float xgb_fsub_preserve(float a, float b) {
  float value = a - b;
  uint aa = as_type<uint>(a) & 0x7fffffffu, bb = as_type<uint>(b) & 0x7fffffffu;
  if ((aa && aa < 0x800000u) || (bb && bb < 0x800000u) ||
      (!(as_type<uint>(value) & 0x7fffffffu) && as_type<uint>(a) != as_type<uint>(b)))
    return xgb_dto_float(xgb_dsub(xgb_dfrom_float(a), xgb_dfrom_float(b)));
  return value;
}
struct QuantileMetricParams {
  ulong count;
  uint targets, levels, weighted, expectile;
};
kernel void xgb_quantile_metric(
    device const float* predictions [[buffer(0)]], device const float* labels [[buffer(1)]],
    device const float* weights [[buffer(2)]], device const float* alpha [[buffer(3)]],
    device ulong2* partial [[buffer(4)]], constant QuantileMetricParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  XGBDouble loss{0}, weight{0};
  if (ulong(tid) < p.count) {
    uint row = tid / (p.targets * p.levels), level = (tid / p.targets) % p.levels,
         target = tid % p.targets;
    float prediction = predictions[tid], label = labels[ulong(row) * p.targets + target];
    float a = alpha[level], w = p.weighted ? weights[row] : 1.0f, value;
    if (p.expectile) {
      float diff = xgb_fsub_preserve(prediction, label);
      float scale = xgb_dlt(xgb_dfrom_float(diff), {0}) ? a : 1.0f - a;
      value = xgb_fmul_preserve(xgb_fmul_preserve(xgb_fmul_preserve(scale, diff), diff), w);
    } else {
      float diff = xgb_fsub_preserve(label, prediction);
      float sign = xgb_dlt(xgb_dfrom_float(diff), {0}) ? 0.0f : 1.0f;
      float left = xgb_fmul_preserve(xgb_fmul_preserve(a, sign), diff);
      float right = xgb_fmul_preserve(xgb_fmul_preserve(1.0f - a, 1.0f - sign), diff);
      value = xgb_fmul_preserve(xgb_fsub_preserve(left, right), w);
    }
    loss = xgb_dfrom_float(value);
    weight = xgb_dfrom_float(w);
  }
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total = xgb_reduce_double_pair(loss, weight, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[group] = total;
}

struct SamplingParams {
  ulong rows, stride0, stride1, seed;
  uint targets, threads;
  float probability;
  uint padding;
};
inline ulong xgb_lcg_skip(ulong index, ulong seed) {
  ulong base = 16807, mask = 0x7ffffffffffffffful;
  while (index) {
    if (index & 1) seed = (seed * base) & mask;
    base = (base * base) & mask;
    index >>= 1;
  }
  return seed;
}
kernel void xgb_uniform_sample(device float2* gradients [[buffer(0)]],
                               constant SamplingParams& p [[buffer(1)]],
                               uint row [[thread_position_in_grid]]) {
  ulong seed = p.seed & 0x7ffffffffffffffful, index = ulong(row) + 1;
  // The CPU creates an LCG at each OpenMP partition. Its zero seed is replaced
  // with one by std::linear_congruential_engine; preserve this rare edge case.
  if (seed == 0) {
    ulong chunk = p.rows / p.threads;
    if (chunk) index = ulong(row) - min(ulong(row) / chunk, ulong(p.threads - 1)) * chunk + 1;
    seed = 1;
  }
  ulong state = xgb_lcg_skip(index, seed);
  // libc++ generate_canonical<double> takes one draw, subtracts engine.min()
  // (one), rounds to double, then divides by the range rounded to 2^63.
  XGBDouble random{xgb_scaled_integer(state - 1, -63)};
  if (!xgb_dlt(random, xgb_dfrom_float(p.probability)))
    for (uint target = 0; target < p.targets; ++target)
      gradients[ulong(row) * p.stride0 + ulong(target) * p.stride1] = float2(0);
}

// IEEE comparisons via ordered integer bits preserve subnormal cut boundaries
// on hardware that flushes subnormal FP32 arithmetic. Signed zeros compare equal.
inline bool xgb_float_less(uint a, uint b) {
  if ((a & 0x7fffffffu) > 0x7f800000u || (b & 0x7fffffffu) > 0x7f800000u) return false;
  if (((a | b) & 0x7fffffffu) == 0) return false;
  uint ka = a & 0x80000000u ? ~a : a ^ 0x80000000u;
  uint kb = b & 0x80000000u ? ~b : b ^ 0x80000000u;
  return ka < kb;
}
struct BinningParams {
  ulong count;
  uint features, width, dense, padding;
};
kernel void xgb_bin_entries(
    device const uint2* entries [[buffer(0)]], device const uint* offsets [[buffer(1)]],
    device const uint* cuts [[buffer(2)]], device const uchar* categorical [[buffer(3)]],
    device uchar* output [[buffer(4)]], device atomic_uint* counts [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]], constant BinningParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.count) return;
  uint2 entry = entries[tid];
  uint feature = entry.x, value = entry.y;
  if ((value & 0x7fffffffu) == 0x7f800000u) {
    atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  if (feature >= p.features || offsets[feature] == offsets[feature + 1]) {
    atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    return;
  }
  uint begin = offsets[feature], end = offsets[feature + 1];
  bool cat = categorical[feature] != 0;
  if (cat) value = as_type<uint>(float(int(as_type<float>(value))));
  uint lo = begin, hi = end;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    bool advance = cat ? xgb_float_less(cuts[mid], value) : !xgb_float_less(value, cuts[mid]);
    if (advance)
      lo = mid + 1;
    else
      hi = mid;
  }
  uint global = min(lo, end - 1);
  uint bin = p.dense ? global - begin : global;
  if (p.width == 1)
    output[tid] = uchar(bin);
  else if (p.width == 2)
    reinterpret_cast<device ushort*>(output)[tid] = ushort(bin);
  else
    reinterpret_cast<device uint*>(output)[tid] = bin;
  atomic_fetch_add_explicit(counts + global, 1u, memory_order_relaxed);
}

struct MeanParams {
  GradientSumParams shape;
  uint weighted, float32_terms;
};
kernel void xgb_mean_weight_reduce(device const uint* weights [[buffer(0)]],
                                   device ulong2* partial [[buffer(1)]],
                                   constant GradientSumParams& p [[buffer(2)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong row = ulong(group) * 256 + tid;
  XGBDouble value = row < p.rows ? xgb_dfrom_float(as_type<float>(weights[row])) : XGBDouble{0};
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total = xgb_reduce_double_pair(value, {0}, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[group] = total;
}
kernel void xgb_mean_reduce(
    device const uint* input [[buffer(0)]], device const uint* weights [[buffer(1)]],
    device const ulong2* weight_sum [[buffer(2)]], device ulong2* partial [[buffer(3)]],
    constant MeanParams& p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong row = ulong(group.x) * 256 + tid;
  XGBDouble denominator = p.weighted
                              ? XGBDouble{weight_sum[0].x}
                              : (p.float32_terms ? xgb_dfrom_float(float(p.shape.rows))
                                                 : XGBDouble{xgb_scaled_integer(p.shape.rows, 0)});
  XGBDouble value{0};
  if (row < p.shape.rows && xgb_dlt({0}, denominator)) {
    ulong index = row * p.shape.stride0 + ulong(group.y) * p.shape.stride1;
    value = xgb_ddiv(xgb_dfrom_float(as_type<float>(input[index])), denominator);
    if (p.weighted) value = xgb_dmul(value, xgb_dfrom_float(as_type<float>(weights[row])));
    if (p.float32_terms) value = xgb_dfrom_float(xgb_dto_float(value));
  }
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total = xgb_reduce_double_pair(value, {0}, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[ulong(group.y) * p.shape.blocks + group.x] = total;
}
kernel void xgb_mean_result(device const ulong2* sums [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
  output[tid] = xgb_dto_float({sums[tid].x});
}

// Stable 4-bit radix sorting of column-wise (ordered-float-key, original-row)
// records. Counts are laid out target/bucket/block, so a single integer prefix
// gives globally disjoint destinations for every target and bucket.
struct RadixParams {
  uint rows, targets, blocks, shift;
};
kernel void xgb_sort_columns_init(device const uint* values [[buffer(0)]],
                                  device uint2* records [[buffer(1)]],
                                  constant GradientSumParams& p [[buffer(2)]],
                                  uint2 tid [[thread_position_in_grid]]) {
  uint bits = values[ulong(tid.x) * p.stride0 + ulong(tid.y) * p.stride1];
  uint key =
      (bits & 0x7fffffffu) == 0 ? 0x80000000u : (bits & 0x80000000u ? ~bits : bits ^ 0x80000000u);
  records[ulong(tid.y) * p.rows + tid.x] = uint2(key, tid.x);
}
kernel void xgb_radix_rank(device const uint2* records [[buffer(0)]],
                           device uint* ranks [[buffer(1)]], device uint* counts [[buffer(2)]],
                           constant RadixParams& p [[buffer(3)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong row = ulong(group.x) * 256 + tid, index = ulong(group.y) * p.rows + row;
  bool valid = row < p.rows;
  uint bucket = valid ? (records[index].x >> p.shift) & 15 : 0;
  uint rank = 0;
  threadgroup uint hist[128];
  for (uint b = 0; b < 16; ++b) {
    uint match = uint(valid && bucket == b);
    uint prefix = simd_prefix_exclusive_sum(match);
    if (bucket == b) rank = prefix;
    if (lane == 31) hist[simd * 16 + b] = prefix + match;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (valid) {
    for (uint prior = 0; prior < simd; ++prior) rank += hist[prior * 16 + bucket];
    ranks[index] = rank;
  }
  if (tid < 16) {
    uint sum = 0;
    for (uint i = 0; i < 8; ++i) sum += hist[i * 16 + tid];
    counts[(ulong(group.y) * 16 + tid) * p.blocks + group.x] = sum;
  }
}
kernel void xgb_prefix_u32(device const uint* input [[buffer(0)]],
                           device uint* output [[buffer(1)]], device uint* totals [[buffer(2)]],
                           constant uint& n [[buffer(3)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong index = ulong(group) * 256 + tid;
  uint value = index < n ? input[index] : 0;
  uint prefix = simd_prefix_exclusive_sum(value);
  threadgroup uint sums[8];
  if (lane == 31) sums[simd] = prefix + value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < simd; ++i) prefix += sums[i];
  if (index < n) output[index] = prefix;
  if (tid == 255) totals[group] = prefix + value;
}
kernel void xgb_prefix_add_u32(device uint* output [[buffer(0)]],
                               device const uint* block_prefix [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {
  output[tid] += block_prefix[tid / 256];
}
kernel void xgb_radix_scatter(device const uint2* input [[buffer(0)]],
                              device const uint* ranks [[buffer(1)]],
                              device const uint* prefix [[buffer(2)]],
                              device uint2* output [[buffer(3)]],
                              constant RadixParams& p [[buffer(4)]],
                              uint2 tid [[thread_position_in_grid]]) {
  ulong index = ulong(tid.y) * p.rows + tid.x;
  uint2 record = input[index];
  uint bucket = (record.x >> p.shift) & 15;
  ulong count_index = (ulong(tid.y) * 16 + bucket) * p.blocks + tid.x / 256;
  output[prefix[count_index] + ranks[index]] = record;
}
inline float xgb_fadd_preserve(float a, float b) {
  uint aa = as_type<uint>(a) & 0x7fffffffu, bb = as_type<uint>(b) & 0x7fffffffu;
  float result = a + b;
  uint rr = as_type<uint>(result) & 0x7fffffffu;
  if ((aa && aa < 0x00800000u) || (bb && bb < 0x00800000u) || (rr < 0x00800000u && aa && bb))
    return xgb_dto_float(xgb_dadd(xgb_dfrom_float(a), xgb_dfrom_float(b)));
  return result;
}
// WeightedQuantile uses a *sequential FP32* cumulative sum on CPU. Parallel
// floating scans can change the selected observation; retain that order while
// gathering and storing each 256-value tile cooperatively.
kernel void xgb_quantile_weight_cdf(device const uint2* records [[buffer(0)]],
                                    device const uint* weights [[buffer(1)]],
                                    device uint* cdf [[buffer(2)]],
                                    constant uint& rows [[buffer(3)]],
                                    uint target [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tile[256];
  threadgroup uint slow_group[8];
  threadgroup uint accumulated;
  if (tid == 0) accumulated = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong begin = 0; begin < rows; begin += 256) {
    ulong index = ulong(target) * rows + begin + tid;
    uint length = uint(min(256ul, ulong(rows) - begin));
    uint bits = tid < length ? weights[records[index].y] : 0;
    tile[tid] = as_type<float>(bits);
    uint magnitude = bits & 0x7fffffffu;
    bool slow = simd_any((magnitude && magnitude < 0x00800000u) ||
                         (magnitude && (bits & 0x80000000u)));
    if (tid % 32 == 0) slow_group[tid / 32] = slow;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint a = accumulated & 0x7fffffffu;
      bool preserve = (a && a < 0x00800000u) || (a && (accumulated & 0x80000000u));
      for (uint i = 0; i < 8; ++i) preserve |= bool(slow_group[i]);
      float sum = as_type<float>(accumulated);
      uint first = 0;
      if (begin == 0) {
        sum = tile[0];  // std::partial_sum copies its first input, including -0.
        first = 1;
      }
      // Normal, nonnegative inputs cannot underflow or cancel. Keeping this
      // branch outside the loop preserves sequential FP32 rounding while
      // avoiding subnormal checks on every addition in the common case.
      if (preserve) {
        for (uint i = first; i < length; ++i) {
          sum = xgb_fadd_preserve(sum, tile[i]);
          tile[i] = sum;
        }
      } else {
        for (uint i = first; i < length; ++i) {
          sum += tile[i];
          tile[i] = sum;
        }
      }
      accumulated = as_type<uint>(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < length) cdf[index] = as_type<uint>(tile[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
struct QuantileRequest {
  uint target, padding;
  ulong alpha;
};
struct QuantileParams {
  uint rows, targets, weighted, requests;
};
inline ulong xgb_positive_floor(XGBDouble value) {
  int exponent = int((value.bits >> 52) & 2047) - 1023;
  if (exponent < 0) return 0;
  ulong significand = (value.bits & kDoubleFrac) | (1ul << 52);
  return exponent < 52 ? significand >> (52 - exponent) : significand << (exponent - 52);
}
kernel void xgb_quantile_select(
    device const uint2* records [[buffer(0)]], device const uint* values [[buffer(1)]],
    device const uint* cdf [[buffer(2)]], device const QuantileRequest* requests [[buffer(3)]],
    device uint* output [[buffer(4)]], constant QuantileParams& p [[buffer(5)]],
    constant GradientSumParams& shape [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  QuantileRequest request = requests[tid];
  ulong base = ulong(request.target) * p.rows;
  XGBDouble alpha{request.alpha};
  uint index;
  if (p.weighted) {
    XGBDouble total = xgb_dfrom_float(as_type<float>(cdf[base + p.rows - 1]));
    uint threshold = as_type<uint>(xgb_dto_float(xgb_dmul(total, alpha)));
    uint lo = 0, hi = p.rows;
    while (lo < hi) {
      uint mid = lo + (hi - lo) / 2;
      if (xgb_float_less(cdf[base + mid], threshold))
        lo = mid + 1;
      else
        hi = mid;
    }
    index = min(lo, p.rows - 1);
  } else {
    XGBDouble one = xgb_dfrom_float(1.0f), n{xgb_scaled_integer(p.rows, 0)},
              n1{xgb_scaled_integer(ulong(p.rows) + 1, 0)};
    if (!xgb_dlt(xgb_ddiv(one, n1), alpha))
      index = 0;
    else if (!xgb_dlt(alpha, xgb_ddiv(n, n1)))
      index = p.rows - 1;
    else {
      XGBDouble x = xgb_dmul(alpha, n1);
      ulong k = xgb_positive_floor(x) - 1;
      XGBDouble d = xgb_dsub(xgb_dsub(x, one), {xgb_scaled_integer(k, 0)});
      uint row0 = records[base + k].y, row1 = records[base + k + 1].y;
      float v0 = as_type<float>(
          values[ulong(row0) * shape.stride0 + ulong(request.target) * shape.stride1]);
      float v1 = as_type<float>(
          values[ulong(row1) * shape.stride0 + ulong(request.target) * shape.stride1]);
      XGBDouble diff = xgb_dfrom_float(xgb_fsub_preserve(v1, v0));
      output[tid] = as_type<uint>(xgb_dto_float(xgb_dadd(xgb_dfrom_float(v0), xgb_dmul(d, diff))));
      return;
    }
  }
  output[tid] = values[ulong(records[base + index].y) * shape.stride0 +
                       ulong(request.target) * shape.stride1];
}

// Binary64 transcendental helpers for survival probability calculations. Metal
// has no native double arithmetic; each operator uses the integer implementation
// above. No reassociation or fused operations are used.
namespace xgb_survival {
using D = XGBDouble;
inline D operator+(D a, D b) { return xgb_dadd(a, b); }
inline D operator-(D a, D b) { return xgb_dsub(a, b); }
inline D operator-(D a) { return xgb_dneg(a); }
inline D operator*(D a, D b) { return xgb_dmul(a, b); }
inline D operator/(D a, D b) { return xgb_ddiv(a, b); }
inline bool operator<(D a, D b) { return xgb_dlt(a, b); }
inline bool operator>(D a, D b) { return xgb_dlt(b, a); }
inline D From(float a) { return xgb_dfrom_float(a); }
inline bool IsNaN(D a) { return (a.bits & ~kDoubleSign) > kDoubleInf; }
inline bool IsInf(D a) { return (a.bits & ~kDoubleSign) == kDoubleInf; }
inline D Abs(D a) { return {a.bits & ~kDoubleSign}; }
constant D zero{0}, one{0x3ff0000000000000ul}, two{0x4000000000000000ul},
    half_value{0x3fe0000000000000ul};
// Split ln(2), so integer multiplication of the high part is exact throughout
// the exp/log ranges. Polynomial coefficients are rounded binary64 constants.
constant D ln2_hi{0x3fe62e42fee00000ul}, ln2_lo{0x3dea39ef35793c76ul};
constant ulong exp_coeff[] = {
    0x3ff0000000000000ul,  // 1.0
    0x3ff0000000000000ul,  // 1.0
    0x3fe0000000000000ul,  // 0.5
    0x3fc5555555555555ul,  // 0.16666666666666666
    0x3fa5555555555555ul,  // 0.041666666666666664
    0x3f81111111111111ul,  // 0.008333333333333333
    0x3f56c16c16c16c17ul,  // 0.001388888888888889
    0x3f2a01a01a01a01aul,  // 0.0001984126984126984
    0x3efa01a01a01a01aul,  // 2.48015873015873e-05
    0x3ec71de3a556c734ul,  // 2.7557319223985893e-06
    0x3e927e4fb7789f5cul,  // 2.755731922398589e-07
    0x3e5ae64567f544e4ul,  // 2.505210838544172e-08
    0x3e21eed8eff8d898ul,  // 2.08767569878681e-09
    0x3de6124613a86d09ul,  // 1.6059043836821613e-10
    0x3da93974a8c07c9dul,  // 1.1470745597729725e-11
    0x3d6ae7f3e733b81ful,  // 7.647163731819816e-13
    0x3d2ae7f3e733b81ful,  // 4.779477332387385e-14
    0x3ce952c77030ad4aul,  // 2.8114572543455206e-15
    0x3ca6827863b97d97ul,  // 1.5619206968586225e-16
};
constant ulong log_coeff[] = {
    0x3ff0000000000000ul,  // 1.0
    0x3fd5555555555555ul,  // 0.3333333333333333
    0x3fc999999999999aul,  // 0.2
    0x3fc2492492492492ul,  // 0.14285714285714285
    0x3fbc71c71c71c71cul,  // 0.1111111111111111
    0x3fb745d1745d1746ul,  // 0.09090909090909091
    0x3fb3b13b13b13b14ul,  // 0.07692307692307693
    0x3fb1111111111111ul,  // 0.06666666666666667
    0x3fae1e1e1e1e1e1eul,  // 0.058823529411764705
    0x3faaf286bca1af28ul,  // 0.05263157894736842
    0x3fa8618618618618ul,  // 0.047619047619047616
    0x3fa642c8590b2164ul,  // 0.043478260869565216
    0x3fa47ae147ae147bul,  // 0.04
    0x3fa2f684bda12f68ul,  // 0.037037037037037035
};

inline D Polynomial(D x, constant ulong* c, uint n) {
  D result{c[n - 1]};
  for (uint i = n - 1; i; --i) result = D{c[i - 1]} + x * result;
  return result;
}
inline D Exp(D x) {
  if (IsNaN(x)) return x;
  if (x > D{0x40862e42fefa39eful}) return {kDoubleInf};  // overflow threshold
  if (x < D{0xc0874910d52d3052ul}) return zero;          // half of minimum subnormal
  // An approximate integer is sufficient: either neighbor leaves a small
  // reduced argument. Compute the remainder and polynomial in binary64.
  int k = int(rint(xgb_dto_float(x) * 1.4426950408889634f));
  D kd = From(float(k));
  D r = (x - kd * ln2_hi) - kd * ln2_lo;
  D value = Polynomial(r, exp_coeff, 19);
  if (k > 1023) return (value * two) * D{0x7fe0000000000000ul};
  if (k < -1022) return (value * D{ulong(k + 1077) << 52}) * D{0x3c90000000000000ul};
  return value * D{ulong(k + 1023) << 52};
}
inline D Log(D x) {
  ulong magnitude = x.bits & ~kDoubleSign;
  if (!magnitude) return {kDoubleSign | kDoubleInf};
  if (IsNaN(x)) return x;
  if (x.bits & kDoubleSign) return {kDoubleNaN};
  if (IsInf(x)) return x;
  int exponent;
  ulong significand;
  xgb_dunpack(x.bits, exponent, significand);
  int k = exponent - 1023;
  D m{0x3ff0000000000000ul | (significand & kDoubleFrac)};
  if (m > D{0x3ff6a09e667f3bcdul}) {
    m = m * half_value;
    ++k;
  }
  D t = (m - one) / (m + one);
  D reduced = two * t * Polynomial(t * t, log_coeff, 14);
  D kd = From(float(k));
  return kd * ln2_hi + (reduced + kd * ln2_lo);
}

/*
 * Error-function approximation adapted from OpenLibm src/s_erf.c (fdlibm).
 * https://github.com/JuliaMath/openlibm/blob/master/src/s_erf.c
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this software is freely
 * granted, provided that this notice is preserved.
 * Coefficients retain their original binary64 bits; arithmetic uses Metal
 * integer emulation. Floating-point exception flags are not exposed on Metal.
 */
constant ulong erf_pp[] = {
    0x3fc06eba8214db68ul,  // 0.12837916709551256
    0xbfd4cd7d691cb913ul,  // -0.3250421072470015
    0xbf9d2a51dbd7194ful,  // -0.02848174957559851
    0xbf77a291236668e4ul,  // -0.005770270296489442
    0xbef8ead6120016acul,  // -2.3763016656650163e-05
};
constant ulong erf_qq[] = {
    0x3ff0000000000000ul,  // 1
    0x3fd97779cddadc09ul,  // 0.39791722395915535
    0x3fb0a54c5536cebaul,  // 0.0650222499887673
    0x3f74d022c4d36b0ful,  // 0.005081306281875766
    0x3f215dc9221c1a10ul,  // 0.00013249473800432164
    0xbed09c4342a26120ul,  // -3.960228278775368e-06
};
constant ulong erf_pa[] = {
    0xbf6359b8bef77538ul,  // -0.0023621185607526594
    0x3fda8d00ad92b34dul,  // 0.41485611868374833
    0xbfd7d240fbb8c3f1ul,  // -0.3722078760357013
    0x3fd45fca805120e4ul,  // 0.31834661990116175
    0xbfbc63983d3e28ecul,  // -0.11089469428239668
    0x3fa22a36599795ebul,  // 0.035478304325618236
    0xbf61bf380a96073ful,  // -0.002166375594868791
};
constant ulong erf_qa[] = {
    0x3ff0000000000000ul,  // 1
    0x3fbb3e6618eee323ul,  // 0.10642088040084423
    0x3fe14af092eb6f33ul,  // 0.540397917702171
    0x3fb2635cd99fe9a7ul,  // 0.07182865441419627
    0x3fc02660e763351ful,  // 0.12617121980876164
    0x3f8bedc26b51dd1cul,  // 0.01363708391202905
    0x3f888b545735151dul,  // 0.011984499846799107
};
constant ulong erf_ra[] = {
    0xbf843412600d6435ul,  // -0.009864944034847148
    0xbfe63416e4ba7360ul,  // -0.6938585727071818
    0xc0251e0441b0e726ul,  // -10.558626225323291
    0xc04f300ae4cba38dul,  // -62.375332450326006
    0xc0644cb184282266ul,  // -162.39666946257347
    0xc067135cebccabb2ul,  // -184.60509290671104
    0xc054526557e4d2f2ul,  // -81.2874355063066
    0xc023a0efc69ac25cul,  // -9.814329344169145
};
constant ulong erf_sa[] = {
    0x3ff0000000000000ul,  // 1
    0x4033a6b9bd707687ul,  // 19.651271667439257
    0x4061350c526ae721ul,  // 137.65775414351904
    0x407b290dd58a1a71ul,  // 434.56587747522923
    0x40842b1921ec2868ul,  // 645.3872717332679
    0x407ad02157700314ul,  // 429.00814002756783
    0x405b28a3ee48ae2cul,  // 108.63500554177944
    0x401a47ef8e484a93ul,  // 6.570249770319282
    0xbfaeeff2ee749a62ul,  // -0.0604244152148581
};
constant ulong erf_rb[] = {
    0xbf84341239e86f4aul,  // -0.0098649429247001
    0xbfe993ba70c285deul,  // -0.799283237680523
    0xc031c209555f995aul,  // -17.757954917754752
    0xc064145d43c5ed98ul,  // -160.63638485582192
    0xc083ec881375f228ul,  // -637.5664433683896
    0xc09004616a2e5992ul,  // -1025.0951316110772
    0xc07e384e9bdc383ful,  // -483.5191916086514
};
constant ulong erf_sb[] = {
    0x3ff0000000000000ul,  // 1
    0x403e568b261d5190ul,  // 30.33806074348246
    0x40745cae221b9f0aul,  // 325.7925129965739
    0x409802eb189d5118ul,  // 1536.729586084437
    0x40a8ffb7688c246aul,  // 3199.8582195085955
    0x40a3f219cedf3be6ul,  // 2553.0504064331644
    0x407da874e79fe763ul,  // 474.52854120695537
    0xc03670e242712d62ul,  // -22.44095244658582
};

inline D Erf(D x) {
  uint high = uint((x.bits & ~kDoubleSign) >> 32);
  bool negative = bool(x.bits & kDoubleSign);
  if (IsNaN(x)) return x;
  if (high >= 0x40180000u) return negative ? -one : one;  // |x| >= 6, including infinity
  if (high < 0x3feb0000u) {
    if (high < 0x3e300000u) {
      if (high < 0x00800000u) return From(0.125f) * (From(8) * x + D{0x3ff06eba8214db69ul} * x);
      return x + D{0x3fc06eba8214db69ul} * x;
    }
    D z = x * x;
    return x + x * (Polynomial(z, erf_pp, 5) / Polynomial(z, erf_qq, 6));
  }
  D a = Abs(x);
  if (high < 0x3ff40000u) {
    D t = a - one;
    D value = D{0x3feb0ac160000000ul} + Polynomial(t, erf_pa, 7) / Polynomial(t, erf_qa, 7);
    return negative ? -value : value;
  }
  D t = one / (a * a);
  D ratio = high < 0x4006db6eu ? Polynomial(t, erf_ra, 8) / Polynomial(t, erf_sa, 9)
                               : Polynomial(t, erf_rb, 7) / Polynomial(t, erf_sb, 8);
  D z{a.bits & 0xffffffff00000000ul};
  D r = Exp(-z * z - From(0.5625f)) * Exp((z - a) * (z + a) + ratio);
  return negative ? r / a - one : one - r / a;
}
}  // namespace xgb_survival

namespace xgb_survival {
struct Distribution {
  D pdf, cdf, grad, hess;
};
inline Distribution Probability(uint kind, D z) {
  if (kind == 0) {
    D pdf = Exp(-z * z / two) / D{0x40040d931ff62705ul};  // sqrt(2*pi)
    D cdf = half_value * (one + Erf(z / D{0x3ff6a09e667f3bcdul}));
    return {pdf, cdf, -z * pdf, (z * z - one) * pdf};
  }
  D w = Exp(z), ww = w * w;
  bool infinite = IsInf(w), squared_infinite = IsInf(ww);
  if (kind == 1) {
    D den = one + w;
    D pdf = infinite || squared_infinite ? zero : w / (den * den);
    D cdf = infinite ? one : w / den;
    D grad = infinite ? zero : pdf * (one - w) / den;
    D hess = infinite || squared_infinite ? zero : pdf * (ww - From(4) * w + one) / (den * den);
    return {pdf, cdf, grad, hess};
  }
  D exp_minus_w = Exp(-w);
  D pdf = infinite ? zero : w * exp_minus_w;
  return {pdf, one - exp_minus_w, infinite ? zero : (one - w) * pdf,
          infinite || squared_infinite ? zero : (ww - From(3) * w + one) * pdf};
}
// Censoring kinds match common::CensoringType: uncensored/right/left/interval.
inline D LimitGradient(uint distribution, uint censoring, bool positive, D sigma) {
  if ((censoring == 1 && !positive) || (censoring == 2 && positive)) return zero;
  if (distribution == 0) return From(positive ? -15 : 15);
  if (distribution == 1) return (positive ? -one : one) / sigma;
  return positive ? From(-15) : one / sigma;
}
constant D min_hessian{0x3c9cd2b297d889bcul}, epsilon{0x3d719799812dea11ul};
inline D LimitHessian(uint distribution, uint censoring, bool positive, D sigma) {
  if (distribution == 1) return min_hessian;
  if (distribution == 2) return censoring != 2 && positive ? From(15) : min_hessian;
  if ((censoring == 1 && !positive) || (censoring == 2 && positive)) return min_hessian;
  return one / (sigma * sigma);
}
inline D Clip(D value, D lower, D upper) {
  return value < lower ? lower : value > upper ? upper : value;
}
struct AFTResult {
  D gradient, hessian, likelihood;
};
inline AFTResult Evaluate(uint kind, D lower, D upper, D prediction, D sigma) {
  D gn, gd, hn, hd, likelihood;
  uint censoring = 0;
  bool positive;
  if (xgb_deq(lower, upper)) {
    D z = (Log(lower) - prediction) / sigma;
    auto d = Probability(kind, z);
    gn = d.grad;
    gd = sigma * d.pdf;
    hn = -(d.pdf * d.hess - d.grad * d.grad);
    hd = sigma * sigma * d.pdf * d.pdf;
    likelihood = d.pdf / (sigma * lower);
    positive = z > zero;
  } else {
    censoring = 3;
    D zu = zero, zl = zero;
    Distribution u{zero, one, zero, zero}, l{zero, zero, zero, zero};
    if (IsInf(upper))
      censoring = 1;
    else {
      zu = (Log(upper) - prediction) / sigma;
      u = Probability(kind, zu);
    }
    if (!(lower > zero))
      censoring = 2;
    else {
      zl = (Log(lower) - prediction) / sigma;
      l = Probability(kind, zl);
    }
    D cdf = u.cdf - l.cdf, pdf = u.pdf - l.pdf, grad = u.grad - l.grad;
    gn = pdf;
    gd = sigma * cdf;
    hn = -(cdf * grad - pdf * pdf);
    hd = gd * gd;
    likelihood = cdf;
    positive = zu > zero || zl > zero;
  }
  D g = gn / gd, h = hn / hd;
  if (gd < epsilon && (IsNaN(g) || IsInf(g))) g = LimitGradient(kind, censoring, positive, sigma);
  if (hd < epsilon && (IsNaN(h) || IsInf(h))) h = LimitHessian(kind, censoring, positive, sigma);
  return {Clip(g, From(-15), From(15)), Clip(h, min_hessian, From(15)), likelihood};
}
inline D Loss(uint kind, D lower, D upper, D prediction, D sigma) {
  // Keep loss evaluation separate so metrics need no PDF derivatives.
  D likelihood;
  if (xgb_deq(lower, upper)) {
    D z = (Log(lower) - prediction) / sigma;
    likelihood = Probability(kind, z).pdf / (sigma * lower);
  } else {
    D u = IsInf(upper) ? one : Probability(kind, (Log(upper) - prediction) / sigma).cdf;
    D l = lower > zero ? Probability(kind, (Log(lower) - prediction) / sigma).cdf : zero;
    likelihood = u - l;
  }
  if (IsNaN(likelihood) || likelihood < epsilon) likelihood = epsilon;  // CPU fmax
  return -Log(likelihood);
}
}  // namespace xgb_survival
struct SurvivalParams {
  ulong rows;
  uint distribution, weighted, accuracy;
  float scale;
};
kernel void xgb_aft_gradient(
    device const uint* predictions [[buffer(0)]], device const uint* lower [[buffer(1)]],
    device const uint* upper [[buffer(2)]], device const uint* weights [[buffer(3)]],
    device uint2* output [[buffer(4)]], constant SurvivalParams& p [[buffer(5)]],
    device atomic_uint* correction_count [[buffer(6)]], device uint* correction_rows [[buffer(7)]],
    uint row [[thread_position_in_grid]]) {
  if (ulong(row) >= p.rows) return;
  using namespace xgb_survival;
  auto value =
      Evaluate(p.distribution, From(as_type<float>(lower[row])), From(as_type<float>(upper[row])),
               From(as_type<float>(predictions[row])), From(p.scale));
  // CPU libm and GPU approximations can differ by a few binary64 ULP. In
  // very small CDF differences that changes gradient/Hessian clipping. Mark
  // only these rows for the original host formula; do not hide the correction.
  if (!xgb_deq(From(as_type<float>(lower[row])), From(as_type<float>(upper[row]))) &&
      !(value.likelihood > From(1e-5f))) {
    uint index = atomic_fetch_add_explicit(correction_count, 1u, memory_order_relaxed);
    correction_rows[index] = row;
  }
  float w = p.weighted ? as_type<float>(weights[row]) : 1.0f;
  output[row] = uint2(as_type<uint>(xgb_fmul_preserve(xgb_dto_float(value.gradient), w)),
                      as_type<uint>(xgb_fmul_preserve(xgb_dto_float(value.hessian), w)));
}
kernel void xgb_aft_metric(
    device const uint* predictions [[buffer(0)]], device const uint* lower [[buffer(1)]],
    device const uint* upper [[buffer(2)]], device const uint* weights [[buffer(3)]],
    device ulong2* partial [[buffer(4)]], constant SurvivalParams& p [[buffer(5)]],
    uint row [[thread_position_in_grid]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  using namespace xgb_survival;
  D w = zero, loss = zero;
  if (ulong(row) < p.rows) {
    D l = From(as_type<float>(lower[row])), u = From(as_type<float>(upper[row])),
      prediction = From(as_type<float>(predictions[row]));
    w = p.weighted ? From(as_type<float>(weights[row])) : one;
    if (p.accuracy) {
      D v = Exp(prediction);
      // Ordered comparisons: NaN must not count as inside the interval.
      loss = (!IsNaN(v) && !IsNaN(l) && !IsNaN(u) && !(v < l) && !(u < v)) ? one : zero;
    } else
      loss = Loss(p.distribution, l, u, prediction, From(p.scale));
  }
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 result = xgb_reduce_double_pair(loss * w, w, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[group] = result;
}
kernel void xgb_aft_transform(device const uint* input [[buffer(0)]],
                              device uint* output [[buffer(1)]],
                              constant uint& to_margin [[buffer(2)]],
                              uint row [[thread_position_in_grid]]) {
  auto value = xgb_survival::From(as_type<float>(input[row]));
  output[row] =
      as_type<uint>(xgb_dto_float(to_margin ? xgb_survival::Log(value) : xgb_survival::Exp(value)));
}

// ROC-AUC: stable descending scores, inclusive binary64 (FP, TP) prefixes,
// and one trapezoid per equal-score group. No per-row host sorting or scan.
kernel void xgb_roc_sort_init(device const uint* values [[buffer(0)]],
                              device uint2* records [[buffer(1)]],
                              constant GradientSumParams& p [[buffer(2)]],
                              uint2 tid [[thread_position_in_grid]]) {
  uint bits = values[ulong(tid.x) * p.stride0 + ulong(tid.y) * p.stride1];
  uint key = (bits & 0x7fffffffu) == 0 ? 0x80000000u
              : (bits & 0x80000000u ? ~bits : bits ^ 0x80000000u);
  records[ulong(tid.y) * p.rows + tid.x] = uint2(~key, tid.x);
}
struct ROCParams {
  ulong rows, label_stride0, label_stride1;
  uint targets, blocks, weighted, multiclass;
};
kernel void xgb_roc_counts(device const uint2* sorted [[buffer(0)]],
                           device const uint* labels [[buffer(1)]],
                           device const uint* weights [[buffer(2)]],
                           device ulong2* counts [[buffer(3)]],
                           constant ROCParams& p [[buffer(4)]],
                           uint2 tid [[thread_position_in_grid]]) {
  ulong idx = ulong(tid.y) * p.rows + tid.x;
  uint row = sorted[idx].y;
  float label = as_type<float>(labels[ulong(row) * p.label_stride0 +
                                     (p.multiclass ? 0 : ulong(tid.y) * p.label_stride1)]);
  if (p.multiclass) {
    uint bits = as_type<uint>(label);
    // Match CPU equality without flushing tiny nonzero labels to class zero.
    label = float((bits & 0x7fffffffu) == 0 ? tid.y == 0 : bits == as_type<uint>(float(tid.y)));
  }
  float w = p.weighted ? as_type<float>(weights[row]) : 1.0f;
  // Upstream promotes only the first negative term before subtraction/product;
  // all subsequent terms and all positive terms round their products to FP32.
  XGBDouble fp = tid.x == 0
      ? xgb_dmul(xgb_dsub(xgb_dfrom_float(1.0f), xgb_dfrom_float(label)), xgb_dfrom_float(w))
      : xgb_dfrom_float(xgb_fmul_preserve(xgb_fsub_preserve(1.0f, label), w));
  XGBDouble tp = xgb_dfrom_float(xgb_fmul_preserve(label, w));
  counts[idx] = ulong2(fp.bits, tp.bits);
}
inline ulong2 xgb_dpair_add(ulong2 a, ulong2 b) {
  return ulong2(xgb_dadd({a.x}, {b.x}).bits, xgb_dadd({a.y}, {b.y}).bits);
}
inline ulong2 xgb_dpair_shuffle_up(ulong2 a, uint delta) {
  uint4 words = as_type<uint4>(a);
  return as_type<ulong2>(uint4(simd_shuffle_up(words.x, delta), simd_shuffle_up(words.y, delta),
                              simd_shuffle_up(words.z, delta), simd_shuffle_up(words.w, delta)));
}
kernel void xgb_prefix_double_pair(device const ulong2* input [[buffer(0)]],
                                   device ulong2* output [[buffer(1)]],
                                   device ulong2* totals [[buffer(2)]],
                                   constant uint2& shape [[buffer(3)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
  uint row = group.x * 256 + tid;
  ulong base = ulong(group.y) * shape.x;
  ulong2 value = row < shape.x ? input[base + row] : ulong2(0);
  for (uint d = 1; d < 32; d <<= 1) {
    ulong2 prior = xgb_dpair_shuffle_up(value, d);
    if (lane >= d) value = xgb_dpair_add(prior, value);
  }
  threadgroup ulong2 subgroup[8];
  if (lane == 31) subgroup[simd] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    ulong2 upper = lane < 8 ? subgroup[lane] : ulong2(0);
    for (uint d = 1; d < 8; d <<= 1) {
      ulong2 prior = xgb_dpair_shuffle_up(upper, d);
      if (lane >= d) upper = xgb_dpair_add(prior, upper);
    }
    if (lane < 8) subgroup[lane] = upper;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd) value = xgb_dpair_add(subgroup[simd - 1], value);
  if (row < shape.x) output[base + row] = value;
  if (tid == 255) totals[ulong(group.y) * ((ulong(shape.x) + 255) / 256) + group.x] = value;
}
kernel void xgb_prefix_double_pair_add(device ulong2* values [[buffer(0)]],
                                       device const ulong2* prefix [[buffer(1)]],
                                       constant uint2& shape [[buffer(2)]],
                                       uint2 tid [[thread_position_in_grid]]) {
  uint block = tid.x / 256;
  if (!block) return;
  ulong idx = ulong(tid.y) * shape.x + tid.x;
  ulong offset = ulong(tid.y) * ((ulong(shape.x) + 255) / 256) + block - 1;
  values[idx] = xgb_dpair_add(prefix[offset], values[idx]);
}
kernel void xgb_roc_area(device const uint2* sorted [[buffer(0)]],
                         device const ulong2* prefix [[buffer(1)]],
                         device ulong2* partial [[buffer(2)]],
                         constant ROCParams& p [[buffer(3)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong row = ulong(group.x) * 256 + tid, base = ulong(group.y) * p.rows;
  XGBDouble area{0};
  if (row < p.rows && (row + 1 == p.rows || sorted[base + row].x != sorted[base + row + 1].x)) {
    uint start = uint(row), key = sorted[base + row].x;
    if (row && sorted[base + row - 1].x == key) {
      uint lo = 0, hi = uint(row);
      while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (sorted[base + mid].x < key) lo = mid + 1;
        else hi = mid;
      }
      start = lo;
    }
    ulong2 prev = start ? prefix[base + start - 1] : ulong2(0), end = prefix[base + row];
    XGBDouble width = xgb_dsub({end.x}, {prev.x});
    width.bits &= ~kDoubleSign;
    area = xgb_dmul(xgb_dmul(width, xgb_dadd({end.y}, {prev.y})), xgb_dfrom_float(0.5f));
  }
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total = xgb_reduce_double_pair(area, {0}, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[ulong(group.y) * p.blocks + group.x] = total;
}

// Stable integral of precision over recall. The CPU's difference of two logs
// can cancel completely for highly unequal weights. This form uses nonnegative
// terms and evaluates 1-log(1+x)/x directly for small x.
constant ulong xgb_pr_log_ratio_coeff[] = {
    0x3fe0000000000000ul,
    0xbfd5555555555555ul,
    0x3fd0000000000000ul,
    0xbfc999999999999aul,
    0x3fc5555555555555ul,
    0xbfc2492492492492ul,
    0x3fc0000000000000ul,
    0xbfbc71c71c71c71cul,
    0x3fb999999999999aul,
    0xbfb745d1745d1746ul,
    0x3fb5555555555555ul,
    0xbfb3b13b13b13b14ul,
    0x3fb2492492492492ul,
    0xbfb1111111111111ul,
    0x3fb0000000000000ul,
    0xbfae1e1e1e1e1e1eul,
    0x3fac71c71c71c71cul,
    0xbfaaf286bca1af28ul,
    0x3fa999999999999aul,
    0xbfa8618618618618ul
};
inline XGBDouble xgb_pr_integral(XGBDouble fp0, XGBDouble fp1, XGBDouble tp0,
                                XGBDouble tp1, XGBDouble total) {
  using namespace xgb_survival;
  D dt = tp1 - tp0, df = fp1 - fp0;
  if (!(dt > zero)) return zero;
  // Parallel positive prefixes can differ by a rounding unit at block edges.
  if (df < zero) df = zero;
  D ds = dt + df, fraction = dt / ds, recall = dt / total, start = fp0 + tp0;
  if (xgb_dzero(start)) return recall * fraction;
  D x = ds / start, q, l;
  if (x < From(0.125f)) {
    q = x * Polynomial(x, xgb_pr_log_ratio_coeff, 20);
    l = one - q;
  } else {
    l = Log(one + x) / x;
    q = one - l;
  }
  return recall * ((tp0 / start) * l + fraction * q);
}

// Continuous precision-recall interpolation, following metric/auc.h.
kernel void xgb_pr_area(device const uint2* sorted [[buffer(0)]],
                         device const ulong2* prefix [[buffer(1)]],
                         device ulong2* partial [[buffer(2)]],
                         constant ROCParams& p [[buffer(3)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
  ulong row = ulong(group.x) * 256 + tid, base = ulong(group.y) * p.rows;
  XGBDouble area{0};
  if (row < p.rows && (row + 1 == p.rows || sorted[base + row].x != sorted[base + row + 1].x)) {
    uint start = uint(row), key = sorted[base + row].x;
    if (row && sorted[base + row - 1].x == key) {
      uint lo = 0, hi = uint(row);
      while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (sorted[base + mid].x < key) lo = mid + 1;
        else hi = mid;
      }
      start = lo;
    }
    ulong2 prev = start ? prefix[base + start - 1] : ulong2(0), end = prefix[base + row];
    using namespace xgb_survival;
    D total{prefix[base + p.rows - 1].y};
    D fp0{prev.x}, tp0{prev.y}, fp1{end.x}, tp1{end.y};
    if (total > zero && D{prefix[base + p.rows - 1].x} > zero && !xgb_deq(tp0, tp1)) {
      area = xgb_pr_integral(fp0, fp1, tp0, tp1, total);
    }
  }
  threadgroup int4 bounds[8];
  threadgroup ulong2 sums[8];
  ulong2 total = xgb_reduce_double_pair(area, {0}, bounds, sums, lane, simd);
  if (simd == 0 && lane == 0) partial[ulong(group.y) * p.blocks + group.x] = total;
}

struct GroupSortParams {
  ulong rows, stride;
  uint groups, padding;
};
inline bool xgb_rank_less(uint2 a, uint2 b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}
// Common ranking queries fit in one threadgroup. Sort score bits and original
// row together, so ties (including signed zeros) retain their original order.
kernel void xgb_group_sort_small(device const uint* values [[buffer(0)]],
                                  device const uint* groups [[buffer(1)]],
                                  device uint2* records [[buffer(2)]],
                                  constant GroupSortParams& p [[buffer(3)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  ulong row = ulong(groups[group]) + tid;
  bool valid = row < groups[group + 1];
  uint2 value = uint2(0xffffffffu);
  if (valid) {
    uint bits = values[row * p.stride];
    uint key = (bits & 0x7fffffffu) == 0 ? 0x80000000u
                 : (bits & 0x80000000u ? ~bits : bits ^ 0x80000000u);
    value = uint2(~key, uint(row));
  }
  threadgroup uint2 shared[256];
  uint padded = 32, count = groups[group + 1] - groups[group];
  while (padded < count) padded <<= 1;
  for (uint size = 2; size <= padded; size <<= 1) {
    for (uint distance = size >> 1; distance; distance >>= 1) {
      uint2 other;
      if (distance < 32) {
        other = simd_shuffle_xor(value, distance);
      } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        other = shared[tid ^ distance];
      }
      bool take_min = ((tid & distance) == 0) == ((tid & size) == 0);
      if (take_min ? xgb_rank_less(other, value) : xgb_rank_less(value, other)) value = other;
    }
  }
  if (valid) records[row] = uint2(group, value.y);
}
// Queries up to 4096 records fit in Apple GPU threadgroup memory. Assign each
// comparator pair to one thread; pairs are disjoint within a network stage.
kernel void xgb_group_sort_medium(device const uint* values [[buffer(0)]],
                                   device const uint* groups [[buffer(1)]],
                                   device uint2* records [[buffer(2)]],
                                   constant GroupSortParams& p [[buffer(3)]],
                                   threadgroup uint2* shared [[threadgroup(0)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  uint begin = groups[group], count = groups[group + 1] - begin;
  if (count <= 1) {
    if (count && tid == 0) records[begin] = uint2(group, begin);
    return;
  }
  uint padded = 2;
  while (padded < count) padded <<= 1;
  for (uint i = tid; i < padded; i += 256) {
    uint2 value = uint2(0xffffffffu);
    if (i < count) {
      uint bits = values[(ulong(begin) + i) * p.stride];
      uint key = (bits & 0x7fffffffu) == 0 ? 0x80000000u
                   : (bits & 0x80000000u ? ~bits : bits ^ 0x80000000u);
      value = uint2(~key, begin + i);
    }
    shared[i] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint size = 2; size <= padded; size <<= 1) {
    for (uint distance = size >> 1; distance; distance >>= 1) {
      for (uint pair = tid; pair < padded / 2; pair += 256) {
        uint low = ((pair & ~(distance - 1)) << 1) | (pair & (distance - 1));
        uint high = low + distance;
        uint2 a = shared[low], b = shared[high];
        bool ascending = (low & size) == 0;
        if (ascending ? xgb_rank_less(b, a) : xgb_rank_less(a, b)) {
          shared[low] = b;
          shared[high] = a;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = tid; i < count; i += 256) records[begin + i] = uint2(group, shared[i].y);
}
// For large/skewed queries, first sort all scores, then stably radix-sort by
// query ID. Binary search handles empty groups without expanding IDs on CPU.
kernel void xgb_group_sort_key(device uint2* records [[buffer(0)]],
                                device const uint* groups [[buffer(1)]],
                                constant GroupSortParams& p [[buffer(2)]],
                                uint tid [[thread_position_in_grid]]) {
  uint row = records[tid].y, lo = 0, hi = p.groups;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (groups[mid] <= row) lo = mid + 1;
    else hi = mid;
  }
  records[tid].x = lo - 1;
}
kernel void xgb_group_sort_export(device const uint2* records [[buffer(0)]],
                                   device const uint* groups [[buffer(1)]],
                                   device ulong* output [[buffer(2)]],
                                   uint tid [[thread_position_in_grid]]) {
  uint2 record = records[tid];
  output[tid] = ulong(record.y - groups[record.x]);
}

struct RankingScoreParams {
  ulong rows, label_stride;
  uint groups, topk, kind, weighted, minus, exponential;
};
struct RankingLabelParams {
  ulong rows, stride;
  uint kind, padding;
};
kernel void xgb_ranking_labels(device const uint* labels [[buffer(0)]],
                                device atomic_uint* result [[buffer(1)]],
                                constant RankingLabelParams& p [[buffer(2)]],
                                uint tid [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  uint error = 0;
  // Uniform threadgroups allow the SIMD reduction to include padded lanes.
  if (ulong(tid) < p.rows) {
    uint bits = labels[ulong(tid) * p.stride], magnitude = bits & 0x7fffffffu;
    float value = as_type<float>(bits);
    if (magnitude >= 0x7f800000u) {
      error = p.kind ? 3 : 1;
    } else if (p.kind == 0) {
      bool near_zero = magnitude < as_type<uint>(1.0e-6f);
      error = !(near_zero || abs(value - 1.0f) < 1.0e-6f);
    } else {
      bool negative = (bits & 0x80000000u) && magnitude;
      error = uint(negative || abs(floor(value) - value) > 1.0e-6f);
      if (value >= 32.0f) error |= 2;
    }
  }
  // Only failing SIMD groups contend on the flag word.
  uint combined = simd_or(error);
  if (lane == 0 && combined) atomic_fetch_or_explicit(result, combined, memory_order_relaxed);
}
kernel void xgb_ndcg_discount(device ulong* discounts [[buffer(0)]],
                               uint tid [[thread_position_in_grid]]) {
  using namespace xgb_survival;
  D n{xgb_scaled_integer(ulong(tid) + 2, 0)};
  discounts[tid] = (D{0x3fe62e42fefa39eful} / Log(n)).bits;
}
kernel void xgb_ndcg_ideal(device const uint2* records [[buffer(0)]],
                            device const uint* labels [[buffer(1)]],
                            device const uint* offsets [[buffer(2)]],
                            device const ulong* discounts [[buffer(3)]],
                            device ulong* inverse [[buffer(4)]],
                            constant RankingScoreParams& p [[buffer(5)]],
                            uint block [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
  uint group = block * 8 + simd;
  if (group >= p.groups) return;
  using namespace xgb_survival;
  uint begin = offsets[group], n = min(offsets[group + 1] - begin, p.topk);
  D sum = zero;
  for (ulong i = lane; i < n; i += 32) {
    uint bits = labels[ulong(records[ulong(begin) + i].y) * p.label_stride];
    D gain = p.exponential ? D{xgb_scaled_integer((1u << uint(as_type<float>(bits))) - 1, 0)}
                           : From(as_type<float>(bits));
    sum = sum + D{discounts[i]} * gain;
  }
  for (uint distance = 16; distance; distance >>= 1) {
    D other = xgb_shuffle_down_double(sum, distance);
    if (lane + distance < 32) sum = sum + other;
  }
  if (lane == 0) inverse[group] = (xgb_dzero(sum) ? zero : one / sum).bits;
}
// One SIMD group per query. Eight queries share a threadgroup without barriers;
// sorted records stay on the GPU and only the final (score, weight) is read back.
kernel void xgb_ranking_score(device const uint2* records [[buffer(0)]],
                               device const uint* labels [[buffer(1)]],
                               device const uint* offsets [[buffer(2)]],
                               device const uint* weights [[buffer(3)]],
                               device const ulong* discounts [[buffer(4)]],
                               device const ulong* inverse_idcg [[buffer(5)]],
                               device ulong2* partial [[buffer(6)]],
                               constant RankingScoreParams& p [[buffer(7)]],
                               uint block [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
  uint group = block * 8 + simd;
  if (group >= p.groups) return;
  using namespace xgb_survival;
  uint begin = offsets[group], count = offsets[group + 1] - begin, n = min(count, p.topk);
  D weight = p.weighted ? From(as_type<float>(weights[group])) : one;
  if (p.kind == 2) {
    uint local_hits = 0;
    bool soft_label = false;
    for (ulong i = lane; i < count; i += 32) {
      uint bits = labels[ulong(records[ulong(begin) + i].y) * p.label_stride];
      local_hits += uint(bits == 0x3f800000u);
      soft_label |= (bits & 0x7fffffffu) != 0 && bits != 0x3f800000u;
    }
    // Upstream validation admits labels within epsilon of zero/one, including
    // tiny negative values. Cancellation can change the zero-hit fallback. Keep
    // CPU's sequential order for these queries, still entirely on the GPU.
    if (simd_any(soft_label)) {
      if (lane == 0) {
        D hits = zero, area = zero;
        for (ulong i = 0; i < count; ++i) {
          D label = From(as_type<float>(labels[ulong(records[ulong(begin) + i].y) * p.label_stride]));
          hits = hits + label;
          if (i < n) area = area + (hits / D{xgb_scaled_integer(i + 1, 0)}) * label;
        }
        D limit{xgb_scaled_integer(p.topk, 0)};
        area = hits > zero ? area / (hits < limit ? hits : limit) : (p.minus ? zero : one);
        partial[group] = ulong2((area * weight).bits, weight.bits);
      }
      return;
    }
    uint total_hits = simd_sum(local_hits), preceding = 0;
    D area = zero;
    for (ulong base = 0; base < n; base += 32) {
      ulong i = base + lane;
      uint hit = i < n ? uint(labels[ulong(records[ulong(begin) + i].y) * p.label_stride] == 0x3f800000u) : 0;
      uint prefix = simd_prefix_exclusive_sum(hit) + hit;
      if (hit) area = area + D{xgb_scaled_integer(preceding + prefix, 0)} /
                                D{xgb_scaled_integer(i + 1, 0)};
      preceding += simd_sum(hit);
    }
    for (uint distance = 16; distance; distance >>= 1) {
      D other = xgb_shuffle_down_double(area, distance);
      if (lane + distance < 32) area = area + other;
    }
    if (lane == 0) {
      area = total_hits ? area / D{xgb_scaled_integer(min(total_hits, p.topk), 0)}
                        : (p.minus ? zero : one);
      partial[group] = ulong2((area * weight).bits, weight.bits);
    }
    return;
  }
  D inverse = p.kind == 1 ? D{inverse_idcg[group]} : one;
  D sum = zero;
  if (p.kind == 0 || inverse > zero) {
    for (ulong i = lane; i < n; i += 32) {
      uint bits = labels[ulong(records[ulong(begin) + i].y) * p.label_stride];
      D label = From(as_type<float>(bits));
      if (p.kind == 0) {
        // CPU precision multiplies labels and query weights in FP32 before
        // accumulating in binary64. Preserve even subnormal product bits.
        sum = sum + From(xgb_dto_float(label * weight));
      } else {
        D gain = p.exponential ? D{xgb_scaled_integer((1u << uint(as_type<float>(bits))) - 1, 0)}
                               : label;
        sum = sum + (D{discounts[i]} * gain) * inverse;
      }
    }
  }
  for (uint distance = 16; distance; distance >>= 1) {
    D other = xgb_shuffle_down_double(sum, distance);
    if (lane + distance < 32) sum = sum + other;
  }
  if (lane == 0) {
    if (p.kind == 0) sum = sum / D{xgb_scaled_integer(n, 0)};
    // Preserve upstream CPU's unweighted zero-IDCG fallback for NDCG.
    else sum = inverse > zero ? sum * weight : (p.minus ? zero : one);
    partial[group] = ulong2(sum.bits, weight.bits);
  }
}

// Ranking gradients retain CPU FP32 accumulation and multiplication boundaries.
// Native arithmetic handles normal results; the integer binary64 path preserves
// inputs/results that Metal is permitted to flush to zero.
inline uint xgb_rank_fadd(uint a, uint b) {
  uint ma = a & 0x7fffffffu, mb = b & 0x7fffffffu;
  if ((!ma || ma >= 0x00800000u) && (!mb || mb >= 0x00800000u)) {
    uint result = as_type<uint>(as_type<float>(a) + as_type<float>(b));
    if ((result & 0x7fffffffu) >= 0x00800000u ||
        (ma == mb && ((a ^ b) & 0x80000000u)) || (!ma && !mb)) return result;
  }
  return as_type<uint>(xgb_dto_float(xgb_dadd(xgb_dfrom_float(as_type<float>(a)),
                                            xgb_dfrom_float(as_type<float>(b)))));
}
inline uint xgb_rank_fmul(uint a, uint b) {
  uint ma = a & 0x7fffffffu, mb = b & 0x7fffffffu;
  if ((!ma || ma >= 0x00800000u) && (!mb || mb >= 0x00800000u)) {
    uint result = as_type<uint>(as_type<float>(a) * as_type<float>(b));
    if ((result & 0x7fffffffu) >= 0x00800000u || !ma || !mb) return result;
  }
  return as_type<uint>(xgb_dto_float(xgb_dmul(xgb_dfrom_float(as_type<float>(a)),
                                            xgb_dfrom_float(as_type<float>(b)))));
}
// Two-float expansion for expf's bounded input. Explicit FMA preserves the
// product residual; low words retain enough precision to round the FP32 result.
inline float2 xgb_rank_add2(float2 a, float2 b) {
  float s = a.x + b.x, v = s - a.x;
  float e = ((a.x - (s - v)) + (b.x - v)) + (a.y + b.y);
  float h = s + e;
  return float2(h, e - (h - s));
}
inline float2 xgb_rank_mul2(float2 a, float2 b) {
  float h = a.x * b.x;
  float e = fma(a.x, b.x, -h) + (a.x * b.y + a.y * b.x);
  e += a.y * b.y;
  float s = h + e;
  return float2(s, e - (s - h));
}
constant float2 xgb_rank_exp_coeff[] = {
  float2(0x1.0000000000000p+0f, 0x0.0p+0f),
  float2(0x1.0000000000000p+0f, 0x0.0p+0f),
  float2(0x1.0000000000000p-1f, 0x0.0p+0f),
  float2(0x1.5555560000000p-3f, -0x1.5555560000000p-28f),
  float2(0x1.5555560000000p-5f, -0x1.5555560000000p-30f),
  float2(0x1.1111120000000p-7f, -0x1.ddddde0000000p-32f),
  float2(0x1.6c16c20000000p-10f, -0x1.27d27e0000000p-35f),
  float2(0x1.a01a020000000p-13f, -0x1.7f97fa0000000p-39f),
  float2(0x1.a01a020000000p-16f, -0x1.7f97fa0000000p-42f),
  float2(0x1.71de3a0000000p-19f, 0x1.55b1cc0000000p-45f),
  float2(0x1.27e4fc0000000p-22f, -0x1.10ec140000000p-47f),
  float2(0x1.ae64560000000p-26f, 0x1.fd51380000000p-52f),
  float2(0x1.1eed8e0000000p-29f, 0x1.ff1b140000000p-54f),
  float2(0x1.6124620000000p-33f, -0x1.8af25e0000000p-58f),
};
inline uint xgb_rank_exp_bits(float x) {
  float k = rint(x * 1.4426950408889634f);
  float2 reduced = xgb_rank_add2(float2(x, 0), xgb_rank_mul2(float2(-k, 0), float2(0x1.62e4300000000p-1f, -0x1.05c6100000000p-29f)));
  float2 value = xgb_rank_exp_coeff[13];
  for (int i = 12; i >= 0; --i) value = xgb_rank_add2(xgb_rank_exp_coeff[i], xgb_rank_mul2(reduced, value));
  // Here -18 <= x <= 88.7, so both the rounded mantissa and scaled result
  // are normal FP32. Adjusting the exponent performs exact power-of-two scaling.
  return as_type<uint>(value.x + value.y) + (uint(int(k)) << 23);
}

// Match the CPU's FP32 exp/add/reciprocal rounding boundaries. Native Metal
// exp/division errors can feed back through sampled ranking over many rounds.
inline uint xgb_rank_sigmoid(uint input) {
  using namespace xgb_survival;
  D x = From(as_type<float>(input ^ 0x80000000u));
  D cap = From(88.7f);
  if (x > cap) x = cap;
  if (x < From(-18.f)) return 0x3f800000u;
  uint e = xgb_rank_exp_bits(xgb_dto_float(x));
  uint denom = xgb_rank_fadd(xgb_rank_fadd(e, 0x3f800000u), as_type<uint>(1e-16f));
  return as_type<uint>(xgb_dto_float(one / From(as_type<float>(denom))));
}
struct RankingPairTask {
  ulong pair_begin, tile_begin;
  uint begin, count, topk, padding;
};
struct RankingGradientParams {
  ulong rows, label_stride, pair_base, tile_base;
  uint group_begin, group_end, row_begin, kind;
  uint normalized, score_normalized, exponential, weighted;
  uint weight_norm, mean, seed, positions;
};
inline ulong xgb_rank_pair_start(ulong i, ulong count) { return i * count - i * (i + 1) / 2; }

// MAP's relevance/precision prefixes. Exact binary labels use parallel integer
// hit prefixes; other accepted labels preserve CPU's ordered arithmetic.
kernel void xgb_ranking_map_stats(device const uint2* records [[buffer(0)]],
                                   device const uint* labels [[buffer(1)]],
                                   device const uint* offsets [[buffer(2)]],
                                   device ulong2* stats [[buffer(3)]],
                                   constant RankingScoreParams& p [[buffer(4)]],
                                   uint block [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
  uint group = block * 8 + simd;
  if (group >= p.groups) return;
  using namespace xgb_survival;
  uint begin = offsets[group], count = offsets[group + 1] - begin;
  bool soft = false;
  for (ulong i = lane; i < count; i += 32) {
    uint bits = labels[ulong(records[ulong(begin) + i].y) * p.label_stride];
    soft |= (bits & 0x7fffffffu) != 0 && bits != 0x3f800000u;
  }
  if (simd_any(soft)) {
    if (lane == 0) {
      D hits = zero, acc = zero;
      for (ulong i = 0; i < count; ++i) {
        D label = From(as_type<float>(labels[ulong(records[ulong(begin) + i].y) * p.label_stride]));
        hits = hits + label;
        acc = acc + label / D{xgb_scaled_integer(i + 1, 0)};
        stats[ulong(begin) + i] = ulong2(hits.bits, acc.bits);
      }
    }
    return;
  }
  uint preceding = 0;
  D carry = zero;
  for (ulong base = 0; base < count; base += 32) {
    ulong i = base + lane;
    uint hit = i < count ? uint(labels[ulong(records[ulong(begin) + i].y) * p.label_stride] == 0x3f800000u) : 0;
    uint prefix = simd_prefix_exclusive_sum(hit) + hit;
    D acc = hit ? one / D{xgb_scaled_integer(i + 1, 0)} : zero;
    for (uint distance = 1; distance < 32; distance <<= 1) {
      ulong2 other = xgb_dpair_shuffle_up(ulong2(acc.bits, 0), distance);
      if (lane >= distance) acc = acc + D{other.x};
    }
    acc = carry + acc;
    if (i < count) stats[ulong(begin) + i] = ulong2(xgb_scaled_integer(preceding + prefix, 0), acc.bits);
    uint lo = simd_broadcast(uint(acc.bits), 31), hi = simd_broadcast(uint(acc.bits >> 32), 31);
    carry = D{(ulong(hi) << 32) | lo};
    preceding += simd_sum(hit);
  }
}

inline uint2 xgb_rank_pair_indices(RankingPairTask task, ulong index, ulong output,
                                     bool mean, device const uint2* sampled) {
  if (mean) return sampled[output];
  uint lo = 0, hi = task.topk;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (xgb_rank_pair_start(mid, task.count) <= index) lo = mid + 1;
    else hi = mid;
  }
  uint i = lo - 1;
  return uint2(i, uint(i + 1 + index - xgb_rank_pair_start(i, task.count)));
}

// Separate instantiations keep expensive bias logarithms out of the ordinary
// pair kernel. Only compacted eligible pairs execute the unbiased instantiation.
template <bool unbiased>
inline uint2 xgb_rank_pair_gradient(device const uint2* records, device const uint* predictions,
    device const uint* labels, device const ulong* discounts, device const ulong* inverse,
    device const ulong2* map_stats, device ulong2* bias_pair, device const ulong* positive,
    device const ulong* negative, constant RankingGradientParams& p, RankingPairTask task,
    uint group, uint i, uint j, ulong output) {
  uint row_i = records[ulong(task.begin) + i].y, row_j = records[ulong(task.begin) + j].y;
  using namespace xgb_survival;
  D yi = From(as_type<float>(labels[ulong(row_i) * p.label_stride]));
  D yj = From(as_type<float>(labels[ulong(row_j) * p.label_stride]));
  if (p.positions) bias_pair[output] = ulong2(0);
  if (xgb_deq(yi, yj)) return uint2(0);
  bool swap = yi < yj;
  uint high = swap ? row_j : row_i, low = swap ? row_i : row_j;
  uint diff_bits = xgb_rank_fadd(predictions[high], predictions[low] ^ 0x80000000u);
  D sigmoid = From(as_type<float>(xgb_rank_sigmoid(diff_bits)));
  D delta = one;
  if (p.kind == 1) {
    D gi = p.exponential ? D{xgb_scaled_integer((1u << uint(xgb_dto_float(yi))) - 1, 0)} : yi;
    D gj = p.exponential ? D{xgb_scaled_integer((1u << uint(xgb_dto_float(yj))) - 1, 0)} : yj;
    D di{discounts[i]}, dj{discounts[j]};
    // CPU evaluates high-relevance terms first, regardless of prediction rank.
    D original = swap ? gj * dj + gi * di : gi * di + gj * dj;
    D changed = swap ? gi * dj + gj * di : gj * di + gi * dj;
    delta = Abs((original - changed) * D{inverse[group]});
  } else if (p.kind == 2) {
    D n{map_stats[ulong(task.begin) + i].x}, m{map_stats[ulong(task.begin) + j].x};
    D b = D{map_stats[ulong(task.begin) + j - 1].y} - D{map_stats[ulong(task.begin) + i].y};
    D ri{xgb_scaled_integer(ulong(i) + 1, 0)}, rj{xgb_scaled_integer(ulong(j) + 1, 0)};
    D total{map_stats[ulong(task.begin) + task.count - 1].x};
    delta = Abs((swap ? (m / rj - (n + one) / ri) - b : (n / ri - m / rj) + b) / total);
  }
  D best = From(as_type<float>(predictions[records[task.begin].y]));
  D worst = From(as_type<float>(predictions[records[ulong(task.begin) + task.count - 1].y]));
  if (p.score_normalized && !xgb_deq(best, worst)) {
    delta = delta / (From(as_type<float>(diff_bits & 0x7fffffffu)) + D{0x3f847ae147ae147bul});
  }
  D gradient = (sigmoid - one) * delta;
  D hessian = sigmoid * (one - sigmoid), epsilon{0x3c9cd2b297d889bcul};  // CPU 1e-16
  if (hessian < epsilon) hessian = epsilon;
  hessian = (hessian * delta) * two;
  uint high_position = high - task.begin, low_position = low - task.begin;
  if (unbiased) {
    D tp{positive[high_position]}, tm{negative[low_position]};
    if ((tp > epsilon || xgb_deq(tp, epsilon)) && (tm > epsilon || xgb_deq(tm, epsilon))) {
      D denominator = tp * tm;
      gradient = gradient / denominator;
      hessian = hessian / denominator;
    }
    D cost = Log(one / (one - sigmoid)) * delta;
    D li = (tm > epsilon || xgb_deq(tm, epsilon)) ? cost / tm : zero;
    D lj = (tp > epsilon || xgb_deq(tp, epsilon)) ? cost / tp : zero;
    // Costs are nonnegative. The sign bit tags the irrelevant endpoint, so
    // the row gather can accumulate li/lj without repeating label comparisons.
    ulong relevant = li.bits & ~kDoubleSign, irrelevant = lj.bits | kDoubleSign;
    bias_pair[output] = swap ? ulong2(irrelevant, relevant) : ulong2(relevant, irrelevant);
  }
  uint g = as_type<uint>(xgb_dto_float(gradient));
  return uint2(swap ? g ^ 0x80000000u : g, as_type<uint>(xgb_dto_float(hessian)));
}

kernel void xgb_ranking_pairs(device const uint2* records [[buffer(0)]],
                               device const uint* predictions [[buffer(1)]],
                               device const uint* labels [[buffer(2)]],
                               device const RankingPairTask* tasks [[buffer(3)]],
                               device const ulong* discounts [[buffer(4)]],
                               device const ulong* inverse [[buffer(5)]],
                               device const ulong2* map_stats [[buffer(6)]],
                               device uint2* pairs [[buffer(7)]],
                               constant RankingGradientParams& p [[buffer(8)]],
                               device const uint2* sampled [[buffer(9)]],
                               device ulong2* bias_pair [[buffer(10)]],
                               device const ulong* positive [[buffer(11)]],
                               device const ulong* negative [[buffer(12)]],
                               device uint* selected [[buffer(13)]],
                               device atomic_uint* selected_count [[buffer(14)]],
                               uint tile [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint query;
  if (tid == 0) {
    uint lo = p.group_begin, hi = p.group_end;
    ulong global_tile = p.tile_base + tile;
    while (lo < hi) {
      uint mid = lo + (hi - lo) / 2;
      if (tasks[mid].tile_begin <= global_tile) lo = mid + 1;
      else hi = mid;
    }
    query = lo - 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint group = query;
  RankingPairTask task = tasks[group];
  ulong index = (p.tile_base + tile - task.tile_begin) * 256 + tid;
  ulong count = p.mean ? ulong(task.topk) * task.count : xgb_rank_pair_start(task.topk, task.count);
  if (index >= count) return;
  ulong output = task.pair_begin - p.pair_base + index;
  uint2 pair = xgb_rank_pair_indices(task, index, output, p.mean, sampled);
  pairs[output] = xgb_rank_pair_gradient<false>(records, predictions, labels, discounts, inverse,
      map_stats, bias_pair, positive, negative, p, task, group, pair.x, pair.y, output);
  if (p.positions) {
    uint a = records[ulong(task.begin) + pair.x].y, b = records[ulong(task.begin) + pair.y].y;
    uint ya = labels[ulong(a) * p.label_stride], yb = labels[ulong(b) * p.label_stride];
    bool active = a - task.begin < p.positions && b - task.begin < p.positions &&
                  ya != yb && ((ya | yb) & 0x7fffffffu) != 0;
    uint count = simd_sum(uint(active)), begin = 0;
    if ((tid & 31) == 0 && count) begin = atomic_fetch_add_explicit(selected_count, count, memory_order_relaxed);
    begin = simd_broadcast(begin, 0);
    uint offset = simd_prefix_exclusive_sum(uint(active));
    if (active) selected[begin + offset] = uint(output);
  }
}

kernel void xgb_ranking_bias_pairs(device const uint2* records [[buffer(0)]],
                               device const uint* predictions [[buffer(1)]],
                               device const uint* labels [[buffer(2)]],
                               device const RankingPairTask* tasks [[buffer(3)]],
                               device const ulong* discounts [[buffer(4)]],
                               device const ulong* inverse [[buffer(5)]],
                               device const ulong2* map_stats [[buffer(6)]],
                               device uint2* pairs [[buffer(7)]],
                               constant RankingGradientParams& p [[buffer(8)]],
                               device const uint2* sampled [[buffer(9)]],
                               device ulong2* bias_pair [[buffer(10)]],
                               device const ulong* positive [[buffer(11)]],
                               device const ulong* negative [[buffer(12)]],
                               device const uint* selected [[buffer(13)]],
                               device const uint* selected_count [[buffer(14)]],
                               uint tid [[thread_position_in_grid]]) {
  if (tid >= selected_count[0]) return;
  ulong output = selected[tid], global_pair = p.pair_base + output;
  uint lo = p.group_begin, hi = p.group_end;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (tasks[mid].pair_begin <= global_pair) lo = mid + 1;
    else hi = mid;
  }
  uint group = lo - 1;
  RankingPairTask task = tasks[group];
  uint2 pair = xgb_rank_pair_indices(task, global_pair - task.pair_begin, output, p.mean, sampled);
  pairs[output] = xgb_rank_pair_gradient<true>(records, predictions, labels, discounts, inverse,
      map_stats, bias_pair, positive, negative, p, task, group, pair.x, pair.y, output);
}
// A nonzero dispatch avoids device-specific zero-grid handling. The correction
// kernel immediately returns when the compacted list is empty.
kernel void xgb_ranking_bias_dispatch(device uint* count [[buffer(0)]]) {
  count[1] = max(1u, count[0] / 256u + uint((count[0] & 255u) != 0)); count[2] = 1; count[3] = 1;
}

kernel void xgb_ranking_normalize(device const uint2* pairs [[buffer(0)]],
                                   device const RankingPairTask* tasks [[buffer(1)]],
                                   device uint* norms [[buffer(2)]],
                                   constant RankingGradientParams& p [[buffer(3)]],
                                   uint block [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
  uint group = p.group_begin + block * 8 + simd;
  if (group >= p.group_end) return;
  using namespace xgb_survival;
  RankingPairTask task = tasks[group];
  if (p.mean) {
    if (lane == 0) norms[group] = as_type<uint>(xgb_dto_float(p.normalized ? one / D{xgb_scaled_integer(task.topk, 0)} : one));
    return;
  }
  ulong count = xgb_rank_pair_start(task.topk, task.count);
  D sum = zero;
  if (p.normalized) {
    for (ulong i = lane; i < count; i += 32) {
      uint bits = pairs[task.pair_begin - p.pair_base + i].x & 0x7fffffffu;
      sum = sum + two * From(as_type<float>(bits));
    }
    for (uint distance = 16; distance; distance >>= 1) {
      D other = xgb_shuffle_down_double(sum, distance);
      if (lane + distance < 32) sum = sum + other;
    }
  }
  if (lane == 0) {
    D norm = sum > zero ? (Log(one + sum) / D{0x3fe62e42fefa39eful}) / sum : one;
    norms[group] = as_type<uint>(xgb_dto_float(norm));
  }
}

inline void xgb_rank_add_bias(ulong tagged, thread xgb_survival::D& li,
                              thread xgb_survival::D& lj) {
  using namespace xgb_survival;
  D value{tagged & ~kDoubleSign};
  if (tagged & kDoubleSign) lj = lj + value;
  else li = li + value;
}

kernel void xgb_ranking_gather(device const uint2* records [[buffer(0)]],
                                device const RankingPairTask* tasks [[buffer(1)]],
                                device const uint2* pairs [[buffer(2)]],
                                device const uint* norms [[buffer(3)]],
                                device const uint* weights [[buffer(4)]],
                                device uint2* output [[buffer(5)]],
                                constant RankingGradientParams& p [[buffer(6)]],
                                device const ulong2* bias_pair [[buffer(9)]],
                                device ulong2* bias_row [[buffer(10)]],
                                uint tid [[thread_position_in_grid]]) {
  ulong index = ulong(p.row_begin) + tid;
  uint2 record = records[index];
  uint group = record.x;
  RankingPairTask task = tasks[group];
  uint rank = uint(index - task.begin);
  uint2 sum(0);
  xgb_survival::D li = xgb_survival::zero, lj = xgb_survival::zero;
  for (uint i = 0; i < min(rank, task.topk); ++i) {
    ulong offset = task.pair_begin - p.pair_base + xgb_rank_pair_start(i, task.count) + rank - i - 1;
    uint2 value = pairs[offset];
    if (p.positions) xgb_rank_add_bias(bias_pair[offset].y, li, lj);
    sum = uint2(xgb_rank_fadd(sum.x, value.x ^ 0x80000000u), xgb_rank_fadd(sum.y, value.y));
  }
  if (rank < task.topk) {
    for (ulong j = ulong(rank) + 1; j < task.count; ++j) {
      ulong offset = task.pair_begin - p.pair_base + xgb_rank_pair_start(rank, task.count) + j - rank - 1;
      uint2 value = pairs[offset];
      if (p.positions) xgb_rank_add_bias(bias_pair[offset].x, li, lj);
      sum = uint2(xgb_rank_fadd(sum.x, value.x), xgb_rank_fadd(sum.y, value.y));
    }
  }
  if (p.positions) bias_row[record.y] = ulong2(li.bits, lj.bits);
  // GradientPair::operator* accepts float, even when callers pass a double.
  uint w = p.weighted ? weights[group] : 0x3f800000u, norm = norms[group];
  sum = uint2(xgb_rank_fmul(sum.x, norm), xgb_rank_fmul(sum.y, norm));
  sum = uint2(xgb_rank_fmul(sum.x, w), xgb_rank_fmul(sum.y, w));
  output[record.y] = uint2(xgb_rank_fmul(sum.x, p.weight_norm), xgb_rank_fmul(sum.y, p.weight_norm));
}


// Match Apple libc++'s minstd_rand + uniform_int_distribution<size_t>.
// Unlike modulo sampling, rejection preserves the CPU sequence and distribution.
inline uint xgb_rank_minstd(thread uint& state) {
  ulong product = ulong(state) * 48271u;
  uint next = uint(product & 0x7ffffffful) + uint(product >> 31);
  state = next >= 0x7fffffffu ? next - 0x7fffffffu : next;
  return state;
}
inline uint xgb_rank_uniform(thread uint& state, uint range) {
  if (range == 1) return 0;  // libc++ consumes no engine output for a singleton.
  uint w = 32 - clz(range - 1), n = (w + 29) / 30, w0 = w / n;
  constexpr uint rp = 2147483646u;
  uint y0 = (rp >> w0) << w0;
  if (rp - y0 > y0 / n) { ++n; w0 = w / n; y0 = (rp >> w0) << w0; }
  uint n0 = n - w % n, y1 = (rp >> (w0 + 1)) << (w0 + 1);
  uint mask0 = (1u << w0) - 1, mask1 = (1u << (w0 + 1)) - 1;
  uint value;
  do {
    value = 0;
    for (uint k = 0; k < n; ++k) {
      uint u, limit = k < n0 ? y0 : y1;
      do { u = xgb_rank_minstd(state) - 1; } while (u >= limit);
      value = (value << (k < n0 ? w0 : w0 + 1)) + (u & (k < n0 ? mask0 : mask1));
    }
  } while (value >= range);
  return value;
}
kernel void xgb_ranking_mean_labels(device const uint2* records [[buffer(0)]],
                                    device const uint* labels [[buffer(1)]],
                                    device uint* ranked [[buffer(2)]],
                                    constant ulong& stride [[buffer(3)]],
                                    uint row [[thread_position_in_grid]]) {
  ranked[row] = labels[ulong(records[row].y) * stride];
}
// Sampling is sequential within a query to retain variable rejection draws and
// pair order. Independent queries execute in parallel. Incidence lists remove
// the need to scan all pairs for each row or to use nondeterministic atomics.
kernel void xgb_ranking_mean_sample(device const uint2* records [[buffer(0)]],
                                    device const uint2* label_order [[buffer(1)]],
                                    device const uint* labels [[buffer(2)]],
                                    device const RankingPairTask* tasks [[buffer(3)]],
                                    device uint2* sampled [[buffer(4)]],
                                    device uint* links [[buffer(5)]],
                                    device uint* heads [[buffer(6)]],
                                    device uint* tails [[buffer(7)]],
                                    constant RankingGradientParams& p [[buffer(8)]],
                                    uint tid [[thread_position_in_grid]]) {
  uint group = p.group_begin + tid;
  RankingPairTask task = tasks[group];
  uint state = (p.seed + group) % 2147483647u;
  if (!state) state = 1;
  for (ulong row = task.begin; row < ulong(task.begin) + task.count; ++row) heads[row] = 0xffffffffu;
  using namespace xgb_survival;
  for (ulong i = 0; i < task.count;) {
    ulong j = i + 1;
    D label = From(as_type<float>(labels[ulong(records[label_order[task.begin + i].y].y) * p.label_stride]));
    while (j < task.count && xgb_deq(label, From(as_type<float>(labels[ulong(records[label_order[task.begin + j].y].y) * p.label_stride])))) ++j;
    uint outside = uint(i + task.count - j);
    for (uint repeat = 0; repeat < task.topk; ++repeat) {
      for (ulong item = i; item < j; ++item) {
        ulong slot = task.pair_begin - p.pair_base + i * task.topk + ulong(repeat) * (j - i) + item - i;
        if (!outside) { sampled[slot] = uint2(0); continue; }
        ulong other = xgb_rank_uniform(state, outside);
        if (other >= i) other = other - i + j;
        uint a = label_order[task.begin + item].y - task.begin;
        uint b = label_order[task.begin + other].y - task.begin;
        uint first = min(a, b), second = max(a, b);
        sampled[slot] = uint2(first, second);
        for (uint side = 0; side < 2; ++side) {
          uint node = uint(2 * slot + side), row = task.begin + (side ? second : first);
          links[node] = 0xffffffffu;
          if (heads[row] == 0xffffffffu) heads[row] = node;
          else links[tails[row]] = node;
          tails[row] = node;
        }
      }
    }
    i = j;
  }
}
kernel void xgb_ranking_mean_gather(device const uint2* records [[buffer(0)]],
                                    device const RankingPairTask* tasks [[buffer(1)]],
                                    device const uint2* pairs [[buffer(2)]],
                                    device const uint* norms [[buffer(3)]],
                                    device const uint* weights [[buffer(4)]],
                                    device uint2* output [[buffer(5)]],
                                    constant RankingGradientParams& p [[buffer(6)]],
                                    device const uint* heads [[buffer(7)]],
                                    device const uint* links [[buffer(8)]],
                                    device const ulong2* bias_pair [[buffer(9)]],
                                    device ulong2* bias_row [[buffer(10)]],
                                    uint tid [[thread_position_in_grid]]) {
  ulong index = ulong(p.row_begin) + tid;
  uint2 record = records[index], sum(0);
  xgb_survival::D li = xgb_survival::zero, lj = xgb_survival::zero;
  for (uint node = heads[index]; node != 0xffffffffu; node = links[node]) {
    uint2 value = pairs[node / 2];
    if (p.positions) xgb_rank_add_bias(bias_pair[node / 2][node & 1], li, lj);
    sum = uint2(xgb_rank_fadd(sum.x, value.x ^ ((node & 1) ? 0x80000000u : 0)), xgb_rank_fadd(sum.y, value.y));
  }
  if (p.positions) bias_row[record.y] = ulong2(li.bits, lj.bits);
  uint w = p.weighted ? weights[record.x] : 0x3f800000u, norm = norms[record.x];
  sum = uint2(xgb_rank_fmul(sum.x, norm), xgb_rank_fmul(sum.y, norm));
  sum = uint2(xgb_rank_fmul(sum.x, w), xgb_rank_fmul(sum.y, w));
  output[record.y] = uint2(xgb_rank_fmul(sum.x, p.weight_norm), xgb_rank_fmul(sum.y, p.weight_norm));
}


struct RankingBiasParams { uint groups, positions; ulong regularizer; uint segments, padding; };
// Sum independent query blocks in parallel. A single thread per position would
// leave most GPU cores idle for large datasets. Within each block, preserve order.
kernel void xgb_ranking_bias_sum(device const ulong2* rows [[buffer(0)]],
                                  device const uint* groups [[buffer(1)]],
                                  device ulong2* sums [[buffer(2)]],
                                  constant RankingBiasParams& p [[buffer(3)]],
                                  uint2 tid [[thread_position_in_grid]]) {
  using namespace xgb_survival;
  uint position = tid.x, segment = tid.y;
  D li = zero, lj = zero;
  ulong end = min(ulong(p.groups), (ulong(segment) + 1) * 128);
  for (uint group = segment * 128; group < end; ++group) {
    uint begin = groups[group], count = groups[group + 1] - begin;
    if (position < count) {
      ulong2 value = rows[ulong(begin) + position];
      li = li + D{value.x}; lj = lj + D{value.y};
    }
  }
  sums[ulong(segment) * p.positions + position] = ulong2(li.bits, lj.bits);
}
kernel void xgb_ranking_bias_finish(device const ulong2* partial [[buffer(0)]],
                                     device ulong2* sums [[buffer(1)]],
                                     constant RankingBiasParams& p [[buffer(2)]],
                                     uint position [[thread_position_in_grid]]) {
  using namespace xgb_survival;
  D li = zero, lj = zero;
  for (uint segment = 0; segment < p.segments; ++segment) {
    ulong2 value = partial[ulong(segment) * p.positions + position];
    li = li + D{value.x}; lj = lj + D{value.y};
  }
  sums[position] = ulong2(li.bits, lj.bits);
}
inline xgb_survival::D xgb_rank_pow(xgb_survival::D x, xgb_survival::D exponent) {
  using namespace xgb_survival;
  if (xgb_deq(exponent, zero) || xgb_deq(x, one)) return one;
  if (xgb_deq(exponent, one)) return x;
  if (xgb_deq(x, zero)) return zero;  // regularizer is nonnegative
  return Exp(Log(x) * exponent);
}
kernel void xgb_ranking_bias_update(device const ulong2* sums [[buffer(0)]],
                                     device ulong* positive [[buffer(1)]],
                                     device ulong* negative [[buffer(2)]],
                                     constant RankingBiasParams& p [[buffer(3)]],
                                     uint position [[thread_position_in_grid]]) {
  using namespace xgb_survival;
  D li0{sums[0].x}, lj0{sums[0].y}, epsilon{0x3c9cd2b297d889bcul}, regularizer{p.regularizer};
  if ((li0 > epsilon || xgb_deq(li0, epsilon))) positive[position] = xgb_rank_pow(D{sums[position].x} / li0, regularizer).bits;
  if ((lj0 > epsilon || xgb_deq(lj0, epsilon))) negative[position] = xgb_rank_pow(D{sums[position].y} / lj0, regularizer).bits;
}
