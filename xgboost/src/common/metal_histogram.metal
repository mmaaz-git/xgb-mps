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

// Privatize one feature histogram per threadgroup. Eight rows per lane amortize
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
    uint3 threads [[threads_per_threadgroup]]) {
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
  ulong begin = node_ptr[node] + (ulong(group.x) - block_ptr[node]) * threads.x * 8;
  ulong end = min(node_ptr[node + 1], begin + threads.x * 8);
  if (begin >= end) return;  // Uniform across the threadgroup, before any barrier.
  uint feature_offset = offsets[feature];
  uint bins = offsets[feature + 1] - feature_offset;
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
kernel void xgb_partition_prefix(device const PartitionSplit* splits [[buffer(0)]],
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
                               device const float2* gradients [[buffer(2)]],
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
    invalid = invalid && gradients[row * p.stride0 + t * p.stride1].y == 0.0f;
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
  SplitResult best{0, 0, 0, 0, 0.0f, -1, 0, 0};
  if (task.kind == 0) {
    for (uint i = tid; i < n; i += width) {
      DoublePair sum = xgb_pair_read(scans[task.begin + i].xy);
      xgb_consider_split(sum, xgb_pair_sub(parent, sum), task, p, int(i), 0, 0, best);
    }
    if (missing)
      for (int i = int(n) - 1 - int(tid); i >= 0; i -= int(width)) {
        DoublePair sum = xgb_pair_read(scans[task.begin + i].zw);
        xgb_consider_split(xgb_pair_sub(parent, sum), sum, task, p, i, 1, 0, best);
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
