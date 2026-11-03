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
    device const ulong* row_begin [[buffer(8)]],
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
    ulong row = rows[row_begin[node] + r - node_ptr[node]];
    ulong i = p.padding ? ulong(feature) * p.page_rows + (row - p.base_rowid)
                        : row_ptr[row - p.base_rowid] + feature;
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
struct QuantizeParams {
  ulong n;
  int grad_exponent;
  int hess_exponent;
};
kernel void xgb_quantize_gradients(device const uint2* input [[buffer(0)]],
                                   device long2* output [[buffer(1)]],
                                   constant QuantizeParams& p [[buffer(2)]],
                                   uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= p.n) return;
  uint2 bits = input[tid];
  output[tid] =
      long2(quantize_float(bits.x, p.grad_exponent), quantize_float(bits.y, p.hess_exponent));
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
    if (p.kind == 6) {
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
