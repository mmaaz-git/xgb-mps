// Independent scalar reading of common/quantile.cu's GPU sketch equations.
// No Metal helpers and no HostSketchContainer / WQSummary merge or prune calls.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cuda_sketch_reference {
struct Entry {
  float rmin, rmax, wmin, value;
};
using Column = std::vector<Entry>;
inline std::size_t Budget(std::size_t rows, std::size_t bins) {
  if (!rows) return 1;
  double epsilon = (1.0 / std::min(bins, rows)) / 2.0;
  for (std::size_t level = 1;; ++level) {
    auto size = std::min(rows, std::size_t(std::ceil(level / epsilon)) + 1);
    if ((std::size_t{1} << level) * size >= rows) return size;
  }
}
inline Column Unique(Column a) {
  Column out;
  for (auto e : a)
    if (out.empty() || e.value - out.back().value != 0.f) out.push_back(e);
  return out;
}
inline Column Prune(Column const& a, std::size_t budget, bool categorical,
                    bool index_unique = false) {
  if (categorical || a.size() <= budget) return a;
  if (budget == 1) return {a.front()};
  Column out{a.front()};
  std::size_t last_index = 0;
  for (std::size_t k = 1; k + 1 < budget; ++k) {
    float width = a.back().rmin - a.front().rmax;
    float rank = float(k) * width;
    rank = rank / (float(budget) - 1.f);
    rank = rank + a.front().rmax;
    rank = rank * 2.f;
    std::size_t chosen;
    if (rank < a.front().rmin + a.front().rmax)
      chosen = 0;
    else if (rank >= a.back().rmin + a.back().rmax)
      chosen = a.size() - 1;
    else {
      // Linear traversal, independent of the GPU's binary search implementation.
      std::size_t i = 0;
      while (i + 2 < a.size() && !(rank < a[i + 1].rmin + a[i + 1].rmax)) ++i;
      float left = a[i].rmin + a[i].wmin;
      float right = a[i + 1].rmax - a[i + 1].wmin;
      chosen = rank < left + right ? i : i + 1;
    }
    if (!index_unique || last_index != chosen) out.push_back(a[chosen]);
    last_index = chosen;
  }
  if (!index_unique || last_index != a.size() - 1) out.push_back(a.back());
  return out;
}
inline Column ScanInput(Column a) {
  if (a.empty()) return a;
  for (std::size_t i = a.size() - 1; i > 0; --i) {
    Entry& l = a[i - 1];
    auto r = a[i];
    if (l.value - r.value == 0.f && l.rmin - r.rmin != 0.f) {
      l.wmin = l.wmin + r.wmin;
      l.rmax = l.rmin + l.wmin;
    }
  }
  return Unique(a);
}
inline Column Merge(Column const& a, Column const& b) {
  Column out;
  std::size_t i = 0, j = 0;
  // Linear stable merge independently enumerates CUDA's partition boundaries.
  // Equal outputs must survive until subtraction-based Unique (inf != inf).
  if (a.empty() || b.empty()) out = a.empty() ? b : a;
  while (!a.empty() && !b.empty() && (i < a.size() || j < b.size())) {
    if (i < a.size() && j < b.size() && a[i].value == b[j].value) {
      out.push_back(
          {a[i].rmin + b[j].rmin, a[i].rmax + b[j].rmax, a[i].wmin + b[j].wmin, a[i].value});
      ++i;
    } else if (i < a.size() && (j == b.size() || a[i].value < b[j].value)) {
      float lo = j ? b[j - 1].rmin + b[j - 1].wmin : 0.f;
      float hi = b.empty() ? 0.f : (j == b.size() ? b.back().rmax : b[j].rmax - b[j].wmin);
      out.push_back({a[i].rmin + lo, a[i].rmax + hi, a[i].wmin, a[i].value});
      ++i;
    } else {
      float lo = i ? a[i - 1].rmin + a[i - 1].wmin : 0.f;
      float hi = a.empty() ? 0.f : (i == a.size() ? a.back().rmax : a[i].rmax - a[i].wmin);
      out.push_back({lo + b[j].rmin, hi + b[j].rmax, b[j].wmin, b[j].value});
      ++j;
    }
  }
  out = Unique(out);
  float lo = 0, hi = 0;
  for (auto& e : out) {
    if (e.rmin < lo) e.rmin = lo;
    if (e.rmax < hi) e.rmax = hi;
    float next = e.rmin + e.wmin;
    if (e.rmax < next) e.rmax = next;
    lo = e.rmin;
    hi = e.rmax;
  }
  return out;
}
struct Raw {
  std::uint32_t feature;
  float value, weight;
};
struct Sketch {
  std::vector<Column> columns;
  std::vector<bool> categorical;
  std::size_t bins, seen = 0;
  Sketch(std::size_t n, std::size_t bins, std::vector<bool> categorical)
      : columns(n), categorical(std::move(categorical)), bins(bins) {}
  void Push(std::vector<Raw> raw, std::size_t rows, bool weighted) {
    if (raw.empty()) return;
    seen += rows;
    std::stable_sort(raw.begin(), raw.end(), [](auto a, auto b) {
      return a.feature < b.feature || (a.feature == b.feature && a.value < b.value);
    });
    std::vector<std::vector<Raw>> by_column(columns.size());
    float cumulative = 0;
    std::uint32_t previous = UINT32_MAX;
    for (auto r : raw) {
      bool first = r.feature != previous;
      previous = r.feature;
      if (weighted) {
        cumulative = first ? r.weight : cumulative + r.weight;
        r.weight = cumulative;
      }
      auto& c = by_column[r.feature];
      if (!categorical[r.feature] || c.empty() || c.back().value != r.value) c.push_back(r);
    }
    for (std::size_t f = 0; f < columns.size(); ++f) {
      Column next;
      auto& raw_column = by_column[f];
      for (std::size_t i = 0; i < raw_column.size(); ++i) {
        float lo = weighted ? (i ? raw_column[i - 1].weight : 0.f) : float(i);
        float hi = weighted ? raw_column[i].weight : float(i + 1);
        float w = weighted ? hi - lo : 1.f;
        if (w < 0) w = 1e-6f;
        next.push_back({lo, hi, w, raw_column[i].value});
      }
      next = ScanInput(Prune(next, std::min(rows, Budget(std::max<std::size_t>(1, rows), bins)),
                             categorical[f]));
      columns[f] = Unique(Prune(Merge(columns[f], next), Budget(seen, bins), categorical[f], true));
    }
  }
};
}  // namespace cuda_sketch_reference
