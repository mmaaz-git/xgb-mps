/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <memory>

#include "../encoder/ordinal.h"

namespace xgboost::cpu_impl {
struct CatContainerImpl;
}
namespace xgboost::common::metal {
// Immutable owned snapshot. View metadata lives on the host; its typed name
// spans address shared Metal storage. Only CopyToHost publishes a host mirror.
class CategoryDictionary {
 public:
  virtual ~CategoryDictionary() = default;
  virtual enc::HostColumnsView View() const = 0;
  virtual std::unique_ptr<CategoryDictionary> Clone() const = 0;
  virtual void CopyToHost(::xgboost::cpu_impl::CatContainerImpl* output) const = 0;
  virtual void Sort(Span<std::int32_t> sorted) const = 0;
  virtual void Recode(Span<std::int32_t const> sorted, CategoryDictionary const& incoming,
                      Span<std::int32_t> mapping) const = 0;
};
std::unique_ptr<CategoryDictionary> MakeCategoryDictionary(enc::HostColumnsView columns);

// Typed dictionaries retain integer widths and signed-byte string comparisons.
void SortCategories(enc::HostColumnsView columns, Span<std::int32_t> sorted);
void RecodeCategories(enc::HostColumnsView original, Span<std::int32_t const> sorted,
                      enc::HostColumnsView incoming, Span<std::int32_t> mapping);
}  // namespace xgboost::common::metal
