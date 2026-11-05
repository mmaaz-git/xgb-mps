/** Copyright 2026, XGBoost Contributors */
#pragma once
#include "xgboost/logging.h"

// Recent gRPC/Protobuf headers expose Abseil's generic LOG/CHECK macros.
// Let those headers use their own definitions, then restore XGBoost's macros
// (notably LOG(CONSOLE) and exception-based fatal checks) for our callers.
#pragma push_macro("LOG")
#undef LOG
#pragma push_macro("LOG_IF")
#undef LOG_IF
#pragma push_macro("VLOG")
#undef VLOG
#pragma push_macro("DLOG")
#undef DLOG
#pragma push_macro("DLOG_IF")
#undef DLOG_IF
#pragma push_macro("CHECK")
#undef CHECK
#pragma push_macro("CHECK_NOTNULL")
#undef CHECK_NOTNULL
#pragma push_macro("CHECK_EQ")
#undef CHECK_EQ
#pragma push_macro("CHECK_NE")
#undef CHECK_NE
#pragma push_macro("CHECK_LT")
#undef CHECK_LT
#pragma push_macro("CHECK_LE")
#undef CHECK_LE
#pragma push_macro("CHECK_GT")
#undef CHECK_GT
#pragma push_macro("CHECK_GE")
#undef CHECK_GE
#pragma push_macro("DCHECK_EQ")
#undef DCHECK_EQ
#pragma push_macro("DCHECK_NE")
#undef DCHECK_NE
#pragma push_macro("DCHECK_LT")
#undef DCHECK_LT
#pragma push_macro("DCHECK_LE")
#undef DCHECK_LE
#pragma push_macro("DCHECK_GT")
#undef DCHECK_GT
#pragma push_macro("DCHECK_GE")
#undef DCHECK_GE
#pragma push_macro("DCHECK")
#undef DCHECK

#include <federated.grpc.pb.h>
#include <federated.pb.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>

#pragma pop_macro("DCHECK")
#pragma pop_macro("DCHECK_GE")
#pragma pop_macro("DCHECK_GT")
#pragma pop_macro("DCHECK_LE")
#pragma pop_macro("DCHECK_LT")
#pragma pop_macro("DCHECK_NE")
#pragma pop_macro("DCHECK_EQ")
#pragma pop_macro("CHECK_GE")
#pragma pop_macro("CHECK_GT")
#pragma pop_macro("CHECK_LE")
#pragma pop_macro("CHECK_LT")
#pragma pop_macro("CHECK_NE")
#pragma pop_macro("CHECK_EQ")
#pragma pop_macro("CHECK_NOTNULL")
#pragma pop_macro("CHECK")
#pragma pop_macro("DLOG_IF")
#pragma pop_macro("DLOG")
#pragma pop_macro("VLOG")
#pragma pop_macro("LOG_IF")
#pragma pop_macro("LOG")
