"""Source instrumentation checks; these do not execute or compile CUDA kernels."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("auc_probe", ROOT / "scripts/prepare_auc_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
SOURCE = ROOT / "xgboost/src/metric/auc.cu"


def test_exact_original_source_reconstruction():
    source = SOURCE.read_bytes()
    generated, record = probe.generate(source)
    restored = generated.decode()
    assert len(record["edits"]) == 10
    assert generated.count(b"::auc_capture::") == 8
    assert b"namespace xgboost::metric::cuda_impl {" not in generated
    assert b"DMLC_REGISTRY_FILE_TAG(auc_gpu);" not in generated
    for edit in reversed(record["edits"]):
        at, after = edit["offset"], edit["after"]
        assert restored[at:at + len(after)] == after
        restored = restored[:at] + edit["before"] + restored[at + len(after):]
    assert restored.encode() == source
    assert record["generated_sha256"] == hashlib.sha256(generated).hexdigest()
    assert record["gpu_execution"] is False


@pytest.mark.parametrize("before,after", [
    (b"float fp = (1.0 - label) * w;", b"float fp = (1.0f - label) * w;"),
    (b"PairPlus<double, double>{}", b"PairPlus<float, float>{}"),
    (b"thrust::reduce(", b"thrust::reduce( /* changed */"),
    (b"n_targets", b"output_count"),
])
def test_unreviewed_arithmetic_or_structure_changes_rejected(before, after):
    source = SOURCE.read_bytes()
    assert before in source
    with pytest.raises(ValueError, match="source changed"):
        probe.generate(source.replace(before, after, 1))


def test_capture_follows_actual_producers_and_precedes_consumers():
    generated, _ = probe.generate(SOURCE.read_bytes())
    text = generated.decode()
    binary = text[text.index("std::tuple<double, double, double> BinaryAUC("):
                  text.index("std::tuple<double, double, double> BinaryROCAUC(")]
    assert binary.index("d_fptp[i] = get_fp_tp(i)") < binary.index("::auc_capture::Terms(")
    assert binary.index("::auc_capture::Terms(") < binary.index("common::InclusiveScan(")
    assert binary.index("common::InclusiveScan(") < binary.index("::auc_capture::Prefixes(")
    pr = text[text.index("std::tuple<double, double, double> BinaryPRAUC("):
              text.index("double MultiPRAUC(")]
    assert pr.index("::auc_capture::BinaryTotals(") < pr.index("if (total_pos <= 0.0")
    multi = text[text.index("double MultiAUC("):text.index("void MultiSortedIdx(")]
    assert multi.index("SegmentedReduceAUC(") < multi.index("::auc_capture::Curves(")
    assert multi.index("if (n_samples == 0)") < multi.index("::auc_capture::Terms(")


def test_generation_does_not_mutate_vendored_source(tmp_path):
    before = SOURCE.read_bytes()
    generated, record = probe.generate(before)
    target = tmp_path / "auc_probe.inc"
    target.write_bytes(generated)
    assert SOURCE.read_bytes() == before
    assert record["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert probe.generate(before) == (generated, record)


def test_capture_presence_order_and_bits_with_host_copy_stub(tmp_path):
    # Exercise the real capture header's serialization/control paths. The CUDA
    # calls below are explicitly host-copy stubs, not GPU execution evidence.
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for capture plumbing checks")
    (tmp_path / "cuda_runtime.h").write_text(r'''
#pragma once
#include <cstring>
enum cudaError_t { cudaSuccess };
enum cudaMemcpyKind { cudaMemcpyDeviceToHost };
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline char const* cudaGetErrorString(cudaError_t) { return "host copy stub"; }
inline cudaError_t cudaMemcpy(void* dst, void const* src, std::size_t bytes, cudaMemcpyKind) {
  std::memcpy(dst, src, bytes); return cudaSuccess;
}
''')
    (tmp_path / "capture.cc").write_text(r'''
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
#include "auc_capture.cuh"
// Match the CUDA replay translation unit: this helper is declared after the
// capture header. cudaError_t is global, so unqualified dependent calls can
// discover this function through argument-dependent lookup at instantiation.
void Cuda(cudaError_t) { throw std::runtime_error("Wrong CUDA error helper"); }
using xgboost::common::Span;
int main() {
  using Pair = std::pair<double, double>;
  Pair points[]{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  std::size_t const indices[]{2, 0, 3, 1};
  Span<Pair> p{points, 4};
  Span<std::size_t const> idx{indices, 4};
  auc_capture::Observation out;
  {
    auc_capture::Scope scope{&out};
    auc_capture::Terms(nullptr, p, idx);
    assert(!out.terms_present && out.terms.empty());
    out.trace = true;
    auc_capture::Terms(nullptr, p, idx);
    auc_capture::Prefixes(nullptr, p);
    auc_capture::MultiTotals(nullptr, p);
    assert(out.terms_present && out.prefixes_present && out.totals_present);
    assert(out.sorted_indices == std::vector<std::uint64_t>({2, 0, 3, 1}));
    assert(out.terms == std::vector<double>({1, 2, 3, 4, 5, 6, 7, 8}));
    assert(out.totals == std::vector<double>({2, 1, 4, 3, 6, 5, 8, 7}));
    double area[]{9, 10}, weights[]{-0.0, std::numeric_limits<double>::quiet_NaN()};
    auc_capture::Curves(nullptr, p, Span<double>{area, 2}, 2);
    assert(out.curves == std::vector<double>({3, 4, 9, 7, 8, 10}));
    auc_capture::Normalization(nullptr, {area, 2}, {weights, 2}, {area, 2});
    assert(out.normalization_present && out.normalization.size() == 6);
    assert(std::signbit(out.normalization[1]) && std::isnan(out.normalization[4]));
    bool rejected = false;
    try { auc_capture::Terms(nullptr, p, idx); } catch (std::runtime_error const&) { rejected = true; }
    assert(rejected);
    auc_capture::Observation empty;
    {
      auc_capture::Scope inner{&empty};
      auc_capture::Normalization(nullptr, {}, {}, {});
      auc_capture::BinaryTotals(0, 0);
      assert(empty.normalization_present && empty.normalization.empty());
      assert(empty.totals_present && !empty.terms_present && !empty.prefixes_present);
    }
    assert(auc_capture::sink == &out);
  }
  assert(auc_capture::sink == nullptr);
}
''')
    executable = tmp_path / "capture"
    subprocess.run([
        compiler, "-std=c++17", "-I", str(tmp_path), "-I", str(ROOT / "tests"),
        "-I", str(ROOT / "xgboost/include"), "-I", str(ROOT / "xgboost/dmlc-core/include"),
        str(tmp_path / "capture.cc"), "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
