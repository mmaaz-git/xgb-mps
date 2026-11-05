"""Synthetic metric report checks; no NVIDIA execution evidence."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct
import sys
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location("metric_compare",Path(__file__).parents[1]/"scripts/compare_metric_contracts.py")
mc=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mc)


@lru_cache
def encoded(kind, accepted_guard=False):
    parts=[mc.MAGIC,struct.pack("<II",kind,mc.COUNT)]
    for s in mc.specs():
        parts += [struct.pack("<15I",*s),bytes(4*(s.predictions+s.labels+s.upper+s.weights+s.alphas))]
        status=2 if kind==2 and s.family==5 and s.mode>=2 else int(s.family==5 and not accepted_guard)
        if kind:
            parts += [struct.pack("<I",status),bytes(4*s.fields)]
            if s.family and not status: parts.append(bytes(16))
            if s.family in (1,2): parts.append(bytes(32))
        if kind==1:
            parts.append(bytes(4*s.fields))
            if s.family in (3,4): parts.append(bytes(16))
    return b"".join(parts)+b"METE"+struct.pack("<I",mc.COUNT)


def reports(): return [mc.decode(bytearray(encoded(k))) for k in (1,2)]


def compare(m,c): return mc.compare(mc.decode(m["data"]),mc.decode(c["data"]))


def find(**fields): return next(s.id for s in mc.specs() if all(getattr(s,k)==v for k,v in fields.items()))


def reconcile(case):
    for i in (0,1):
        total,magnitude=mc.neumaier(case["actual"][i::2])
        case["reductions"][2*i:2*i+2]=[mc.bits(total,64),mc.bits(magnitude,64)]
        case["sums"][i]=mc.bits(total,64)


def test_complete_matrix_preserves_uncounted_math_guards_and_empties():
    m,c=reports(); result=compare(m,c)
    assert [sum(s.family==f for s in mc.specs()) for f in range(7)]==[1,287,32,24,6,12,13]
    assert sum(s.fields for s in mc.specs())==15628618+500285
    assert result["status"]=="common-domain-identical-with-exclusions" and not result["reference_failures"]
    assert result["cuda_executed_cases"]==367 and len(result["domain_exclusions"])==8
    assert result["compared_fields_including_status"]==16129994
    mc.match_inputs(mc.decode(encoded(0)),m)
    with pytest.raises(ValueError,match="Metal and actual CUDA"): mc.compare(m,m)


def test_identical_wrong_terms_fail_even_when_their_reductions_are_consistent():
    m,c=reports()
    for report in (m,c): report["cases"][1]["actual"][0]=mc.bits(1.); reconcile(report["cases"][1])
    result=compare(m,c)
    assert [e["policy"] for e in result["reference_failures"]]==["original-row-ulp"]*2
    assert not result["different_fields"]


def test_identical_wrong_sums_fail_independent_checks():
    m,c=reports()
    for report in (m,c): report["cases"][1]["sums"][0]=mc.bits(1.,64)
    result=compare(m,c)
    assert len(result["reference_failures"])==2 and not result["different_fields"]


def test_cuda_reduction_uses_cuda_terms_not_metal_terms():
    m,c=reports(); index=find(family=1,kind=4,rows=1,mode=0)
    for report in (m,c): report["cases"][index]["actual"][0]=mc.bits(1.)
    m["cases"][index]["reference"][0]=mc.bits(1.)
    c["cases"][index]["actual"][0]+=1
    for report in (m,c): reconcile(report["cases"][index])
    result=compare(m,c)
    assert not result["reference_failures"] and result["different_fields"]==2


def test_original_loss_and_weight_ulp_budgets():
    m,c=reports(); index=find(family=1,kind=4,rows=1,mode=0)
    c["cases"][index]["actual"][0]=4; reconcile(c["cases"][index])
    assert not compare(m,c)["reference_failures"]
    c["cases"][index]["actual"][:2]=[5,1]; reconcile(c["cases"][index])
    errors=compare(m,c)["reference_failures"]
    assert len(errors)==1 and errors[0]["policy"]=="original-row-ulp" and errors[0]["count"]==2
    c["cases"][1]["actual"][0]=1; reconcile(c["cases"][1])
    assert len(compare(m,c)["reference_failures"])==2


def test_original_math_budgets_and_signed_zero_requirements():
    m,c=reports()
    for report in (m,c): report["cases"][0]["inputs"]["predictions"][0]=mc.bits(1.)
    c["cases"][0]["actual"][:5]=[1,1,1,0,6]
    assert not compare(m,c)["reference_failures"]
    c["cases"][0]["actual"][[3,4]]=[1,7]
    assert compare(m,c)["reference_failures"][0]["count"]==2
    m,c=reports()
    for report in (m,c): report["cases"][0]["inputs"]["predictions"][0]=mc.bits(-0.)
    m["cases"][0]["reference"][[1,3]]=mc.bits(-0.)
    m["cases"][0]["actual"][[1,3]]=mc.bits(-0.)
    result=compare(m,c)
    assert {e["policy"] for e in result["reference_failures"]}=={"original-math-signed-zero-1","original-math-signed-zero-3"}


def test_neumaier_compensation_is_recomputed_from_observed_terms():
    m,c=reports()
    for report in (m,c):
        report["cases"][1]["actual"][::2]=[mc.bits(1e20),mc.bits(1.),mc.bits(-1e20)]
        reconcile(report["cases"][1])
    m["cases"][1]["reference"][:]=m["cases"][1]["actual"]
    assert mc.value(c["cases"][1]["reductions"][0])==1.
    assert not compare(m,c)["reference_failures"]
    c["cases"][1]["reductions"][0]=mc.bits(0.,64)
    assert compare(m,c)["reference_failures"][0]["policy"]=="own-terms-diagnostic-0"


def test_original_reduction_bound_not_widened():
    m,c=reports()
    for report in (m,c): report["cases"][1]["actual"][0]=mc.bits(1.); reconcile(report["cases"][1])
    m["cases"][1]["reference"][0]=mc.bits(1.)
    c["cases"][1]["sums"][0]=mc.bits(1.+31*2.**-52,64)
    assert not compare(m,c)["reference_failures"]
    c["cases"][1]["sums"][0]=mc.bits(1.+33*2.**-52,64)
    assert compare(m,c)["reference_failures"][0]["policy"]=="original-own-terms-reduction-0"


def test_forged_diagnostics_cannot_create_a_looser_bound():
    m,c=reports(); c["cases"][1]["reductions"][1]=mc.bits(1e99,64)
    c["cases"][1]["sums"][0]=mc.bits(123.,64)
    result=compare(m,c)
    assert {e["policy"] for e in result["reference_failures"]}=={"own-terms-diagnostic-0","original-own-terms-reduction-0"}


def test_nonfinite_terms_and_nan_payload_differences_are_retained():
    m,c=reports()
    for report in (m,c): report["cases"][1]["actual"][[0,2]]=[mc.bits(float("inf")),mc.bits(-float("inf"))]; reconcile(report["cases"][1])
    m["cases"][1]["reference"][:]=m["cases"][1]["actual"]
    assert not compare(m,c)["reference_failures"]
    c["cases"][1]["sums"][0]=mc.bits(float("inf"),64)
    assert compare(m,c)["reference_failures"]
    m,c=reports(); index=find(family=1,kind=4,rows=1,mode=0)
    for report in (m,c): report["cases"][index]["actual"][0]=0x7fc00001 if report is m else 0xffc00002; reconcile(report["cases"][index])
    m["cases"][index]["reference"][0]=0x7fc00000
    result=compare(m,c)
    assert not result["reference_failures"] and result["different_fields"]>=1


def test_survival_analytic_and_exact_weight_checks():
    m,c=reports(); index=find(family=3)
    m["cases"][index]["analytic"][:]=[mc.bits(1.,64),mc.bits(2.,64)]
    for report in (m,c): report["cases"][index]["sums"][:]=m["cases"][index]["analytic"]
    c["cases"][index]["sums"][0]=mc.bits(1.+1e-14,64)
    assert not compare(m,c)["reference_failures"]
    c["cases"][index]["sums"][:]=[mc.bits(1.+3e-14,64),mc.bits(2.,64)+1]
    assert compare(m,c)["reference_failures"][0]["count"]==2
    index=find(family=4); c["cases"][index]["sums"][0]=mc.bits(1e-14,64)
    assert len(compare(m,c)["reference_failures"])==1  # Guard's original absolute floor is retained.


def test_missing_cuda_label_rejection_is_an_observed_failure():
    m=mc.decode(bytearray(encoded(1))); c=mc.decode(bytearray(encoded(2,True)))
    result=compare(m,c)
    assert len(result["reference_failures"])==4 and result["different_fields"]==4
    assert all(e["policy"]=="original-invalid-label-rejection" for e in result["reference_failures"])


def test_empty_calls_keep_exact_zero_and_signed_zero_differences():
    m,c=reports(); index=find(family=6)
    c["cases"][index]["sums"][0]=mc.bits(-0.,64)
    result=compare(m,c)
    assert not result["reference_failures"] and result["different_fields"]==1
    c["cases"][index]["sums"][0]=mc.bits(1e-300,64)
    assert compare(m,c)["reference_failures"][0]["policy"]=="original-empty-sums"


def test_every_input_stream_and_parameter_identity_is_checked():
    for name,family in (("predictions",1),("labels",1),("upper",4),("weights",4),("alphas",2)):
        m,c=reports(); index=find(family=family)
        c["cases"][index]["inputs"][name][0]^=np.uint32(1)
        with pytest.raises(ValueError,match="input bits differ"): compare(m,c)
    raw=bytearray(encoded(0)); raw[16+56]^=1
    with pytest.raises(ValueError,match="identity"): mc.decode(raw)


def test_all_bit_differences_counted_beyond_examples():
    m,c=reports(); index=find(family=1,kind=4,rows=257,mode=0)
    c["cases"][index]["actual"][:100:2]=1; reconcile(c["cases"][index])
    result=compare(m,c)
    assert not result["reference_failures"] and result["different_fields"]==51
    assert result["differences"][0]["count"]==50 and len(result["differences"][0]["first_fields"])==32


def test_framing_status_completion_and_output_overwrite(tmp_path,monkeypatch):
    raw=encoded(0)
    for bad in (raw[:-1],raw+b"!",raw[:8]+struct.pack("<I",3)+raw[12:]):
        with pytest.raises(ValueError): mc.decode(bad)
    bad=bytearray(encoded(2)); offset=16+60+mc.specs()[0].predictions*4
    struct.pack_into("<I",bad,offset,1)
    with pytest.raises(ValueError,match="status"): mc.decode(bad)
    output=tmp_path/"keep"; output.write_bytes(b"retained")
    monkeypatch.setattr(mc,"load",lambda _:mc.decode(raw))
    monkeypatch.setattr(sys,"argv",["compare","unpack","input","--output",str(output)])
    assert mc.main()==1 and output.read_bytes()==b"retained"


def test_undefined_casts_cannot_be_reported_as_cuda_passes_and_require_review(tmp_path,monkeypatch):
    m,c=reports(); index=find(family=5,mode=2)
    case=c['cases'][index]
    assert case['status']==2 and case['actual'].size==0 and case['sums'].size==0
    raw=bytearray(c['data']); decoded=mc.decode(raw); item=decoded['cases'][index]
    offset=item['actual'].__array_interface__['data'][0]-np.frombuffer(raw,dtype='u1').__array_interface__['data'][0]-4
    struct.pack_into('<I',raw,offset,1)
    with pytest.raises(ValueError,match='domain status'): mc.decode(raw)
    monkeypatch.setattr(mc,'load',lambda path:m if str(path)=='metal' else c)
    monkeypatch.setattr(sys,'argv',['compare','compare','metal','cuda','--output',str(tmp_path/'review.json')])
    assert mc.main()==2


def test_lossless_raw_gzip_and_xz_loading(tmp_path):
    import gzip
    import lzma
    raw=encoded(0)
    for suffix,packed in (('.bin',raw),('.bin.gz',gzip.compress(raw)),('.bin.xz',lzma.compress(raw))):
        p=tmp_path/('input'+suffix); p.write_bytes(packed)
        report=mc.load(p)
        assert report['data']==raw and report['kind']==0 and len(report['cases'])==375
