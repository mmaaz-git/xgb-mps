"""Synthetic interchange/comparator adversarial checks, never CUDA execution evidence."""
from functools import lru_cache
import gzip
import importlib.util
import lzma
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
SPEC=importlib.util.spec_from_file_location("stump_compare",ROOT/"scripts/compare_stump_contracts.py")
sc=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(sc)


@lru_cache
def zeros(): return [np.zeros(2*s.pairs,dtype="<u4") for s in sc.specs()]


def reports():
    result=[]
    for kind in (1,2):
        cases=[]
        for s,storage in zip(sc.specs(),zeros()):
            case={"spec":s,"storage":storage,"sums":np.zeros(2*s.targets,dtype="<u8"),
                  "weights":np.full(s.targets,sc.bits(-0.,32),dtype="<u4")}
            if kind==1: case["references"]=np.zeros(4*s.targets,dtype="<u8")
            else: case["api_weights"]=case["weights"].copy()
            cases.append(case)
        result.append({"kind":kind,"cases":cases})
    return result


def seed(m,c,index,g=1.,h=1.):
    a,b=m["cases"][index],c["cases"][index]
    data=a["storage"].copy(); data[:2]=[sc.bits(g,32),sc.bits(h,32)]
    a["storage"]=data; b["storage"]=data.copy()
    ref=sc.references(a); a["references"][:]=ref.view("<u8")
    for case in (a,b):
        case["sums"][:]=ref[::2].copy().view("<u8")
        for t in range(a["spec"].targets):
            case["weights"][t]=sc.bits(sc.weight(*ref[4*t:4*t+4:2]),32)
    b["api_weights"][:]=b["weights"]
    return a,b


def test_complete_original_case_matrix_and_independent_zeros():
    specs=sc.specs()
    assert len(specs)==405 and sum(s.rows*s.targets for s in specs)==11964810
    assert sum(s.targets for s in specs)==1485
    m,c=reports(); result=sc.compare(m,c)
    assert result["status"]=="observed-bit-identical" and result["compared_fields"]==5940
    assert not sc.validate_metal(m)
    with pytest.raises(ValueError,match="actual CUDA"): sc.compare(m,m)


def test_identical_wrong_sums_and_forged_references_cannot_pass():
    m,c=reports()
    for case in (m["cases"][0],c["cases"][0]): case["sums"][0]=sc.bits(1.)
    m["cases"][0]["references"][0]=sc.bits(1.)
    result=sc.compare(m,c)
    assert {r["policy"] for r in result["reference_failures"]} >= {"original-input-reference","original-exact-sum"}
    assert not result["different_fields"]


def test_exact_modes_and_original_8e14_bound_are_unchanged():
    m,c=reports(); a,b=seed(m,c,0)
    b["sums"][0]=sc.bits(np.nextafter(1.,2.))
    assert "original-exact-sum" in {r["policy"] for r in sc.compare(m,c)["reference_failures"]}
    m,c=reports(); a,b=seed(m,c,2)
    b["sums"][0]=sc.bits(1.+.9*8e-14)
    assert not sc.compare(m,c)["reference_failures"]
    b["sums"][0]=sc.bits(1.+1.1*8e-14)
    assert "original-sum-bound" in {r["policy"] for r in sc.compare(m,c)["reference_failures"]}


def test_each_weight_uses_own_observed_sums_and_signed_zero():
    m,c=reports(); c["cases"][0]["weights"][0]=sc.bits(0.,32)
    result=sc.compare(m,c)
    assert [r["policy"] for r in result["reference_failures"]]==["original-weight-from-own-sums"]
    assert result["different_fields"]==1 and result["api_probe_different_fields"]==1


def test_api_corruption_fails_even_when_diagnostic_passes():
    m,c=reports(); seed(m,c,2)
    c["cases"][2]["api_weights"][0]=sc.bits(-2.,32)
    result=sc.compare(m,c)
    assert [r["policy"] for r in result["reference_failures"]]==["api-weight-original-sum-interval"]
    assert result["api_probe_different_fields"]==1


def test_cancellation_api_interval_allows_original_sum_error_but_reports_difference():
    m,c=reports(); index=next(s.id for s in sc.specs() if s.rows==31 and s.mode==2)
    a,b=seed(m,c,index)
    for case in (a,b):
        data=case["storage"]; stride=2*case["spec"].stride0
        data[[0,stride,2*stride]]=[sc.bits(2.**70,32),sc.bits(-2.**70,32),sc.bits(1.,32)]
    a["references"][:]=sc.references(a).view("<u8")
    # Actual API can differ from a separately executed sum trace within the
    # same original compensated-sum bound. Preserve that as review, not parity.
    b["api_weights"][0]=sc.bits(-2.,32)
    result=sc.compare(m,c)
    assert not result["reference_failures"] and result["status"]=="review-differences"
    assert result["different_fields"]==1 and result["api_probe_different_fields"]==1
    assert sc.compensated([2.**70,1.,-2.**70])==(1.,2.**71)


def test_input_padding_bits_must_match_and_nonfinite_terms_are_rejected():
    m,c=reports(); c["cases"][0]["storage"]=c["cases"][0]["storage"].copy()
    c["cases"][0]["storage"][-1]=1
    with pytest.raises(ValueError,match="input bits"): sc.compare(m,c)
    m,c=reports()
    with pytest.raises(ValueError,match="Nonfinite"):
        seed(m,c,0,float('nan'))


def encode(report,kind=None):
    kind=report["kind"] if kind is None else kind
    chunks=[sc.MAGIC,struct.pack("<II",kind,sc.COUNT)]
    for case in report["cases"]:
        chunks.extend([struct.pack("<8I",*case["spec"]),case["storage"].tobytes()])
        if kind: chunks.extend([case["sums"].tobytes(),case["weights"].tobytes()])
        if kind==1: chunks.append(case["references"].tobytes())
        if kind==2: chunks.append(case["api_weights"].tobytes())
    return b"".join(chunks)+b"STPE"+struct.pack("<I",sc.COUNT)


def test_framing_identities_count_and_lossless_compression(tmp_path,monkeypatch):
    m,c=reports(); raw=encode(m,0)
    assert sc.decode(raw)["kind"]==0
    for bad in (raw[:-1],raw+b'x',raw[:12]+struct.pack('<I',404)+raw[16:],raw[:16]+struct.pack('<I',7)+raw[20:]):
        with pytest.raises(ValueError): sc.decode(bad)
    for suffix,compress in [('.bin',lambda v:v),('.gz',gzip.compress),('.xz',lambda v:lzma.compress(v,preset=1))]:
        path=tmp_path/('inputs'+suffix); path.write_bytes(compress(raw))
        assert sc.load(path)["data"]==raw
    output=tmp_path/'new.bin'; source=tmp_path/'inputs.xz'
    monkeypatch.setattr(sys,'argv',['compare','unpack',str(source),str(output)])
    assert sc.main()==0 and output.read_bytes()==raw
    with pytest.raises(FileExistsError): sc.main()
    metal=sc.decode(encode(m)); cuda=sc.decode(encode(c))
    sc.match_inputs(metal,cuda)
    assert not sc.validate_metal(metal)


def test_cli_retains_review_failures_and_never_overwrites(tmp_path,monkeypatch):
    m,c=reports(); a,b=seed(m,c,2)
    b['sums'][0]=sc.bits(1.+.9*8e-14)
    monkeypatch.setattr(sc,'load',lambda p:m if str(p)=='metal' else c)
    output=tmp_path/'review.json'
    monkeypatch.setattr(sys,'argv',['compare','compare','metal','cuda','--output',str(output)])
    assert sc.main()==2
    with pytest.raises(ValueError,match='already exists'): sc.main()
    b['sums'][0]=sc.bits(2.)
    monkeypatch.setattr(sys,'argv',['compare','compare','metal','cuda','--output',str(tmp_path/'failure.json')])
    assert sc.main()==1
