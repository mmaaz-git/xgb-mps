"""Synthetic report integrity and independent arithmetic witnesses, not GPU runs."""
import gzip
import importlib.util
import json
import lzma
from pathlib import Path
import struct
import sys
import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
SPEC=importlib.util.spec_from_file_location('linear_compare',ROOT/'scripts/compare_linear_contracts.py')
lc=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(lc)


def pairwise(source):
    if not len(source):
        return np.zeros(source.shape[1:],dtype='<f4')
    values=source.copy()
    while len(values)>1:
        if len(values)%2:
            values=np.concatenate((values,np.zeros((1,*values.shape[1:]),dtype='<f4')))
        values=np.add(values[::2],values[1::2],dtype='<f4')
    return values[0]


@pytest.fixture
def reports(monkeypatch):
    # Keep all arithmetic/domain families, but no million-row allocations in
    # synthetic corruption tests. The real full corpus is validated separately.
    selected=[lc.specs()[i] for i in (0,1,2,3,4,12,13,14,15,20,28,36,53,54,55)]
    monkeypatch.setattr(lc,'specs',lambda:selected)
    monkeypatch.setattr(lc,'COUNT',len(selected))
    monkeypatch.setattr(lc,'STEPS',sum(len(lc.actions(s)) for s in selected))
    reports=[dict(kind=1,cases=[]),dict(kind=2,cases=[])]
    with np.errstate(all='ignore'):
        for s in selected:
            case=dict(spec=s,**lc.source_inputs(s))
            original=case['original'][2:-2] if s.family==0 else case['original']
            state=original.view('<f4').reshape(s.rows,s.groups,2).copy()
            steps=[]
            for a in lc.actions(s):
                step=dict(action=a,state=state.ravel().view('<u4').copy())
                gradient=pairwise(lc.terms(case,state,a.feature)).ravel().view('<u4') if a.gradient and a.domain!=2 else np.array([],dtype='<u4')
                lc.update(case,state,a)
                residual=state.ravel().view('<u4').copy() if a.copy else np.array([],dtype='<u4')
                if a.copy and s.family==0:
                    guard=np.array([-999.,998.],dtype='<f4').view('<u4')
                    residual=np.concatenate((guard,residual,guard))
                steps.append(dict(step,gradients=gradient,residuals=residual))
            for report in reports:
                copy=dict(case,steps=[])
                for step in steps:
                    out=dict(step,gradients=step['gradients'].copy(),residuals=step['residuals'].copy(),state=step['state'].copy(),
                             status=lc.status(report['kind'],step['action']))
                    if report['kind']==2 and step['action'].domain:
                        out['residuals']=np.array([],dtype='<u4')
                    copy['steps'].append(out)
                report['cases'].append(copy)
    return reports


def encode(report,kind=None):
    kind=report['kind'] if kind is None else kind
    out=bytearray(lc.MAGIC+struct.pack('<II',kind,lc.COUNT))
    for case in report['cases']:
        out.extend(struct.pack('<6I',*case['spec']))
        for key in lc.INPUTS:
            out.extend(case[key].tobytes())
        for i,step in enumerate(case['steps']):
            a=step['action']
            out.extend(struct.pack('<7I',i,a.feature&0xffffffff,a.gradient,a.update,a.copy,a.delta,a.domain))
            out.extend(step['state'].tobytes())
            if kind:
                out.extend(struct.pack('<I',step['status']))
                out.extend(step['gradients'].tobytes());out.extend(step['residuals'].tobytes())
    return bytes(out)+b'LINE'+struct.pack('<II',lc.COUNT,lc.STEPS)


def test_full_original_matrix_actions_fields_and_exclusions():
    specs=lc.specs()
    assert len(specs)==56 and sum(s.family==0 for s in specs)==49
    assert sum(len(lc.actions(s)) for s in specs)==893
    assert sum(s.rows*s.groups for s in specs if s.family==0)==4118809
    assert sum(a.domain==1 for s in specs for a in lc.actions(s))==123
    assert sum(a.domain==2 for s in specs for a in lc.actions(s))==3
    sums=sum(2*s.groups for s in specs for a in lc.actions(s) if a.gradient and a.domain!=2)
    residuals=sum(2*s.rows*s.groups+(4 if s.family==0 else 0) for s in specs for a in lc.actions(s) if a.copy)
    cuda_residuals=sum(2*s.rows*s.groups+(4 if s.family==0 else 0) for s in specs for a in lc.actions(s) if a.copy and a.domain==0)
    assert sums==3570 and residuals==148280662 and cuda_residuals==123567316


def test_all_synthetic_families_match_original_bounds(reports):
    result=lc.validate(*reports)
    assert not result['reference_failures'] and not result['different_actual_fields']
    assert result['domain_exclusions'] and result['status']=='review-differences-and-domains'
    with pytest.raises(ValueError,match='actual CUDA'):
        lc.validate(reports[0],reports[0])


def test_analytical_reduction_exactness_bound_and_nonfinite_classification():
    source=np.array([[[2.**24,1]],[[1,1]],[[-2.**24,1]]],dtype='<f4')
    good,exact,_=lc.sum_checks(np.array([[0,3]],dtype='<f4'),source)
    assert good.all() and exact.tolist()==[[False,True]]
    good,_,_=lc.sum_checks(np.array([[100,4]],dtype='<f4'),source)
    assert not good.any()
    source=np.array([[[np.inf,1]],[[-np.inf,1]]],dtype='<f4')
    with np.errstate(all='ignore'):
        assert lc.sum_checks(np.array([[np.nan,2]],dtype='<f4'),source)[0].all()
        assert not lc.sum_checks(np.array([[np.inf,2]],dtype='<f4'),source)[0][0,0]


def test_analytical_residual_product_order_and_duplicate_sequence():
    h=np.array(0x401758e3,dtype='<u4').view('<f4')
    step=np.array(0x3fd4c3f0,dtype='<u4').view('<f4')
    x=np.array(0x3f807d7e,dtype='<u4').view('<f4')
    assert lc.bits((h*step)*x)==0x407c898a
    assert lc.bits((h*x)*step)==0x407c898b
    case=dict(offsets=np.array([0,2],dtype='<u8'),rows=np.array([0,0],dtype='<u4'),
              entries=np.array([3.,-.5],dtype='<f4').view('<u4'))
    state=np.array([[[7.,-2.]]],dtype='<f4')
    terms=lc.terms(case,state,0)
    assert terms.sum(axis=0).tolist()==[[17.5,-18.5]]
    lc.update(case,state,lc.Action(0,0,1,1,lc.bits(.5),1))
    assert state.tolist()==[[[4.5,-2.]]]


def test_zero_bias_skips_but_zero_feature_changes_exceptional_state():
    case=lc.source_inputs(lc.specs()[3])
    state=case['original'].view('<f4').reshape(1,1,2).copy()
    before=state.view('<u4').copy()
    lc.update(case,state,lc.actions(lc.specs()[3])[0])
    assert np.array_equal(state.view('<u4'),before)
    with np.errstate(all='ignore'):
        lc.update(case,state,lc.actions(lc.specs()[3])[1])
    assert np.isnan(state[0,0,0]) and state.view('<u4')[0,0,1]==0


@pytest.mark.parametrize('field,index',[('gradients',0),('residuals',2)])
def test_identical_wrong_outputs_cannot_pass(reports,field,index):
    for report in reports:
        report['cases'][0]['steps'][index][field][0]^=np.uint32(1)
    result=lc.validate(*reports)
    assert result['different_actual_fields']==0
    assert {f['backend'] for f in result['reference_failures']}=={'metal','cuda'}


def test_identical_wrong_intermediate_inputs_are_reconstructed(reports):
    for report in reports:
        report['cases'][0]['steps'][2]['state'][0]^=np.uint32(1)
    failures=lc.validate(*reports)['reference_failures']
    assert any(f['backend']=='input' and f['field']=='state' for f in failures)


@pytest.mark.parametrize('field',['offsets','rows','entries','original'])
def test_source_bits_must_match_and_preserve_original_generator(reports,field):
    for report in reports:
        report['cases'][0][field]=report['cases'][0][field].copy()
    reports[1]['cases'][0][field][-1]^=np.array(1,dtype=reports[1]['cases'][0][field].dtype)
    with pytest.raises(ValueError,match='input bits differ'):
        lc.validate(*reports)
    reports[0]['cases'][0][field][-1]^=np.array(1,dtype=reports[0]['cases'][0][field].dtype)
    with pytest.raises(ValueError,match='generator bits differ'):
        lc.validate(*reports)


def test_output_copy_guards_remain_exact(reports):
    reports[1]['cases'][5]['steps'][0]['residuals'][0]=lc.bits(-998.)
    assert any(f['field']=='residuals' for f in lc.validate(*reports)['reference_failures'])


def test_nan_payload_and_numeric_zero_bits_remain_visible(reports):
    # The zero-feature output is NaN; native tests accept its classification.
    reports[1]['cases'][3]['steps'][1]['residuals'][0]=0x7fc00123
    # An empty reduction is numerically zero regardless of its sign bit.
    reports[1]['cases'][4]['steps'][0]['gradients'][0]=0x80000000
    result=lc.validate(*reports)
    assert not result['reference_failures'] and result['different_actual_fields']==2


@pytest.mark.parametrize('case_index,step_index,status',[(5,3,0),(-1,0,1)])
def test_cuda_exclusions_cannot_be_relabelled_as_executed_or_rejected(reports,case_index,step_index,status):
    reports[1]['cases'][case_index]['steps'][step_index]['status']=status
    with pytest.raises(ValueError,match='exclusion status'):
        lc.decode(encode(reports[1]))


def test_input_states_match_exactly_across_backends(reports):
    reports[1]['cases'][0]['steps'][0]['state'][0]^=np.uint32(1)
    with pytest.raises(ValueError,match='input state bits differ'):
        lc.validate(*reports)


def test_complete_framing_and_compression(reports,tmp_path):
    raw=encode(reports[0],0)
    for broken in (raw[:-1],raw+b'x',b'BADMAGIC'+raw[8:],raw[:8]+struct.pack('<I',9)+raw[12:],
                   raw[:12]+struct.pack('<I',lc.COUNT-1)+raw[16:],raw[:16]+struct.pack('<I',99)+raw[20:]):
        with pytest.raises(ValueError):
            lc.decode(broken)
    for suffix,compress in (('.bin',lambda b:b),('.gz',gzip.compress),('.xz',lzma.compress)):
        path=tmp_path/('source'+suffix);path.write_bytes(compress(raw))
        assert bytes(lc.load(path)['data'])==raw
    assert not lc.validate(*(lc.decode(encode(r)) for r in reports))['reference_failures']


def test_input_only_unpack_and_no_overwrite(reports,tmp_path,monkeypatch):
    source,out=tmp_path/'source.bin',tmp_path/'out.bin';source.write_bytes(encode(reports[0]))
    monkeypatch.setattr(sys,'argv',['compare','unpack',str(source),str(out)])
    with pytest.raises(ValueError,match='input-only'):
        lc.main()
    assert not out.exists()
    source.write_bytes(encode(reports[0],0))
    assert lc.main()==0 and out.read_bytes()==source.read_bytes()
    with pytest.raises(FileExistsError):
        lc.main()


def test_cli_preserves_exclusion_review_and_failed_checks(reports,tmp_path,monkeypatch):
    monkeypatch.setattr(lc,'load',lambda p:reports[int(str(p)=='cuda')])
    out=tmp_path/'review.json'
    monkeypatch.setattr(sys,'argv',['compare','compare','metal','cuda','--output',str(out)])
    assert lc.main()==2 and json.loads(out.read_text())['domain_exclusions']
    with pytest.raises(ValueError,match='already exists'):
        lc.main()
    reports[1]['cases'][0]['steps'][0]['gradients'][0]^=np.uint32(1)
    monkeypatch.setattr(sys,'argv',['compare','compare','metal','cuda','--output',str(tmp_path/'failed.json')])
    assert lc.main()==1
