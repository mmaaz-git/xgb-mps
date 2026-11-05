"""Report-integrity tests with synthetic observations; these are not GPU evidence."""
import importlib.util
from pathlib import Path
import struct
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('ranking_metric_compare',ROOT/'scripts/compare_ranking_metrics.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def encode(c,kind):
    pieces=[m.MAGIC,struct.pack('<II',kind,1),struct.pack('<13I',*c['spec'])]
    pieces.extend(c[k].tobytes() for k in m.INPUTS)
    if kind:
        pieces.append(struct.pack('<I',c['guards']));pieces.extend(c[k].tobytes() for k in m.FIELDS)
    return b''.join(pieces+[b'RMTE',struct.pack('<I',1)])


def synthetic(monkeypatch,shape):
    s,ptr=shape
    monkeypatch.setattr(m,'COUNT',1);monkeypatch.setattr(m,'shapes',lambda:[shape])
    src=next(m.source_inputs());inputs={k:src[k].view(f'<u{src[k].dtype.itemsize}').copy() for k in m.INPUTS}
    base=dict(spec=s,**inputs);ref=m.reference(base);common=m.reference(base,True)
    reports=[]
    for kind in (1,2):
        c=dict(base,guards=m.guards(s) if kind==2 else 0)
        values=dict(original=ref['pair'],common=common['pair'],reducer=common['reducer'],prefix=ref['prefix'],order=ref['order'])
        for name,n in m.field_sizes(s,kind).items():
            a=np.asarray(values[name],dtype='<u8' if name=='order' else '<f8') if n else np.array([],dtype='<u8')
            c[name]=a.view('<u8').copy()
        reports.append(m.decode(bytearray(encode(c,kind))))
    return reports


@pytest.fixture
def reports(monkeypatch):
    shape=next((s,p) for s,p in m.shapes() if s.family==m.SCORE and s.kind==3 and s.rows==289 and s.topk==10 and s.mode==2 and s.stride==1)
    return synthetic(monkeypatch,shape)


def test_complete_matrix():
    shapes=m.shapes()
    assert len(shapes)==1031
    assert [sum(s.family==f for s,p in shapes) for f in range(3)]==[15,120,896]
    assert sum(s.rows for s,p in shapes if s.family==m.SCORE)==9511680
    assert sum(bool(m.guards(s)&1) for s,p in shapes)==227
    assert sum(bool(m.guards(s)&2) for s,p in shapes)==64
    assert sum(s.family==m.SORT and s.stride==3 for s,p in shapes)==60


def test_synthetic_agreement_requires_review(reports):
    a,b=reports
    assert not m.validate(a)['failures'] and not m.validate(b)['failures']
    result=m.compare(a,b)
    assert result['status']=='review-required' and not result['runtime_equivalence_certified']


def test_identical_wrong_gpus_fail(reports):
    a,b=reports
    for r in (a,b):r['cases'][0]['common'].view('<f8')[0]=123456
    assert m.compare(a,b)['status']=='failed-reference-or-budget'


def test_wrong_prefix_with_consistent_downstream_still_fails(reports):
    a,b=reports;c=b['cases'][0];c['prefix'].view('<f8')[:]+=1
    ref=m.reference(c,True,c['prefix'].view('<f8'))
    c['reducer'][:]=ref['reducer'].view('<u8')
    c['common'][:]=m.reference(c,True,c['prefix'].view('<f8'),c['reducer'].view('<f8'))['pair'].view('<u8')
    assert any(f['field']=='prefix' for f in m.validate(b)['failures'])
    assert m.compare(a,b)['status']=='failed-reference-or-budget'


def test_changed_sort_rejected(reports):
    a,b=reports;b['cases'][0]['order'][0]=999999
    assert any(f['field']=='order' for f in m.validate(b)['failures'])


@pytest.mark.parametrize('field',['predictions','labels','weights','discounts','inverse','ptr'])
def test_changed_inputs_fail_even_if_gpus_agree(reports,field):
    a,b=reports
    for r in (a,b):r['cases'][0][field][0]^=np.array(1,dtype=r['cases'][0][field].dtype)
    with pytest.raises(ValueError,match='changed source input'):m.compare(a,b)


@pytest.mark.parametrize('change',['magic','kind','count','identity','guard','truncated','trailing','footer'])
def test_malformed_report_rejected(reports,change):
    raw=bytearray(reports[1]['data'])
    if change=='magic':raw[0]^=1
    elif change=='kind':struct.pack_into('<I',raw,8,8)
    elif change=='count':struct.pack_into('<I',raw,12,2)
    elif change=='identity':raw[16]^=1
    elif change=='guard':
        c=reports[1]['cases'][0];offset=16+52+sum(c[k].nbytes for k in m.INPUTS)
        struct.pack_into('<I',raw,offset,255)
    elif change=='truncated':raw=raw[:-9]
    elif change=='trailing':raw+=b'x'
    else:raw[-8]^=1
    with pytest.raises(ValueError):m.decode(bytes(raw))


def test_custom_discount_original_and_comparable_call(monkeypatch):
    a,b=synthetic(monkeypatch,m.shapes()[11]);c=a['cases'][0]
    assert c['original'][0]!=c['common'][0]
    assert not m.validate(a)['failures'] and not m.validate(b)['failures']
    c['original'][:]=c['common']
    assert m.validate(a)['failures']


def test_zero_group_ndcg_has_no_fabricated_cuda_outputs(monkeypatch):
    shape=next((s,p) for s,p in m.shapes() if m.no_groups(s) and s.topk==0)
    a,b=synthetic(monkeypatch,shape)
    assert b['cases'][0]['guards']==3
    assert all(not len(b['cases'][0][k]) for k in m.FIELDS)
    assert len(a['cases'][0]['common'])==2
    assert m.compare(a,b)['status']=='review-required'


def test_exact_edge_order_cannot_use_score_tolerance(monkeypatch):
    a,b=synthetic(monkeypatch,m.shapes()[8]);c=a['cases'][0]
    c['original'].view('<f8')[0]=np.nextafter(c['original'].view('<f8')[0],np.inf)
    assert any(f['field']=='original' for f in m.validate(a)['failures'])


def test_unpack_only_inputs(reports,tmp_path):
    a,b=reports;path=tmp_path/'inputs.bin';m.unpack(a,path)
    result=m.load(path)
    assert result['kind']==0 and not any(k in result['cases'][0] for k in m.FIELDS)
    assert not m.validate(result)['failures']
    with pytest.raises(ValueError,match='already exists'):m.unpack(b,path)
