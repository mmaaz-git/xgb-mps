"""Synthetic report integrity and independent pair-oracle tests, not GPU evidence."""
import importlib.util
from pathlib import Path
import struct
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('grouped_roc_compare',ROOT/'scripts/compare_grouped_roc_contracts.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def encode(cases,kind):
    parts=[m.MAGIC,struct.pack('<II',kind,len(cases))]
    for c in cases:
        parts.append(struct.pack('<7I',*c['spec']));parts.extend(c[k].tobytes() for k in m.INPUTS)
        if kind:
            parts.append(struct.pack('<II',c['guarded'],c['cpu_overflow_rejected']))
            parts.extend(c[k].tobytes() for k in m.FIELDS)
    return b''.join(parts+[b'GROE',struct.pack('<I',len(cases))])


@pytest.fixture
def reports(monkeypatch):
    # One real small control shape; observations below are synthetic host values.
    shape=next((s,p) for s,p in m.shapes() if s.family==m.MAIN and s.rows==41 and s.mode==0)
    s,ptr=shape
    monkeypatch.setattr(m,'COUNT',1);monkeypatch.setattr(m,'shapes',lambda:[shape])
    src=m.source_inputs(s,ptr);stride,offset,_=m.layout(s)
    p=src['predictions'][offset:offset+s.rows*stride:stride];y=src['labels'][offset:offset+s.rows*stride:stride]
    ref=np.array([m.pair_reference(p[a:b],y[a:b],np.float32(1)) for a,b in zip(ptr[:-1],ptr[1:])])
    total=np.cumsum(ref,axis=0)[-1];result=[]
    for kind in (1,2):
        c=dict(spec=s,ptr=ptr.copy(),**{k:v.view('<u4').copy() for k,v in src.items()})
        c.update(guarded=0,cpu_overflow_rejected=0)
        values=dict(original=total if kind==1 else [],common=total,isolated=ref.ravel(),
                    per_query_guards=np.vstack(([-397.,973.],[-397.,973.],ref,[-397.,973.],[-397.,973.])).ravel() if kind==1 else [],
                    cpu=ref[np.diff(ptr)>=3,0] if kind==1 else [])
        c.update({k:np.asarray(v,dtype='<f8').view('<u8') for k,v in values.items()})
        result.append(m.decode(encode([c],kind)))
    return result


def test_full_control_matrix():
    shapes=m.shapes()
    assert len(shapes)==278
    assert [sum(s.family==f for s,p in shapes) for f in range(5)]==[1,1,1,272,3]
    assert shapes[2][0].groups==390 and shapes[2][0].rows==19110
    assert sum(s.rows for s,p in shapes if s.family==m.MAIN)==161680
    assert sum(s.groups for s,p in shapes if s.family!=m.LARGE)==2567
    assert all(int(p[-1])*(int(p[-1])-1)//2==500002500003 for s,p in shapes if s.family==m.LARGE)


def test_synthetic_roundtrip_and_reference(reports):
    a,b=reports
    assert not m.validate(a)['reference_failures'] and not m.validate(b)['reference_failures']
    assert m.unpack(a)==m.unpack(b)
    assert m.decode(m.unpack(a))['kind']==0
    result=m.compare(a,b)
    assert result['status']=='review-required' and not result['bit_differences']


def test_identical_wrong_results_cannot_pass(reports):
    bad=[]
    for r in reports:
        c=r['cases'][0];c['common']=c['common'].copy();c['common'][0]=np.array([99.]).view('<u8')[0]
        bad.append(m.decode(encode([c],r['kind'])))
    assert m.compare(*bad)['status']=='reference-failure'


@pytest.mark.parametrize('field',['isolated','per_query_guards','cpu'])
def test_individual_observation_corruption_detected(reports,field):
    r=reports[0];c=r['cases'][0];c[field]=c[field].copy();c[field][0]=np.array([99.]).view('<u8')[0]
    assert m.validate(m.decode(encode([c],1)))['reference_failures']


@pytest.mark.parametrize('damage',['truncated','trailing','count','identity','ptr','guard','cpu-guard','missing'])
def test_malformed_reports_rejected(reports,damage):
    r=reports[0];c=r['cases'][0];data=encode([c],1)
    if damage=='truncated':data=data[:-1]
    elif damage=='trailing':data+=b'x'
    elif damage=='count':data=data[:12]+struct.pack('<I',2)+data[16:]
    elif damage=='identity':data=data[:16]+struct.pack('<I',999)+data[20:]
    else:
        if damage=='ptr':c['ptr']=c['ptr'].copy();c['ptr'][1]+=1
        if damage=='guard':c['guarded']=1
        if damage=='cpu-guard':c['cpu_overflow_rejected']=1
        if damage=='missing':c['common']=np.array([],dtype='<u8')
        data=encode([c],1)
    with pytest.raises(ValueError):m.decode(data)


def test_original_input_bits_required(reports):
    r=reports[0];c=r['cases'][0];c['predictions']=c['predictions'].copy();c['predictions'][0]^=1
    with pytest.raises(ValueError,match='original predictions bits'):m.validate(m.decode(encode([c],1)))


def serial_pairs(p,y,w):
    # Scalar input-order enumeration from the original native oracle.
    square=np.float32(w)*np.float32(w);area=total=0.
    if len(p)<3:return np.array([0.,0.])
    with np.errstate(all='ignore'):
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                d=np.float32(p[j]-p[i]) if y[i]<y[j] else np.float32(p[i]-p[j])
                value=np.float32(1 if d>0 else .5 if d==0 else 0)
                area+=float(np.float32(value*square));total+=float(square)
        return np.array([area/total if total>0 else 0.,1.])


@pytest.mark.parametrize('weight',[.7,-.7,0.,float.fromhex('0x1.8p-75'),float.fromhex('0x1.4p-74'),1e30,np.nan])
def test_pair_oracle_preserves_fp32_products_and_ties(weight):
    p=np.array([np.inf,-np.inf,np.nan,0.,-0.,2.**-149,1.,1.],dtype='<f4')
    y=np.array([1,0,1,2,2,1,0,0],dtype='<f4')
    with np.errstate(all='ignore'):
        assert np.all(m.near(m.pair_reference(p,y,np.float32(weight)),serial_pairs(p,y,weight),True))


def test_tied_subnormal_weight_square_changes_half_credit():
    p=np.ones(3,dtype='<f4');y=np.ones(3,dtype='<f4')
    assert m.pair_reference(p,y,np.float32(float.fromhex('0x1.8p-75')))[0]==0.
    assert m.pair_reference(p,y,np.float32(float.fromhex('0x1.4p-74')))[0]==2./3.


def test_large_cuda_guard_has_no_fabricated_outputs():
    s,ptr=m.shapes()[-1]
    assert all(v==0 for v in m.field_sizes(s,ptr,2).values())
    assert m.field_sizes(s,ptr,1)['original']==2
