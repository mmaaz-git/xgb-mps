"""Interchange/reference corruption checks; synthetic reports are not GPU evidence."""
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('auc_comparison',ROOT/'scripts/compare_auc_contracts.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def encode(cases,kind):
    parts=[m.MAGIC,struct.pack('<II',kind,len(cases))]
    for c in cases:
        parts.append(struct.pack('<10I',*c['spec']))
        parts.extend(c[k].tobytes() for k in m.INPUTS)
        if kind:
            for obs in c['calls']:
                parts.append(struct.pack('<I',obs['status']))
                for k in m.FIELDS:
                    parts.extend((struct.pack('<Q',len(obs[k])),obs[k].tobytes()))
    return b''.join(parts+[b'AUCE',struct.pack('<I',len(cases))])


@pytest.fixture
def edge_reports(monkeypatch):
    specs=m.specs()[:4]
    monkeypatch.setattr(m,'COUNT',4);monkeypatch.setattr(m,'specs',lambda:specs)
    reports=[]
    fp=2.*float(np.float32((1.-float(np.float32(.25)))*float(np.float32(.7))))
    tp=2.*float(np.float32(.25)*np.float32(.7))
    expected=([0.,1.,0.],[0.,0.,0.],[fp,tp,fp*tp*.5],[1.,1.,1./(fp/tp+1.)])
    for kind in (1,2):
        cases=[]
        for s in specs:
            src=m.source_inputs(s,np.random.RandomState(718))
            c=dict(spec=s,**{k:v.view('<u8' if k=='summaries' else '<u4') for k,v in src.items()})
            obs=dict(status=0,**{k:np.array([],dtype='<u8') for k in m.FIELDS})
            obs['curves']=np.array(expected[s.id],dtype='<f8').view('<u8')
            if kind==2:
                obs['linked']=obs['curves'].copy()
                if not s.roc:
                    y,w=src['labels'],src['weights'] if s.weights else np.ones(s.rows,dtype='<f4')
                    obs['totals']=np.array([np.sum((np.float32(1)-y)*w,dtype=np.float64),np.sum(y*w,dtype=np.float64)],dtype='<f8').view('<u8')
            c['calls']=[obs];cases.append(c)
        reports.append(m.decode(encode(cases,kind)))
    return reports


def test_complete_matrix_and_special_shapes():
    s=m.specs()
    assert len(s)==528
    assert [sum(x.family==f for x in s) for f in range(7)]==[4,22,8,1,6,486,1]
    assert s[20].targets==65535 and s[23].targets==262145
    assert [s[i].targets for i in (35,37,39)]==[65535,65536,65537]
    assert m.storage(s[41],1)>0  # Empty main case retains backing guards.


def test_synthetic_reference_and_roundtrip(edge_reports):
    metal,cuda=edge_reports
    assert not m.validate(metal)['reference_failures']
    assert not m.validate(cuda)['reference_failures']
    assert m.unpack(metal)==m.unpack(cuda)
    assert m.decode(m.unpack(metal))['kind']==0
    result=m.compare(metal,cuda)
    assert result['status']=='review-required' and not result['bit_differences']


def test_two_identical_wrong_gpus_fail_reference(edge_reports):
    reports=[]
    for r in edge_reports:
        cases=r['cases']
        obs=cases[2]['calls'][0]
        obs['curves']=obs['curves'].copy();obs['curves'][2]=np.array([999.]).view('<u8')[0]
        if r['kind']==2:obs['linked']=obs['curves'].copy()
        reports.append(m.decode(encode(cases,r['kind'])))
    assert m.compare(*reports)['status']=='reference-failure'


def test_linked_library_disagreement_rejected(edge_reports):
    r=edge_reports[1];obs=r['cases'][2]['calls'][0]
    obs['linked']=obs['linked'].copy();obs['linked'][2]+=np.uint64(1)
    out=m.validate(m.decode(encode(r['cases'],2)))
    assert any(x['field']=='probe/linked' for x in out['reference_failures'])


@pytest.mark.parametrize('damage',['truncated','trailing','count','identity','missing','status','oversize'])
def test_report_corruption_rejected(edge_reports,damage):
    r=edge_reports[0];data=encode(r['cases'],1)
    if damage=='truncated':data=data[:-1]
    elif damage=='trailing':data+=b'x'
    elif damage=='count':data=data[:12]+struct.pack('<I',5)+data[16:]
    elif damage=='identity':data=data[:16]+struct.pack('<I',999)+data[20:]
    else:
        c=r['cases'];obs=c[0]['calls'][0]
        if damage=='missing':obs['curves']=np.array([],dtype='<u8')
        if damage=='status':obs['status']=1
        if damage=='oversize':obs['curves']=np.zeros(10,dtype='<u8')
        data=encode(c,1)
    with pytest.raises(ValueError):m.decode(data)


def test_changed_input_rejected_even_when_both_reports_match(edge_reports):
    r=edge_reports[0];c=r['cases'][0];c['predictions']=c['predictions'].copy();c['predictions'][0]=1
    report=m.decode(encode(r['cases'],1))
    with pytest.raises(ValueError,match='original predictions bits'):m.validate(report)


def test_cuda_skipped_stages_are_not_zero_filled():
    s=m.Spec(0,m.MAIN,4,0,1,1,0,2,1,0)
    status,shape=m.observation_shape(s,1,2,np.array([0.,1.]))
    assert status==0 and shape['totals']==2 and shape['terms']==shape['prefixes']==shape['order']==0
    empty=s._replace(rows=0,weights=0)
    status,shape=m.observation_shape(empty,0,2,np.array([]))
    assert status==1 and sum(shape.values())==0
    status,shape=m.observation_shape(empty._replace(targets=3),1,2,np.zeros(6))
    assert status==0 and shape['metadata']==9 and shape['source_curves']==0 and shape['totals']==6


def test_exact_prefix_keeps_tiny_mass_before_rounding():
    values=np.array([2.**53,1.,1.],dtype='<f4')
    assert np.cumsum(values.astype(np.float64))[-1]==2.**53
    assert m.exact_prefix(values)[-1]==2.**53+2


def test_skipped_cuda_integral_must_still_return_zero():
    s=m.Spec(0,m.MAIN,4,0,1,1,0,2,1,0)
    obs={k:np.array([],dtype='<u8') for k in ('terms','prefixes','order')}
    obs['totals']=np.array([0.,1.]).view('<u8')
    errors=[]
    def check(name,a,b,exact=False):
        if not np.all(m.near(a,b,exact=exact)):errors.append(name)
    m.trace_checks(s,0,np.array([0.],dtype='<f4'),np.array([1.],dtype='<f4'),
                   np.array([1.],dtype='<f4'),obs,2,42.,check,dict(bounded_fields=0))
    assert errors==['trace/degenerate-integral']


@pytest.mark.parametrize('damage',['none','term','prefix','total','area','order'])
def test_pathological_trace_stage_corruption(damage):
    s=m.Spec(0,m.MAIN,4,0,2,1,0,2,2,0)
    p=np.array([1,0],dtype='<f4');y=np.array([0,1],dtype='<f4');w=np.ones(2,dtype='<f4')
    obs={k:np.array(v,dtype='<f8').view('<u8') for k,v in
         dict(terms=[1,0,0,1],prefixes=[1,0,1,1],totals=[1,1]).items()}
    obs['order']=np.array([0,1],dtype='<u8');area=1.-np.log(2.)
    if damage in ('term','prefix','total'):
        key={'term':'terms','prefix':'prefixes','total':'totals'}[damage]
        obs[key]=obs[key].copy();obs[key][0]=np.array([2.]).view('<u8')[0]
    if damage=='area':area=.5
    if damage=='order':obs['order']=obs['order'][::-1].copy()
    errors=[]
    def check(name,a,b,exact=False):
        if not np.all(m.near(a,b,exact=exact)):errors.append(name)
    counts=dict(bounded_fields=0,exact_terms=0,source_integrals=0,source_nonfinite_integrals=0)
    m.trace_checks(s,0,p,y,w,obs,2,area,check,counts)
    assert bool(errors)==(damage!='none')
