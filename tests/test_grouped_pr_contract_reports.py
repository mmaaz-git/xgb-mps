"""Framing, exact inputs and analytical PR checks; no GPU execution."""
import importlib.util
import json
import lzma
from pathlib import Path
import struct
import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('grouped_pr_reports',ROOT/'scripts/compare_grouped_pr_contracts.py')
gp=importlib.util.module_from_spec(spec);spec.loader.exec_module(gp)


@pytest.fixture
def small(monkeypatch):
    # Keep every original family/mode/layout, with small synthetic row counts.
    shapes=[]
    for s,ptr in gp.shapes():
        lengths=np.minimum(np.diff(ptr),3)
        ptr=np.r_[0,np.cumsum(lengths)].astype('<u4')
        shapes.append((s._replace(rows=int(ptr[-1])),ptr))
    monkeypatch.setattr(gp,'shapes',lambda:shapes)
    return shapes


def encode(kind):
    pieces=[gp.MAGIC,struct.pack('<II',kind,gp.COUNT)]
    for s,ptr in gp.shapes():
        inputs=gp.source_inputs(s,ptr)
        pieces.extend([struct.pack('<7I',*s),ptr.tobytes()])
        pieces.extend(inputs[k].tobytes() for k in ('predictions','labels','weights'))
        if not kind: continue
        pieces.append(struct.pack('<I',int(s.family==2)))
        if s.family==2: continue
        stride,offset,_=gp.layout(s)
        p=inputs['predictions'][offset:offset+s.rows*stride:stride]
        y=inputs['labels'][offset:offset+s.rows*stride:stride]
        refs=np.array([gp.curve_reference(p[a:b],y[a:b],inputs['weights'][g] if s.weights else 1.)
                       for g,(a,b) in enumerate(zip(ptr[:-1],ptr[1:]))]).reshape(-1,2)
        aggregate=np.cumsum(refs,axis=0)[-1] if len(refs) else np.zeros(2)
        if kind==1:
            pieces.append(aggregate.astype('<f8').tobytes())
            if s.family==1:
                pieces.append(np.vstack(([-397.,973.],[-397.,973.],refs,[-397.,973.],[-397.,973.])).astype('<f8').tobytes())
        pieces.extend([aggregate.astype('<f8').tobytes(),refs[np.diff(ptr)!=0].astype('<f8').tobytes()])
    pieces.extend([b'GPRE',struct.pack('<I',gp.COUNT)])
    return b''.join(pieces)


def report(kind):
    return gp.decode(bytearray(encode(kind)))


@pytest.mark.parametrize('scores,labels,expected',[
    ([2,1],[1,0],1.),([2,1],[0,1],1-np.log(2.)),([0,0],[1,0],.5),
    ([1],[.5],0.),([1,0],[1,1],0.),([1,0],[0,0],0.)])
def test_analytical_integrals(scores,labels,expected):
    result=gp.curve_reference(np.array(scores,dtype='<f4'),np.array(labels,dtype='<f4'),1.)
    assert abs(result[0]-expected)<1e-15
    assert result[1]==float(len(labels)>=2 and 0<sum(labels)<len(labels))


def test_fp32_source_cast_and_nonfinite_weights():
    p=np.array([0,0],dtype='<f4');y=np.full(2,.1,dtype='<f4');w=np.float32(.7)
    # Single tied interval: independent formula from the rounded source terms.
    positive=float(np.float32(y[0]*w))
    negative=float(np.float32((1.-float(y[0]))*float(w)))
    assert gp.curve_reference(p,y,w)[0]==1/(1+negative/positive)
    for weight in (0.,np.nan):
        assert np.array_equal(gp.curve_reference(p,y,weight),[0.,0.])
    assert np.isnan(gp.curve_reference(p,y,np.inf)[0])


def test_full_shape_counts_and_empty_query_adaptation():
    shapes=gp.shapes()
    assert len(shapes)==319 and sum(s.rows for s,_ in shapes if s.family==1)==14198354
    assert sum(s.groups for s,_ in shapes if s.family==1)==4660
    assert sum(bool(s.rows) and np.count_nonzero(np.diff(ptr))!=s.groups for s,ptr in shapes)==301


def test_all_modes_roundtrip_and_no_fabricated_completion(small):
    metal,cuda=report(1),report(2)
    assert not gp.validate(metal)['reference_failures']
    assert not gp.validate(cuda)['reference_failures']
    result=gp.compare(metal,cuda)
    assert result['status']=='review-required' and not result['bit_differences']
    unpacked=gp.decode(gp.unpack(metal))
    gp.match_inputs(unpacked,metal)
    assert bytes(unpacked['data'])==encode(0)


@pytest.mark.parametrize('field',['predictions','labels','weights'])
def test_original_input_mutation_rejected_even_if_both_reports_agree(small,field):
    a,b=report(1),report(2)
    for r in (a,b): r['cases'][1][field][0]^=1
    with pytest.raises(ValueError,match='original .* bits differ'): gp.compare(a,b)


def test_matching_wrong_outputs_fail_independent_reference(small):
    a,b=report(1),report(2)
    for r in (a,b): r['cases'][1]['common'].view('<f8')[0]=.9
    result=gp.compare(a,b)
    assert result['status']=='reference-failure' and not result['bit_differences']


def test_guards_counts_and_tiny_relative_contract(small):
    r=report(1)
    main=next(c for c in r['cases'] if c['spec'].family==1)
    main['guards'].view('<f8')[0]=0
    main['common'].view('<f8')[1]+=.00001
    tiny=next(c for c in r['cases'] if c['spec'].family==1 and c['spec'].mode==11)
    tiny['guards'].view('<f8')[2*4]=0  # First query with two documents.
    failures=gp.validate(r)['reference_failures']
    assert {'guard','common/valid','original-tiny-tied-relative-accuracy'} <= {f['field'] for f in failures}


def test_signed_zero_differences_remain_visible(small):
    a,b=report(1),report(2)
    b['cases'][0]['common'].view('<f8')[0]=-0.
    result=gp.compare(a,b)
    assert result['bit_differences']==[dict(case=0,field='common',bit_differences=1,outside_original_tolerance=0)]


@pytest.mark.parametrize('mutation',['magic','count','truncated','trailing','identity','ptr','status'])
def test_corrupt_framing_or_status_rejected(small,mutation):
    data=bytearray(encode(1))
    if mutation=='magic': data[0]^=1
    if mutation=='count': struct.pack_into('<I',data,12,318)
    if mutation=='truncated': del data[-1:]
    if mutation=='trailing': data.extend(b'x')
    if mutation=='identity': data[16]^=1
    if mutation=='ptr': data[16+28]^=1
    if mutation=='status': struct.pack_into('<I',data,16+28+8+8,1)
    with pytest.raises(ValueError): gp.decode(data)


def test_compressed_unpack_and_cli_never_silently_passes_adaptation(small,tmp_path):
    metal=tmp_path/'metal.bin.xz';cuda=tmp_path/'cuda.bin';output=tmp_path/'comparison.json'
    metal.write_bytes(lzma.compress(encode(1)));cuda.write_bytes(encode(2))
    unpacked=tmp_path/'input.bin'
    assert gp.main(['unpack',str(metal),str(unpacked)])==0 and unpacked.read_bytes()==encode(0)
    with pytest.raises(FileExistsError): gp.main(['unpack',str(metal),str(unpacked)])
    assert gp.main(['compare',str(metal),str(cuda),'--output',str(output)])==2
    assert json.loads(output.read_text())['status']=='review-required'
    with pytest.raises(ValueError,match='already exists'): gp.main(['compare',str(metal),str(cuda),'--output',str(output)])
