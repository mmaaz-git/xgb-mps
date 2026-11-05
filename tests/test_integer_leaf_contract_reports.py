"""Synthetic report corruption checks and analytical integer witnesses, not GPU runs."""
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
SPEC=importlib.util.spec_from_file_location('integer_leaf_compare',ROOT/'scripts/compare_integer_leaf_contracts.py')
lc=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(lc)


def test_complete_matrix_preserves_uncounted_guards_and_all_pages():
    specs=lc.specs()
    assert len(specs)==256
    assert [sum(s.family==f for s in specs) for f in range(4)]==[242,8,1,5]
    assert [s.id for s in specs if s.domain]==[253,254,255]
    assert sum(2*lc.layout(s).leaves*s.targets*lc.layout(s).snapshots for s in specs)==20466
    assert sum(lc.layout(s).weight_words for s in specs)==19430
    assert sum(s.rows*s.targets for s in specs if not s.family)==11594018
    assert [s.extra for s in specs[-5:]]==[0xc0800000,0x407fffff,0x40800000,0x7f800000,0x7fc00000]


def case(s):
    shape=lc.layout(s)
    storage=np.full((shape.storage,2),[17.,-23.],dtype='<f4')
    if s.family==0:
        for r in range(shape.rows):
            for t in range(s.targets):storage[shape.offset+r*shape.stride0+t*shape.stride1]=[(r-7)/8,(t+1)/4]
        sampled=storage.copy()
        for r in range(shape.rows):
            for t in range(s.targets):
                i=shape.offset+r*shape.stride0+t*shape.stride1
                sampled[i]=storage[i]*np.float32((0,1,1.25,1.5)[r%4])
    elif s.family==1:
        storage[:]=[[2.**54,0],[1,1],[1,1],[-2.**54,0]];sampled=storage.copy()
    elif s.family==2:
        storage[:]=[[0,2.**65],[-1,2.**-149],[-1,2.**-100],[0,0]]
        sampled=storage.copy();sampled[0]=0;sampled[1:3]*=np.float32(1.5)
    else:
        storage[:]=[1,1];sampled=storage.copy()
        sampled[0,0]=np.array([s.extra],dtype='<u4').view('<f4')[0]
    return {'spec':s,'original':storage.ravel().view('<u4'),'sampled':sampled.ravel().view('<u4')}


@pytest.fixture
def reports(monkeypatch):
    original=lc.specs()
    specs=[original[80],original[100],original[242],original[243],original[250],*original[251:]]
    # Use small positive row cases, one with target broadcast, and retain exact
    # cancellation/curvature/range parameters. Full corpus validated separately.
    specs[0]=specs[0]._replace(rows=5,targets=3,layout=2)
    specs[1]=specs[1]._replace(rows=5,targets=3,layout=4)
    specs=[s._replace(id=i) for i,s in enumerate(specs)]
    monkeypatch.setattr(lc,'COUNT',len(specs));monkeypatch.setattr(lc,'specs',lambda:specs)
    results=[]
    for kind in (1,2):
        cases=[]
        for s in specs:
            c=case(s);refs=lc.references(c)
            c.update(status=(1 if kind==1 else 2) if s.domain else 0,**{k:v.copy() for k,v in refs.items()})
            if kind==1:c.update(reference_sums=refs['sums'].copy(),reference_weights=refs['weights'].copy())
            cases.append(c)
        results.append({'kind':kind,'cases':cases})
    return results


def encode(report,kind=None):
    kind=report['kind'] if kind is None else kind;chunks=[lc.MAGIC,struct.pack('<II',kind,lc.COUNT)]
    for c in report['cases']:
        chunks.extend((struct.pack('<8I',*c['spec']),c['original'].tobytes(),c['sampled'].tobytes()))
        if kind:chunks.extend((struct.pack('<I',c['status']),c['sums'].tobytes(),c['weights'].tobytes(),c['integers'].tobytes()))
        if kind==1:chunks.extend((c['reference_sums'].tobytes(),c['reference_weights'].tobytes()))
    return b''.join(chunks)+b'ILFE'+struct.pack('<I',lc.COUNT)


def test_analytical_cancellation_curvature_and_int64_boundary(reports):
    m,c=reports
    for index in (2,3):
        sums=m['cases'][index]['sums'].view('<f8').reshape(-1,2,1,2)
        assert (sums[-1,:,0,:]==[[0,0],[2,2]]).all()
    paged=m['cases'][3]
    assert paged['sums'].view('<f8')[2]==2.**54  # Two small terms lost only in materialization.
    assert int(paged['integers'][2])==2**61+256  # Retained across pages.
    assert m['cases'][4]['sums'].view('<f8').tolist()==[-3.,32.]
    assert int(m['cases'][5]['integers'][0])==2**63
    assert m['cases'][5]['sums'].view('<f8').tolist()==[-4.,1.]
    assert m['cases'][6]['sums'].view('<f8')[0]==float(np.nextafter(np.float32(4),np.float32(0)))
    assert not lc.validate_metal(m)


def test_actual_integer_diagnostics_are_only_from_cuda(reports):
    result=lc.compare(*reports)
    assert result['cuda_executed_cases']==7 and len(result['domain_exclusions'])==3
    assert result['compared_actual_fields']==sum(len(c['sums'])+len(c['weights']) for c in reports[0]['cases'])
    assert result['cuda_integer_reference_checks']==sum(len(c['integers']) for c in reports[1]['cases'])
    assert result['status']=='common-domain-identical-with-exclusions'
    with pytest.raises(ValueError,match='actual CUDA'):lc.compare(reports[0],reports[0])


@pytest.mark.parametrize('field',['sums','weights','integers'])
def test_identical_wrong_outputs_and_saved_references_cannot_pass(reports,field):
    m,c=reports
    for report in reports:report['cases'][0][field][0]^=np.array(1,dtype=report['cases'][0][field].dtype)
    if field!='integers':m['cases'][0]['reference_'+field][0]=m['cases'][0][field][0]
    result=lc.compare(m,c)
    assert not result['different_actual_fields']
    assert {'saved-reference','cuda'} <= {r['backend'] for r in result['reference_failures']}
    if field=='integers':assert 'metal' not in {r['backend'] for r in result['reference_failures']}


def test_numeric_zero_still_records_bit_difference(reports):
    # An empty leaf has numeric zero, but raw sign differences remain reviewable.
    c=reports[1]['cases'][2];c['sums'][0]=1<<63
    result=lc.compare(*reports)
    assert not result['reference_failures'] and result['different_actual_fields']==1


def test_weight_guards_and_bounds_remain_exact(reports):
    m,c=reports;n=lc.layout(m['cases'][0]['spec']).weight_words//5
    bounded=m['cases'][0]['weights'][-n:].view('<f4')
    assert bounded[0]==bounded[-1]==-913.
    assert np.all(bounded[1:-1]>=np.float32(.125)*np.float32(.3))
    assert np.all(bounded[1:-1]<=np.float32(.25)*np.float32(.3))
    c['cases'][0]['weights'][-1]=lc.bits(-912.)
    assert lc.compare(m,c)['reference_failures'][0]['field']=='weights'


@pytest.mark.parametrize('key',['original','sampled'])
def test_input_backing_and_padding_match_exactly(reports,key):
    reports[1]['cases'][0][key][-1]^=np.uint32(1)
    with pytest.raises(ValueError,match='input bits'):lc.compare(*reports)


def test_common_domain_cannot_hide_undefined_cast(reports):
    c=reports[0]['cases'][5]
    c['sampled'][0]=lc.bits(4.)
    with pytest.raises(ValueError,match='int64 range'):lc.references(c)


def test_corrupted_backend_status_and_reference_words_are_rejected(reports):
    m,c=reports;c['cases'][-1]['status']=1
    with pytest.raises(ValueError,match='exclusion status'):lc.decode(encode(c))
    m['cases'][0]['integers'][0]^=np.uint64(1)
    assert lc.validate_metal(lc.decode(encode(m)))


def test_framing_identities_and_compression(reports,tmp_path):
    raw=encode(reports[0],0)
    for broken in (raw[:-1],raw+b'x',b'BADMAGIC'+raw[8:],raw[:8]+struct.pack('<I',9)+raw[12:],
                   raw[:12]+struct.pack('<I',lc.COUNT-1)+raw[16:],raw[:16]+struct.pack('<I',99)+raw[20:]):
        with pytest.raises(ValueError):lc.decode(broken)
    for suffix,compress in (('.bin',lambda b:b),('.gz',gzip.compress),('.xz',lzma.compress)):
        p=tmp_path/('inputs'+suffix);p.write_bytes(compress(raw));assert lc.load(p)['data']==raw
    assert not lc.validate_metal(lc.decode(encode(reports[0])))
    assert not lc.compare(lc.decode(encode(reports[0])),lc.decode(encode(reports[1])))['reference_failures']


def test_input_only_export_and_no_overwrite(reports,tmp_path,monkeypatch):
    source,out=tmp_path/'source.bin',tmp_path/'out.bin';source.write_bytes(encode(reports[0]))
    monkeypatch.setattr(sys,'argv',['compare','unpack',str(source),str(out)])
    with pytest.raises(ValueError,match='input-only'):lc.main()
    assert not out.exists();raw=encode(reports[0],0);source.write_bytes(raw)
    assert lc.main()==0 and out.read_bytes()==raw
    with pytest.raises(FileExistsError):lc.main()


def test_cli_preserves_review_and_failures(reports,tmp_path,monkeypatch):
    monkeypatch.setattr(lc,'load',lambda p:reports[int(str(p)=='cuda')])
    out=tmp_path/'review.json';monkeypatch.setattr(sys,'argv',['compare','compare','metal','cuda','--output',str(out)])
    assert lc.main()==2 and json.loads(out.read_text())['status']=='common-domain-identical-with-exclusions'
    with pytest.raises(ValueError,match='already exists'):lc.main()
    reports[1]['cases'][0]['integers'][0]^=np.uint64(1)
    monkeypatch.setattr(sys,'argv',['compare','compare','metal','cuda','--output',str(tmp_path/'failure.json')])
    assert lc.main()==1
