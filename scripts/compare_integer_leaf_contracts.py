#!/usr/bin/env python3
"""Compare actual leaf outputs, independently reconstructing original integer references.

Metal integer words are reference-only. CUDA integer words are actual API state.
Undefined conversion cases are exclusions, never successful CUDA rejections.
"""
import argparse
from collections import namedtuple
import gzip
import hashlib
import json
import lzma
from pathlib import Path
import struct
import sys
import numpy as np

MAGIC, COUNT = b'XGBILF1\0', 256
Spec = namedtuple('Spec', 'id family rows targets layout mode extra domain')
Shape = namedtuple('Shape', 'rows stride0 stride1 offset extent storage leaves snapshots weight_words')


def require(ok, message):
    if not ok: raise ValueError(message)


def bits(value): return struct.unpack('<I', struct.pack('<f', value))[0]


def specs():
    out = []
    def add(n,t,layout,mode): out.append(Spec(len(out),0,n,t,layout,mode,0,0))
    for n in (0,1,31,257,4097,65539):
        for t in (1,3):
            for layout in range(5):
                for mode in range(4): add(n,t,layout,mode)
    add(1000003,3,0,0); add(1000003,3,1,0)
    for exponent in (-80,0,54,100):
        for paged in (0,1): out.append(Spec(len(out),1,4,1,0,paged,exponent & 0xffffffff,0))
    out.append(Spec(250,2,4,1,0,0,0,0))
    for value in (0xc0800000,0x407fffff,0x40800000,0x7f800000,0x7fc00000):
        out.append(Spec(len(out),3,1,1,0,0,value,int(len(out)>=253)))
    require(len(out)==COUNT,'Incomplete integer leaf matrix');return out


def layout(s):
    if s.family: return Shape(s.rows,1,1,0,s.rows,s.rows,2 if s.family==1 else 1,
                              0 if s.domain else 2 if s.family==1 and s.mode else 1,0)
    n=s.rows+11; a=1 if s.layout==1 else s.targets*2+3 if s.layout==2 else s.targets
    b=n+3 if s.layout==1 else 2 if s.layout==2 else 1
    if s.layout==3: a=0
    if s.layout==4: a,b=1,0
    extent=(n-1)*a+(s.targets-1)*b+1
    return Shape(n,a,b,4,extent,extent+8,7,3,5*(7*s.targets+2))


def decode(data):
    require(len(data)>=16 and data[:8]==MAGIC,'Invalid integer leaf header')
    kind,count=struct.unpack_from('<II',data,8)
    require(kind in (0,1,2) and count==COUNT,'Wrong integer leaf kind/count')
    offset,cases=16,[]
    def array(n,width):
        nonlocal offset
        require(offset+n*width<=len(data),'Truncated integer leaf array')
        result=np.frombuffer(data,dtype=f'<u{width}',count=n,offset=offset);offset+=n*width;return result
    for s in specs():
        require(offset+32<=len(data),'Truncated integer leaf identity')
        require(struct.unpack_from('<8I',data,offset)==s,'Wrong integer leaf identity/order/domain');offset+=32
        shape=layout(s);case={'spec':s,'original':array(2*shape.storage,4),'sampled':array(2*shape.storage,4)}
        n=2*shape.leaves*s.targets*shape.snapshots
        if kind:
            status=int(array(1,4)[0]);require(status==((1 if kind==1 else 2) if s.domain else 0),'Wrong integer leaf rejection/exclusion status')
            case.update(status=status,sums=array(n,8),weights=array(shape.weight_words,4),integers=array(n,8))
        if kind==1:case.update(reference_sums=array(n,8),reference_weights=array(shape.weight_words,4))
        cases.append(case)
    require(len(data)==offset+8 and data[offset:offset+4]==b'ILFE' and struct.unpack_from('<I',data,offset+4)[0]==COUNT,
            'Missing integer leaf completion marker or trailing bytes')
    return {'kind':kind,'cases':cases,'data':data,'sha256':hashlib.sha256(data).hexdigest()}


def load(path):
    opener=lzma.open if path.suffix=='.xz' else gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as stream:return decode(stream.read())


def match_inputs(a,b):
    require(len(a['cases'])==len(b['cases'])==COUNT,'Incomplete integer leaf corpus')
    for x,y in zip(a['cases'],b['cases']):
        require(x['spec']==y['spec'] and all(np.array_equal(x[k],y[k]) for k in ('original','sampled')),
                f"Integer leaf case {x['spec'].id}: input bits differ")


def logical(case,key):
    s=case['spec'];shape=layout(s)
    # Bounds are determined by the validated fixed spec, including overlapping views.
    pairs=case[key].view('<f4').reshape(-1,2)
    idx=shape.offset+np.arange(shape.rows)[:,None]*shape.stride0+np.arange(s.targets)[None,:]*shape.stride1
    return pairs[idx].astype('<f8')


def references(case):
    s=case['spec'];shape=layout(s)
    if s.domain:return {'sums':np.array([],dtype='<u8'),'weights':np.array([],dtype='<u4'),'integers':np.array([],dtype='<u8')}
    original=logical(case,'original');sampled=logical(case,'sampled')
    require(np.all(np.isfinite(original)),'Nonfinite original integer leaf input')
    # Accumulate in original row order (not numpy.sum's pairwise reduction).
    positive=np.add.accumulate(np.where(original>=0,original,0.),axis=0)[-1]
    negative=np.add.accumulate(np.where(original<0,-original,0.),axis=0)[-1]
    bound=np.maximum(positive,negative)/(1.-2.*shape.rows*np.finfo(np.float64).eps)
    scales=62-np.frexp(bound)[1]
    fixed=np.ldexp(sampled,scales)
    # Only selected logical rows are converted by the actual leaf API.
    selected=fixed[5:5+s.rows] if s.family==0 else fixed
    require(np.all(np.isfinite(selected)&(selected>=-2.**63)&(selected<2.**63)),
            'Common-domain leaf input exceeds original quantizer int64 range')
    integers=selected.astype('<i8').view('<u8')
    hess=(sampled[5:5+s.rows] if s.family==0 else sampled)[...,1]
    integers[...,1]=np.where((hess>0)&(integers[...,1]==0),np.uint64(1),integers[...,1])
    running=np.zeros((shape.leaves,s.targets,2),dtype='<u8');snapshots=[]
    if s.family==0:
        boundaries=(0,s.rows//3,s.rows*2//3,s.rows)
        for begin,end in zip(boundaries,boundaries[1:]):
            for leaf in range(7):
                lo=max(begin,s.rows*min(leaf,5)//5);hi=min(end,s.rows*min(leaf+1,5)//5)
                if hi>lo:running[(leaf*3+2)%7]+=integers[lo:hi].sum(axis=0,dtype='<u8')
            snapshots.append(running.copy())
    elif s.family==1:
        if s.mode:
            running[1]=integers[:3].sum(axis=0,dtype='<u8');snapshots.append(running.copy())
            running[1]+=integers[3]
        else:running[1]=integers.sum(axis=0,dtype='<u8')
        snapshots.append(running.copy())
    else:
        running[0]=integers.sum(axis=0,dtype='<u8');snapshots.append(running.copy())
    exact=np.asarray(snapshots,dtype='<u8')
    sums=np.ldexp(exact.view('<i8').astype('<f8'),-scales)
    weights=[]
    if s.family==0:
        g,h=sums[-1,...,0],sums[-1,...,1]
        for kind in range(5):
            alpha=.75 if kind==2 else 0.;lam=0. if kind==0 else 1.25
            threshold=np.where(g>alpha,g-alpha,np.where(g<-alpha,g+alpha,0.))
            w=np.zeros_like(g);np.divide(-threshold,h+lam,out=w,where=h>0)
            if kind==3:w=np.where(np.abs(w)>.125,np.copysign(.125,w),w)
            w=w.astype('<f4')
            if kind==4:w=np.minimum(np.maximum(w,np.float32(.125)),np.float32(.25))
            w=np.multiply(w,np.float32(.3),dtype='<f4')
            weights.extend([bits(-913.),*w.ravel().view('<u4').tolist(),bits(-913.)])
    return {'sums':sums.ravel().view('<u8'),'weights':np.asarray(weights,dtype='<u4'),'integers':exact.ravel()}


def failures(case,expected,backend,check_integers):
    out=[]
    for key in ('sums','weights','integers'):
        if key=='integers' and not check_integers:continue
        a,b=case[key],expected[key]
        good=a.view('<f8')==b.view('<f8') if key=='sums' else a==b
        bad=np.flatnonzero(~good)
        if len(bad):out.append({'case':case['spec'].id,'backend':backend,'field':key,
                               'policy':'original-numeric-double' if key=='sums' else 'original-exact',
                               'count':len(bad),'first_indices':bad[:32].tolist()})
    return out


def validate_metal_case(case,expected):
    ref={'spec':case['spec'],'sums':case['reference_sums'],'weights':case['reference_weights'],'integers':case['integers']}
    return failures(ref,expected,'saved-reference',True)+failures(case,expected,'metal',False)


def validate_metal(metal,progress=False):
    require(metal['kind']==1,'Need Metal integer leaf observations');out=[]
    for i,case in enumerate(metal['cases']):
        out.extend(validate_metal_case(case,references(case)))
        if progress and ((i+1)%25==0 or i+1==COUNT):print(f'Validated integer leaf {i+1}/{COUNT}',flush=True)
    return out


def compare(metal,cuda):
    require(metal['kind']==1 and cuda['kind']==2,'Need Metal and actual CUDA observation kinds');match_inputs(metal,cuda)
    errors=[];differences=[];different=fields=integer_fields=0
    for m,c in zip(metal['cases'],cuda['cases']):
        expected=references(m);errors.extend(validate_metal_case(m,expected));errors.extend(failures(c,expected,'cuda',True))
        integer_fields+=len(c['integers'])
        for key in ('sums','weights'):
            indices=np.flatnonzero(m[key]!=c[key]);fields+=len(m[key]);different+=len(indices)
            if len(indices):differences.append({'case':m['spec'].id,'field':key,'count':len(indices),
                'first_fields':[{'index':int(i),'metal':int(m[key][i]),'cuda':int(c[key][i])} for i in indices[:32]]})
    return {'suite':'integer-leaf-v1','cases':COUNT,'cuda_executed_cases':sum(not c['spec'].domain for c in cuda['cases']),
            'domain_exclusions':[c['spec'].id for c in cuda['cases'] if c['spec'].domain],
            'compared_actual_fields':fields,'cuda_integer_reference_checks':integer_fields,
            'different_actual_fields':different,'differences':differences,'reference_failures':errors,
            'status':'reference-failure' if errors else 'review-differences-and-domains' if different else 'common-domain-identical-with-exclusions'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('inspect','unpack','validate-metal'):
        c=sub.add_parser(name);c.add_argument('input',type=Path)
        if name=='unpack':c.add_argument('output',type=Path)
    c=sub.add_parser('compare');c.add_argument('metal',type=Path);c.add_argument('cuda',type=Path);c.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='compare':
        require(not args.output.exists(),'Output already exists');result=compare(load(args.metal),load(args.cuda))
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='differences'}))
        return 1 if result['reference_failures'] else 2
    report=load(args.input)
    if args.command=='unpack':
        require(report['kind']==0,'Only input-only integer leaf records may be unpacked for CUDA')
        with args.output.open('xb') as stream:stream.write(report['data'])
    result={'kind':report['kind'],'cases':COUNT,'raw_sha256':report['sha256']}
    if args.command=='validate-metal':
        result['reference_failures']=validate_metal(report,True);result['status']='reference-failure' if result['reference_failures'] else 'complete; no CUDA equivalence claim'
    print(json.dumps(result));return 1 if result.get('reference_failures') else 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,EOFError,struct.error) as error:
        print(f'FAIL integer leaf report: {error}',file=sys.stderr);sys.exit(1)
