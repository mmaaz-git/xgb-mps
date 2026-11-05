#!/usr/bin/env python3
"""Validate complete grouped ROC observations and compare actual CUDA/Metal calls.

Keep original Metal storage and per-query outputs. Mixed CUDA calls omit only
zero-pair queries; every original query also runs alone. Large CUDA inputs must
hit the actual pair-count guard, not emit invented successful results.
"""
import argparse
from collections import namedtuple
import gzip
import hashlib
import json
import lzma
import mmap
from pathlib import Path
import struct
import sys
import numpy as np

MAGIC,COUNT=b'XGBGRO1\0',278
EMPTY,OVERFLOW,EDGES,MAIN,LARGE=range(5)
Spec=namedtuple('Spec','id family layout mode rows groups weights')
INPUTS=('ptr','predictions','labels','weights')
FIELDS=('original','per_query_guards','common','isolated','cpu')


def require(ok,message):
    if not ok:raise ValueError(message)


def shapes():
    out=[]
    def add(family,layout,mode,lengths,weighted):
        ptr=np.r_[0,np.cumsum(lengths,dtype=np.uint32)].astype('<u4')
        s=Spec(len(out),family,layout,mode,int(ptr[-1]),len(lengths),len(lengths) if weighted else 0)
        out.append((s,ptr))
    add(EMPTY,0,0,[],False);add(OVERFLOW,0,0,[3],True)
    add(EDGES,0,0,[n for n in (0,1,2,3,31,257) for w in range(13) for mode in range(5)],True)
    for n in (0,1,2,3,31,256,257,1025):
        for layout in range(3):
            for mode in range(11):add(MAIN,layout,mode,[n,0,n//2,1,2,3,31,0],mode!=0)
    for n in (4097,8193):
        for mode in (0,1,6,9):add(MAIN,2,mode,[n,0,n//2,1,2,3,31,0],mode!=0)
    for mode in range(3):add(LARGE,0,mode,[1000003],False)
    require(len(out)==COUNT,'Incomplete grouped ROC matrix')
    return out


def layout(s):
    stride=(1,3,7)[s.layout] if s.family==MAIN else 1
    return stride,4 if s.family==MAIN else 0,s.rows*stride+(8 if s.family==MAIN else 0)


def field_sizes(s,ptr,kind):
    metal=kind==1;normal=s.family!=LARGE
    guards=2*(s.groups+4) if s.family==MAIN else 2*(s.groups+2) if s.family==EDGES else 0
    return dict(original=2 if metal else 0,per_query_guards=guards if metal else 0,
                common=2 if normal else 0,isolated=2*s.groups if normal else 0,
                cpu=int(np.count_nonzero(np.diff(ptr)>=3)) if metal and s.family==MAIN else 0)


def decode(data):
    require(len(data)>=16 and data[:8]==MAGIC,'Invalid grouped ROC header')
    kind,count=struct.unpack_from('<II',data,8)
    require(kind in (0,1,2) and count==COUNT,'Wrong grouped ROC kind/count')
    offset,cases=16,[]
    def array(n,width):
        nonlocal offset
        require(offset+n*width<=len(data),'Truncated grouped ROC array')
        a=np.frombuffer(data,dtype=f'<u{width}',count=n,offset=offset);offset+=n*width;return a
    for s,ptr in shapes():
        require(tuple(array(7,4))==s,'Wrong grouped ROC identity/order')
        observed=array(s.groups+1,4)
        require(np.array_equal(observed,ptr),'Wrong grouped ROC query boundaries')
        size=layout(s)[2]
        c=dict(spec=s,ptr=observed,predictions=array(size,4),labels=array(size,4),weights=array(s.weights,4))
        if kind:
            c['guarded']=int(array(1,4)[0]);c['cpu_overflow_rejected']=int(array(1,4)[0])
            require(c['guarded']==int(kind==2 and s.family==LARGE) and
                    c['cpu_overflow_rejected']==int(kind==1 and s.family==OVERFLOW),
                    'Wrong grouped ROC observed guard status')
            for k,n in field_sizes(s,ptr,kind).items():c[k]=array(n,8)
        cases.append(c)
    require(len(data)==offset+8 and data[offset:offset+4]==b'GROE' and
            struct.unpack_from('<I',data,offset+4)[0]==COUNT,'Incomplete/trailing grouped ROC report')
    return dict(kind=kind,cases=cases,data=data,sha256=hashlib.sha256(data).hexdigest())


def load(path):
    if path.suffix in ('.xz','.gz'):
        with (lzma.open if path.suffix=='.xz' else gzip.open)(path,'rb') as stream:return decode(stream.read())
    with path.open('rb') as stream:
        require(path.stat().st_size>0,'Empty grouped ROC report')
        return decode(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))


def source_inputs(s,ptr):
    """Every original generator word, including guard storage and unused draws."""
    f=np.float32;stride,offset,size=layout(s)
    p=np.zeros(size,dtype='<f4');y=np.zeros(size,dtype='<f4');w=np.zeros(s.weights,dtype='<f4')
    if s.family==OVERFLOW:p[:]=[0,1,2];y[:]=[0,1,2];w[:]=[1e30]
    elif s.family==EDGES:
        weights=np.array([0.,-0.,1.,.7,-.7,2.**-100,float.fromhex('0x1.8p-75'),
                          float.fromhex('0x1.4p-74'),1e19,1e30,np.inf,-np.inf,np.nan],dtype='<f4')
        for g,(begin,end) in enumerate(zip(ptr[:-1],ptr[1:])):
            begin,end=int(begin),int(end);i=np.arange(end-begin,dtype=np.int64);mode=g%5
            pp=((i*7)%13).astype('<f4');yy=((i*3)%11).astype('<f4')
            if mode==1:pp=np.where(i%2,f(-0.),f(0.)).astype('<f4')
            if mode==2:pp=np.ldexp((i%5-2).astype('<f4'),-149)
            if mode==3:pp=np.array([np.inf,-np.inf,np.nan,np.inf,-np.inf,0],dtype='<f4')[i%6]
            if mode==4:yy.fill(1)
            p[begin:end]=pp;y[begin:end]=yy;w[g]=weights[(g//5)%13]
    elif s.family==MAIN:
        p.fill(-971);y.fill(-193);rows=np.arange(s.rows,dtype=np.int64)
        rng=np.random.RandomState(1179+s.mode)
        draws=rng.randint(0,2**32,size=(s.rows,3 if s.mode==1 else 2),dtype=np.uint32)
        pp=((draws[:,0]%8191).astype(np.int64)-4095).astype('<f4')/f(93)
        yy=(draws[:,1]%11).astype('<f4')
        if s.mode==1:pp=(draws[:,2]%5).astype('<f4')
        if s.mode==2:pp=np.where(rows%2,f(-0.),f(0.)).astype('<f4')
        if s.mode==3:pp=np.ldexp(pp,-145);yy=np.ldexp(yy,-149)
        if s.mode==4:
            pp[rows%5==0]=np.inf;pp[rows%5==1]=-np.inf;pp[rows%5==2]=np.nan
        if s.mode==9:yy.fill(1)
        if s.mode==10:yy/=f(11)
        p[offset+rows*stride]=pp;y[offset+rows*stride]=yy
        for g in range(s.weights):
            value=f(g+1)/f(7)
            if s.mode==5:value=f(-0.) if g%2 else f(0.)
            if s.mode==6:value=np.ldexp(f(1+g%3),-74)
            if s.mode==7:value=np.ldexp(f(g+1),55)
            if s.mode==8:value=np.ldexp(f(g+1),-100)
            if s.mode==10:value=np.ldexp(value,int(g*17%121)-60)
            w[g]=value
    elif s.family==LARGE:
        i=np.arange(s.rows,dtype=np.int64)
        p[:]=i if s.mode==0 else s.rows-i if s.mode==1 else 1
    return dict(predictions=p,labels=y,weights=w)


def pair_reference(p,y,weight):
    """Independent input-order pair enumeration; no sort/merge/inversion oracle.

    Count full/tied FP32 products separately. Each finite sum is exact here:
    fewer than 2^28 pairs on a common at-most-25-bit exponent grid, including
    rounded subnormal halves. This is equivalent to
    the original serial double additions without allocating a quadratic matrix.
    """
    n=len(p)
    if n<3:return np.array([0.,0.])
    pairs=n*(n-1)//2
    require(pairs<2**28,'Pair reference requires its documented exact-sum range')
    with np.errstate(all='ignore'):
        square=np.float32(weight)*np.float32(weight)
        half=np.float32(.5)*square
    if not square>0:return np.array([0.,1.])  # Includes NaN source total.
    if np.isinf(square):return np.array([np.nan,1.])
    wins=ties=0;j=np.arange(n)
    with np.errstate(all='ignore'):
        for begin in range(0,n,32):
            end=min(begin+32,n);i=np.arange(begin,end)
            reverse=y[i,None]<y[None,:]
            delta=np.where(reverse,p[None,:]-p[i,None],p[i,None]-p[None,:])
            valid=i[:,None]<j[None,:]
            wins+=int(np.count_nonzero(valid&(delta>0)))
            ties+=int(np.count_nonzero(valid&(delta==0)))
    return np.array([(wins*float(square)+ties*float(half))/(pairs*float(square)),1.])


def near(actual,expected,exact=False):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    same=(a==b)|(np.isnan(a)&np.isnan(b))
    with np.errstate(all='ignore'):
        return same if exact else same|(np.isfinite(a)&np.isfinite(b)&(np.abs(a-b)<=2e-12*np.maximum(1.,np.abs(b))))


def validate(report,progress=False):
    failures=[];fields=words=isolated=adapted=removed=short=guards=0;cache={}
    logical_pairs=0
    for c in report['cases']:
        s=c['spec'];ptr=c['ptr'];src=source_inputs(s,ptr)
        for k in ('predictions','labels','weights'):
            words+=len(c[k]);require(np.array_equal(c[k],src[k].view('<u4')),
                                    f'Grouped ROC case {s.id}: original {k} bits differ')
        words+=len(ptr)
        lengths=np.diff(ptr);drop=int(np.count_nonzero(lengths<2));removed+=drop;adapted+=bool(drop)
        if report['kind']==0:continue
        def check(name,observed,wanted,exact=False):
            nonlocal fields
            good=near(observed,wanted,exact);fields+=np.size(good)
            if not np.all(good):failures.append(dict(case=s.id,field=name,count=int(np.count_nonzero(~good))))
        if s.family==LARGE:
            if report['kind']==1:check('million-row-analytic',c['original'].view('<f8'),[(0,1,.5)[s.mode],1],True)
            else:guards+=1
            continue
        stride,offset,_=layout(s)
        p=src['predictions'][offset:offset+s.rows*stride:stride]
        y=src['labels'][offset:offset+s.rows*stride:stride]
        refs=[]
        for g,(begin,end) in enumerate(zip(ptr[:-1],ptr[1:])):
            begin,end=int(begin),int(end);n=end-begin
            w=src['weights'][g] if s.weights else np.float32(1)
            key=(p[begin:end].tobytes(),y[begin:end].tobytes(),w.tobytes())
            if key not in cache:cache[key]=pair_reference(p[begin:end],y[begin:end],w)
            refs.append(cache[key]);logical_pairs+=n*(n-1)//2 if n>=3 else 0
        ref=np.asarray(refs,dtype=np.float64).reshape(-1,2)
        total=np.cumsum(ref,axis=0)[-1] if len(ref) else np.zeros(2)
        for k in ('original','common'):
            if len(c[k]):
                v=c[k].view('<f8');check(k+'/area',v[0],total[0]);check(k+'/valid',v[1],total[1],True)
        values=c['isolated'].view('<f8').reshape(-1,2)
        check('isolated/area',values[:,0],ref[:,0]);check('isolated/valid',values[:,1],ref[:,1],True)
        isolated+=s.groups;short+=int(np.count_nonzero(lengths<3))
        if len(c['per_query_guards']):
            v=c['per_query_guards'].view('<f8').reshape(-1,2);pad=2 if s.family==MAIN else 1
            check('guards',np.concatenate((v[:pad],v[-pad:])),np.tile([-397.,973.],(2*pad,1)),True)
            check('original-queries/area',v[pad:-pad,0],ref[:,0]);check('original-queries/valid',v[pad:-pad,1],ref[:,1],True)
        if len(c['cpu']):check('actual-CPU',c['cpu'].view('<f8'),ref[lengths>=3,0])
        if progress and ((s.id+1)%24==0 or s.id+1==COUNT):
            print(f'Validated grouped ROC {s.id+1}/{COUNT}; {words} input words; {logical_pairs} reference pairs; {len(failures)} failures',file=sys.stderr,flush=True)
    return dict(cases=len(report['cases']),input_words=words,checked_fields=fields,isolated_queries=isolated,
                isolated_short_queries=short,adapted_records=adapted,removed_zero_pair_groups=removed,
                logical_reference_pairs=logical_pairs,unique_reference_queries=len(cache),observed_cuda_pair_guards=guards,
                reference_failures=failures)


def match_inputs(a,b):
    require(len(a['cases'])==len(b['cases'])==COUNT,'Incomplete grouped ROC corpus')
    for x,y in zip(a['cases'],b['cases']):
        require(x['spec']==y['spec'] and all(np.array_equal(x[k],y[k]) for k in INPUTS),
                f"Grouped ROC case {x['spec'].id}: input bits differ")


def compare(metal,cuda):
    require(metal['kind']==1 and cuda['kind']==2,'Need Metal and actual CUDA observations')
    match_inputs(metal,cuda);checks=[validate(metal),validate(cuda)];differences=[];fields=outside=0
    for a,b in zip(metal['cases'],cuda['cases']):
        for k in ('common','isolated'):
            fields+=len(a[k]);changed=a[k]!=b[k]
            if np.any(changed):
                material=int(np.count_nonzero(~near(a[k].view('<f8'),b[k].view('<f8'))));outside+=material
                differences.append(dict(case=a['spec'].id,field=k,bit_differences=int(np.count_nonzero(changed)),outside_original_tolerance=material))
    return dict(status='reference-failure' if any(c['reference_failures'] for c in checks) else 'review-required',
                metal_sha256=metal['sha256'],cuda_sha256=cuda['sha256'],validation=checks,
                common_observed_fields=fields,bit_differences=differences,outside_original_tolerance=outside,
                scope='Mixed CUDA calls remove only length0/1 groups; all original groups execute separately. Three million-row inputs must hit the actual CUDA pair-count guard. Source/Metal extensions and original per-query diagnostics are retained, not fabricated as CUDA outputs.')


def unpack(report):
    parts=[MAGIC,struct.pack('<II',0,COUNT)]
    for c in report['cases']:
        parts.append(struct.pack('<7I',*c['spec']));parts.extend(c[k].tobytes() for k in INPUTS)
    return b''.join(parts+[b'GROE',struct.pack('<I',COUNT)])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate-metal');v.add_argument('metal',type=Path)
    u=sub.add_parser('unpack');u.add_argument('input',type=Path);u.add_argument('output',type=Path)
    c=sub.add_parser('compare');c.add_argument('metal',type=Path);c.add_argument('cuda',type=Path);c.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='unpack':
        r=load(args.input)
        with args.output.open('xb') as stream:stream.write(unpack(r))
        return 0
    if args.command=='validate-metal':
        r=load(args.metal);require(r['kind']==1,'Expected Metal grouped ROC observations')
        result=validate(r,True);print(json.dumps(result));return int(bool(result['reference_failures']))
    require(not args.output.exists(),'Comparison output already exists')
    result=compare(load(args.metal),load(args.cuda))
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result));return 1 if result['status']=='reference-failure' else 2


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,struct.error) as error:
        print(f'FAIL grouped ROC report: {error}',file=sys.stderr);sys.exit(1)
