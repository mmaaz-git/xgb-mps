#!/usr/bin/env python3
"""Validate the complete grouped PR corpus and compare actual GPU observations.

Original strided/empty-query Metal calls remain in the report. Common replay
packs values and removes empty queries explicitly. Isolated-query calls are
separate observations on both backends. No CUDA result is inferred from Metal.
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

MAGIC, COUNT = b'XGBGPR1\0', 319
Spec = namedtuple('Spec', 'id family layout mode rows groups weights')
INPUTS = ('ptr','predictions','labels','weights')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def shapes():
    out = []
    def add(family, layout, mode, lengths, weighted):
        ptr = np.concatenate(([0], np.cumsum(lengths, dtype=np.uint32))).astype('<u4')
        s = Spec(len(out), family, layout, mode, int(ptr[-1]), len(lengths), len(lengths) if weighted else 0)
        out.append((s, ptr))
    add(0,0,0,[1],False)
    for mode in range(1,16):
        add(0,0,mode,[2],True)
    for n in (0,1,31,255,256,257,4097,65537):
        for layout in range(3):
            for mode in range(12):
                add(1,layout,mode,[0,1,2,3,31,32,33,255,256,257,513,0,n,n+1,n//2,0],mode!=0)
    for mode in (0,2,3,7,8,10,11):
        add(1,2,mode,[1,31,1000003,0,65537,257,1],mode!=0)
    add(1,0,0,[],False)
    add(1,0,0,[0,0,0],False)
    for mode in range(6):
        add(2,0,mode,[1,0,2],False)
    return out


def layout(s):
    stride = (1,3,7)[s.layout] if s.family == 1 else 1
    offset = 4 if s.family == 1 else 0
    return stride, offset, s.rows*stride+(8 if s.family == 1 else 0)


def decode(data):
    require(len(data)>=16 and data[:8]==MAGIC, 'Invalid grouped PR header')
    kind,count = struct.unpack_from('<II',data,8)
    require(kind in (0,1,2) and count==COUNT, 'Wrong grouped PR kind/count')
    offset,cases = 16,[]
    def array(n,width):
        nonlocal offset
        require(offset+n*width<=len(data), 'Truncated grouped PR array')
        a=np.frombuffer(data,dtype=f'<u{width}',count=n,offset=offset)
        offset+=n*width
        return a
    for s,ptr in shapes():
        require(tuple(array(7,4))==s, 'Wrong grouped PR identity/order')
        observed_ptr=array(s.groups+1,4)
        require(np.array_equal(observed_ptr,ptr), 'Wrong grouped PR query boundaries')
        size=layout(s)[2]
        case=dict(spec=s,ptr=observed_ptr,predictions=array(size,4),labels=array(size,4),weights=array(s.weights,4))
        if kind:
            case['rejected']=int(array(1,4)[0])
            require(case['rejected']==int(s.family==2), 'Wrong grouped PR rejection status')
            normal=s.family!=2
            case['original']=array(2 if normal and kind==1 else 0,8)
            case['guards']=array(2*(s.groups+4) if normal and kind==1 and s.family==1 else 0,8)
            case['common']=array(2 if normal else 0,8)
            case['isolated']=array(2*np.count_nonzero(np.diff(ptr)) if normal else 0,8)
        cases.append(case)
    require(len(data)==offset+8 and data[offset:offset+4]==b'GPRE' and
            struct.unpack_from('<I',data,offset+4)[0]==COUNT, 'Incomplete/trailing grouped PR corpus')
    return dict(kind=kind,cases=cases,data=data,sha256=hashlib.sha256(data).hexdigest())


def load(path):
    if path.suffix in ('.xz','.gz'):
        opener=lzma.open if path.suffix=='.xz' else gzip.open
        with opener(path,'rb') as stream:
            return decode(stream.read())
    with path.open('rb') as stream:
        require(path.stat().st_size>0,'Empty grouped PR report')
        return decode(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))


def source_inputs(s,ptr):
    """Independent reconstruction of every original generator word and guard."""
    f=np.float32
    tiny=np.nextafter(f(0),f(1))
    if s.family==0:
        p=np.zeros(s.rows,dtype='<f4')
        if s.mode==0:
            y=np.array([.5],dtype='<f4');w=np.array([],dtype='<f4')
        elif s.mode<=12:
            label=(f(.25),f(.1),np.nextafter(f(1),f(0)),f(1e-30))[(s.mode-1)//3]
            y=np.full(2,label,dtype='<f4');w=np.array([(f(.7),f(1),tiny)[(s.mode-1)%3]],dtype='<f4')
        else:
            y=np.array([.25,.75],dtype='<f4');w=np.array([(np.inf,np.nan,-1)[s.mode-13]],dtype='<f4')
        return dict(predictions=p,labels=y,weights=w)
    if s.family==2:
        bad=(-tiny,np.nextafter(f(1),f(2)),f(-1),f(2),f(np.inf),f(-np.inf))[s.mode]
        return dict(predictions=np.array([1,0,2],dtype='<f4'),labels=np.array([0,1,bad],dtype='<f4'),weights=np.array([],dtype='<f4'))
    stride,offset,size=layout(s)
    p=np.full(size,-971,dtype='<f4');y=np.full(size,-193,dtype='<f4');weights=[]
    rng=np.random.RandomState(2871+s.mode)
    for g,(begin,end) in enumerate(zip(ptr[:-1],ptr[1:])):
        begin,end=int(begin),int(end)
        w=f(g+1)/f(7)
        if s.mode==2: w=np.ldexp(w,int(g*31%221)-110)
        if s.mode==3: w=tiny*f(1+g%7)
        if s.mode==4: w=f(1e37)*f(1+g%17)
        if s.mode==6: w=f(0)
        if s.mode==7: w=np.array([1e-30,1e30,1,tiny,1e-10,3],dtype='<f4')[g%6]
        if s.mode==9: w=np.array([np.inf,np.nan,-1,0,1],dtype='<f4')[g%5]
        if s.mode==10: w=tiny
        if s.mode==11: w=f(1)
        if s.weights: weights.append(w)
        rows=np.arange(begin,end,dtype=np.int64)
        draws=rng.randint(0,2**32,size=(end-begin,4 if s.mode==2 else 2),dtype=np.uint32)
        pred=((draws[:,0]%8191).astype(np.int64)-4095).astype('<f4')/f(93)
        label=(draws[:,1]%2).astype('<f4')
        if s.mode in (1,11): pred=np.where(rows%2,f(-0.),f(0.)).astype('<f4')
        if s.mode==2:
            pred=(draws[:,2]%7).astype('<f4');label=(draws[:,3]%17).astype('<f4')/f(16)
        if s.mode==3: pred=np.ldexp(pred,-145)
        if s.mode==4:
            pred[rows%7==0]=f(np.inf);pred[rows%7==1]=f(-np.inf)
        if s.mode==5: label.fill(f(g%2))
        if s.mode==7: label=np.array([0,1,np.nextafter(f(1),f(0)),1e-30,tiny,.5],dtype='<f4')[rows%6]
        if s.mode==8 and g%2 and len(label): label[0]=f(np.nan)
        if s.mode==10: label.fill(np.nextafter(f(1),f(0)))
        if s.mode==11: label.fill(f(1e-30))
        p[offset+rows*stride]=pred;y[offset+rows*stride]=label
    return dict(predictions=p,labels=y,weights=np.asarray(weights,dtype='<f4'))


def curve_reference(predictions,labels,weight):
    """Original source terms and serial binary64 reference, not a CUB emulator."""
    if len(labels)<2:
        return np.array([0.,0.])
    # The FP term has a binary64 subtraction/product then FP32 cast; TP has
    # FP32 multiplication. Keep this distinction even for subnormal weights.
    with np.errstate(all='ignore'):
        tp_terms=(labels*np.float32(weight)).astype(np.float64)
        total=np.cumsum(tp_terms,dtype=np.float64)[-1]
        fp_terms=((1.-labels.astype(np.float64))*float(weight)).astype('<f4').astype(np.float64)
        order=np.argsort(-predictions,kind='stable')
        fp=np.cumsum(fp_terms[order],dtype=np.float64)
        tp=np.cumsum(tp_terms[order],dtype=np.float64)
        if not fp[-1]*tp[-1]>0:
            return np.array([0.,0.])
        sorted_p=predictions[order]
        ends=np.r_[sorted_p[:-1]!=sorted_p[1:],True]
        fp,tp=fp[ends],tp[ends]
        fp0,tp0=np.r_[0.,fp[:-1]],np.r_[0.,tp[:-1]]
        previous,recall=tp0/total,tp/total
        h=np.where(tp==tp0,0.,(fp-fp0)/(tp-tp0))
        a=h+1.
        b=np.where(tp==tp0,0.,(fp0-h*tp0)/total)
        terms=np.where(b!=0.,(recall-previous-b/a*(np.log(a*recall+b)-np.log(a*previous+b)))/a,(recall-previous)/a)
        return np.array([np.cumsum(terms,dtype=np.float64)[-1],1.])


def near(actual,expected,exact=False):
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    equal=(actual==expected)|(np.isnan(actual)&np.isnan(expected))
    with np.errstate(all='ignore'):
        return equal if exact else equal|(np.isfinite(actual)&np.isfinite(expected)&(np.abs(actual-expected)<=2e-12+2e-12*np.abs(expected)))


def validate(report, progress=False):
    failures=[];fields=0;words=0;isolated=0;adapted=0
    for case in report['cases']:
        s=case['spec'];ptr=case['ptr'];expected=source_inputs(s,ptr)
        for key in ('predictions','labels','weights'):
            words+=len(case[key])
            require(np.array_equal(case[key],expected[key].view('<u4')),f'Grouped PR case {s.id}: original {key} bits differ')
        if s.rows and np.count_nonzero(np.diff(ptr))!=s.groups: adapted+=1
        if report['kind']==0 or s.family==2: continue
        stride,offset,_=layout(s)
        p=expected['predictions'][offset:offset+s.rows*stride:stride]
        y=expected['labels'][offset:offset+s.rows*stride:stride]
        references=[]
        for g,(begin,end) in enumerate(zip(ptr[:-1],ptr[1:])):
            w=expected['weights'][g] if s.weights else np.float32(1.)
            references.append(curve_reference(p[begin:end],y[begin:end],w))
        ref=np.asarray(references,dtype=np.float64).reshape(-1,2)
        nonempty=np.diff(ptr)!=0
        aggregate=np.cumsum(ref,axis=0)[-1] if len(ref) else np.zeros(2)
        isolated+=int(np.count_nonzero(nonempty))
        def check(name,observed,wanted,exact=False):
            nonlocal fields
            observed=np.asarray(observed,dtype=np.float64);wanted=np.asarray(wanted,dtype=np.float64)
            fields+=wanted.size
            good=near(observed,wanted,exact)
            if not np.all(good):
                failures.append(dict(case=s.id,field=name,count=int(np.count_nonzero(~good))))
        for key in ('common','original'):
            if not len(case[key]): continue
            value=case[key].view('<f8')
            check(key+'/area',value[0],aggregate[0],exact=s.family==0 and 1<=s.mode<=12)
            check(key+'/valid',value[1],aggregate[1],exact=True)
        value=case['isolated'].view('<f8').reshape(-1,2)
        check('isolated/area',value[:,0],ref[nonempty,0],exact=s.family==0 and 1<=s.mode<=12)
        check('isolated/valid',value[:,1],ref[nonempty,1],exact=True)
        if len(case['guards']):
            value=case['guards'].view('<f8').reshape(-1,2)
            check('guard',value[[0,1,-2,-1]],np.tile([-397.,973.],(4,1)),exact=True)
            check('original-queries/area',value[2:-2,0],ref[:,0])
            check('original-queries/valid',value[2:-2,1],ref[:,1],exact=True)
        if s.family==1 and s.mode==11:
            exact=float(np.float32(1e-30))/(1+float(np.float32(1e-30)))
            selected=np.diff(ptr)[nonempty]>=2
            values=case['isolated'].view('<f8').reshape(-1,2)[selected,0]
            if np.any(np.abs(values/exact-1)>=2e-12): failures.append(dict(case=s.id,field='tiny-tied-relative-accuracy'))
            if len(case['guards']):
                values=case['guards'].view('<f8').reshape(-1,2)[2:-2,0][np.diff(ptr)>=2]
                if np.any(np.abs(values/exact-1)>=2e-12): failures.append(dict(case=s.id,field='original-tiny-tied-relative-accuracy'))
        if progress and (s.id%24==0 or s.id+1==COUNT):
            print(f'Validated grouped PR {s.id+1}/{COUNT}; {words} input words',file=sys.stderr,flush=True)
    return dict(cases=len(report['cases']),input_words=words,checked_fields=fields,
                isolated_queries=isolated,empty_group_adaptations=adapted,reference_failures=failures)


def match_inputs(a,b):
    require(len(a['cases'])==len(b['cases'])==COUNT,'Incomplete grouped PR corpus')
    for x,y in zip(a['cases'],b['cases']):
        require(x['spec']==y['spec'] and all(np.array_equal(x[k],y[k]) for k in INPUTS),
                f"Grouped PR case {x['spec'].id}: source input bits differ")


def compare(metal,cuda):
    require(metal['kind']==1 and cuda['kind']==2,'Need Metal and actual CUDA observation reports')
    match_inputs(metal,cuda)
    checks=[validate(metal),validate(cuda)]
    differences=[];fields=0;classification_differences=0
    for a,b in zip(metal['cases'],cuda['cases']):
        for key in ('common','isolated'):
            fields+=len(a[key]);different=a[key]!=b[key]
            if np.any(different):
                n=int(np.count_nonzero(different))
                material=int(np.count_nonzero(~near(a[key].view('<f8'),b[key].view('<f8'))))
                classification_differences+=material
                differences.append(dict(case=a['spec'].id,field=key,bit_differences=n,outside_original_tolerance=material))
    return dict(status='reference-failure' if any(c['reference_failures'] for c in checks) else 'review-required',
                metal_sha256=metal['sha256'],cuda_sha256=cuda['sha256'],validation=checks,
                common_observed_fields=fields,bit_differences=differences,
                outside_original_tolerance=classification_differences,
                domain_limit='Empty queries removed explicitly; original layouts with empty groups are not executed on CUDA. Isolated-query outputs are separate calls, not aggregate internals.')


def unpack(report):
    parts=[MAGIC,struct.pack('<II',0,COUNT)]
    for case in report['cases']:
        parts.append(struct.pack('<7I',*case['spec']))
        parts.extend(case[key].tobytes() for key in INPUTS)
    parts.extend([b'GPRE',struct.pack('<I',COUNT)])
    return b''.join(parts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate-metal');v.add_argument('metal',type=Path)
    u=sub.add_parser('unpack');u.add_argument('input',type=Path);u.add_argument('output',type=Path)
    c=sub.add_parser('compare');c.add_argument('metal',type=Path);c.add_argument('cuda',type=Path);c.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='unpack':
        report=load(args.input)
        with args.output.open('xb') as stream: stream.write(unpack(report))
        return 0
    if args.command=='validate-metal':
        report=load(args.metal);require(report['kind']==1,'Expected Metal grouped PR observations')
        result=validate(report,True);print(json.dumps(result));return int(bool(result['reference_failures']))
    require(not args.output.exists(),'Comparison output already exists')
    result=compare(load(args.metal),load(args.cuda))
    with args.output.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result));return 1 if result['status']=='reference-failure' else 2


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError,struct.error) as error:
        print(f'FAIL grouped PR report: {error}',file=sys.stderr);sys.exit(1)
