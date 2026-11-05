#!/usr/bin/env python3
"""Strict full-corpus ranking metric input validation and independent references.

CUDA report outputs must be actual observations. Exit 2 after a valid two-GPU
comparison: numerical differences and explicit API adaptations require review.
No tolerance is increased from the original 2e-12 scaled score budget.
"""
import argparse
from collections import namedtuple
import gzip
import hashlib
import json
import lzma
import math
import mmap
from pathlib import Path
import struct
import sys
import numpy as np

MAGIC,COUNT=b'XGBRMT1\0',1031
EDGE,SORT,SCORE=range(3)
Spec=namedtuple('Spec','id family layout kind stride topk mode minus rows groups weights discounts inverse')
INPUTS=('ptr','predictions','labels','weights','discounts','inverse')
FIELDS=('original','common','reducer','prefix','order')


def require(ok,message):
    if not ok:raise ValueError(message)


def ndcg(s):return s.kind in (1,2)
def no_groups(s):return ndcg(s) and not s.groups

def guards(s):return int(s.family!=SORT and s.topk==0) | (2 if no_groups(s) else 0)


def shapes():
    out=[]
    def add(family,layout,kind,stride,topk,mode,minus,lengths,weighted,discounts,inverse):
        ptr=np.r_[0,np.cumsum(lengths,dtype=np.uint32)].astype('<u4')
        out.append((Spec(len(out),family,layout,kind,stride,topk,mode,int(minus),int(ptr[-1]),len(lengths),
                         len(lengths) if weighted else 0,discounts,inverse),ptr))
    add(EDGE,0,0,1,10,0,False,[0,0],True,0,0)
    add(EDGE,0,0,1,0,0,False,[1],True,0,0)
    add(EDGE,0,0,1,1,0,False,[1],True,0,0)
    add(EDGE,0,0,1,3,0,False,[3],True,0,0)
    for minus in (False,True):
        for mode in (0,1):add(EDGE,0,3,1,10,mode,minus,[0,2,0],True,0,0)
    add(EDGE,0,3,1,4,0,False,[4],True,0,0)
    add(EDGE,0,3,1,0,0,False,[1],True,0,0)
    add(EDGE,0,3,1,0,1,False,[1],True,0,0)
    add(EDGE,0,2,1,2,0,False,[2],False,2,1)
    for mode in range(3):add(EDGE,0,2,1,1,mode,False,[1],True,1,1)
    sorts=[[],[0,0,0],[1],[31,0,32,33],[255,256,0,1],[257,0,1,256],[0,65537,0],
           [65537,1,4097,0,257],[0,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096,0],[4097,1],
           [g%257 for g in range(4097)],[1]*32768+[4097]+[1]*32768]
    for layout,lengths in enumerate(sorts):
        for stride in (1,3):
            for mode in range(5):add(SORT,layout,4,stride,0,mode,False,lengths,False,0,0)
    scores=[[],[0,0],[1],[0,31,0,257,1],[4096,32,0],[0,4097,1],[65537,1,256]]
    for layout,lengths in enumerate(scores):
        for kind in range(4):
            for stride in (1,3):
                for topk in (0,1,10,2**32-1):
                    for mode in range(4):add(SCORE,layout,kind,stride,topk,mode,mode&1,lengths,mode!=0,max(lengths,default=0),len(lengths))
    require(len(out)==COUNT,'Incomplete ranking metric matrix');return out


def field_sizes(s,kind):
    metal=kind==1;score=s.family!=SORT;live=kind==2 and not no_groups(s)
    return dict(original=2 if metal and score else 0,common=2 if score and (metal or live) else 0,
                reducer=s.groups if live and score else 0,prefix=s.rows if live and s.kind==3 else 0,
                order=s.rows if (metal and not score) or live else 0)


def decode(data):
    require(len(data)>=16 and data[:8]==MAGIC,'Invalid ranking metric header')
    kind,count=struct.unpack_from('<II',data,8)
    require(kind in (0,1,2) and count==COUNT,'Wrong ranking metric backend/count')
    offset=16;cases=[]
    def array(n,width):
        nonlocal offset
        require(offset+n*width<=len(data),'Truncated ranking metric array')
        a=np.frombuffer(data,dtype=f'<u{width}',count=n,offset=offset);offset+=n*width;return a
    for s,ptr in shapes():
        require(tuple(array(13,4))==s,'Wrong ranking metric identity/order')
        actual=array(s.groups+1,4);require(np.array_equal(actual,ptr),'Wrong ranking query boundaries')
        c=dict(spec=s,ptr=actual,predictions=array(s.rows*s.stride,4),labels=array(0 if s.family==SORT else s.rows*s.stride,4),
               weights=array(s.weights,4),discounts=array(s.discounts,8),inverse=array(s.inverse,8))
        if kind:
            c['guards']=int(array(1,4)[0]);require(c['guards']==(guards(s) if kind==2 else 0),'Wrong observed ranking guard status')
            for name,n in field_sizes(s,kind).items():c[name]=array(n,8)
        cases.append(c)
    require(len(data)==offset+8 and data[offset:offset+4]==b'RMTE' and struct.unpack_from('<I',data,offset+4)[0]==COUNT,
            'Incomplete/trailing ranking metric report')
    return dict(kind=kind,cases=cases,data=data,sha256=hashlib.sha256(data).hexdigest())


def load(path):
    if path.suffix in ('.xz','.gz'):
        with (lzma.open if path.suffix=='.xz' else gzip.open)(path,'rb') as stream:return decode(stream.read())
    with path.open('rb') as stream:
        require(path.stat().st_size>0,'Empty ranking metric report')
        return decode(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))


def serial_sum(x):
    return float(np.cumsum(x,dtype=np.float64)[-1]) if len(x) else 0.


def discounts(n):return np.array([1/math.log2(i+2) for i in range(n)],dtype='<f8')


def gain(y,exponential):
    return ((np.left_shift(np.uint64(1),y.astype(np.uint64))-np.uint64(1)).astype('<f8')
            if exponential else y.astype('<f8'))


def source_inputs():
    """Reproduce original C++ mt19937 draws, including discarded/guard words."""
    f=np.float32;sort_rng=np.random.RandomState(761);score_rng=np.random.RandomState(7182)
    discount_cache={}
    binary=np.array([0,1,.99999994,1e-7,-1e-7],dtype='<f4')
    linear=np.array([-.4,0,.7,1.2,2.8,6.4],dtype='<f4')
    exponential=np.array([0,1,2,5,15,31,1e-7,1.0000001],dtype='<f4')
    edge_labels=[[],[1],[np.nextafter(f(0),f(1))],[.99999994,0,0],[0,0],[0,0],[0,0],[0,0],
                 [1,1,0,1],[1],[0],[1,1],[1],[1],[1]]
    edge_weights=[[2,3],[2],[.5],[.7],[0,0,0],[2,2,2],[0,0,0],[2,2,2],[.7],[2],[2],[],[2],[2],[2]]
    sort_bits=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
                        0x7f800000,0xff800000,0x7f7fffff,0xff7fffff],dtype='<u4').view('<f4')
    def draws(rng,n):return rng.randint(0,2**32,size=n,dtype=np.uint32)
    for s,ptr in shapes():
        p=np.full(s.rows*s.stride,np.nan,dtype='<f4');y=np.full(0 if s.family==SORT else s.rows*s.stride,np.nan,dtype='<f4')
        w=np.zeros(s.weights,dtype='<f4');d=np.zeros(s.discounts,dtype='<f8');inv=np.zeros(s.inverse,dtype='<f8')
        if s.family==EDGE:
            p[:]=-np.arange(s.rows,dtype='<f4');y[:]=edge_labels[s.id];w[:]=edge_weights[s.id]
            if s.id==11:d[:]=[1,.1];inv[:]=[.3]
            elif s.id>=12:d[:]=[1];inv[:]=[np.nan if s.id==12 else 0 if s.id==13 else -1]
        elif s.family==SORT:
            r=draws(sort_rng,(s.rows,2 if s.mode==4 else 1));v=((r[:,0]%101).astype(np.int64)-50).astype('<f4')/f(7)
            if s.mode==1:v=np.where(np.arange(s.rows)%3,f(-0.),f(0.)).astype('<f4')
            if s.mode==2:v=np.arange(s.rows,dtype='<f4')
            if s.mode==3:v=-np.arange(s.rows,dtype='<f4')
            if s.mode==4:v=sort_bits[r[:,1]%12]
            p[::s.stride]=v
        else:
            if s.mode and s.mode!=1:w[:]=(draws(score_rng,s.weights)%101).astype('<f4')/f(7)
            if s.mode==3:w[:]=np.ldexp(np.ones(s.weights,dtype='<f4'),(draws(score_rng,s.weights)%241).astype(np.int64)-120)
            r=draws(score_rng,(s.rows,3 if s.kind==3 and s.mode==2 else 2))
            p[::s.stride]=((r[:,0]%101).astype(np.int64)-50).astype('<f4')/f(7)
            vals=binary[r[:,1]%5] if s.kind in (0,3) else exponential[r[:,1]%8] if s.kind==1 else linear[r[:,1]%6]
            if s.kind==0:vals[::17]=np.nextafter(f(0),f(1))
            if s.kind==3 and s.mode==2:vals=(r[:,2]%2).astype('<f4')
            if s.mode==1:vals.fill(0)
            y[::s.stride]=vals
            if s.discounts not in discount_cache:discount_cache[s.discounts]=discounts(s.discounts)
            d[:]=discount_cache[s.discounts]
            if ndcg(s):
                for g,(begin,end) in enumerate(zip(ptr[:-1],ptr[1:])):
                    values=vals[int(begin):int(end)];order=np.argsort(-values,kind='stable');k=min(s.topk,len(values))
                    ideal=serial_sum(d[:k]*gain(values[order[:k]],s.kind==1));inv[g]=0 if ideal==0 else 1/ideal
        yield dict(spec=s,ptr=ptr,predictions=p,labels=y,weights=w,discounts=d,inverse=inv)


def reference(c,common=False,observed_prefix=None,observed_reducer=None):
    """Independent stable ordering and original serial double formulas.

    Optional observed producer arrays check CUDA's actual downstream consumer
    separately; they never replace the independent full-score check.
    """
    s=c['spec'];p=c['predictions'].view('<f4')[::s.stride];y=c['labels'].view('<f4')[::s.stride]
    w=c['weights'].view('<f4');d=c['discounts'].view('<f8');inv=c['inverse'].view('<f8')
    if common and s.id==11:d=discounts(len(d))
    order=np.empty(s.rows,dtype='<u8');prefix=np.empty(s.rows if s.kind==3 else 0,dtype='<f8')
    reducer=[];scores=[];weights=[]
    for g,(begin,end) in enumerate(zip(c['ptr'][:-1],c['ptr'][1:])):
        begin,end=int(begin),int(end);n=end-begin
        idx=np.argsort(-p[begin:end],kind='stable') if n>1 else np.arange(n,dtype=np.uint64)
        order[begin:end]=idx
        if s.family==SORT:continue
        values=y[begin:end][idx].astype('<f8');k=min(s.topk,n);weight=float(w[g]) if len(w) else 1.
        if s.kind==0:
            subtotal=serial_sum(values[:k]/k*weight) if k else 0.;score=subtotal
        elif ndcg(s):
            subtotal=serial_sum(d[:k]*gain(values[:k],s.kind==1))
            actual=observed_reducer[g] if observed_reducer is not None else subtotal
            score=float(1-s.minus) if inv[g]<=0 else (actual*inv[g])*weight
        else:
            hits=np.cumsum(values,dtype='<f8');prefix[begin:end]=hits
            actual_hits=observed_prefix[begin:end] if observed_prefix is not None else hits
            subtotal=serial_sum(actual_hits[:k]/np.arange(1,k+1,dtype='<f8')*values[:k])
            actual=observed_reducer[g] if observed_reducer is not None else subtotal
            total=actual_hits[-1] if n else 0
            with np.errstate(divide='ignore',invalid='ignore'):
                score=float(np.divide(actual*weight,min(total,s.topk))) if total>0 else float(1-s.minus)
        if s.kind==0 and observed_reducer is not None:score=observed_reducer[g]
        reducer.append(subtotal);scores.append(score);weights.append(weight)
    return dict(pair=np.array([serial_sum(scores),serial_sum(weights)],dtype='<f8'),order=order,
                reducer=np.array(reducer,dtype='<f8'),prefix=prefix)


def check_array(actual,expected,exact=False):
    a=np.asarray(actual);e=np.asarray(expected)
    require(a.shape==e.shape,'Reference array shape mismatch')
    if exact:return int(np.count_nonzero(~((a==e) | (np.isnan(a)&np.isnan(e)))))
    with np.errstate(invalid='ignore',over='ignore'):
        same=(a==e) | (np.isnan(a)&np.isnan(e));error=np.abs(a-e)
        ok=same | (np.isfinite(a)&np.isfinite(e)&(error<=2e-12*np.maximum(1,np.abs(e))))
    return int(np.count_nonzero(~ok))


def validate(report,progress=False):
    failed=[];fields=0;input_words=0;observed_guards=0
    def check(s,name,a,e,exact=False):
        nonlocal fields
        fields+=len(a);count=check_array(a,e,exact)
        if count:failed.append(dict(case=s.id,field=name,count=count))
    for c,expected in zip(report['cases'],source_inputs()):
        s=c['spec']
        for name in INPUTS:
            want=expected[name];want=want.view(f'<u{want.dtype.itemsize}')
            require(np.array_equal(c[name],want),f'Case {s.id}: changed source input {name}')
            input_words+=len(want)
        if report['kind']:
            ref=reference(c);observed_guards+=bool(c['guards'])
            if len(c['order']):check(s,'order',c['order'],ref['order'],True)
            if len(c['original']):check(s,'original',c['original'].view('<f8'),ref['pair'],s.family==EDGE)
            if len(c['common']):
                common=reference(c,True) if s.id==11 else ref
                check(s,'common',c['common'].view('<f8'),common['pair'],s.family==EDGE and s.id!=11)
            if report['kind']==2 and not no_groups(s) and s.family!=SORT:
                # Check producers independently, then the final result from its
                # own producers. MAP prefixes use the unchanged score budget.
                if s.kind==3:check(s,'prefix',c['prefix'].view('<f8'),ref['prefix'])
                downstream=reference(c,True,c['prefix'].view('<f8') if s.kind==3 else None,c['reducer'].view('<f8'))
                check(s,'reducer',c['reducer'].view('<f8'),downstream['reducer'])
                check(s,'own-producer-consumer',c['common'].view('<f8'),downstream['pair'])
        if progress and ((s.id+1)%64==0 or s.id+1==COUNT):print(f"Validated {s.id+1}/{COUNT} records; {fields} output fields",flush=True)
    return dict(status='failed-reference' if failed else 'passed-reference',cases=COUNT,input_words=input_words,
                output_fields=fields,guarded_records=observed_guards,failures=failed,sha256=report['sha256'])


def compare(metal,cuda):
    require(metal['kind']==1 and cuda['kind']==2,'Comparison requires Metal then CUDA reports')
    checks=[validate(metal),validate(cuda)];differences=[]
    for m,c in zip(metal['cases'],cuda['cases']):
        s=m['spec']
        for field in INPUTS:require(np.array_equal(m[field],c[field]),f'Case {s.id}: GPU inputs differ')
        for field in ('common','order'):
            if len(m[field]) and len(c[field]):
                a=m[field] if field=='order' else m[field].view('<f8');b=c[field] if field=='order' else c[field].view('<f8')
                changed=check_array(a,b,True)
                if changed:differences.append(dict(case=s.id,field=field,changed=changed,outside_budget=check_array(a,b,field=='order')))
    failed=any(r['failures'] for r in checks) or any(d['outside_budget'] for d in differences)
    return dict(status='failed-reference-or-budget' if failed else 'review-required',references=checks,differences=differences,
                runtime_equivalence_certified=False,adaptations='topk0 parameter rejection then explicit raw cache input; supplied inverse-IDCG; custom-discount edge11 separately compared with standard discounts; zero-group NDCG guard')


def unpack(report,path):
    require(not path.exists(),'Output already exists')
    with path.open('xb') as stream:
        stream.write(MAGIC+struct.pack('<II',0,COUNT))
        for c in report['cases']:
            stream.write(struct.pack('<13I',*c['spec']))
            for field in INPUTS:stream.write(c[field].tobytes())
        stream.write(b'RMTE'+struct.pack('<I',COUNT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('validate','unpack','compare'):
        p=sub.add_parser(name);p.add_argument('report',type=Path)
        if name=='compare':p.add_argument('cuda',type=Path)
        p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Output already exists');r=load(args.report)
    if args.command=='unpack':validate_result=validate(r,True);require(not validate_result['failures'],'Invalid source report');unpack(r,args.output);return 0
    result=compare(r,load(args.cuda)) if args.command=='compare' else validate(r,True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return 1 if result['status'].startswith('failed') else 2 if args.command=='compare' else 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as error:print(f'FAIL ranking metric: {error}',file=sys.stderr);sys.exit(1)
