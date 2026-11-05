#!/usr/bin/env python3
"""Validate/replay the complete AUC corpus with original numerical contracts.

Reports keep original storage bits and separate raw curves, normalization
metadata, linked-library outputs and actual source-produced diagnostics.
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

MAGIC, COUNT = b'XGBAUC1\0', 528
EDGE, NORMALIZE, METADATA, MALFORMED, WIDE, MAIN, EXTREME = range(7)
Spec = namedtuple('Spec', 'id family mode layout rows targets multiclass roc weights summaries')
INPUTS = ('predictions', 'labels', 'weights', 'summaries')
FIELDS = ('curves', 'source_curves', 'metadata', 'final', 'cpu', 'linked', 'guards',
          'terms', 'prefixes', 'totals', 'order')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def specs():
    out = []
    def add(family, mode=0, layout=0, rows=0, targets=0, multiclass=0, roc=0, weights=0, summaries=0):
        out.append(Spec(len(out), family, mode, layout, rows, targets, multiclass, roc, weights, summaries))
    for i in range(4):
        add(EDGE, i, rows=1 if i < 2 else 2, targets=1, roc=int(i % 2 == 0), weights=0 if i < 2 else 2)
    for i, n in enumerate((3,6,3,3,3,3,3,3,6,3,3,3)):
        add(NORMALIZE, i, targets=n//3, summaries=n)
    for i, n in enumerate((1,2,31,257,65535,65536,65537,262145), 12):
        add(NORMALIZE, i, targets=n, summaries=3*n)
    for multi in (0,1):
        for roc in (0,1):
            for rows in (0,4):
                add(METADATA, rows=rows, targets=3, multiclass=multi, roc=roc, weights=rows)
    add(NORMALIZE, 20)
    add(NORMALIZE, 21, targets=65536, summaries=3*65536)
    add(MALFORMED, summaries=2)
    for n in (65535,65536,65537):
        for roc in (1,0):
            add(WIDE, rows=2, targets=n, roc=roc)
    for rows in (0,1,2,31,255,256,257,4097,65537):
        for targets in (1,3):
            for layout in range(3):
                for mode in range(9):
                    add(MAIN, mode, layout, rows, targets, int(targets==3 and mode%2), 2, rows if mode>=2 else 0)
    add(EXTREME, rows=2, targets=1, weights=2)
    require(len(out)==COUNT, 'Incomplete AUC case matrix')
    return out


def strides(s, columns):
    if s.family != MAIN:
        return columns,1
    return (1 if s.layout==1 else columns*(2 if s.layout==2 else 1)+(3 if s.layout==2 else 0),
            max(s.rows,1) if s.layout==1 else 2 if s.layout==2 else 1)


def storage(s, columns):
    a,b=strides(s,columns)
    return s.rows*a+columns*b+1 if s.family==MAIN else s.rows*columns


def multi(s):
    return s.targets>1 or bool(s.multiclass)


def roc(s, call):
    return call==0 if s.roc==2 else bool(s.roc)


def trace(s, call):
    return s.family==MAIN and s.mode==4 and not roc(s,call)


def lengths(s):
    return storage(s,s.targets),storage(s,1 if s.multiclass else s.targets),s.weights,s.summaries


def observation_shape(s, call, kind, totals):
    sizes=dict.fromkeys(FIELDS,0)
    if kind==1:
        curve=s.family in (EDGE,WIDE,MAIN,EXTREME)
        metadata=s.family==METADATA or (curve and multi(s))
        sizes.update(curves=3*s.targets if curve else 0, metadata=3*s.targets if metadata else 0,
                     final=int(s.family in (NORMALIZE,WIDE) or metadata), cpu=int(s.family in (WIDE,EXTREME)),
                     guards=4 if s.family==METADATA else 0)
        if trace(s,call) and s.rows:
            sizes.update(terms=2*s.rows*s.targets,prefixes=2*s.rows*s.targets,totals=2*s.targets)
        return int(s.family==MALFORMED),sizes
    if s.family==MALFORMED:
        return 2,sizes
    if s.family==NORMALIZE:
        sizes.update(metadata=3*s.targets,final=1,linked=1)
        return 0,sizes
    is_multi=multi(s)
    if not is_multi and roc(s,call) and not s.rows:
        return 1,sizes
    if is_multi:
        sizes.update(source_curves=3*s.targets if s.rows else 0, metadata=3*s.targets,final=1,linked=1)
    else:
        sizes.update(curves=3,linked=3)
    if not roc(s,call):
        sizes['totals']=2*s.targets
    if trace(s,call) and s.rows and (is_multi or (len(totals)==2 and np.all(totals>0))):
        sizes.update(terms=2*s.rows*s.targets,prefixes=2*s.rows*s.targets,order=s.rows*s.targets)
    return 0,sizes


def decode(data):
    require(len(data)>=16 and data[:8]==MAGIC,'Invalid AUC header')
    kind,count=struct.unpack_from('<II',data,8)
    require(kind in (0,1,2) and count==COUNT,'Wrong AUC kind/count')
    offset,cases=16,[]
    def array(n,width):
        nonlocal offset
        require(0<=n and offset+n*width<=len(data),'Truncated AUC array')
        value=np.frombuffer(data,dtype=f'<u{width}',count=n,offset=offset)
        offset+=n*width
        return value
    for s in specs():
        require(tuple(array(10,4))==s,'Wrong AUC identity/order')
        case=dict(spec=s)
        for key,n,width in zip(INPUTS,lengths(s),(4,4,4,8)):
            case[key]=array(n,width)
        case['calls']=[]
        if kind:
            for call in range(2 if s.family==MAIN else 1):
                obs=dict(status=int(array(1,4)[0]))
                for field in FIELDS:
                    n=int(array(1,8)[0])
                    # Validate the recorded size before interpreting any floating values.
                    require(n<=max(3*s.targets,2*s.rows*s.targets,4),'Excessive AUC observation length')
                    obs[field]=array(n,8)
                status,sizes=observation_shape(s,call,kind,obs['totals'].view('<f8'))
                require(obs['status']==status and all(len(obs[k])==v for k,v in sizes.items()),
                        f'AUC case {s.id} call {call}: missing/unexpected observations or status')
                case['calls'].append(obs)
        cases.append(case)
    require(len(data)==offset+8 and data[offset:offset+4]==b'AUCE' and
            struct.unpack_from('<I',data,offset+4)[0]==COUNT,'Incomplete/trailing AUC corpus')
    return dict(kind=kind,cases=cases,data=data,sha256=hashlib.sha256(data).hexdigest())


def load(path):
    if path.suffix in ('.xz','.gz'):
        with (lzma.open if path.suffix=='.xz' else gzip.open)(path,'rb') as stream:
            return decode(stream.read())
    with path.open('rb') as stream:
        require(path.stat().st_size>0,'Empty AUC report')
        return decode(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))


def source_inputs(s,rng):
    """Reconstruct every original C++ generator draw, including unused draws."""
    f=np.float32
    p=np.zeros(storage(s,s.targets),dtype='<f4')
    columns=1 if s.multiclass else s.targets
    y=np.zeros(storage(s,columns),dtype='<f4')
    w=np.zeros(s.weights,dtype='<f4');summaries=np.zeros(s.summaries,dtype='<f8')
    if s.family==EDGE:
        y.fill(1 if s.mode<2 else .25)
        if s.weights:w.fill(.7)
    elif s.family==NORMALIZE:
        fixed=([2,3,1],[2,3,1,4,1,3],[0,1,1],[-1,2,1],[np.nan,1,1],[2,0,1],
               [2,1,np.nan],[2,-1,1],[2,1,1,2,-1,1],[np.inf,1,1],[2,np.inf,1],[1e-300,1e-200,5e-301])
        if s.mode<12:summaries=np.asarray(fixed[s.mode],dtype='<f8')
        elif s.mode<20:
            t=np.arange(s.targets,dtype=np.int64)
            summaries[0::3]=1.+t%17;summaries[1::3]=1.+t%5;summaries[2::3]=(t%11)/19.
    elif s.family==MALFORMED:
        summaries[:]=[1,2]
    elif s.family==METADATA:
        p[:]=np.arange(len(p))%5
        y[:]=np.arange(len(y))%3
        if not s.multiclass:y/=f(3)
        w.fill(.7)
    elif s.family==WIDE:
        t=np.arange(s.targets)
        p[:s.targets]=(t%3==1);p[s.targets:]=(t%3==0);y[s.targets:]=1
    elif s.family==EXTREME:
        p[:]=[1,0];y[:]=[0,1];w[:]=[1e30,1]
    else:
        p.fill(-999);y.fill(-999)
        draws_per_row=s.targets*(1+int(s.mode==2))+columns*(1+int(s.mode==8 or (s.mode==5 and not s.multiclass)))
        draws_per_row+=int(s.mode>=2)*(1+int(s.mode in (3,4,8)))
        draws=rng.randint(0,2**32,size=(s.rows,draws_per_row),dtype=np.uint32)
        at=0;rows=np.arange(s.rows,dtype=np.int64)
        ps,pt=strides(s,s.targets);ys,yt=strides(s,columns)
        for t in range(s.targets):
            v=((draws[:,at]%32767).astype(np.int64)-16383).astype('<f4');at+=1
            if s.mode==1:v=np.where(rows%2,f(-0.),f(0.)).astype('<f4')
            if s.mode==2:v=(draws[:,at]%7).astype('<f4');at+=1
            if s.mode==3:v=np.ldexp(v,-149)
            if s.mode==4:v[rows%37==0]=np.where(rows[rows%37==0]%2,f(np.inf),f(-np.inf))
            p[rows*ps+t*pt]=v
        for t in range(columns):
            v=(draws[:,at]%(s.targets if s.multiclass else 2)).astype('<f4');at+=1
            if s.mode==5 and not s.multiclass:v=(draws[:,at]%17).astype('<f4')/f(16);at+=1
            if s.multiclass and s.mode==3:v[rows%7==0]=np.nextafter(f(0),f(1))
            if s.mode==6:v.fill(1)
            if s.mode==8:v=(draws[:,at]%1001).astype('<f4')/f(1000);at+=1
            y[rows*ys+t*yt]=v
        if s.mode>=2:
            w=(draws[:,at]%31).astype('<f4')/f(8);at+=1
            if s.mode==3:w=np.ldexp((draws[:,at]%127).astype('<f4'),-149);at+=1
            if s.mode==4:w=np.ldexp(w,(draws[:,at]%241).astype(np.int32)-120);at+=1
            if s.mode==7:w.fill(0)
            if s.mode==8:w=(draws[:,at]%101).astype('<f4')/f(7);at+=1
        require(at==draws_per_row,'AUC generator draw count')
    return dict(predictions=p,labels=y,weights=w,summaries=summaries)


def validate_inputs(report,progress=False):
    rng=np.random.RandomState(718);words=0
    for case in report['cases']:
        s=case['spec'];expected=source_inputs(s,rng)
        for name in INPUTS:
            words+=len(case[name]);width=8 if name=='summaries' else 4
            require(np.array_equal(case[name],expected[name].view(f'<u{width}')),
                    f'AUC case {s.id}: original {name} bits differ')
        if progress and ((s.id+1)%48==0 or s.id+1==COUNT):
            print(f'Validated AUC inputs {s.id+1}/{COUNT}; {words} words',file=sys.stderr,flush=True)
    return dict(cases=len(report['cases']),input_words=words)


def match_inputs(a,b):
    require(len(a['cases'])==len(b['cases'])==COUNT,'Incomplete AUC reports')
    for x,y in zip(a['cases'],b['cases']):
        require(x['spec']==y['spec'] and all(np.array_equal(x[k],y[k]) for k in INPUTS),
                f"AUC case {x['spec'].id}: input bits differ")


def unpack(report):
    parts=[MAGIC,struct.pack('<II',0,COUNT)]
    for case in report['cases']:
        parts.append(struct.pack('<10I',*case['spec']))
        parts.extend(case[name].tobytes() for name in INPUTS)
    parts.extend([b'AUCE',struct.pack('<I',COUNT)])
    return b''.join(parts)


def near(actual,expected,atol=0,exact=False):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    same=(a==b)|(np.isnan(a)&np.isnan(b))
    with np.errstate(all='ignore'):
        return same if exact else same|(np.isfinite(a)&np.isfinite(b)&(np.abs(a-b)<=atol+2e-12*np.abs(b)))


def serial(values):
    values=np.asarray(values,dtype=np.float64)
    return np.cumsum(values,dtype=np.float64)[-1] if len(values) else np.float64(0.)


def normalize(metadata):
    v=np.asarray(metadata,dtype=np.float64).reshape(-1,3)
    with np.errstate(all='ignore'):
        area,weight,auc=v.T
        numerator=serial(np.where(area>0,(auc/area)*weight,np.nan))
        denominator=serial(np.where(area>0,weight,0.))
        return numerator/denominator if denominator!=0 and not np.isnan(numerator) else np.nan


def column(case,target):
    s=case['spec'];r=np.arange(s.rows,dtype=np.int64)
    a,b=strides(s,s.targets);p=case['predictions'].view('<f4')[r*a+target*b]
    a,b=strides(s,1 if s.multiclass else s.targets)
    y=case['labels'].view('<f4')[r*a+(0 if s.multiclass else target*b)]
    if s.multiclass:y=(y==target).astype('<f4')
    w=case['weights'].view('<f4') if s.weights else np.ones(s.rows,dtype='<f4')
    return p,y,w


def curve_reference(p,y,w,is_multi):
    """Original serial source expression; FP32 terms are rounded separately."""
    if not len(p):
        return np.zeros(3),np.zeros(3)
    order=np.argsort(-p,kind='stable');p=p[order];y=y[order];w=w[order]
    with np.errstate(all='ignore'):
        terms=np.column_stack((((1.-y.astype(np.float64))*w.astype(np.float64)).astype('<f4').astype(np.float64),
                               (y*w).astype(np.float64)))
        prefixes=np.cumsum(terms,axis=0,dtype=np.float64)
        fp,tp=prefixes[-1]
        totals=[serial(((np.float32(1)-y)*w).astype(np.float64)),serial((y*w).astype(np.float64))]
        ends=np.r_[p[:-1]!=p[1:],True]
        x,z=prefixes[ends].T;x0,z0=np.r_[0.,x[:-1]],np.r_[0.,z[:-1]]
        roc_area=serial(np.abs(x-x0)*(z+z0)*.5)
        pr_area,_=integral(prefixes,p,totals[1])
        invalid=totals[0]<=0 or totals[1]<=0
        if invalid:pr_area=np.nan
        pr=np.zeros(3) if invalid and not is_multi else np.array([1.,1.,pr_area])
        return np.array([fp,tp,roc_area]),pr


def integral(prefixes,sorted_p,total_tp):
    """Source difference-of-logs on observed endpoints and its original budget."""
    ends=np.r_[sorted_p[:-1]!=sorted_p[1:],True]
    fp,tp=np.asarray(prefixes,dtype=np.float64)[ends].T
    fp0,tp0=np.r_[0.,fp[:-1]],np.r_[0.,tp[:-1]]
    with np.errstate(all='ignore'):
        previous,recall=tp0/total_tp,tp/total_tp
        h=np.where(tp==tp0,0.,(fp-fp0)/(tp-tp0));a=h+1
        b=np.where(tp==tp0,0.,(fp0-h*tp0)/total_tp)
        l1,l0=np.log(a*recall+b),np.log(a*previous+b)
        term=np.where(b!=0,(recall-previous-b/a*(l1-l0))/a,(recall-previous)/a)
        ulp0=np.abs(np.nextafter(l0,np.inf)-l0);ulp1=np.abs(np.nextafter(l1,np.inf)-l1)
        log_bound=serial(np.where(b!=0,np.abs((b/a)/a)*4*(ulp0+ulp1),0.))
        n=len(sorted_p);u=2.**-53;depth=n+(n+255)//256+32
        gamma=depth*u/(1-depth*u)
        bound=log_bound+(gamma+64*u)*(serial(np.abs(term))+log_bound)
        return serial(term),bound


def exact_prefix(values):
    """Exact nonnegative FP32 sums in 2^-149 units, then binary64 rounding."""
    words=np.asarray(values,dtype='<f4').view('<u4');total=0;out=[]
    for word in words:
        word=int(word);exponent=(word>>23)&255;mantissa=word&0x7fffff
        require(not (word>>31) and exponent!=255,'Exact prefix input domain')
        if exponent:mantissa|=0x800000
        total+=mantissa<<max(0,exponent-1)
        out.append(float(total)*2.**-149)
    return np.asarray(out,dtype=np.float64)


def trace_checks(s,target,p,y,w,obs,kind,actual,check,counts):
    """Each backend is judged using its own observed prefixes and totals."""
    n=len(p)
    if not n:return
    order=np.argsort(-p,kind='stable')
    if kind==2 and len(obs['order']):
        recorded=obs['order'].reshape(s.targets,n)[target]
        # CUDA segmented sorting stores global transposed indices.
        check('trace/order',recorded,order+target*n,exact=True)
    with np.errstate(all='ignore'):
        fp=((1.-y.astype(np.float64))*w.astype(np.float64)).astype('<f4')
        tp=(y*w).astype('<f4');original_fp=((np.float32(1)-y)*w).astype('<f4')
    total_ref=[exact_prefix(original_fp[order])[-1],exact_prefix(tp[order])[-1]]
    totals=obs['totals'].view('<f8').reshape(s.targets,2)[target]
    u=2.**-53
    def bound_check(name,values,reference,depth):
        gamma=depth*u/(1-depth*u);bound=(gamma+2*u)*reference
        counts['bounded_fields']+=np.size(reference)
        check(name,np.abs(values-reference)<=bound,True,exact=True)
    bound_check('trace/totals',totals,np.asarray(total_ref),(n+255)//256+8)
    if totals[0]<=0 or totals[1]<=0:
        check('trace/degenerate-integral',actual,np.nan if multi(s) else 0.,exact=True)
    if not len(obs['terms']):
        require(kind==2 and not multi(s) and np.any(totals<=0),'Missing executed AUC trace')
        return
    terms=obs['terms'].view('<f8').reshape(s.targets,n,2)[target]
    prefixes=obs['prefixes'].view('<f8').reshape(s.targets,n,2)[target]
    expected_terms=np.column_stack((fp[order],tp[order])).astype(np.float64)
    check('trace/terms',terms,expected_terms,exact=True);counts['exact_terms']+=2*n
    levels=1;blocks=(n+255)//256
    while blocks>1:levels+=1;blocks=(blocks+255)//256
    reference=np.column_stack((exact_prefix(fp[order]),exact_prefix(tp[order])))
    bound_check('trace/prefixes',prefixes,reference,10*levels)
    if totals[0]<=0 or totals[1]<=0:return
    expected,bound=integral(prefixes,p[order],totals[1]);counts['source_integrals']+=1
    if np.isfinite(expected):
        check('trace/source-integral',abs(actual-expected)<=bound,True,exact=True)
    else:
        check('trace/source-integral',actual,expected,exact=True)
        counts['source_nonfinite_integrals']+=1


def validate(report,progress=False):
    result=validate_inputs(report,progress)
    failures=[];fields=0
    counts=dict(exact_terms=0,bounded_fields=0,source_integrals=0,source_nonfinite_integrals=0,
                independent_pair_curves=0,independent_pr_integrals=0)
    if report['kind']==0:return dict(**result,checked_fields=0,reference_failures=[])
    kind=report['kind']
    for case in report['cases']:
        s=case['spec']
        for call,obs in enumerate(case['calls']):
            def check(name,actual,expected,atol=0,exact=False):
                nonlocal fields
                good=near(actual,expected,atol,exact);fields+=np.size(good)
                if not np.all(good):
                    failures.append(dict(case=s.id,call=call,field=name,count=int(np.count_nonzero(~good))))
            if obs['status']:continue
            value={k:obs[k].view('<f8') for k in FIELDS if k!='order'}
            if kind==2:
                a=value['final'] if len(value['final']) else value['curves']
                # Source capture and the original linked implementation must agree exactly.
                check('probe/linked',a,value['linked'],exact=True)
            if s.family==NORMALIZE:
                summaries=case['summaries'].view('<f8')
                check('normalization',value['final'][0],normalize(summaries),atol=2e-12)
                if kind==2:check('captured-normalization-input',value['metadata'],summaries,exact=True)
                continue
            if len(value['guards']):check('guards',value['guards'],[-971.]*4,exact=True)
            if len(value['metadata']):
                check('normalization-of-own-metadata',value['final'][0],normalize(value['metadata']),atol=2e-12)
            if s.family==WIDE:
                t=np.arange(s.targets)
                area=np.where(t%3==0,1.,np.where(t%3==2,.5,0. if roc(s,call) else 1.-np.log(2.)))
                triples=np.column_stack((np.ones(s.targets),np.ones(s.targets),area))
                observed=value['curves'] if kind==1 else value['source_curves']
                check('wide-curves',observed.reshape(-1,3),triples,atol=2e-12)
                check('wide-metadata',value['metadata'].reshape(-1,3),triples,atol=2e-12)
                if kind==2 and not roc(s,call):check('wide-original-totals',value['totals'],1.,exact=True)
                expected=serial(area)/s.targets
                check('wide-public-metric',value['final'],expected,atol=2e-12)
                if kind==1:check('wide-public-CPU',value['cpu'],expected,atol=2e-12)
                continue
            if s.family==EXTREME:
                check('extreme-cancellation',value['curves'],[1,1,1],exact=True)
                if kind==1:check('extreme-CPU',value['cpu'],[1],exact=True)
                continue
            for t in range(s.targets):
                p,y,w=column(case,t)
                r,pr=curve_reference(p,y,w,multi(s));expected=r if roc(s,call) else pr
                if kind==2 and not roc(s,call) and not trace(s,call):
                    order=np.argsort(-p,kind='stable')
                    totals=[serial(((np.float32(1)-y[order])*w[order]).astype(np.float64)),
                            serial((y[order]*w[order]).astype(np.float64))]
                    check(f'original-totals/{t}',value['totals'].reshape(-1,2)[t],totals)
                if kind==1:
                    curve=value['curves'].reshape(-1,3)[t] if len(value['curves']) else None
                elif multi(s):
                    # CUDA multi PR exposes normalization metadata; raw source
                    # endpoints are separately checked below, never relabeled.
                    curve=value['source_curves'].reshape(-1,3)[t] if roc(s,call) and s.rows else None
                else:curve=value['curves']
                if curve is not None:
                    check(f'curve/{t}/fp-tp',curve[:2],expected[:2],exact=s.family==EDGE)
                    if trace(s,call):pass  # Integral uses actual observed endpoints below.
                    elif s.family==MAIN and s.mode==4 and roc(s,call) and r[0]>0 and r[1]>0:
                        check(f'curve/{t}/normalized-ROC',curve[2]/(r[0]*r[1]),r[2]/(r[0]*r[1]),atol=2e-12)
                    else:check(f'curve/{t}/area',curve[2],expected[2],atol=0 if roc(s,call) else 2e-10,exact=s.family==EDGE)
                if len(value['source_curves']):
                    raw=value['source_curves'].reshape(-1,3)[t]
                    check(f'source/{t}/fp-tp',raw[:2],r[:2])
                if len(value['metadata']):
                    meta=value['metadata'].reshape(-1,3)[t]
                    ref=[r[0]*r[1] if roc(s,call) else float(bool(s.rows)),
                         (r[1] if roc(s,call) and s.multiclass else 1.) if s.rows else 0.,expected[2]]
                    check(f'metadata/{t}/area-weight',meta[:2],ref[:2])
                    if not trace(s,call):check(f'metadata/{t}/integral',meta[2],ref[2],atol=0 if roc(s,call) else 2e-10)
                    if kind==2 and s.rows:check(f'metadata/{t}/source-integral',meta[2],raw[2],exact=True)
                if trace(s,call):
                    actual=value['metadata'].reshape(-1,3)[t,2] if multi(s) else curve[2]
                    trace_checks(s,t,p,y,w,obs,kind,actual,check,counts)
                    if kind==1 and multi(s):check(f'raw/metadata/{t}',curve[2],actual,exact=True)
                # Retain the independent small-case pair and stable-log1p integral
                # checks instead of relying only on agreement between two GPUs.
                if s.family==MAIN and s.rows<=257 and s.mode<5 and roc(s,call):
                    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
                    pairs=0.
                    for i in positive:
                        pairs+=serial(w[i].astype(np.float64)*w[negative].astype(np.float64)*
                                      np.where(p[i]>p[negative],1.,np.where(p[i]==p[negative],.5,0.)))
                    observed=(curve[2] if curve is not None else 0.)
                    if s.mode==4 and r[0]>0 and r[1]>0:
                        check(f'independent-pair/{t}',observed/(r[0]*r[1]),pairs/(r[0]*r[1]),atol=2e-12)
                    else:check(f'independent-pair/{t}',observed,pairs)
                    counts['independent_pair_curves']+=1
                if s.family==MAIN and s.rows<=257 and s.mode!=4 and not roc(s,call) and pr[0]==1:
                    observed=curve[2] if curve is not None else value['metadata'].reshape(-1,3)[t,2]
                    check(f'independent-log1p/{t}',observed,stable_pr(p,y,w),atol=2e-10)
                    counts['independent_pr_integrals']+=1
        if progress and ((s.id+1)%48==0 or s.id+1==COUNT):
            print(f'Validated AUC observations {s.id+1}/{COUNT}; {fields} fields; {len(failures)} failures',file=sys.stderr,flush=True)
    return dict(**result,checked_fields=fields,**counts,reference_failures=failures)


def stable_pr(p,y,w):
    order=np.argsort(-p,kind='stable');p=p[order];y=y[order];w=w[order]
    total=serial((y*w).astype(np.float64));negative=serial(((np.float32(1)-y)*w).astype(np.float64))
    if total<=0 or negative<=0:return np.nan
    fp=tp=area=0.
    ends=np.r_[np.flatnonzero(p[:-1]!=p[1:])+1,len(p)];begin=0
    for end in ends:
        df=serial(((1.-y[begin:end].astype(np.float64))*w[begin:end].astype(np.float64)).astype('<f4'))
        dt=serial((y[begin:end]*w[begin:end]).astype(np.float64))
        if dt>0:
            a=1+df/dt
            area+=dt/(a*total) if fp+tp==0 else (dt-(fp-(a-1)*tp)/a*np.log1p(a*dt/(fp+tp)))/(a*total)
        fp+=df;tp+=dt;begin=end
    return area


def compare(metal,cuda):
    require(metal['kind']==1 and cuda['kind']==2,'Requires Metal and actual CUDA reports')
    match_inputs(metal,cuda)
    validation=[validate(metal),validate(cuda)];differences=[];fields=0;outside=0
    for a,b in zip(metal['cases'],cuda['cases']):
        s=a['spec']
        for call,(x,y) in enumerate(zip(a['calls'],b['calls'])):
            if y['status']:continue
            pairs=[(k,x[k],y[k]) for k in ('final','metadata') if len(x[k]) and len(y[k])]
            if not multi(s) and s.family!=NORMALIZE:pairs.append(('curves',x['curves'],y['curves']))
            elif roc(s,call) and len(x['curves']) and len(y['source_curves']):
                pairs.append(('raw-ROC',x['curves'],y['source_curves']))
            if len(x['terms']) and len(y['terms']):
                pairs.extend((k,x[k],y[k]) for k in ('terms','prefixes','totals'))
            for name,v,w in pairs:
                fields+=len(v);different=v!=w
                if np.any(different):
                    # This display budget does not certify parity. Both reference
                    # validations must pass; every bit difference requires review.
                    n=int(np.count_nonzero(~near(v.view('<f8'),w.view('<f8'),atol=0 if name=='terms' else 2e-12,exact=name=='terms')))
                    outside+=n;differences.append(dict(case=s.id,call=call,field=name,bit_differences=int(np.count_nonzero(different)),outside_display_budget=n))
    return dict(status='reference-failure' if any(v['reference_failures'] for v in validation) else 'review-required',
                metal_sha256=metal['sha256'],cuda_sha256=cuda['sha256'],validation=validation,
                common_fields=fields,bit_differences=differences,outside_display_budget=outside,
                scope='Empty binary ROC is a CUDA guard; malformed Metal triplets have no matching CUDA API. Original strided storage is packed for CUDA. Pathological PR integrals use each backend\'s own observed prefixes; no CUB reduction topology is assumed.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate-metal');v.add_argument('metal',type=Path)
    u=sub.add_parser('unpack');u.add_argument('input',type=Path);u.add_argument('output',type=Path)
    c=sub.add_parser('compare');c.add_argument('metal',type=Path);c.add_argument('cuda',type=Path);c.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='unpack':
        report=load(args.input)
        with args.output.open('xb') as stream:stream.write(unpack(report))
        return 0
    if args.command=='validate-metal':
        report=load(args.metal);require(report['kind']==1,'Expected Metal AUC observations')
        result=validate(report,True);print(json.dumps(result));return int(bool(result['reference_failures']))
    require(not args.output.exists(),'Comparison output already exists')
    result=compare(load(args.metal),load(args.cuda))
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result));return 1 if result['status']=='reference-failure' else 2


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,struct.error) as error:
        print(f'FAIL AUC report: {error}',file=sys.stderr);sys.exit(1)
