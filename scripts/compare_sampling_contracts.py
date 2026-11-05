#!/usr/bin/env python3
"""Compare the complete current CUDA-derived sampler corpus without widening checks.

Metal reports retain the original independent source-oracle outputs; neither
backend receives them as inputs. Undefined conversions are explicit exclusions,
never fabricated CUDA rejections or successful GPU executions.
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

MAGIC, COUNT = b"XGBSMP1\0", 199
Spec = namedtuple("Spec", "id family rows targets layout mode fraction seed threads sequence step domain")
Shape = namedtuple("Shape", "stride0 stride1 offset extent storage replay_targets replay_stride replay_storage")
EXCLUDED = {114,117,126,129,132,135,138,198}


def require(ok, message):
    if not ok: raise ValueError(message)


def bits(value): return struct.unpack('<I', struct.pack('<f', value))[0]
def fraction(s): return struct.unpack('<f', struct.pack('<I', s.fraction))[0]


def specs():
    result, accepted = [], 0
    def add(n,t,layout,mode,p,seed):
        nonlocal accepted
        index=len(result); domain=int(index in EXCLUDED)
        result.append(Spec(index,0,n,t,layout,mode,bits(p),seed,0,0,0,domain)); accepted+=not domain
    for n in (0,1,2,255,256,257,1027):
        for t in (1,3):
            for mode in range(8): add(n,t,accepted%4,mode,(0,.001,.37,.999,1)[accepted%5],accepted*7919)
    for t in (16,17,31,32,33,511,512,513,1027):
        for mode in (0,3,4): add(33,t,mode%3,mode,.6,793)
    for n in (65537,1000003):
        for t in (1,3): add(n,t,0 if t==1 else 1,0,.4,9939)
    require(len(result)==143 and accepted==136,"Incomplete primitive sampling matrix")
    sequence=0
    for n in (0,1,257):
        for p in (0,.6,1):
            for threads in (1,3):
                for step in range(3): result.append(Spec(len(result),1,n,1,0,0,bits(p),701,threads,sequence,step,0))
                sequence+=1
    for p in (0,.5): result.append(Spec(len(result),2,4,1,0,0,bits(p),1,0,0,0,int(p>0)))
    require(len(result)==COUNT,"Incomplete sampling corpus")
    return result


def layout(s):
    if s.family==2: return Shape(1,1,0,4,4,0,0,0)
    if s.family==1: return Shape(1,1,0,s.rows,s.rows,3,3,3*s.rows)
    a=1 if s.layout==1 else 2*s.targets+1 if s.layout==2 else s.targets+2
    if s.layout==3: a=0
    b=s.rows+3 if s.layout==1 else 2 if s.layout==2 else 1
    extent=(s.rows-1)*a+(s.targets-1)*b+1 if s.rows else 0
    rt=3 if s.targets==1 else 1
    return Shape(a,b,4,extent,extent+8,rt,rt+2,0 if s.domain else s.rows*(rt+2)+8)


def sections(s):
    if s.domain or s.family==2: return []
    shape=layout(s); result=[]
    if s.family==0 and fraction(s)<1: result += [('norms','float-nan',s.rows),('threshold','float-nan',1)]
    result += [('roots_before','double-numeric',2*s.targets),('integers_before','integer',2*s.targets)]
    if s.family==1: result.append(('sampled','float-nan',2*shape.storage))
    result.append(('replay','float-exact' if s.family==0 else 'float-nan',2*shape.replay_storage))
    if s.family==0: result += [('roots_after','double-numeric',2*s.targets),('integers_after','integer',2*s.targets)]
    else: result.append(('next_rng','rng',1))
    return result


def split(s, words):
    result={}; offset=0
    for name,policy,count in sections(s): result[name]=words[offset:offset+count]; offset+=count
    require(len(words)==offset,"Wrong sampling output shape")
    return result


def decode(data):
    require(len(data)>=16 and data[:8]==MAGIC,"Invalid sampling header")
    kind,count=struct.unpack_from('<II',data,8)
    require(kind in (0,1,2) and count==COUNT,"Wrong sampling backend/count")
    offset,cases=16,[]
    def array(n,width):
        nonlocal offset
        require(offset+n*width<=len(data),"Truncated sampling array")
        result=np.frombuffer(data,dtype=f'<u{width}',count=n,offset=offset);offset+=n*width;return result
    for s in specs():
        require(offset+48<=len(data),"Truncated sampling identity")
        identity=struct.unpack_from('<12I',data,offset);offset+=48
        require(identity==s,"Wrong sampling identity/order/parameters/domain")
        shape=layout(s); case={'spec':s,'storage':array(2*shape.storage,4),'replay_input':array(2*shape.replay_storage,4)}
        fields=sum(n for _,_,n in sections(s))
        if kind:
            status=int(array(1,4)[0]); expected=(1 if kind==1 else 2) if s.domain else 0
            require(status==expected,"Wrong sampling rejection/exclusion status")
            case.update(status=status,actual=array(fields,8))
        if kind==1: case['reference']=array(fields,8)
        for key in ('actual','reference'):
            if key not in case: continue
            parts=split(s,case[key])
            for name,policy,_ in sections(s):
                if policy.startswith('float') or policy=='rng':
                    require(np.all(parts[name]<=0xffffffff),"Nonzero upper sampling FP32/RNG bits")
        cases.append(case)
    require(len(data)==offset+8 and data[offset:offset+4]==b'SMPE' and struct.unpack_from('<I',data,offset+4)[0]==COUNT,
            "Missing sampling completion marker or trailing bytes")
    return {'kind':kind,'cases':cases,'data':data,'sha256':hashlib.sha256(data).hexdigest()}


def load(path):
    opener=lzma.open if path.suffix=='.xz' else gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as stream: return decode(stream.read())


def match_inputs(a,b):
    require(len(a['cases'])==len(b['cases'])==COUNT,"Incomplete sampling corpus")
    for x,y in zip(a['cases'],b['cases']):
        require(x['spec']==y['spec'] and all(np.array_equal(x[k],y[k]) for k in ('storage','replay_input')),
                f"Sampling case {x['spec'].id}: input bits or parameters differ")


def guards(s):
    shape=layout(s); mask=np.ones(2*shape.replay_storage,dtype=bool)
    offset=4 if s.family==0 else 0
    for target in range(shape.replay_targets):
        for channel in (0,1):
            start=2*(offset+target)+channel
            mask[start:start+2*s.rows*shape.replay_stride:2*shape.replay_stride]=False
    return mask


def input_checks(case,words,backend):
    s=case['spec'];parts=split(s,words);errors=[]
    if s.domain or s.family==2:return errors
    def check(name,valid,policy):
        count=int(np.count_nonzero(~valid))
        if count: errors.append({'case':s.id,'backend':backend,'field':name,'policy':policy,'count':count})
    mask=guards(s)
    check('replay',parts['replay'][mask]==case['replay_input'][mask],'untouched-replay-padding')
    if fraction(s)==1:
        check('replay',parts['replay']==case['replay_input'],'unsampled-input-identity')
        if s.family==1: check('sampled',parts['sampled']==case['storage'],'unsampled-input-identity')
    budget=int(np.float32(np.float32(s.rows)*np.float32(fraction(s))))
    if not s.rows or (fraction(s)<1 and budget==0):
        for name,part in parts.items():
            if name.startswith('roots_'):check(name,part.view('<f8')==0,'empty-or-zero-budget-root')
            if name.startswith('integers_'):check(name,part==0,'empty-or-zero-budget-integer')
        if fraction(s)<1:
            check('replay',parts['replay'][~mask]==0,'zero-budget-replay')
            if s.family==1:check('sampled',parts['sampled']==0,'zero-budget-sampled')
            if 'threshold' in parts:check('threshold',parts['threshold']==bits(float('inf')),'zero-budget-threshold')
    return errors


def validate(case,reference,backend):
    s=case['spec']; actual=split(s,case['actual']);ref=split(s,reference);failures=[]
    for name,policy,_ in sections(s):
        a,r=actual[name],ref[name]
        if policy=='double-numeric':good=a.view('<f8')==r.view('<f8')
        elif policy=='float-nan':
            good=(a==r)|(np.isnan(a.astype('<u4').view('<f4'))&np.isnan(r.astype('<u4').view('<f4')))
        else:good=a==r
        bad=np.flatnonzero(~good)
        if len(bad): failures.append({'case':s.id,'backend':backend,'field':name,'policy':'original-'+policy,
                                     'count':len(bad),'first_indices':bad[:32].tolist()})
    failures.extend(input_checks(case,case['actual'],backend))
    return failures


def validate_metal(metal,progress=False):
    require(metal['kind']==1,"Need Metal observations")
    failures=[]
    for case in metal['cases']:
        failures.extend(input_checks(case,case['reference'],'reference'))
        failures.extend(validate(case,case['reference'],'metal'))
        if progress and ((case['spec'].id+1)%20==0 or case['spec'].id+1==COUNT):
            print(f"Validated Metal sampling {case['spec'].id+1}/{COUNT}",flush=True)
    return failures


def compare(metal,cuda):
    require(metal['kind']==1 and cuda['kind']==2,"Need Metal and actual CUDA observation kinds")
    match_inputs(metal,cuda);failures=[];differences=[];fields=different=0
    for m,c in zip(metal['cases'],cuda['cases']):
        failures.extend(input_checks(m,m['reference'],'reference'))
        failures.extend(validate(m,m['reference'],'metal'));failures.extend(validate(c,m['reference'],'cuda'))
        a,b=split(m['spec'],m['actual']),split(c['spec'],c['actual'])
        for name,_,_ in sections(m['spec']):
            indices=np.flatnonzero(a[name]!=b[name]);fields+=len(a[name]);different+=len(indices)
            if len(indices):differences.append({'case':m['spec'].id,'field':name,'count':len(indices),
                'first_fields':[{'index':int(i),'metal':int(a[name][i]),'cuda':int(b[name][i])} for i in indices[:32]]})
    return {'suite':'sampling-v1','cases':COUNT,
        'cuda_executed_cases':sum(not c['spec'].domain for c in cuda['cases']),
        'domain_exclusions':[{'case':c['spec'].id,'reason':'nonfinite-norm conversion' if c['spec'].family==2 else 'out-of-int64 rescaling'}
                             for c in cuda['cases'] if c['spec'].domain],
        'compared_fields':fields,'different_fields':different,'differences':differences,'reference_failures':failures,
        'status':'reference-failure' if failures else 'review-differences-and-domains' if different else 'common-domain-identical-with-exclusions'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('inspect','unpack','validate-metal'):
        c=sub.add_parser(name);c.add_argument('input',type=Path)
        if name=='unpack':c.add_argument('output',type=Path)
    c=sub.add_parser('compare');c.add_argument('metal',type=Path);c.add_argument('cuda',type=Path);c.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.command=='compare':
        require(not args.output.exists(),'Output already exists');result=compare(load(args.metal),load(args.cuda))
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='differences'}));return 1 if result['reference_failures'] else 2
    report=load(args.input)
    if args.command=='unpack':
        require(report['kind']==0,'Only input-only sampling records can be unpacked for CUDA')
        with args.output.open('xb') as stream:stream.write(report['data'])
    result={'kind':report['kind'],'cases':COUNT,'raw_sha256':report['sha256']}
    if args.command=='validate-metal':
        result['reference_failures']=validate_metal(report,True);result['status']='reference-failure' if result['reference_failures'] else 'complete; no CUDA equivalence claim'
    print(json.dumps(result));return 1 if result.get('reference_failures') else 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,EOFError,struct.error) as error:
        print(f'FAIL sampling report: {error}',file=sys.stderr);sys.exit(1)
