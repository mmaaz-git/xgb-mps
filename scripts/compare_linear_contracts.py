#!/usr/bin/env python3
"""Validate full linear operation replays using the original FP32 contracts.

Actions have explicit input states. CUDA duplicate-column residual writes and
unsafe rejection calls are exclusions, not successful CUDA observations.
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

MAGIC, COUNT, STEPS = b'XGBLIN1\0', 56, 893
Spec = namedtuple('Spec', 'id family rows groups mode features')
Action = namedtuple('Action', 'feature gradient update copy delta domain')
INPUTS = ('offsets', 'rows', 'entries', 'original')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def specs():
    out = [Spec(0,1,1,1,0,1), Spec(1,2,1,1,0,1), Spec(2,3,2,1,0,0), Spec(3,4,1,1,0,1)]
    for rows in (0,1,31,257,4097,65539):
        for groups in (1,3):
            for mode in range(4):
                out.append(Spec(len(out),0,rows,groups,mode,5))
    out.append(Spec(len(out),0,1000003,3,0,5))
    for family in (5,6,7):
        out.append(Spec(len(out),family,1,1,0,1))
    return out


def actions(s):
    if s.family == 0:
        return [Action(f,1,1,1,bits(delta),int(f == 2 and s.rows > 0))
                for delta in (0.,.125,-.5) for f in range(-1,5)]
    return {
        1: [Action(-1,1,0,0,0,0), Action(0,1,0,0,0,0),
            Action(-1,0,1,1,bits(.5),0), Action(0,0,1,1,bits(.5),0)],
        2: [Action(0,0,1,1,0x3fd4c3f0,0)],
        3: [Action(-1,1,0,0,0,0)],
        4: [Action(-1,0,1,1,0,0), Action(0,0,1,1,0,0)],
        5: [Action(-2,1,0,0,0,2)],
        6: [Action(1,1,0,0,0,2)],
        7: [Action(0,0,0,0,0,2)]}[s.family]


def entries(s):
    return 2*s.rows+(s.rows+1)//2+(s.rows+6)//7+(s.rows+96)//97 if s.family == 0 else s.features


def status(kind, a):
    return 0 if kind == 0 else (1 if kind == 1 else 2) if a.domain == 2 else 3 if kind == 2 and a.domain == 1 else 0


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, 'Invalid linear header')
    kind, count = struct.unpack_from('<II', data, 8)
    require(kind in (0,1,2) and count == COUNT, 'Wrong linear kind/count')
    offset, cases, total = 16, [], 0

    def array(n, width):
        nonlocal offset
        require(offset+n*width <= len(data), 'Truncated linear array')
        result = np.frombuffer(data, dtype=f'<u{width}', count=n, offset=offset)
        offset += n*width
        return result

    for s in specs():
        require(tuple(array(6,4)) == s, 'Wrong linear identity/order')
        n = 2*s.rows*s.groups
        case = dict(spec=s, offsets=array(s.features+1,8), rows=array(entries(s),4),
                    entries=array(entries(s),4), original=array(n+(4 if s.family == 0 else 0),4), steps=[])
        for i, a in enumerate(actions(s)):
            identity = (i,a.feature & 0xffffffff,a.gradient,a.update,a.copy,a.delta,a.domain)
            require(tuple(array(7,4)) == identity, 'Wrong linear action/parameters/domain')
            step = dict(action=a, state=array(n,4))
            if kind:
                observed_status = int(array(1,4)[0])
                require(observed_status == status(kind,a), 'Wrong linear rejection/exclusion status')
                grad = 2*s.groups if a.gradient and a.domain != 2 else 0
                residual = n+(4 if s.family == 0 else 0) if a.copy and not (kind == 2 and a.domain) else 0
                step.update(status=observed_status, gradients=array(grad,4), residuals=array(residual,4))
            case['steps'].append(step)
            total += 1
        cases.append(case)
    require(total == STEPS and len(data) == offset+12 and data[offset:offset+4] == b'LINE' and
            struct.unpack_from('<II',data,offset+4) == (COUNT,STEPS),
            'Missing linear completion marker or trailing bytes')
    return dict(kind=kind,cases=cases,data=data,sha256=hashlib.sha256(data).hexdigest())


def load(path):
    if path.suffix in ('.xz','.gz'):
        opener = lzma.open if path.suffix == '.xz' else gzip.open
        with opener(path,'rb') as stream:
            return decode(stream.read())
    with path.open('rb') as stream:
        require(path.stat().st_size > 0, 'Empty linear file')
        return decode(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))


def match_inputs(a,b):
    require(len(a['cases']) == len(b['cases']) == COUNT, 'Incomplete linear corpus')
    for x,y in zip(a['cases'],b['cases']):
        require(x['spec'] == y['spec'] and all(np.array_equal(x[k],y[k]) for k in INPUTS),
                f"Linear case {x['spec'].id}: source input bits differ")
        require(len(x['steps']) == len(y['steps']) == len(actions(x['spec'])), 'Incomplete linear actions')
        for i,(u,v) in enumerate(zip(x['steps'],y['steps'])):
            require(u['action'] == v['action'] and np.array_equal(u['state'],v['state']),
                    f"Linear case {x['spec'].id} action {i}: input state bits differ")


def source_inputs(s):
    """Reconstruct the unchanged source generator, including legacy MT19937."""
    if s.family == 0:
        r = np.arange(s.rows,dtype=np.int64)
        offsets, ids, values = [0], [], []
        for f in range(5):
            selected = r[:0] if f == 0 else r[r%2 == 0] if f == 1 else r[r%97 == 0] if f == 3 else r
            x = (((selected*13+f*3)%63)-31).astype('<f4')/np.float32(8.)
            if f == 2:
                indices = np.empty(s.rows+(s.rows+6)//7,dtype='<u4')
                column = np.empty(len(indices),dtype='<f4')
                positions = r+(r+6)//7
                indices[positions], column[positions] = r,x
                duplicate = r[r%7 == 0]
                positions = duplicate+(duplicate+6)//7+1
                indices[positions], column[positions] = duplicate,np.float32(-.5)
            else:
                indices, column = selected.astype('<u4'),x
            ids.extend(indices.tolist())
            values.extend(column.tolist())
            offsets.append(len(ids))
        if s.mode == 3:
            special = np.array([0.,-0.,1e-40,-1e-40,np.inf,-np.inf,np.nan],dtype='<f4')
            index = r[:,None]+np.arange(s.groups)[None,:]
            pairs = np.stack((special[index%7],special[(index+2)%7]),axis=-1)
        else:
            raw = np.random.RandomState(137).randint(0,2**32,size=(s.rows,s.groups,2),dtype=np.uint32)
            pairs = np.empty(raw.shape,dtype='<f4')
            pairs[...,0] = ((raw[...,0]%1023).astype(np.int32)-511).astype('<f4')/np.float32(16.)
            pairs[...,1] = (raw[...,1]%127).astype('<f4')/np.float32(8.)
            pairs[r%17 == 0,:,1] = -1.
            if s.mode in (1,2):
                pairs = np.ldexp(pairs,-140 if s.mode == 1 else 100)
        sentinel = np.array([-917.,918.],dtype='<f4')
        original = np.concatenate((sentinel,pairs.ravel(),sentinel)).view('<u4')
        return dict(offsets=np.asarray(offsets,dtype='<u8'),rows=np.asarray(ids,dtype='<u4'),
                    entries=np.asarray(values,dtype='<f4').view('<u4'),original=original)
    rows = np.array([1 if s.family == 7 else 0],dtype='<u4') if s.features else np.array([],dtype='<u4')
    column = np.array([3. if s.family == 1 else np.inf if s.family == 4 else 1.],dtype='<f4').view('<u4') if s.features else np.array([],dtype='<u4')
    if s.family == 2:
        column[0] = 0x3f807d7e
        original = np.array([0,0x401758e3],dtype='<u4')
    else:
        original = np.array([7.,-2.] if s.family == 1 else [np.finfo(np.float32).max,1.]*2 if s.family == 3 else
                            [3.,-0.] if s.family == 4 else [1.,1.],dtype='<f4').view('<u4')
    return dict(offsets=np.array([0,1] if s.features else [0],dtype='<u8'),rows=rows,entries=column,original=original)


def same_float_bits(a,b):
    return (a == b) | (np.isnan(a.view('<f4')) & np.isnan(b.view('<f4')))


def discrepancy(case,step,backend,field,good,policy):
    bad = np.flatnonzero(~good)
    return [] if not len(bad) else [dict(case=case,step=step,backend=backend,field=field,policy=policy,
                                       count=len(bad),first_indices=bad[:32].tolist())]


def terms(case,state,feature):
    if feature < 0:
        return state
    begin,end = map(int,case['offsets'][feature:feature+2])
    ids = case['rows'][begin:end]
    x = case['entries'][begin:end].view('<f4')[:,None]
    result = state[ids].copy()
    result[...,0] = np.multiply(result[...,0],x,dtype='<f4')
    result[...,1] = np.multiply(np.multiply(result[...,1],x,dtype='<f4'),x,dtype='<f4')
    return result


def sum_checks(actual,source):
    """Original sequential binary64 term reference and fixed FP32 error bound."""
    count = len(source)
    shape = source.shape[1:]
    if not count:
        return actual == 0.,np.ones(shape,dtype=bool),0.
    wide = source.astype('<f8')
    value = np.add.accumulate(wide,axis=0)[-1]
    magnitude = np.add.accumulate(np.abs(wide),axis=0)[-1]
    word = source.view('<u4') & 0x7fffffff
    exponent = (word >> 23).astype(np.int32)
    mantissa = (word & 0x7fffff) | np.where(exponent != 0,0x800000,0).astype('<u4')
    lowest = mantissa & (~mantissa+np.uint32(1))
    trailing = np.frexp(np.where(lowest != 0,lowest,1).astype('<f8'))[1]-1
    quantum = np.where((word > 0) & (word < 0x7f800000),
                       np.where(exponent != 0,exponent-150,-149)+trailing,10000).min(axis=0)
    exact = (magnitude == 0.) | (np.ldexp(magnitude,-quantum) <= 2.**24)
    depth = 0
    while True:
        depth += 8
        count = (count+255)//256
        if count <= 1:
            break
    gamma = depth*2.**-24/(1.-depth*2.**-24)
    bound, error = gamma*magnitude,np.abs(actual.astype('<f8')-value)
    good = np.where(exact,actual == value,error <= bound)
    finite = np.isfinite(value)
    good = np.where(finite,good,np.where(np.isnan(value),np.isnan(actual),actual == value))
    fractions = np.divide(error,bound,out=np.zeros_like(error),where=finite & ~exact & (bound > 0))
    return good,exact | ~finite,float(np.max(fractions,initial=0.))


def update(case,state,a):
    if not a.update or a.domain == 2 or (a.feature < 0 and a.delta & 0x7fffffff == 0):
        return
    delta = np.array(a.delta,dtype='<u4').view('<f4')

    def apply(ids,x=None):
        selected = state[ids].copy()
        product = np.multiply(selected[...,1],delta,dtype='<f4')
        if x is not None:
            product = np.multiply(product,x[:,None],dtype='<f4')
        selected[...,0] = np.add(selected[...,0],product,dtype='<f4')
        selected[...,1] = np.add(selected[...,1],np.float32(0.),dtype='<f4')
        state[ids] = selected

    if a.feature < 0:
        apply(np.arange(len(state)))
    else:
        begin,end = map(int,case['offsets'][a.feature:a.feature+2])
        ids = case['rows'][begin:end]
        x = case['entries'][begin:end].view('<f4')
        # Each original duplicate run has two adjacent entries. Reorder across
        # independent rows only, preserving first-then-second updates within a row.
        first = np.r_[True,ids[1:] != ids[:-1]] if len(ids) else np.array([],dtype=bool)
        apply(ids[first],x[first])
        apply(ids[~first],x[~first])


def validate_case(case,backend,peer=None):
    s = case['spec']
    canonical = source_inputs(s)
    require(all(np.array_equal(case[k],canonical[k]) for k in INPUTS),
            f'Linear case {s.id}: source generator bits differ')
    original = case['original'][2:-2] if s.family == 0 else case['original']
    state = original.view('<f4').reshape(s.rows,s.groups,2).copy()
    failures, differences = [],[]
    fields = different = exact_sums = sums = 0
    max_fraction = 0.
    for i,step in enumerate(case['steps']):
        a = step['action']
        failures.extend(discrepancy(s.id,i,'input','state',same_float_bits(step['state'],state.ravel().view('<u4')),
                                    'original finite bits / NaN classification'))
        if a.domain != 2 and a.gradient:
            source = terms(case,state,a.feature)
            for name,out in ((backend,step),('cuda',peer['steps'][i])) if peer is not None else ((backend,step),):
                actual = out['gradients'].view('<f4').reshape(s.groups,2)
                if s.family == 3:
                    good = (actual == np.array([[np.inf,2.]],dtype='<f4'))
                    exact, fraction = np.ones(good.shape,dtype=bool),0.
                else:
                    good,exact,fraction = sum_checks(actual,source)
                failures.extend(discrepancy(s.id,i,name,'gradients',good.ravel(),'original FP32 exact/nonfinite/bound'))
                if name == backend:
                    sums += actual.size
                    exact_sums += int(exact.sum())
                    max_fraction = max(max_fraction,fraction)
        update(case,state,a)
        if a.copy:
            expected = state.ravel().view('<u4')
            if s.family == 0:
                guard = np.array([-999.,998.],dtype='<f4').view('<u4')
                expected = np.concatenate((guard,expected,guard))
            for name,out in ((backend,step),('cuda',peer['steps'][i])) if peer is not None else ((backend,step),):
                if name == 'cuda' and a.domain:
                    require(not len(out['residuals']),'Excluded CUDA residual contains fabricated outputs')
                    continue
                failures.extend(discrepancy(s.id,i,name,'residuals',same_float_bits(out['residuals'],expected),
                                            'original finite bits / NaN classification; exact guards'))
        if peer is not None:
            c = peer['steps'][i]
            for key in ('gradients','residuals'):
                if a.domain == 2 or (key == 'residuals' and a.domain == 1):
                    continue
                indices = np.flatnonzero(step[key] != c[key])
                fields += len(step[key])
                different += len(indices)
                if len(indices):
                    differences.append(dict(case=s.id,step=i,field=key,count=len(indices),
                        first_fields=[dict(index=int(j),metal=int(step[key][j]),cuda=int(c[key][j])) for j in indices[:32]]))
    return dict(reference_failures=failures,differences=differences,compared_actual_fields=fields,
                different_actual_fields=different,sum_fields=sums,exact_or_nonfinite_sum_fields=exact_sums,
                max_forward_error_fraction=max_fraction)


def validate(report,peer=None,progress=False):
    require(report['kind'] == 1, 'Need Metal linear observations')
    if peer is not None:
        require(peer['kind'] == 2,'Need actual CUDA linear observations')
        match_inputs(report,peer)
    result = dict(suite='linear-v1',cases=COUNT,steps=STEPS,reference_failures=[],differences=[],
                  compared_actual_fields=0,different_actual_fields=0,sum_fields=0,
                  exact_or_nonfinite_sum_fields=0,max_forward_error_fraction=0.,domain_exclusions=[])
    with np.errstate(all='ignore'):
        for index,case in enumerate(report['cases']):
            r = validate_case(case,'metal',None if peer is None else peer['cases'][index])
            for key in ('reference_failures','differences'):
                result[key].extend(r[key])
            for key in ('compared_actual_fields','different_actual_fields','sum_fields','exact_or_nonfinite_sum_fields'):
                result[key] += r[key]
            result['max_forward_error_fraction'] = max(result['max_forward_error_fraction'],r['max_forward_error_fraction'])
            result['domain_exclusions'].extend(dict(case=case['spec'].id,step=i,domain=a.domain)
                                               for i,a in enumerate(actions(case['spec'])) if a.domain)
            if progress and ((index+1)%12 == 0 or index+1 == COUNT):
                print(f'Validated linear {index+1}/{COUNT}; sum fields={result["sum_fields"]}; failures={len(result["reference_failures"])}',flush=True)
    result['status'] = 'reference-failure' if result['reference_failures'] else 'complete; no CUDA equivalence claim' if peer is None else 'review-differences-and-domains'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('inspect','unpack','validate-metal'):
        command = sub.add_parser(name)
        command.add_argument('input',type=Path)
        if name == 'unpack':
            command.add_argument('output',type=Path)
    command = sub.add_parser('compare')
    command.add_argument('metal',type=Path)
    command.add_argument('cuda',type=Path)
    command.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        require(not args.output.exists(),'Output already exists')
        result = validate(load(args.metal),load(args.cuda),True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'differences'}))
        return 1 if result['reference_failures'] else 2
    report = load(args.input)
    if args.command == 'unpack':
        require(report['kind'] == 0,'Only input-only linear records may be unpacked for CUDA')
        with args.output.open('xb') as stream:
            stream.write(report['data'])
    result = dict(kind=report['kind'],cases=COUNT,steps=STEPS,raw_sha256=report['sha256'])
    if args.command == 'validate-metal':
        result.update(validate(report,progress=True))
    print(json.dumps(result))
    return 1 if result.get('reference_failures') else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError,EOFError,struct.error) as error:
        print(f'FAIL linear report: {error}',file=sys.stderr)
        sys.exit(1)
