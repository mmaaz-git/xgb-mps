#!/usr/bin/env python3
"""Compare the complete integer histogram-cache lifecycle corpus.

CUDA integer storage and Metal materialized storage have different layouts.
Validate each independently; only decoded bins and converted values are common
GPU observations. No private Metal integer buffer is represented in this format.
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

MAGIC, COUNT = b'XGBHIC1\0', 48
Spec = namedtuple('Spec', 'id dense max_bin exponent bins')
INPUTS = ('data', 'cuts', 'first', 'second')
OUTPUTS = ('values', 'prefix_before', 'prefix_after', 'guards', 'integers')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def specs():
    out = []
    for dense in (0, 1):
        for max_bin in (16, 512, 65537):
            for exponent in (-80, 0, 54, 100):
                for bins in (1, 2):
                    out.append(Spec(len(out), dense, max_bin, exponent & 0xffffffff, bins))
    return out


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, 'Invalid integer cache header')
    kind, count = struct.unpack_from('<II', data, 8)
    require(kind in (0, 1, 2) and count == COUNT, 'Wrong integer cache kind/count')
    offset, cases = 16, []

    def array(n, width):
        nonlocal offset
        require(offset + n * width <= len(data), 'Truncated integer cache array')
        result = np.frombuffer(data, dtype=f'<u{width}', count=n, offset=offset)
        offset += n * width
        return result

    for s in specs():
        require(tuple(array(5, 4)) == s, 'Wrong integer cache identity/order/parameters')
        case = dict(spec=s, data=array(4, 4), cuts=array(s.bins, 4),
                    first=array(8, 4), second=array(8, 4))
        if kind:
            width = s.bins if kind == 2 else 1
            case['bins'] = array(4, 4)
            for key, n in zip(OUTPUTS, (18*s.bins+6, 32*width, 32*width,
                                       20*width, 18*s.bins+6 if kind == 2 else 0)):
                case[key] = array(n, 8)
        cases.append(case)
    require(len(data) == offset + 8 and data[offset:offset+4] == b'HICE' and
            struct.unpack_from('<I', data, offset+4)[0] == COUNT,
            'Missing integer cache completion marker or trailing bytes')
    return dict(kind=kind, cases=cases, data=data, sha256=hashlib.sha256(data).hexdigest())


def load(path):
    opener = lzma.open if path.suffix == '.xz' else gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        return decode(stream.read())


def fixture(s):
    """Reconstruct the original 48-case source generator, without GPU outputs."""
    exponent = s.exponent if s.exponent < 2**31 else s.exponent - 2**32
    big, small = np.ldexp(1., exponent), np.ldexp(1., exponent - 54)
    first = np.array([big, 0., small, 1., small, 1., -big, 0.], dtype='<f4')
    return dict(spec=s, data=np.array([0., 0., 0., float(s.bins == 2)], dtype='<f4').view('<u4'),
                cuts=np.array([.5, 2.] if s.bins == 2 else [2.], dtype='<f4').view('<u4'),
                first=first.view('<u4'), second=np.multiply(first, np.float32(2.)).view('<u4'))


def match_inputs(a, b):
    require(len(a['cases']) == len(b['cases']) == COUNT, 'Incomplete integer cache corpus')
    for x, y in zip(a['cases'], b['cases']):
        require(x['spec'] == y['spec'] and all(np.array_equal(x[k], y[k]) for k in INPUTS),
                f"Integer cache case {x['spec'].id}: input bits differ")


def references(case):
    s = case['spec']
    original = fixture(s)
    require(all(np.array_equal(case[k], original[k]) for k in INPUTS),
            f'Integer cache case {s.id}: input does not preserve the original corpus')
    # Numeric SearchBin uses upper_bound and clips values at/above the final cut.
    bins = np.minimum(np.searchsorted(case['cuts'].view('<f4'),
                                     case['data'].view('<f4'), side='right'), s.bins-1).astype('<u4')
    values, integers = [], []

    def quantize(key):
        raw = case[key].view('<f4').reshape(4, 2).astype('<f8')
        positive = np.add.accumulate(np.where(raw >= 0., raw, 0.), axis=0)[-1]
        negative = np.add.accumulate(np.where(raw < 0., -raw, 0.), axis=0)[-1]
        bound = np.maximum(positive, negative)/(1. - 8.*np.finfo(np.float64).eps)
        scale = 62 - np.frexp(bound)[1]
        fixed = np.ldexp(raw, scale)
        require(np.all(np.isfinite(fixed) & (fixed >= -2.**63) & (fixed < 2.**63)),
                'Cache quantizer conversion outside int64 domain')
        q = fixed.astype('<i8').view('<u8')
        q[:, 1] = np.where((raw[:, 1] > 0) & (q[:, 1] == 0), np.uint64(1), q[:, 1])
        return q, scale

    def histogram(rows, q):
        out = np.zeros((s.bins, 2), dtype='<u8')
        for row in rows:
            out[bins[row]] += q[row]
        return out

    def converted(q, scale):
        return np.ldexp(q.view('<i8').astype('<f8'), -scale).view('<u8')

    def observe(q, scale):
        integers.extend(q.ravel().tolist())
        values.extend(converted(q, scale).ravel().tolist())

    def root(q, scale):
        # Sum fixed-point bins before conversion; summing rounded doubles loses
        # the two small terms in the two-bin cancellation witness.
        observe(q.sum(axis=0, dtype='<u8'), scale)

    q, scale = quantize('first')
    parent, sibling = histogram((0, 1, 2), q), histogram((0,), q)
    observe(parent, scale)
    observe(sibling, scale)
    metal_prefix = np.tile(np.array([17.5, -23.25], dtype='<f8').view('<u8'), (16, 1))
    metal_prefix[3:3+s.bins] = converted(parent, scale)
    metal_prefix[8:8+s.bins] = converted(sibling, scale)
    cuda_prefix = np.tile(np.array([17, -23], dtype='<i8').view('<u8'), (16, s.bins, 1))
    cuda_prefix[3], cuda_prefix[8] = parent, sibling
    child = parent - sibling
    observe(child, scale)
    root(child, scale)
    parent += histogram((3,), q)
    observe(parent, scale)
    observe(parent, scale)  # Empty subsequent page preserves the slot.
    root(parent, scale)
    parent = histogram((), q)
    observe(parent, scale)  # Empty first page clears the reused slot.
    parent += histogram((1,), q)
    observe(parent, scale)
    q, scale = quantize('second')  # A new tree must use its new quantizer.
    parent = histogram((0, 1, 2), q)
    observe(parent, scale)
    parent += histogram((3,), q)
    observe(parent, scale)
    root(parent, scale)
    common = dict(bins=bins, values=np.array(values, dtype='<u8'))
    require(len(values) == len(integers) == 18*s.bins+6, 'Incomplete cache lifecycle reference')
    return {
        'metal': dict(common, prefix_before=metal_prefix.ravel(), prefix_after=metal_prefix.ravel(),
                      guards=np.tile(np.array([17.5, -23.25], dtype='<f8').view('<u8'), 10),
                      integers=np.array([], dtype='<u8')),
        'cuda': dict(common, prefix_before=cuda_prefix.ravel(), prefix_after=cuda_prefix.ravel(),
                     guards=np.tile(np.array([17, -23], dtype='<i8').view('<u8'), 10*s.bins),
                     integers=np.array(integers, dtype='<u8'))}


def failures(case, expected, backend):
    out = []
    for key in ('bins', *OUTPUTS):
        a, b = case[key], expected[key]
        require(a.shape == b.shape, f'Wrong {backend} cache {key} shape')
        good = a.view('<f8') == b.view('<f8') if key == 'values' else a == b
        bad = np.flatnonzero(~good)
        if len(bad):
            out.append(dict(case=case['spec'].id, backend=backend, field=key,
                            policy='original-numeric-double' if key == 'values' else 'exact',
                            count=len(bad), first_indices=bad[:32].tolist()))
    return out


def validate_metal(report):
    require(report['kind'] == 1, 'Need Metal integer cache observations')
    return [failure for case in report['cases']
            for failure in failures(case, references(case)['metal'], 'metal')]


def compare(metal, cuda):
    require(metal['kind'] == 1 and cuda['kind'] == 2, 'Need Metal and actual CUDA observation kinds')
    match_inputs(metal, cuda)
    errors, differences = [], []
    fields = different = integer_fields = metal_storage = cuda_storage = 0
    for m, c in zip(metal['cases'], cuda['cases']):
        expected = references(m)
        errors.extend(failures(m, expected['metal'], 'metal'))
        errors.extend(failures(c, expected['cuda'], 'cuda'))
        integer_fields += len(c['integers'])
        metal_storage += sum(len(m[k]) for k in ('prefix_before', 'prefix_after', 'guards'))
        cuda_storage += sum(len(c[k]) for k in ('prefix_before', 'prefix_after', 'guards'))
        for key in ('bins', 'values'):
            indices = np.flatnonzero(m[key] != c[key])
            fields += len(m[key])
            different += len(indices)
            if len(indices):
                differences.append(dict(case=m['spec'].id, field=key, count=len(indices),
                    first_fields=[dict(index=int(i), metal=int(m[key][i]), cuda=int(c[key][i]))
                                  for i in indices[:32]]))
    return dict(suite='integer-cache-v1', cases=COUNT, compared_actual_fields=fields,
                cuda_integer_reference_checks=integer_fields,
                metal_storage_checks=metal_storage, cuda_storage_checks=cuda_storage,
                different_actual_fields=different, differences=differences, reference_failures=errors,
                status='reference-failure' if errors else 'review-differences' if different else 'identical')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'unpack', 'validate-metal'):
        command = sub.add_parser(name)
        command.add_argument('input', type=Path)
        if name == 'unpack':
            command.add_argument('output', type=Path)
    command = sub.add_parser('compare')
    command.add_argument('metal', type=Path)
    command.add_argument('cuda', type=Path)
    command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        require(not args.output.exists(), 'Output already exists')
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'differences'}))
        return 1 if result['reference_failures'] else 2 if result['different_actual_fields'] else 0
    report = load(args.input)
    if args.command == 'unpack':
        require(report['kind'] == 0, 'Only input-only integer cache records may be unpacked for CUDA')
        with args.output.open('xb') as stream:
            stream.write(report['data'])
    result = dict(kind=report['kind'], cases=COUNT, raw_sha256=report['sha256'])
    if args.command == 'validate-metal':
        result['reference_failures'] = validate_metal(report)
        result['status'] = 'reference-failure' if result['reference_failures'] else 'complete; no CUDA equivalence claim'
    print(json.dumps(result))
    return 1 if result.get('reference_failures') else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f'FAIL integer cache report: {error}', file=sys.stderr)
        sys.exit(1)
