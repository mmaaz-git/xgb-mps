"""Synthetic report corruption and analytical witnesses; these are not GPU runs."""
import gzip
import importlib.util
import json
import lzma
from pathlib import Path
import struct
import sys
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('cache_compare', ROOT/'scripts/compare_integer_cache_contracts.py')
hc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hc)


def encode(report, kind=None):
    kind = report['kind'] if kind is None else kind
    out = bytearray(hc.MAGIC + struct.pack('<II', kind, hc.COUNT))
    for case in report['cases']:
        out.extend(struct.pack('<5I', *case['spec']))
        for key in hc.INPUTS:
            out.extend(case[key].tobytes())
        if kind:
            out.extend(case['bins'].tobytes())
            for key in hc.OUTPUTS:
                out.extend(case[key].tobytes())
    return bytes(out) + b'HICE' + struct.pack('<I', hc.COUNT)


@pytest.fixture
def reports():
    metal, cuda = dict(kind=1, cases=[]), dict(kind=2, cases=[])
    for s in hc.specs():
        case = hc.fixture(s)
        expected = hc.references(case)
        for report, backend in ((metal, 'metal'), (cuda, 'cuda')):
            report['cases'].append(dict(case, **{k: v.copy() for k, v in expected[backend].items()}))
    return metal, cuda


@pytest.mark.parametrize('bins', [1, 2])
def test_complete_lifecycle_against_analytical_cancellation(bins):
    s = next(s for s in hc.specs() if s.exponent == 0 and s.bins == bins)
    reference = hc.references(hc.fixture(s))
    small = 2.**-54
    expected = []

    def hist(first, second=(0., 0.)):
        expected.extend(first)
        if bins == 2:
            expected.extend(second)

    hist((1., 2.))
    hist((1., 0.))
    hist((2*small, 2.))
    expected.extend((2*small, 2.))
    for _ in range(2):
        hist((2*small, 2.) if bins == 1 else (1., 2.), (-1., 0.))
    expected.extend((2*small, 2.))
    hist((0., 0.))
    hist((small, 1.))
    hist((2., 4.))
    hist((4*small, 4.) if bins == 1 else (2., 4.), (-2., 0.))
    expected.extend((4*small, 4.))
    assert np.array_equal(reference['metal']['values'].view('<f8'), expected)
    # Integer conversion retains two 128-unit terms underneath the large value.
    assert reference['cuda']['integers'][0] == 2**61 + 256
    assert reference['cuda']['integers'][4*bins] == 256
    assert reference['cuda']['integers'][-2] == 256


def test_all_layouts_widths_exponents_and_actual_field_counts(reports):
    specs = hc.specs()
    assert len(specs) == len(set(specs)) == 48
    assert {s.max_bin for s in specs} == {16, 512, 65537}
    result = hc.compare(*reports)
    assert result['status'] == 'identical'
    assert result['compared_actual_fields'] == 1776
    assert result['cuda_integer_reference_checks'] == 1584
    assert result['metal_storage_checks'] == 4032
    assert result['cuda_storage_checks'] == 6048
    assert not hc.validate_metal(reports[0])
    with pytest.raises(ValueError, match='actual CUDA'):
        hc.compare(reports[0], reports[0])


@pytest.mark.parametrize('field', ['bins', 'values'])
def test_identical_wrong_actual_outputs_cannot_pass(reports, field):
    for report in reports:
        case = report['cases'][0]
        case[field][0] ^= np.array(1, dtype=case[field].dtype)
    result = hc.compare(*reports)
    assert result['different_actual_fields'] == 0
    assert {f['backend'] for f in result['reference_failures']} == {'metal', 'cuda'}


@pytest.mark.parametrize('backend', [0, 1])
def test_unchanged_but_forged_prefix_and_guard_are_rejected(reports, backend):
    case = reports[backend]['cases'][1]
    for key in ('prefix_before', 'prefix_after', 'guards'):
        case[key][0] ^= np.uint64(1)
    assert np.array_equal(case['prefix_before'], case['prefix_after'])
    errors = hc.compare(*reports)['reference_failures']
    assert {e['field'] for e in errors} == {'prefix_before', 'prefix_after', 'guards'}


def test_prefix_growth_and_cuda_integer_state_checked_independently(reports):
    m, c = (report['cases'][1] for report in reports)
    assert len(m['prefix_before']) == 32 and len(c['prefix_before']) == 64
    assert len(m['integers']) == 0 and len(c['integers']) == 42
    c['prefix_after'][-1] ^= np.uint64(1)
    c['integers'][8] ^= np.uint64(1)
    errors = hc.compare(*reports)['reference_failures']
    assert {e['field'] for e in errors} == {'prefix_after', 'integers'}


@pytest.mark.parametrize('field', ['data', 'cuts', 'first', 'second'])
def test_inputs_must_match_and_preserve_original_source(reports, field):
    # Copy because the synthetic input objects are otherwise shared.
    for report in reports:
        report['cases'][0][field] = report['cases'][0][field].copy()
    reports[1]['cases'][0][field][-1] ^= np.uint32(1)
    with pytest.raises(ValueError, match='input bits differ'):
        hc.compare(*reports)
    reports[0]['cases'][0][field][-1] ^= np.uint32(1)
    with pytest.raises(ValueError, match='original corpus'):
        hc.compare(*reports)


def test_numeric_zero_sign_difference_remains_reviewable(reports):
    # The sibling Hessian is zero. The original equality permits signed zero,
    # but cross-backend output bits must still disclose the difference.
    reports[1]['cases'][0]['values'][3] = 1 << 63
    result = hc.compare(*reports)
    assert not result['reference_failures'] and result['different_actual_fields'] == 1


def test_complete_framing_and_compression(reports, tmp_path):
    raw = encode(reports[0], 0)
    for broken in (raw[:-1], raw+b'x', b'BADMAGIC'+raw[8:],
                   raw[:8]+struct.pack('<I', 9)+raw[12:],
                   raw[:12]+struct.pack('<I', 47)+raw[16:],
                   raw[:16]+struct.pack('<I', 99)+raw[20:]):
        with pytest.raises(ValueError):
            hc.decode(broken)
    for suffix, compress in (('.bin', lambda b: b), ('.gz', gzip.compress), ('.xz', lzma.compress)):
        path = tmp_path/('cache'+suffix)
        path.write_bytes(compress(raw))
        assert hc.load(path)['data'] == raw
    assert hc.compare(*(hc.decode(encode(r)) for r in reports))['status'] == 'identical'


def test_unpack_rejects_observations_and_never_overwrites(reports, tmp_path, monkeypatch):
    source, out = tmp_path/'source.bin', tmp_path/'out.bin'
    source.write_bytes(encode(reports[0]))
    monkeypatch.setattr(sys, 'argv', ['compare', 'unpack', str(source), str(out)])
    with pytest.raises(ValueError, match='input-only'):
        hc.main()
    assert not out.exists()
    source.write_bytes(encode(reports[0], 0))
    assert hc.main() == 0 and out.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        hc.main()


def test_cli_exit_codes_preserve_actual_differences_and_failures(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(hc, 'load', lambda p: reports[int(str(p) == 'cuda')])

    def run(name):
        out = tmp_path/(name+'.json')
        monkeypatch.setattr(sys, 'argv', ['compare', 'compare', 'metal', 'cuda', '--output', str(out)])
        return hc.main(), json.loads(out.read_text())

    assert run('identical')[0] == 0
    with pytest.raises(ValueError, match='already exists'):
        hc.main()
    reports[1]['cases'][0]['values'][3] = 1 << 63
    assert run('review')[0] == 2
    reports[1]['cases'][0]['integers'][0] ^= np.uint64(1)
    code, result = run('failure')
    assert code == 1 and result['reference_failures']
