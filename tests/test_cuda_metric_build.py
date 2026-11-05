"""Mock compiler-driver lifecycle/isolation gates; no compiler or GPU execution."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).parents[1]


@pytest.fixture
def build(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("cuda_driver",ROOT/"scripts/build_cuda_comparison.py")
    driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    targets=re.search(r"foreach\(contract ([^)]+)\)",(ROOT/"tests/cuda_comparison/CMakeLists.txt").read_text()).group(1).split()
    assert targets==["numerical","aft","rng","math","ranking","metric"]
    for name in driver.INPUTS:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/name).read_bytes())
    (tmp_path/"xgboost").mkdir(exist_ok=True)
    cmake=tmp_path/"build"; cmake.mkdir()
    (cmake/"CMakeCache.txt").write_text("\n".join([
        "USE_CUDA:BOOL=ON","USE_METAL:BOOL=OFF","USE_NVCOMP:BOOL=OFF",
        "KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR:BOOL=ON","HIDE_CXX_SYMBOLS:BOOL=OFF",
        "BUILD_STATIC_LIB:BOOL=OFF","PLUGIN_RMM:BOOL=OFF",f"CMAKE_HOME_DIRECTORY:PATH={tmp_path/'xgboost'}"]))
    output=tmp_path/"report"; state={"bad":None,"calls":[]}
    def command(args,**kwargs):
        state["calls"].append(args)
        if args[0].endswith("cuda_nvcomp_link_check"):
            if state["bad"]=="nvcomp-loader": raise subprocess.CalledProcessError(127,args)
            return SimpleNamespace(stdout="",returncode=0)
        if "--metric-registry" in args:
            if state["bad"]=="command":
                return SimpleNamespace(stdout="partial diagnostic\n",stderr="loader failure retained\n",returncode=1)
            text="rmse factory_signature\n"
            if state["bad"]=="registry" and args[0].endswith("cuda_metric_contracts"): text="rmse changed_factory\n"
            if state["bad"]=="empty": text=""
            return SimpleNamespace(stdout=text,stderr="",returncode=0)
        if args[0]=="nm":
            # A CUB specialization mentioning a policy is not itself a metric
            # definition. Actual metric methods/vtables must remain hidden.
            text="00000001 T xgboost::metric::EvalRowRMSE()\n" if state["bad"]=="symbols" else "00000001 W thrust::Kernel<xgboost::metric::EvalRowRMSE>()\n"
            return SimpleNamespace(stdout=text,stderr="",returncode=0)
        if "--sampling-rng" in args:
            text = "partial check\n" if state["bad"] == "sampling-rng" else "PASS sampling MT19937 row-seed alignment: 10005 cases; no GPU execution\n"
            return SimpleNamespace(stdout=text,stderr="",returncode=0)
        if "-S" in args:
            p=Path(args[args.index("-B")+1]); p.mkdir(); (p/"compile_commands.json").write_text("[]")
        elif args[2]==str(cmake):
            (cmake/"lib").mkdir(); (cmake/"lib/libxgboost.so").write_bytes(b"fake library")
        elif "--target" not in args:
            for name in targets: (output/f"cmake/cuda_{name}_contracts").write_bytes(f"fake {name}".encode())
        return SimpleNamespace(stdout="")
    def git(*args):
        return b"" if args[0]=="diff" else b"fixture-tree\n" if args[1]=="HEAD:xgboost" else b"fixture-head\n"
    monkeypatch.setattr(driver,"ROOT",tmp_path); monkeypatch.setattr(driver,"git",git)
    monkeypatch.setattr(driver.subprocess,"run",command)
    def run(phase):
        monkeypatch.setattr(sys,"argv",["driver","--build-dir",str(cmake),"--output-dir",str(output),"--phase",phase])
        driver.main()
        return json.loads((output/"build.json").read_text())
    return run,state,tmp_path,output,driver


def test_six_binaries_complete_hashes_and_host_only_isolation_gate(build):
    run,state,root,output,driver=build
    record=run("compile"); assert record["status"]=="passed-object-compile"
    assert not any("--metric-registry" in c for c in state["calls"])
    record=run("link")
    assert record["status"]=="passed-compile-link" and not record["gpu_execution"]
    assert len(record["binaries"])==6 and len(record["inputs"])==43
    assert {"tests/cuda_metric_source.h","tests/cuda_metric_contracts.cu","tests/metric_contract_io.h",
            "tests/metric_registry_probe.h","tests/cuda_metric_reference.h",
            "xgboost/src/metric/elementwise_metric.cu","xgboost/src/metric/multiclass_metric.cu",
            "xgboost/src/metric/survival_metric.cu"} <= set(record["inputs"])
    isolation=record["metric_source_isolation"]
    assert isolation["status"]=="passed" and not isolation["gpu_execution"] and len(isolation["files"])==3
    assert sum("--metric-registry" in c for c in state["calls"])==2
    assert any(c[0]=="nm" for c in state["calls"])
    assert record['sampling_rng_alignment']['cases'] == 10005
    assert record['sampling_rng_alignment']['status'] == 'passed'
    assert not record['sampling_rng_alignment']['gpu_execution']
    assert sum('--sampling-rng' in c for c in state['calls']) == 1
    assert {'tests/sampling_contract_io.h','tests/cuda_sampling_contracts.cuh',
            'xgboost/src/tree/gpu_hist/sampler.cu','xgboost/src/tree/gpu_hist/sampler.cuh',
            'xgboost/src/tree/gpu_hist/quantiser.cu','xgboost/src/tree/gpu_hist/quantiser.cuh',
            'xgboost/src/tree/updater_gpu_hist.cuh','xgboost/src/tree/updater_gpu_hist.cu'} <= set(record['inputs'])
    cmake=(root/"tests/cuda_comparison/CMakeLists.txt").read_text()
    assert '-Xcompiler=-fvisibility=hidden -Xcompiler=-fvisibility-inlines-hidden' in cmake
    workflow=(ROOT/'.github/workflows/cuda-compile.yml').read_text()
    for name in ('metric-registry-baseline.txt','metric-registry-probe.txt','metric-defined-dynamic-symbols.txt','sampling-rng-alignment.txt'):
        assert f'results/cuda-comparison-build/{name}' in workflow


@pytest.mark.parametrize("bad,pattern",[("registry","registered names or factory"),("empty","registered names or factory"),
                                        ("symbols","exports production metric"),("source","changed after object"),
                                        ("sampling-rng","Sampling RNG alignment check")])
def test_isolation_and_source_mismatch_retain_failure(build,bad,pattern):
    run,state,root,output,driver=build
    run("compile"); state["bad"]=bad
    if bad=="source":
        with (root/"tests/cuda_metric_source.h").open("a") as stream: stream.write("// changed\n")
    with pytest.raises(ValueError,match=pattern): run("link")
    record=json.loads((output/"build.json").read_text())
    assert record["status"]=="failed" and not record["gpu_execution"]


def test_failed_host_command_retains_stdout_stderr_and_failed_record(build,capsys):
    run,state,root,output,driver=build
    run('compile'); state['bad']='command'
    with pytest.raises(RuntimeError,match='Host isolation command failed'): run('link')
    assert (output/'metric-registry-baseline.txt').read_text()=='partial diagnostic\n'
    assert 'loader failure retained' in capsys.readouterr().out
    assert json.loads((output/'build.json').read_text())['status']=='failed'


@pytest.mark.parametrize('fail',[False,True])
def test_nvcomp_loader_gate_precedes_production_build(build,fail):
    run,state,root,output,driver=build
    cache=root/'build/CMakeCache.txt'
    cache.write_text(cache.read_text().replace('USE_NVCOMP:BOOL=OFF','USE_NVCOMP:BOOL=ON'))
    if fail:
        state['bad']='nvcomp-loader'
        with pytest.raises(subprocess.CalledProcessError):run('compile')
        record=json.loads((output/'build.json').read_text())
        assert record['status']=='failed' and 'nvcomp_loader_probe' not in record
    else:
        record=run('compile')
        assert record['status']=='passed-object-compile'
        assert record['nvcomp_loader_probe']=={'status':'passed','gpu_execution':False}
    assert state['calls'][-1]==[str(output/'cmake/cuda_nvcomp_link_check')]
    assert not (root/'build/lib/libxgboost.so').exists()
