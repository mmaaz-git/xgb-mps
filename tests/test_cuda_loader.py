"""Exercise the actual harness CMake helper with a synthetic Linux ELF library.

This tests SONAME/symlink and indirect dependency loading, not NVIDIA software
or GPU execution. Requires Linux, CMake, Ninja, a C++ compiler and binutils.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith('linux'),'ELF loader behavior requires Linux')
class LoaderTest(unittest.TestCase):
    def test_real_helper_resolves_soname_and_retains_direct_dependency(self):
        for command in ('cmake','ninja','c++','readelf'):
            self.assertIsNotNone(shutil.which(command),f'{command} is required')
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
        with tempfile.TemporaryDirectory(prefix='xgb-loader-') as directory:
            root=Path(directory)
            for name in ('runtime','dev','include/nvcomp'):(root/name).mkdir(parents=True)
            (root/'dep.cc').write_text('extern "C" int nvcompBatchedSnappyDecompressAsync(){return 7;}\n')
            (root/'middle.cc').write_text('extern "C" int nvcompBatchedSnappyDecompressAsync(); int middle(){return nvcompBatchedSnappyDecompressAsync();}\n')
            (root/'main.cc').write_text('int middle();int main(){return middle()==7?0:1;}\n')
            (root/'include/nvcomp/snappy.h').write_text('extern "C" int nvcompBatchedSnappyDecompressAsync();\n')
            def run(*args):
                result=subprocess.run(args,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                if result.returncode:print(result.stdout,flush=True)
                result.check_returncode()
            run('c++','-shared','-fPIC','dep.cc','-Wl,-soname,libnvcomp.so.5','-o','runtime/libnvcomp.so.5.1')
            (root/'runtime/libnvcomp.so.5').symlink_to('libnvcomp.so.5.1')
            (root/'dev/libnvcomp.so').symlink_to('../runtime/libnvcomp.so.5.1')
            run('c++','-shared','-fPIC','middle.cc','-Ldev','-lnvcomp',f'-Wl,-rpath,{root}/dev','-o','libmiddle.so')
            (root/'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(LoaderRegression LANGUAGES CXX)
set(nvcomp_LIBRARY_DIR "{root}/dev")
add_library(nvcomp::nvcomp SHARED IMPORTED)
set_target_properties(nvcomp::nvcomp PROPERTIES IMPORTED_LOCATION "{root}/dev/libnvcomp.so")
include("{ROOT}/tests/cuda_comparison/nvcomp_runtime.cmake")
foreach(name IN ITEMS original path_only fixed)
  add_executable(${{name}} main.cc)
  target_link_libraries(${{name}} PRIVATE "{root}/libmiddle.so")
  if(name STREQUAL "fixed")
    xgb_link_nvcomp_runtime(${{name}})
  else()
    target_link_options(${{name}} PRIVATE "LINKER:--as-needed")
    target_link_libraries(${{name}} PRIVATE nvcomp::nvcomp)
  endif()
endforeach()
set_target_properties(original PROPERTIES BUILD_RPATH "")
add_executable(probe "{ROOT}/tests/cuda_comparison/nvcomp_link_check.cc")
target_include_directories(probe PRIVATE "{root}/include")
target_link_libraries(probe PRIVATE nvcomp::nvcomp)
''')
            run('cmake','-S',str(root),'-B',str(root/'build'),'-G','Ninja')
            run('cmake','--build',str(root/'build'),'--parallel','2')
            for name in ('original','path_only'):
                result=subprocess.run([str(root/'build'/name)],env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('libnvcomp.so.5',result.stderr)
            for name in ('fixed','probe'):run(str(root/'build'/name))
            dynamic=subprocess.check_output(['readelf','-d',str(root/'build/fixed')],text=True)
            self.assertIn('Shared library: [libnvcomp.so.5]',dynamic)
            self.assertEqual((root/'build/nvcomp-runtime-path.txt').read_text().splitlines(),
                             [str(root/'runtime/libnvcomp.so.5.1'),str(root/'runtime')])
            print('PASS real CMake helper: both broken controls fail; corrected executable and address-only probe load. Synthetic ELF libraries; no NVIDIA/GPU execution.',flush=True)


if __name__=='__main__':unittest.main()
