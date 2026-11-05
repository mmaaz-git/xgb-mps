cmake_minimum_required(VERSION 3.24)
project(XgbCudaComparison LANGUAGES CXX CUDA)

# Consume the independently configured/built vendored CUDA library. This test
# project does not change production sources or their CMake target properties.
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(CCCL CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(cuda_contract_objects OBJECT ../cuda_numerical_contracts.cu)
set_target_properties(cuda_contract_objects PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Static)
target_include_directories(cuda_contract_objects PRIVATE
  "${XGB_SOURCE_DIR}/include" "${XGB_SOURCE_DIR}/dmlc-core/include"
  "${XGB_BINARY_DIR}/dmlc-core/include")
target_compile_definitions(cuda_contract_objects PRIVATE
  XGBOOST_USE_CUDA=1 DMLC_CORE_USE_CMAKE DMLC_LOG_CUSTOMIZE=1)
target_compile_options(cuda_contract_objects PRIVATE
  --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread
  -Xfatbin=-compress-all)
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_flags)
  target_compile_options(cuda_contract_objects PRIVATE "-Xcompiler=${flag}")
endforeach()
target_link_libraries(cuda_contract_objects PRIVATE CCCL::CCCL CUDA::cudart_static OpenMP::OpenMP_CXX)
add_executable(cuda_numerical_contracts $<TARGET_OBJECTS:cuda_contract_objects>)
set_target_properties(cuda_numerical_contracts PROPERTIES
  LINKER_LANGUAGE CUDA CUDA_RUNTIME_LIBRARY Static)
target_link_libraries(cuda_numerical_contracts PRIVATE
  "${XGB_LIBRARY}" CCCL::CCCL CUDA::cudart_static OpenMP::OpenMP_CXX)

if(XGB_USE_NVCOMP)
  # Linking libxgboost by path loses the production CMake target's dependency
  # information. Import the same optional package so its nonstandard library
  # directory participates in the consumer's link and build RPATH.
  find_package(nvcomp REQUIRED)
  target_link_libraries(cuda_numerical_contracts PRIVATE nvcomp::nvcomp)
  add_executable(cuda_nvcomp_link_check nvcomp_link_check.cc)
  target_compile_features(cuda_nvcomp_link_check PRIVATE cxx_std_17)
  target_link_libraries(cuda_nvcomp_link_check PRIVATE nvcomp::nvcomp CUDA::cudart_static)
  # Compile/link only: no NVIDIA runtime is present on the CI runner. Check
  # package dependencies before the expensive production-library compilation.
  add_dependencies(cuda_contract_objects cuda_nvcomp_link_check)
endif()
