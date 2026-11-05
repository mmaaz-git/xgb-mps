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
