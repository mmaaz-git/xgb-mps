cmake_minimum_required(VERSION 3.24)
project(XgbCudaComparison LANGUAGES CXX CUDA)

# Consume the independently configured/built vendored CUDA library. This test
# project does not change production sources or their CMake target properties.
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(CCCL CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
if(XGB_USE_NVCOMP)
  # Linking libxgboost by path loses the production CMake target's dependency
  # information. Import the same optional package so its nonstandard library
  # directory participates in the consumer's link and build RPATH.
  find_package(nvcomp REQUIRED)
  add_executable(cuda_nvcomp_link_check nvcomp_link_check.cc)
  target_compile_features(cuda_nvcomp_link_check PRIVATE cxx_std_17)
  target_link_libraries(cuda_nvcomp_link_check PRIVATE nvcomp::nvcomp CUDA::cudart_static)
  # Compile/link only: no NVIDIA runtime is present on the CI runner. Check
  # package dependencies before the expensive production-library compilation.
endif()

add_custom_target(cuda_comparison_objects)
foreach(contract numerical aft rng)
  set(object cuda_${contract}_objects)
  set(binary cuda_${contract}_contracts)
  add_library(${object} OBJECT ../cuda_${contract}_contracts.cu)
  set_target_properties(${object} PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Static)
  target_include_directories(${object} PRIVATE
    "${XGB_SOURCE_DIR}/include" "${XGB_SOURCE_DIR}/dmlc-core/include"
    "${XGB_BINARY_DIR}/dmlc-core/include")
  target_compile_definitions(${object} PRIVATE
    XGBOOST_USE_CUDA=1 DMLC_CORE_USE_CMAKE DMLC_LOG_CUSTOMIZE=1)
  target_compile_options(${object} PRIVATE
    --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread
    -Xfatbin=-compress-all)
  foreach(flag IN LISTS openmp_flags)
    target_compile_options(${object} PRIVATE "-Xcompiler=${flag}")
  endforeach()
  target_link_libraries(${object} PRIVATE CCCL::CCCL CUDA::cudart_static OpenMP::OpenMP_CXX)
  add_dependencies(cuda_comparison_objects ${object})
  add_executable(${binary} $<TARGET_OBJECTS:${object}>)
  set_target_properties(${binary} PROPERTIES LINKER_LANGUAGE CUDA CUDA_RUNTIME_LIBRARY Static)
  target_link_libraries(${binary} PRIVATE "${XGB_LIBRARY}" CCCL::CCCL CUDA::cudart_static OpenMP::OpenMP_CXX)
  if(XGB_USE_NVCOMP)
    target_link_libraries(${binary} PRIVATE nvcomp::nvcomp)
    add_dependencies(${object} cuda_nvcomp_link_check)
  endif()
endforeach()
