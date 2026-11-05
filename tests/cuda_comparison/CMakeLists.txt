cmake_minimum_required(VERSION 3.24)
project(XgbCudaComparison LANGUAGES CXX CUDA)

# Consume the independently configured/built vendored CUDA library. This test
# project does not change production sources or their CMake target properties.
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(CCCL CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../../scripts/prepare_auc_probe.py"
    --source "${XGB_SOURCE_DIR}/src/metric/auc.cu"
    --output "${CMAKE_CURRENT_BINARY_DIR}/auc_probe.inc"
  COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/../../scripts/prepare_auc_probe.py"
  "${XGB_SOURCE_DIR}/src/metric/auc.cu")
if(XGB_USE_NVCOMP)
  # Linking libxgboost by path loses the production CMake target's dependency
  # information. Import the same optional package so its nonstandard library
  # directory participates in the consumer's link and build RPATH.
  find_package(nvcomp REQUIRED)
  include("${CMAKE_CURRENT_LIST_DIR}/nvcomp_runtime.cmake")
  add_executable(cuda_nvcomp_link_check nvcomp_link_check.cc)
  target_compile_features(cuda_nvcomp_link_check PRIVATE cxx_std_17)
  target_link_libraries(cuda_nvcomp_link_check PRIVATE nvcomp::nvcomp CUDA::cudart_static)
  # This host-only program takes a function address; it never calls nvCOMP or
  # CUDA. Its loader check runs before expensive production compilation.
endif()

add_custom_target(cuda_comparison_objects)
foreach(contract numerical aft rng math ranking metric linear auc)
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
  if(contract STREQUAL "auc")
    # Retain original relative include lookup for the generated source copy.
    target_include_directories(${object} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
      "${XGB_SOURCE_DIR}/src/metric")
  endif()
  if(contract STREQUAL "metric" OR contract STREQUAL "linear" OR contract STREQUAL "auc")
    # Source policy/reducer probes must not interpose on the production library.
    target_compile_options(${object} PRIVATE
      -Xcompiler=-fvisibility=hidden -Xcompiler=-fvisibility-inlines-hidden)
  endif()
  foreach(flag IN LISTS openmp_flags)
    target_compile_options(${object} PRIVATE "-Xcompiler=${flag}")
  endforeach()
  target_link_libraries(${object} PRIVATE CCCL::CCCL CUDA::cudart_static OpenMP::OpenMP_CXX)
  add_dependencies(cuda_comparison_objects ${object})
  add_executable(${binary} $<TARGET_OBJECTS:${object}>)
  set_target_properties(${binary} PROPERTIES LINKER_LANGUAGE CUDA CUDA_RUNTIME_LIBRARY Static)
  target_link_libraries(${binary} PRIVATE "${XGB_LIBRARY}" CCCL::CCCL CUDA::cudart_static OpenMP::OpenMP_CXX)
  if(XGB_USE_NVCOMP)
    xgb_link_nvcomp_runtime(${binary})
    add_dependencies(${object} cuda_nvcomp_link_check)
  endif()
endforeach()
