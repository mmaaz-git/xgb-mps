file(GLOB_RECURSE CPU_SOURCES *.cc *.h)
list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/cli_main.cc)

if(PLUGIN_SYCL)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/regression_obj.cc)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/hinge.cc)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/quantile_obj.cc)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/multiclass_obj.cc)
endif()

#-- Object library
# Object library is necessary for jvm-package, which creates its own shared library.
add_library(objxgboost OBJECT)
target_sources(objxgboost PRIVATE ${CPU_SOURCES})
if(USE_METAL)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/common/metal_histogram.metal")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/common/metal_histogram.metal" XGBOOST_METAL_SHADER)
  configure_file(common/metal_shader.h.in "${CMAKE_CURRENT_BINARY_DIR}/metal_shader.h" @ONLY)
  target_sources(objxgboost PRIVATE common/metal_histogram.mm)
  set_source_files_properties(common/metal_histogram.mm PROPERTIES
    COMPILE_OPTIONS "-fobjc-arc" SKIP_UNITY_BUILD_INCLUSION ON)
  target_include_directories(objxgboost PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(objxgboost PUBLIC XGBOOST_USE_METAL=1)
  target_link_libraries(objxgboost PUBLIC "-framework Metal" "-framework Foundation")
endif()
# Skip files with factory object
set_source_files_properties(
  predictor/predictor.cc gbm/gbm.cc tree/tree_updater.cc metric/metric.cc objective/objective.cc
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES *.cu)
  target_sources(objxgboost PRIVATE ${CUDA_SOURCES})
endif()

if(PLUGIN_SYCL)
  target_compile_definitions(objxgboost PRIVATE -DXGBOOST_USE_SYCL=1)
endif()

target_include_directories(objxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include)

if(LOG_CAPI_INVOCATION)
  target_compile_definitions(objxgboost PRIVATE -DLOG_CAPI_INVOCATION=1)
endif()

# This grouping organises source files nicely in visual studio
auto_source_group("${CUDA_SOURCES}")
auto_source_group("${CPU_SOURCES}")

#-- End object library
