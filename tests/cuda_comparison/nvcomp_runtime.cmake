# Resolve the runtime SONAME location, which can differ from the development
# symlink directory in NVIDIA's Debian packages.
find_library(XGB_NVCOMP_LINK_LIBRARY NAMES nvcomp
  PATHS "${nvcomp_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
file(REAL_PATH "${XGB_NVCOMP_LINK_LIBRARY}" nvcomp_runtime_library)
get_filename_component(nvcomp_runtime_directory "${nvcomp_runtime_library}" DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH "${nvcomp_runtime_directory}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/nvcomp-runtime-path.txt"
  "${nvcomp_runtime_library}\n${nvcomp_runtime_directory}\n")

function(xgb_link_nvcomp_runtime target)
  # ELF DT_RUNPATH is not transitive. Keep the library directly needed by this
  # executable even when only libxgboost references its functions.
  target_link_options(${target} PRIVATE "LINKER:--no-as-needed")
  target_link_libraries(${target} PRIVATE nvcomp::nvcomp)
endfunction()
