# OpenSplat test suite. Built only when -DOPENSPLAT_BUILD_TESTS=ON (default OFF).
# Levels: unit/ (isolated logic), regression/ (locked-in behaviour), integration/ (modules wired).
# Framework: doctest (header-only). Torch/OpenCV/nanoflann/gsplat_cpu come from the parent scope.
#
# Run:  ctest --test-dir <build-dir> --output-on-failure

include(FetchContent)
# doctest v2.4.11 declares cmake_minimum_required(VERSION 3.0); CMake >= 4.0 rejects minimums
# below 3.5, so allow it (the documented escape hatch). Harmless on older CMake.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(doctest
    URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.zip
)
FetchContent_MakeAvailable(doctest)

set(SRC ${CMAKE_SOURCE_DIR}/src)
# Reuse the parent's single source-of-truth for the src/ include layout (no drift).
set(TEST_INCLUDE_DIRS ${OPENSPLAT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/rasterizer ${TORCH_INCLUDE_DIRS})

# add_opensplat_test(<name> SOURCES ... [LIBS ...] [NO_TORCH] [HEAVY] [WORKING_DIR <dir>])
# One recipe for every test target — links doctest (+ Torch unless NO_TORCH) + pthread.
# Each target is recorded in GLOBAL properties so the aggregate build targets below stay in
# sync automatically (CI never hand-lists individual test names). HEAVY = excluded from the
# lightweight unit aggregate (e.g. the integration target that recompiles the whole pipeline).
function(add_opensplat_test name)
    cmake_parse_arguments(T "NO_TORCH;HEAVY" "WORKING_DIR" "SOURCES;LIBS" ${ARGN})
    add_executable(${name} ${T_SOURCES})
    target_include_directories(${name} PRIVATE ${TEST_INCLUDE_DIRS})
    set(_libs doctest::doctest ${T_LIBS})
    if(NOT T_NO_TORCH)
        list(APPEND _libs ${TORCH_LIBRARIES})
    endif()
    if(NOT WIN32)
        list(APPEND _libs pthread)
    endif()
    target_link_libraries(${name} PRIVATE ${_libs})
    set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
    if(T_WORKING_DIR)
        add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${T_WORKING_DIR})
    else()
        add_test(NAME ${name} COMMAND ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY OPENSPLAT_ALL_TESTS ${name})
    if(NOT T_HEAVY)
        set_property(GLOBAL APPEND PROPERTY OPENSPLAT_UNIT_TESTS ${name})
    endif()
endfunction()

# --- Unit tests -------------------------------------------------------------
add_opensplat_test(test_utils            NO_TORCH SOURCES unit/test_utils.cpp)
add_opensplat_test(test_tensor_math      SOURCES unit/test_tensor_math.cpp     ${SRC}/common/tensor_math.cpp)
add_opensplat_test(test_ssim             SOURCES unit/test_ssim.cpp            ${SRC}/render/ssim.cpp)
add_opensplat_test(test_optim_scheduler  SOURCES unit/test_optim_scheduler.cpp ${SRC}/model/optim_scheduler.cpp)
add_opensplat_test(test_kdtree           SOURCES unit/test_kdtree.cpp          ${SRC}/model/kdtree_tensor.cpp
                                         LIBS nanoflann::nanoflann)
# Link the same OpenCV_LIBS the main opensplat target uses. (cv_utils.cpp also references
# cv::imread/imwrite from opencv_imgcodecs, resolved transitively via highgui — same as the
# main target; kept identical so the tests don't diverge from how the app links OpenCV.)
add_opensplat_test(test_cv_utils         SOURCES unit/test_cv_utils.cpp        ${SRC}/common/cv_utils.cpp
                                         LIBS ${OpenCV_LIBS})
# Resource-aware process contract: pure resolvePolicy() + JSON/env parsing (no Torch).
add_opensplat_test(test_resource         NO_TORCH SOURCES unit/test_resource.cpp ${SRC}/common/resource.cpp
                                         LIBS nlohmann_json::nlohmann_json)

# --- Regression / characterization -----------------------------------------
add_opensplat_test(test_regression_math  SOURCES regression/test_regression_math.cpp
                                         ${SRC}/common/tensor_math.cpp ${SRC}/model/optim_scheduler.cpp)
add_opensplat_test(test_regression_resource NO_TORCH SOURCES regression/test_regression_resource.cpp
                                         ${SRC}/common/resource.cpp
                                         LIBS nlohmann_json::nlohmann_json)

# --- Integration (full pipeline on the CPU backend) -------------------------
# CORE_SRC is derived from the parent's OPENSPLAT_SRC_FILES (single source of truth), minus the
# main (opensplat.cpp) and the optional Pangolin visualizer. No USE_* defines => CPU rasterizer.
set(CORE_SRC "")
foreach(f IN LISTS OPENSPLAT_SRC_FILES)
    if(NOT f MATCHES "/(opensplat|visualizer)\\.cpp$")
        list(APPEND CORE_SRC ${CMAKE_SOURCE_DIR}/${f})
    endif()
endforeach()

# WORKING_DIR = repo root so the data-gated test resolves the relative "data/<scene>" path
# (ctest's default CWD is the build tree). Override the scene with env OPENSPLAT_TEST_DATA.
add_opensplat_test(test_integration HEAVY
    SOURCES integration/test_pipeline.cpp ${CORE_SRC}
    LIBS gsplat_cpu ${OpenCV_LIBS} nlohmann_json::nlohmann_json nanoflann::nanoflann ${STDPPFS_LIBRARY}
    WORKING_DIR ${CMAKE_SOURCE_DIR})

# CUDA integration test — only in a CUDA build. Recompiles CORE_SRC with USE_CUDA
# and links the CUDA `gsplat` library so it exercises the real GPU kernels. Skips
# at runtime if no CUDA device is present (so it is safe on GPU-less CI runners).
if(GPU_RUNTIME STREQUAL "CUDA")
    # CORE_SRC's project/rasterize/SH .cpp files reference BOTH the CPU and the CUDA
    # rasterizer symbols (apply() branches on device), so link gsplat AND gsplat_cpu.
    add_opensplat_test(test_integration_cuda HEAVY
        SOURCES integration/test_pipeline_cuda.cpp ${CORE_SRC}
        LIBS gsplat gsplat_cpu ${GPU_LIBRARIES} ${OpenCV_LIBS} nlohmann_json::nlohmann_json nanoflann::nanoflann ${STDPPFS_LIBRARY}
        WORKING_DIR ${CMAKE_SOURCE_DIR})
    target_compile_definitions(test_integration_cuda PRIVATE USE_CUDA)
    target_include_directories(test_integration_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Aggregate build targets so CI never hand-lists individual test names:
#   opensplat_unit_tests  — all suites except HEAVY (fast; what macOS CI builds)
#   opensplat_all_tests   — everything incl. integration (what Linux CI builds)
get_property(_opensplat_all_tests  GLOBAL PROPERTY OPENSPLAT_ALL_TESTS)
get_property(_opensplat_unit_tests GLOBAL PROPERTY OPENSPLAT_UNIT_TESTS)
add_custom_target(opensplat_unit_tests DEPENDS ${_opensplat_unit_tests})
add_custom_target(opensplat_all_tests  DEPENDS ${_opensplat_all_tests})
