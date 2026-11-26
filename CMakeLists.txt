# twinfer, CMake build: Windows (Visual Studio 2022 / Build Tools, MSVC) and Linux (GCC / Clang).
# The Makefile does the same on Linux; use whichever you like.
#
#   cmake -B build                          configure (picks up CUDA if nvcc is installed)
#   cmake --build build --config Release    build: build/twinfer-cpu, build/twinfer-cuda, ...
#   ctest --test-dir build -C Release -LE gpu    CPU-side tests (python + numpy; node for the JS tokenizer)
#   ctest --test-dir build -C Release -L gpu     CUDA kernel + end-to-end tests (needs an NVIDIA GPU)
#
# Options:
#   -DCMAKE_CUDA_ARCHITECTURES=120   GPU arch to compile for (default: native, the GPU in this machine).
#                                    120 = RTX 50 series, 89 = RTX 40, 86 = RTX 30, 75 = T4 / RTX 20
#   -DTWINFER_CUDA=OFF               CPU engine only, even if CUDA is installed
#   -DTWINFER_NATIVE=OFF             don't use -march=native for the CPU engine (GCC/Clang)
cmake_minimum_required(VERSION 3.24)
project(twinfer LANGUAGES CXX)

option(TWINFER_CUDA "Build the CUDA engine, kernel tests and benchmarks when CUDA is available" ON)
option(TWINFER_NATIVE "Optimize the CPU engine for the build machine (-march=native, GCC/Clang only)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Every binary goes straight into the build folder, also with multi-config generators such as
# Visual Studio (no Release/ subfolder), so the test scripts always find build/twinfer-cpu(.exe).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} ${CMAKE_BINARY_DIR})
endforeach()

if(MSVC)
  # /utf-8: sources and string literals are UTF-8. C4068: `#pragma omp simd` isn't known to MSVC's OpenMP 2.0.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3>
                      $<$<COMPILE_LANGUAGE:CXX>:/wd4068>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

set(COMMON_SRC common/model_files.cpp common/tokenizer.cpp common/platform.cpp)

# CPU reference engine
add_executable(twinfer-cpu cpu/main.cpp cpu/cpu_model.cpp ${COMMON_SRC})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(twinfer-cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
if(TWINFER_NATIVE AND NOT MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native TWINFER_HAS_MARCH_NATIVE)
  if(TWINFER_HAS_MARCH_NATIVE)
    target_compile_options(twinfer-cpu PRIVATE -march=native)
  endif()
endif()

# CUDA engine, kernel tests, kernel benchmarks
if(TWINFER_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)  # needs a GPU at configure time; pass e.g. 120 otherwise
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)

    add_executable(twinfer-cuda cuda/main.cu cuda/cuda_model.cu ${COMMON_SRC})
    add_executable(test-kernels cuda/test_kernels.cu)
    add_executable(bench-kernels cuda/bench_kernels.cu)
    target_link_libraries(bench-kernels PRIVATE CUDA::cublas)
    foreach(t twinfer-cuda test-kernels bench-kernels)
      target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
      if(MSVC)
        target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
      endif()
    endforeach()
    message(STATUS "twinfer: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "twinfer: no CUDA compiler found, building the CPU engine only")
  endif()
endif()

# Tests
enable_testing()
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME cpu COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_tests.py --cpu
                            --build-dir ${CMAKE_BINARY_DIR})
  if(TARGET twinfer-cuda)
    add_test(NAME cuda-e2e COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_tests.py --cuda
                                   --build-dir ${CMAKE_BINARY_DIR})
    set_tests_properties(cuda-e2e PROPERTIES LABELS gpu)
  endif()
endif()
if(TARGET test-kernels)
  add_test(NAME cuda-kernels COMMAND test-kernels)
  set_tests_properties(cuda-kernels PROPERTIES LABELS gpu)
endif()
