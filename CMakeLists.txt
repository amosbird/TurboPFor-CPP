cmake_minimum_required(VERSION 3.16)

# Prefer clang compiler if available
find_program(CLANG_C_COMPILER clang)
find_program(CLANG_CXX_COMPILER clang++)

if(CLANG_C_COMPILER AND CLANG_CXX_COMPILER AND NOT CMAKE_C_COMPILER AND NOT CMAKE_CXX_COMPILER)
    set(CMAKE_C_COMPILER ${CLANG_C_COMPILER} CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER ${CLANG_CXX_COMPILER} CACHE STRING "C++ compiler" FORCE)
endif()

project(turbopfor_cpp LANGUAGES C CXX)

include(FetchContent)

FetchContent_Declare(
  turbopfor_upstream
  URL https://github.com/powturbo/TurboPFor-Integer-Compression/archive/refs/heads/master.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(turbopfor_upstream)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(ENABLE_SSE42 "Enable SSE4.2 optimizations for p4enc128v32 and p4d1dec128v32" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations for p4enc256v32 and p4d1dec256v32" ON)

include(CheckCXXCompilerFlag)

set(TURBOPFOR_BASE_COMPILE_OPTIONS
  -O3
  -DNDEBUG
  -g
  -ffp-contract=off
  -ffunction-sections
  -fdata-sections
  -fstrict-aliasing
  -falign-functions=64
  -fomit-frame-pointer
  -ftree-vectorize
  -funroll-loops
  -msse2
  -mssse3
  -msse4.1
  -msse4.2
  -mpopcnt
)

# Add Clang-specific vectorize flag
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND TURBOPFOR_BASE_COMPILE_OPTIONS -fvectorize)
endif()

check_cxx_compiler_flag("-mbranches-within-32B-boundaries" TURBOPFOR_HAS_BRANCH_BOUNDS)
if(TURBOPFOR_HAS_BRANCH_BOUNDS)
  list(APPEND TURBOPFOR_BASE_COMPILE_OPTIONS -mbranches-within-32B-boundaries)
endif()

check_cxx_compiler_flag("-falign-loops" TURBOPFOR_HAS_ALIGN_LOOPS)
if(TURBOPFOR_HAS_ALIGN_LOOPS)
  list(APPEND TURBOPFOR_BASE_COMPILE_OPTIONS -falign-loops)
endif()

# Base library sources (always included)
set(TURBOPFOR_SOURCES
  src/dispatch.cpp
  src/scalar/p4enc32.cpp
  src/scalar/p4d1dec32.cpp
  src/scalar/p4_scalar_bitpack32.cpp
  src/scalar/p4_scalar_bitunpack32.cpp
  src/scalar/p4_scalar_bitunpackd1_32.cpp
  src/scalar/p4_scalar_internal.cpp
  src/scalar/bitpack128v32_scalar.cpp
  src/scalar/p4enc128v32_scalar.cpp
  src/scalar/p4d1dec128v32_scalar.cpp
  src/scalar/bitpack256v32_scalar.cpp
  src/scalar/p4enc256v32_scalar.cpp
  src/scalar/p4d1dec256v32_scalar.cpp
)

# SSE4.2 optimized sources for 128v32 functions (optional)
if(ENABLE_SSE42)
  list(APPEND TURBOPFOR_SOURCES
    src/simd/p4enc128v32.cpp
    src/simd/p4d1dec128v32.cpp
    src/simd/p4_simd_internal.cpp
    src/simd/bitpack128v32_simd.cpp
    src/simd/bitunpack128v32_simd.cpp
  )
endif()

if(ENABLE_AVX2)
  list(APPEND TURBOPFOR_SOURCES
    src/simd/p4enc256v32.cpp
    src/simd/p4d1dec256v32.cpp
    src/simd/p4_simd_internal_256v.cpp
    src/simd/bitpack256v32_simd.cpp
    src/simd/bitunpack256v32_simd.cpp
  )
endif()

add_library(turbopfor STATIC ${TURBOPFOR_SOURCES})

target_include_directories(turbopfor
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(turbopfor PRIVATE ${TURBOPFOR_BASE_COMPILE_OPTIONS})

# Enable ENABLE_SSE42 preprocessor definition
if(ENABLE_SSE42)
  target_compile_definitions(turbopfor PRIVATE ENABLE_SSE42)
endif()

if(ENABLE_AVX2)
  target_compile_definitions(turbopfor PRIVATE ENABLE_AVX2)
endif()

# Set SSE4.2 flags only for SIMD source files
if(ENABLE_SSE42)
  set_source_files_properties(
    src/simd/p4enc128v32.cpp
    src/simd/p4d1dec128v32.cpp
    src/simd/p4_simd_internal.cpp
    src/simd/bitpack128v32_simd.cpp
    src/simd/bitunpack128v32_simd.cpp
    PROPERTIES COMPILE_OPTIONS "-mssse3;-msse4.1;-msse4.2;-mpopcnt"
  )
endif()

if(ENABLE_AVX2)
  set_source_files_properties(
    src/simd/p4enc256v32.cpp
    src/simd/p4d1dec256v32.cpp
    src/simd/p4_simd_internal_256v.cpp
    src/simd/bitpack256v32_simd.cpp
    src/simd/bitunpack256v32_simd.cpp
    PROPERTIES COMPILE_OPTIONS "-mssse3;-msse4.1;-msse4.2;-mpopcnt;-mavx2"
  )
endif()

set(TURBOPFOR_SOURCE_DIR ${turbopfor_upstream_SOURCE_DIR})

# Non-AVX2 reference library for basic functions (p4enc32, p4d1dec32, bitpack32, etc.)
# These are compiled WITHOUT -mavx2 because the basic functions are in #ifndef __AVX2__ blocks
add_library(turbopfor_ref_base
    ${TURBOPFOR_SOURCE_DIR}/lib/v8.c
    ${TURBOPFOR_SOURCE_DIR}/lib/vint.c
    ${TURBOPFOR_SOURCE_DIR}/lib/trlec.c
    ${TURBOPFOR_SOURCE_DIR}/lib/trled.c
    ${TURBOPFOR_SOURCE_DIR}/lib/vsimple.c
    ${TURBOPFOR_SOURCE_DIR}/lib/bitutil.c
    ${TURBOPFOR_SOURCE_DIR}/lib/bitpack.c
    ${TURBOPFOR_SOURCE_DIR}/lib/bitunpack.c
    ${TURBOPFOR_SOURCE_DIR}/lib/vp4c.c
    ${TURBOPFOR_SOURCE_DIR}/lib/vp4d.c
    ${TURBOPFOR_SOURCE_DIR}/lib/transpose.c
)
target_compile_options(turbopfor_ref_base PRIVATE ${TURBOPFOR_BASE_COMPILE_OPTIONS} -w -DSSE2_ON -mssse3 -msse4.1 -msse4.2 -mpopcnt)
target_include_directories(turbopfor_ref_base SYSTEM PUBLIC ${TURBOPFOR_SOURCE_DIR}/lib/include_)

# AVX2 reference library for 256v functions (p4enc256v32, p4d1dec256v32, etc.)
# These are compiled WITH -mavx2 because the 256v functions are in #ifdef __AVX2__ blocks
add_library(turbopfor_ref_avx2 OBJECT
    ${TURBOPFOR_SOURCE_DIR}/lib/bitpack.c
    ${TURBOPFOR_SOURCE_DIR}/lib/bitunpack.c
    ${TURBOPFOR_SOURCE_DIR}/lib/bitutil.c
    ${TURBOPFOR_SOURCE_DIR}/lib/vp4c.c
    ${TURBOPFOR_SOURCE_DIR}/lib/vp4d.c
)
target_compile_options(turbopfor_ref_avx2 PRIVATE ${TURBOPFOR_BASE_COMPILE_OPTIONS} -w -DSSE2_ON -mssse3 -msse4.1 -msse4.2 -mpopcnt -mavx2)
target_include_directories(turbopfor_ref_avx2 SYSTEM PUBLIC ${TURBOPFOR_SOURCE_DIR}/lib/include_)

# Combined reference library
add_library(turbopfor_ref INTERFACE)
target_link_libraries(turbopfor_ref INTERFACE turbopfor_ref_base $<TARGET_OBJECTS:turbopfor_ref_avx2>)

target_link_libraries(turbopfor PRIVATE turbopfor_ref_base)

# Test and benchmark executables always compile with both scalar and SIMD versions
# so they can test the actual implementations regardless of what the library exports
add_executable(binary_compat_test
  tests/binary_compat_test.cpp
  src/simd/p4enc128v32.cpp
  src/simd/p4d1dec128v32.cpp
  src/simd/p4_simd_internal.cpp
  src/simd/bitpack128v32_simd.cpp
  src/simd/bitunpack128v32_simd.cpp
)
target_link_libraries(binary_compat_test PRIVATE turbopfor turbopfor_ref_base $<TARGET_OBJECTS:turbopfor_ref_avx2>)

target_include_directories(binary_compat_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(binary_compat_test PRIVATE ${TURBOPFOR_BASE_COMPILE_OPTIONS})

# Set SSE4.2 flags for SIMD source files in test
set_source_files_properties(
  src/simd/p4enc128v32.cpp
  src/simd/p4d1dec128v32.cpp
  src/simd/p4_simd_internal.cpp
  src/simd/bitpack128v32_simd.cpp
  src/simd/bitunpack128v32_simd.cpp
  PROPERTIES COMPILE_OPTIONS "-mssse3;-msse4.1;-msse4.2;-mpopcnt"
)





# Benchmark executable
add_executable(ab_test benchmarks/ab_test.cpp)
target_link_libraries(ab_test PRIVATE turbopfor turbopfor_ref_base $<TARGET_OBJECTS:turbopfor_ref_avx2>)
target_include_directories(ab_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ab_test PRIVATE ${TURBOPFOR_BASE_COMPILE_OPTIONS})
if(ENABLE_AVX2)
    target_compile_definitions(ab_test PRIVATE ENABLE_AVX2)
endif()
if(ENABLE_SSE42)
    target_compile_definitions(ab_test PRIVATE ENABLE_SSE42)
endif()

