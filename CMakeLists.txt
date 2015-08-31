cmake_minimum_required(VERSION 2.6)
project(caf-cl-bench CXX)

set(CAF_CL_BENCH_VERSION_MAJOR 0)
set(CAF_CL_BENCH_VERSION_MINOR 1)
set(CAF_CL_BENCH_VERSION_PATCH 0)

# prohibit in-source builds
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please use "
                        "./configure to choose a build directory and "
                        "initialize the build configuration.")
endif ()

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS_DEBUG          "")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "")
  set(CMAKE_CXX_FLAGS_RELEASE        "")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
else()
  set(CMAKE_CXX_FLAGS "-std=c++11 -Wextra -Wall -pedantic ${EXTRA_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
endif()
# set build default build type to RelWithDebInfo if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# check for g++ >= 4.7 or clang++ > = 3.2
try_run(ProgramResult
        CompilationSucceeded
        ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/get_compiler_version.cpp
        RUN_OUTPUT_VARIABLE CompilerVersion)
if (NOT CompilationSucceeded OR NOT ProgramResult EQUAL 0)
  message(FATAL_ERROR "Cannot determine compiler version")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  if (CompilerVersion VERSION_GREATER 4.6)
    message(STATUS "Found g++ version ${CompilerVersion}")
  else ()
    message(FATAL_ERROR "g++ >= 4.7 required (found: ${CompilerVersion}.")
  endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if (CompilerVersion VERSION_GREATER 3.1)
    message(STATUS "Found clang++ version ${CompilerVersion}")
  else ()
    message(FATAL_ERROR "clang++ >= 3.2 required (found: ${CompilerVersion}.")
  endif ()
  if (NOT CXXFLAGS_PROVIDED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif ()
else ()
  message(FATAL_ERROR "Your C++ compiler does not support C++11.")
endif ()

if (CAF_LOG_LEVEL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAF_LOG_LEVEL=${CAF_LOG_LEVEL}")
endif(CAF_LOG_LEVEL)

# set build type (evaluate ENABLE_DEBUG flag)
if (ENABLE_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAF_DEBUG")
elseif (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if (PRINT_IMAGE)
  FIND_PACKAGE(Qt4 REQUIRED)
  INCLUDE(${QT_USE_FILE})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRINT_IMAGE")
  set(WITH_IMAGE "Yes")
else ()
  set(WITH_IMAGE "No")
endif ()

if (NOT NO_CPU)
  set(CPU_MEASUREMENTS "Yes")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_CPU")
else ()
  set(CPU_MEASUREMENTS "No")
endif ()

if (NOT NO_GPU)
  set(GPU_MEASUREMENTS "Yes")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_GPU")
else ()
  set(GPU_MEASUREMENTS "No")
endif ()

find_package(OPENCL REQUIRED)
find_package(CAF COMPONENTS core opencl REQUIRED)

INCLUDE_DIRECTORIES(
  .
  include
  ${CMAKE_BINARY_DIR}
  ${CAF_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
)

set(SOURCES
  src/main.cpp
  src/config.cpp
)

file(GLOB HEADERS "include/*.hpp")

add_executable(cl-bench ${SOURCES} ${HEADERS})
target_link_libraries(cl-bench ${CMAKE_LD_LIBS} ${CAF_LIBRARIES} ${QT_LIBRARIES} ${OpenCL_LIBRARIES})

set(LD_FLAGS ${LD_FLAGS} ${CMAKE_LD_LIBS} ${CAF_LIBRARIES} ${QT_LIBRARIES} ${OpenCL_LIBRARIES})

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
set(ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${UPPER_BUILD_TYPE}}")
message(STATUS
        "\n====================|  Build Summary  |===================="
        "\nBuild type:        ${CMAKE_BUILD_TYPE}"
        "\nCXX:             ${CMAKE_CXX_COMPILER}"
        "\nCXXFLAGS:          ${ALL_CXX_FLAGS}"
        "\nLIBRARIES:         ${LD_FLAGS}"
        "\nPRINT_IMAGE:       ${WITH_IMAGE}"
        "\nCPU_MEASUREMENTS:  ${CPU_MEASUREMENTS}"
        "\nGPU_MEASUREMENTS:  ${GPU_MEASUREMENTS}"
        "\n===========================================================\n")
