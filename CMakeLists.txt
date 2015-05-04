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

# check if the user provided CXXFLAGS on the command line
if (CMAKE_CXX_FLAGS)
    set(CXXFLAGS_PROVIDED true)
endif ()
if (CXXFLAGS_PROVIDED)
  set(CMAKE_CXX_FLAGS_DEBUG          "")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "")
  set(CMAKE_CXX_FLAGS_RELEASE        "")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
else ()
  set(CMAKE_CXX_FLAGS                "-std=c++11 -Wextra -Wall -pedantic")
  set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O4")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
endif ()

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
else ()
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

find_package(Libcaf COMPONENTS core io opencl REQUIRED)
find_package(OPENCL REQUIRED)
add_definitions("-DENABLE_OPENCL")
message(STATUS "LIBCAF_LIBRARIES: ${LIBCAF_LIBRARIES}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_OPENCL")

FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})

INCLUDE_DIRECTORIES(. include ${CMAKE_BINARY_DIR} ${LIBCAF_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIRS})

set(SOURCES
    src/main.cpp
    src/client.cpp
    src/config.cpp
    src/q_byte_array_info.cpp)

file(GLOB HEADERS "include/*.hpp")

add_executable(cl-bench ${SOURCES} ${HEADERS})
target_link_libraries(cl-bench ${CMAKE_DL_LIBS} ${LIBCAF_LIBRARIES} ${QT_LIBRARIES} ${OPENCL_LIBRARIES})

