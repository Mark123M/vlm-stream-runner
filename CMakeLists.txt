cmake_minimum_required(VERSION 3.20)
project(vlm_stream_runner LANGUAGES C CXX OBJC OBJCXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# llama.cpp-omni is consumed as a library. Keep unrelated programs/tests out of
# the default build while retaining its omni and llama-common targets.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp-omni EXCLUDE_FROM_ALL)

add_library(vlm-runner-support
    src/runner_support.cpp
)
target_include_directories(vlm-runner-support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(vlm-runner-support PUBLIC cxx_std_17)

add_executable(vlm-stream-runner
    src/main.cpp
)
target_include_directories(vlm-stream-runner PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-omni"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-omni/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-omni/tools/omni"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-omni/vendor"
)
target_compile_definitions(vlm-stream-runner PRIVATE
    VLM_STREAM_RUNNER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(vlm-stream-runner PRIVATE
    vlm-runner-support
    omni
    llama-common
    Threads::Threads
)

include(CTest)
if(BUILD_TESTING)
    add_executable(vlm-stream-runner-tests tests/test_runner.cpp)
    target_link_libraries(vlm-stream-runner-tests PRIVATE vlm-runner-support)
    add_test(NAME vlm-stream-runner-unit COMMAND vlm-stream-runner-tests)
endif()
