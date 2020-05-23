cmake_minimum_required (VERSION 2.8.12)
project(start_tf_simple)

include("${PROJECT_SOURCE_DIR}/help.cmake")
define_colors()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wno-shadow -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)

option(BUILD_DEBUG "build project for debug or development propurse..." ON)

# 编译结果中的库和可执行文件的路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/../../inference_root/paddle/include
    ${PROJECT_SOURCE_DIR}/../../inference_root/third_party/
    ${PROJECT_SOURCE_DIR}/../../inference_root/third_party/install/glog/include/
    ${PROJECT_SOURCE_DIR}/../../inference_root/third_party/install/gflags/include/
)
include_directories( ${INCLUDE_DIR} )

link_directories(
    ${PROJECT_SOURCE_DIR}/../../inference_root/paddle/lib
    ${PROJECT_SOURCE_DIR}/../../inference_root/third_party/install/glog/lib/
)

set (EXTRA_LIBS ${EXTRA_LIBS} paddle_fluid)
set (EXTRA_LIBS ${EXTRA_LIBS} glog)
set (EXTRA_LIBS ${EXTRA_LIBS} pthread)


add_executable(simple_lr_inference native_main.cc)
target_link_libraries(
    simple_lr_inference
    ${EXTRA_LIBS}
)
