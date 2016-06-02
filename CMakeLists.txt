cmake_minimum_required(VERSION 2.8)
project(HOGViewer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
include(dlib/dlib/cmake)

add_executable(viewer src/viewer.cpp)
target_link_libraries(viewer dlib ${OpenCV_LIBS})
