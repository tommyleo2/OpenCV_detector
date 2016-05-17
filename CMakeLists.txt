cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O0 -g -std=c++11 -pthread")
project(detector)
find_package(OpenCV REQUIRED)
aux_source_directory(src SRC)

add_executable(detector test.cpp ${SRC})
target_link_libraries(detector ${OpenCV_LIBS})