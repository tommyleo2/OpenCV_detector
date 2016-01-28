cmake_minimum_required(VERSION 2.8)
project(OpenCV_detector)
find_package(OpenCV REQUIRED)
add_executable(OpenCV_detector main.cpp)
target_link_libraries(OpenCV_detector ${OpenCV_LIBS})

