cmake_minimum_required(VERSION 3.28)
project(OpenCV1_0)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR E:\\OpenCV\\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\\x64\\mingw\\lib)

add_executable(OpenCV1_0 main.cpp
        opencvfunction.h
        opencvdemo.cpp
        opencvexample.cpp)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
FIND_PACKAGE(OpenCV 4.5.5 REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})
TARGET_LINK_LIBRARIES(OpenCV1_0 ${OpenCV_LIBS})
