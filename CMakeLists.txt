cmake_minimum_required(VERSION 3.16)
project(autodf)

set(CMAKE_CXX_STANDARD 20)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE .)

add_subdirectory(test)
