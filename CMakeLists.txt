cmake_minimum_required(VERSION 3.16)
project(autodf)

set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE .)

enable_testing()
add_subdirectory(test)
