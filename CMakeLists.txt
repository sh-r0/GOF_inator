cmake_minimum_required(VERSION 3.20)

project(GOF_inator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20) 

find_package(CUDA REQUIRED)

set(SOURCES src/main.cpp src/gof.cpp src/gof_io.cpp src/gof_parallel.cpp src/gof_cuda.cu)

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/stb)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20")