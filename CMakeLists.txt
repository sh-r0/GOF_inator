cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE DEBUG)

project(GOF_inator LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20) 

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTKMM REQUIRED gtkmm-4.0)

set(SOURCES src/main.cpp src/gof.cpp src/gof_io.cpp src/gof_parallel.cpp src/gof_window.cpp src/gof_cuda.cu)

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/stb)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(PRIVATE ${GTKMM_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES}) 
target_link_libraries(${PROJECT_NAME} PRIVATE ${GTKMM_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
#target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} PkgConfig::GTK4)

#target_compile_definitions(${PROJECT_NAME} PRIVATE ${GTKMM_CFLAGS_OTHER})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_GXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

target_include_directories(${PROJECT_NAME} PRIVATE ${GTKMM_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${GTKMM_LIBRARY_DIRS})
