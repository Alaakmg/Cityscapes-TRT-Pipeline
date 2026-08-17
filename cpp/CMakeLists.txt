cmake_minimum_required(VERSION 3.18)
project(segdeploy_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT: headers/libs come from JetPack (/usr/include/aarch64-linux-gnu) or
# a TENSORRT_DIR hint on desktop.
find_path(TENSORRT_INCLUDE NvInfer.h HINTS ${TENSORRT_DIR}/include /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu)
find_library(TENSORRT_LIB nvinfer HINTS ${TENSORRT_DIR}/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)

add_executable(segdeploy_infer main.cpp)
target_include_directories(segdeploy_infer PRIVATE ${TENSORRT_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(segdeploy_infer PRIVATE ${TENSORRT_LIB} CUDA::cudart ${OpenCV_LIBS})
