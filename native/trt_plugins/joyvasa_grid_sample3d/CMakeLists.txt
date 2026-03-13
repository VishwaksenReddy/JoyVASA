cmake_minimum_required(VERSION 3.20)
project(joyvasa_trt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED TENSORRT_INCLUDE_DIR OR NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferRuntime.h")
    message(FATAL_ERROR "TENSORRT_INCLUDE_DIR must point to a TensorRT SDK include directory containing NvInferRuntime.h")
endif()

if(NOT DEFINED TENSORRT_LIBRARY_DIR)
    message(FATAL_ERROR "TENSORRT_LIBRARY_DIR must point to a TensorRT SDK library directory")
endif()

find_package(CUDAToolkit REQUIRED)

if(WIN32)
    find_library(NVINFER_LIBRARY NAMES nvinfer nvinfer_10 PATHS "${TENSORRT_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
else()
    find_library(NVINFER_LIBRARY NAMES nvinfer libnvinfer.so libnvinfer.so.10 PATHS "${TENSORRT_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
endif()

add_library(joyvasa_trt_plugins SHARED
    src/grid_sample3d_plugin.cpp
    src/grid_sample3d_kernel.cu
)

target_include_directories(joyvasa_trt_plugins PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(joyvasa_trt_plugins PRIVATE
    "${NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(joyvasa_trt_plugins PROPERTIES
    OUTPUT_NAME joyvasa_trt_plugins
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(DEFINED TRT_PLUGIN_OUTPUT_DIR)
    set_target_properties(joyvasa_trt_plugins PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${TRT_PLUGIN_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${TRT_PLUGIN_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${TRT_PLUGIN_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${TRT_PLUGIN_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${TRT_PLUGIN_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TRT_PLUGIN_OUTPUT_DIR}"
    )
endif()

if(MSVC)
    target_compile_options(joyvasa_trt_plugins PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    )
else()
    target_compile_options(joyvasa_trt_plugins PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()
