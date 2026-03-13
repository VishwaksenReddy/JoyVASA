#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

int enqueueGridSample3D(
    nvinfer1::PluginTensorDesc const& inputDesc,
    nvinfer1::PluginTensorDesc const& gridDesc,
    nvinfer1::PluginTensorDesc const& outputDesc,
    void const* input,
    void const* grid,
    void* output,
    cudaStream_t stream) noexcept;
