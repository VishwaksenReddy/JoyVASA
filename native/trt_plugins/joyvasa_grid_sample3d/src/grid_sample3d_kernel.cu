#include "grid_sample3d_kernel.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

template <typename T>
__device__ inline float toFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ inline float toFloat<__half>(__half value)
{
    return __half2float(value);
}

template <typename T>
__device__ inline T fromFloat(float value)
{
    return static_cast<T>(value);
}

template <>
__device__ inline __half fromFloat<__half>(float value)
{
    return __float2half(value);
}

__device__ inline float unnormalize(float coord, int size)
{
    // JoyVASA uses align_corners=False and zero padding.
    return ((coord + 1.0F) * static_cast<float>(size) - 1.0F) * 0.5F;
}

template <typename InputT, typename GridT>
__global__ void gridSample3dKernel(
    InputT const* input,
    GridT const* grid,
    InputT* output,
    int batch,
    int channels,
    int inputDepth,
    int inputHeight,
    int inputWidth,
    int outputDepth,
    int outputHeight,
    int outputWidth)
{
    int64_t linearIndex = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(batch) * channels * outputDepth * outputHeight * outputWidth;
    if (linearIndex >= total)
    {
        return;
    }

    int64_t valueIndex = linearIndex;
    int outW = static_cast<int>(valueIndex % outputWidth);
    valueIndex /= outputWidth;
    int outH = static_cast<int>(valueIndex % outputHeight);
    valueIndex /= outputHeight;
    int outD = static_cast<int>(valueIndex % outputDepth);
    valueIndex /= outputDepth;
    int channel = static_cast<int>(valueIndex % channels);
    int sample = static_cast<int>(valueIndex / channels);

    int64_t gridIndex = (((static_cast<int64_t>(sample) * outputDepth + outD) * outputHeight + outH) * outputWidth + outW) * 3;
    float x = unnormalize(toFloat(grid[gridIndex]), inputWidth);
    float y = unnormalize(toFloat(grid[gridIndex + 1]), inputHeight);
    float z = unnormalize(toFloat(grid[gridIndex + 2]), inputDepth);

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int z0 = static_cast<int>(floorf(z));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float wx1 = x - static_cast<float>(x0);
    float wy1 = y - static_cast<float>(y0);
    float wz1 = z - static_cast<float>(z0);
    float wx0 = 1.0F - wx1;
    float wy0 = 1.0F - wy1;
    float wz0 = 1.0F - wz1;

    float accum = 0.0F;
    int zs[2] = {z0, z1};
    int ys[2] = {y0, y1};
    int xs[2] = {x0, x1};
    float wzs[2] = {wz0, wz1};
    float wys[2] = {wy0, wy1};
    float wxs[2] = {wx0, wx1};

    for (int zi = 0; zi < 2; ++zi)
    {
        if (zs[zi] < 0 || zs[zi] >= inputDepth)
        {
            continue;
        }
        for (int yi = 0; yi < 2; ++yi)
        {
            if (ys[yi] < 0 || ys[yi] >= inputHeight)
            {
                continue;
            }
            for (int xi = 0; xi < 2; ++xi)
            {
                if (xs[xi] < 0 || xs[xi] >= inputWidth)
                {
                    continue;
                }
                float weight = wzs[zi] * wys[yi] * wxs[xi];
                int64_t inputIndex = ((((static_cast<int64_t>(sample) * channels + channel) * inputDepth + zs[zi]) * inputHeight
                    + ys[yi]) * inputWidth)
                    + xs[xi];
                accum += toFloat(input[inputIndex]) * weight;
            }
        }
    }

    output[linearIndex] = fromFloat<InputT>(accum);
}

template <typename InputT, typename GridT>
int launchKernel(
    InputT const* input,
    GridT const* grid,
    InputT* output,
    int batch,
    int channels,
    int inputDepth,
    int inputHeight,
    int inputWidth,
    int outputDepth,
    int outputHeight,
    int outputWidth,
    cudaStream_t stream) noexcept
{
    int64_t total = static_cast<int64_t>(batch) * channels * outputDepth * outputHeight * outputWidth;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    gridSample3dKernel<<<blocks, threads, 0, stream>>>(
        input,
        grid,
        output,
        batch,
        channels,
        inputDepth,
        inputHeight,
        inputWidth,
        outputDepth,
        outputHeight,
        outputWidth);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

} // namespace

int enqueueGridSample3D(
    nvinfer1::PluginTensorDesc const& inputDesc,
    nvinfer1::PluginTensorDesc const& gridDesc,
    nvinfer1::PluginTensorDesc const& outputDesc,
    void const* input,
    void const* grid,
    void* output,
    cudaStream_t stream) noexcept
{
    if (inputDesc.dims.nbDims != 5 || gridDesc.dims.nbDims != 5 || outputDesc.dims.nbDims != 5)
    {
        return 1;
    }

    int batch = inputDesc.dims.d[0];
    int channels = inputDesc.dims.d[1];
    int inputDepth = inputDesc.dims.d[2];
    int inputHeight = inputDesc.dims.d[3];
    int inputWidth = inputDesc.dims.d[4];
    int outputDepth = outputDesc.dims.d[2];
    int outputHeight = outputDesc.dims.d[3];
    int outputWidth = outputDesc.dims.d[4];

    if (inputDesc.type == nvinfer1::DataType::kFLOAT)
    {
        if (gridDesc.type == nvinfer1::DataType::kFLOAT)
        {
            return launchKernel(
                static_cast<float const*>(input),
                static_cast<float const*>(grid),
                static_cast<float*>(output),
                batch,
                channels,
                inputDepth,
                inputHeight,
                inputWidth,
                outputDepth,
                outputHeight,
                outputWidth,
                stream);
        }
        return 1;
    }

    if (inputDesc.type == nvinfer1::DataType::kHALF)
    {
        if (gridDesc.type == nvinfer1::DataType::kHALF)
        {
            return launchKernel(
                static_cast<__half const*>(input),
                static_cast<__half const*>(grid),
                static_cast<__half*>(output),
                batch,
                channels,
                inputDepth,
                inputHeight,
                inputWidth,
                outputDepth,
                outputHeight,
                outputWidth,
                stream);
        }
        if (gridDesc.type == nvinfer1::DataType::kFLOAT)
        {
            return launchKernel(
                static_cast<__half const*>(input),
                static_cast<float const*>(grid),
                static_cast<__half*>(output),
                batch,
                channels,
                inputDepth,
                inputHeight,
                inputWidth,
                outputDepth,
                outputHeight,
                outputWidth,
                stream);
        }
    }

    return 1;
}
