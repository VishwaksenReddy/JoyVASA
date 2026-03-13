# Build Engines

This document describes the JoyVASA TensorRT build pipeline for:

1. Building the custom 5D `grid_sample` TensorRT plugin used by the warping module.
2. Exporting ONNX artifacts.
3. Building TensorRT engines from those ONNX files.

It is written to cover both Windows and Linux, since the same flow will be reused when this pipeline is dockerized.

## What Gets Built

The TensorRT path in this repo relies on a custom plugin for 5D `grid_sample`:

- Custom ONNX op: `com.joyvasa::JoyVASAGridSample3D`
- Native plugin source: `native/trt_plugins/joyvasa_grid_sample3d`
- Plugin library output:
  - Windows: `pretrained_weights/trt_artifacts/plugins/joyvasa_trt_plugins.dll`
  - Linux: `pretrained_weights/trt_artifacts/plugins/libjoyvasa_trt_plugins.so`

Default artifact roots come from `tools/config.json`:

- ONNX: `pretrained_weights/trt_artifacts/onnx`
- Engines: `pretrained_weights/trt_artifacts/engines`
- Plugins: `pretrained_weights/trt_artifacts/plugins`

## Build Order

The normal build order is:

1. Build the custom TensorRT plugin.
2. Export missing ONNX models.
3. Build TensorRT engines.

If `human_warping_module` is selected in `tools/config.json`, `tools/build_trt_engines.py` will build the plugin first and then pass it into TensorRT engine building.

## Prerequisites

Required on both Windows and Linux:

- Python environment with the repo dependencies installed
- CUDA Toolkit
- TensorRT SDK
- `trtexec`
- CMake

Additional Windows requirements:

- Visual Studio 2022 Build Tools or Visual Studio 2022
- CUDA Visual Studio integration
- Run from a Developer PowerShell / Developer Command Prompt if CMake cannot find the CUDA toolset

Additional Linux requirements:

- `g++`
- `make` or `ninja`
- CUDA and TensorRT runtime libraries visible to the linker and loader

## Environment Variables

The build scripts can auto-discover some tools, but the most reliable setup is to define the relevant paths explicitly.

### TensorRT

Recognized by the plugin build script:

- `TENSORRT_ROOT`
- `TensorRT_ROOT`
- `TRT_INCLUDE_DIR`
- `TENSORRT_INCLUDE_DIR`
- `TRT_LIB_DIR`
- `TENSORRT_LIBRARY_DIR`

Recommended:

- Set `TENSORRT_ROOT`
- Make sure `trtexec` is on `PATH`, or available under `TENSORRT_ROOT/bin`

### CUDA

Recognized by the plugin build script:

- `CUDA_PATH`
- `CUDA_PATH_V*` on Windows
- `CUDAToolkit_ROOT`
- `CMAKE_CUDA_COMPILER`
- `CUDACXX`

Recommended:

- Set `CUDA_PATH` on Windows
- Set `CUDAToolkit_ROOT` or `CUDACXX` on Linux if auto-discovery is unreliable

### CMake

Recognized by the plugin build script:

- `CMAKE_EXECUTABLE`
- `CMAKE_GENERATOR`

Recommended:

- Set `CMAKE_GENERATOR` only if you need to override the default
- On Windows, the default generator is `Visual Studio 17 2022`

## PATH Setup

These are the paths that typically need to be visible.

### Windows

Add these to `PATH`:

- `C:\Program Files\CMake\bin`
- `%CUDA_PATH%\bin`
- `%TENSORRT_ROOT%\bin`
- Optionally `%TENSORRT_ROOT%\lib` for runtime loading

Example:

```powershell
$env:TENSORRT_ROOT = "C:\TensorRT-10.x.x.x"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
$env:Path = "C:\Program Files\CMake\bin;$env:CUDA_PATH\bin;$env:TENSORRT_ROOT\bin;$env:Path"
```

If auto-discovery still fails, also set:

```powershell
$env:TRT_INCLUDE_DIR = "$env:TENSORRT_ROOT\include"
$env:TRT_LIB_DIR = "$env:TENSORRT_ROOT\lib\x64"
```

### Linux

Add these to `PATH` and `LD_LIBRARY_PATH`:

- `/usr/local/cuda/bin`
- `$TENSORRT_ROOT/bin`
- `/usr/local/cuda/lib64`
- `$TENSORRT_ROOT/lib`
- `$TENSORRT_ROOT/lib64`

Example:

```bash
export TENSORRT_ROOT=/opt/TensorRT
export CUDAToolkit_ROOT=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$TENSORRT_ROOT/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$TENSORRT_ROOT/lib:$TENSORRT_ROOT/lib64:$LD_LIBRARY_PATH
```

If auto-discovery still fails, also set:

```bash
export TRT_INCLUDE_DIR=$TENSORRT_ROOT/include
export TRT_LIB_DIR=$TENSORRT_ROOT/lib
```

## Build The Custom 5D GridSample Plugin

Run:

```bash
python tools/build_trt_plugins.py --plugin-root pretrained_weights/trt_artifacts/plugins --force
```

This builds:

- Windows: `pretrained_weights/trt_artifacts/plugins/joyvasa_trt_plugins.dll`
- Linux: `pretrained_weights/trt_artifacts/plugins/libjoyvasa_trt_plugins.so`

It also writes a sidecar build-info file:

- `pretrained_weights/trt_artifacts/plugins/joyvasa_trt_plugins.build.json`

If you want to pass paths explicitly instead of relying on environment variables:

```bash
python tools/build_trt_plugins.py \
  --plugin-root pretrained_weights/trt_artifacts/plugins \
  --tensorrt-root /path/to/TensorRT \
  --cuda-root /path/to/cuda \
  --force
```

Or:

```bash
python tools/build_trt_plugins.py \
  --plugin-root pretrained_weights/trt_artifacts/plugins \
  --tensorrt-include-dir /path/to/include \
  --tensorrt-library-dir /path/to/lib \
  --cuda-compiler /path/to/nvcc \
  --force
```

## Build Engines

The main engine build command is:

```bash
python tools/build_trt_engines.py --config tools/config.json --force
```

What this does:

1. Loads the model list from `tools/config.json`
2. Builds the plugin first if a selected model needs it
3. Exports missing ONNX models
4. Builds TensorRT engines

The default config currently targets:

- `human_appearance_feature_extractor`
- `human_motion_extractor`
- `human_warping_module`
- `human_spade_generator`
- `human_stitching`
- `human_lip`
- `human_eye`
- `motion_audio_feature`
- `motion_denoiser`

Precision selection is controlled in `tools/config.json` under:

- `precision.fp16`
- `precision.fp32`

## Optional: Export ONNX Only

This is useful for debugging ONNX export separately from TensorRT engine building.

```bash
python tools/export_onnx.py --scope human --force
```

This is especially useful when diagnosing:

- ONNX exporter errors
- attention/export issues in the motion denoiser
- custom op export issues for the 5D `grid_sample` warping path

## Validate The Warping Path

Run:

```bash
python tools/validate_warping_trt.py --precision fp32 --force
python tools/validate_warping_trt.py --precision fp16 --force
```

This checks:

1. The exported ONNX graph contains the `JoyVASAGridSample3D` custom nodes
2. TensorRT can build the warping engine with the custom plugin loaded
3. TensorRT output stays close to the PyTorch reference

## Notes For Docker / Linux Automation

For future containerization:

- Install CMake inside the image
- Install CUDA Toolkit inside the image
- Install TensorRT SDK inside the image
- Ensure `trtexec` is available on `PATH`
- Ensure `LD_LIBRARY_PATH` includes TensorRT and CUDA runtime libraries
- Prefer explicit environment variables over host-dependent auto-discovery

Recommended container environment:

```bash
export TENSORRT_ROOT=/opt/TensorRT
export CUDAToolkit_ROOT=/usr/local/cuda
export TRT_INCLUDE_DIR=$TENSORRT_ROOT/include
export TRT_LIB_DIR=$TENSORRT_ROOT/lib
export PATH=/usr/local/cuda/bin:$TENSORRT_ROOT/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

## Troubleshooting

### CMake configure fails on Windows

Common causes:

- wrong generator cache in the plugin build directory
- CUDA Visual Studio integration missing
- running outside a Developer PowerShell / Developer Prompt

### Plugin build fails with only an MSBuild banner

The plugin build script now emits verbose CMake build output. Re-run the command and inspect the full error text.

### ONNX export fails for motion denoiser attention

The exporter already contains a compatibility path for `nn.MultiheadAttention` during ONNX export. If this breaks again, test first with:

```bash
python tools/export_onnx.py --scope human --force
```

### TensorRT SDK is not found

Set one of:

- `TENSORRT_ROOT`
- `TRT_INCLUDE_DIR` and `TRT_LIB_DIR`

### CUDA toolkit is not found

Set one of:

- `CUDA_PATH`
- `CUDAToolkit_ROOT`
- `CMAKE_CUDA_COMPILER`
- `CUDACXX`
