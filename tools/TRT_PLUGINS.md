# JoyVASA TensorRT Plugins

## Build the plugin library

Windows or Linux:

```bash
python tools/build_trt_plugins.py --plugin-root pretrained_weights/trt_artifacts/plugins
```

If TensorRT SDK headers and import libraries are not discoverable automatically, pass one of:

```bash
python tools/build_trt_plugins.py --plugin-root pretrained_weights/trt_artifacts/plugins --tensorrt-root /path/to/TensorRT
python tools/build_trt_plugins.py --plugin-root pretrained_weights/trt_artifacts/plugins --tensorrt-include-dir /path/to/include --tensorrt-library-dir /path/to/lib
```

## Build engines

```bash
python tools/build_trt_engines.py --config tools/config.json --force
```

If `human_warping_module` is selected, `build_trt_engines.py` will build the plugin library first and then pass it to `trtexec` with dynamic plugin loading enabled.

## Validate the warping module

```bash
python tools/validate_warping_trt.py --precision fp32 --force
python tools/validate_warping_trt.py --precision fp16 --force
```

The validation script checks three things:

1. The warping ONNX graph contains the expected `JoyVASAGridSample3D` custom nodes.
2. The TensorRT engine builds successfully with the plugin library.
3. TensorRT output drift stays within the configured max-abs thresholds for `occlusion_map`, `deformation`, and `out`.
