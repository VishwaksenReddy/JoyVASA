import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"


def load_validate_warping_trt_module():
    onnx_stub = types.ModuleType("onnx")
    onnx_stub.load = lambda path: None

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object

    yaml_stub = types.ModuleType("yaml")
    yaml_stub.SafeLoader = object
    yaml_stub.load = lambda *args, **kwargs: {}

    build_trt_plugins_stub = types.ModuleType("build_trt_plugins")
    build_trt_plugins_stub.build_plugin_library = lambda *args, **kwargs: None

    export_onnx_stub = types.ModuleType("export_onnx")
    export_onnx_stub.export_liveportrait_stack = lambda *args, **kwargs: None

    inference_config_stub = types.ModuleType("src.config.inference_config")
    inference_config_stub.InferenceConfig = object

    runtime_stub = types.ModuleType("src.runtime")
    runtime_stub.ensure_engine_artifact = lambda *args, **kwargs: None
    runtime_stub.resolve_engine_artifact = lambda *args, **kwargs: None

    export_wrappers_stub = types.ModuleType("src.runtime.export_wrappers")
    export_wrappers_stub.WarpingModuleExportWrapper = object

    runners_stub = types.ModuleType("src.runtime.runners")
    runners_stub.TensorRTRunner = object

    trt_plugins_stub = types.ModuleType("src.runtime.trt_plugins")
    trt_plugins_stub.PLUGIN_NAMESPACE = "com.joyvasa"
    trt_plugins_stub.PLUGIN_OP_TYPE = "JoyVASAGridSample3D"

    helper_stub = types.ModuleType("src.utils.helper")
    helper_stub.load_model = lambda *args, **kwargs: None

    rprint_stub = types.ModuleType("src.utils.rprint")
    rprint_stub.rlog = lambda *args, **kwargs: None

    module_name = "test_validate_warping_trt_module"
    module_path = TOOLS_ROOT / "validate_warping_trt.py"
    previous_modules = {
        "onnx": sys.modules.get("onnx"),
        "torch": sys.modules.get("torch"),
        "yaml": sys.modules.get("yaml"),
        "build_trt_plugins": sys.modules.get("build_trt_plugins"),
        "export_onnx": sys.modules.get("export_onnx"),
        "src.runtime": sys.modules.get("src.runtime"),
        "src.config.inference_config": sys.modules.get("src.config.inference_config"),
        "src.runtime.export_wrappers": sys.modules.get("src.runtime.export_wrappers"),
        "src.runtime.runners": sys.modules.get("src.runtime.runners"),
        "src.runtime.trt_plugins": sys.modules.get("src.runtime.trt_plugins"),
        "src.utils.helper": sys.modules.get("src.utils.helper"),
        "src.utils.rprint": sys.modules.get("src.utils.rprint"),
        module_name: sys.modules.get(module_name),
    }

    try:
        sys.modules["onnx"] = onnx_stub
        sys.modules["torch"] = torch_stub
        sys.modules["yaml"] = yaml_stub
        sys.modules["build_trt_plugins"] = build_trt_plugins_stub
        sys.modules["export_onnx"] = export_onnx_stub
        sys.modules["src.runtime"] = runtime_stub
        sys.modules["src.config.inference_config"] = inference_config_stub
        sys.modules["src.runtime.export_wrappers"] = export_wrappers_stub
        sys.modules["src.runtime.runners"] = runners_stub
        sys.modules["src.runtime.trt_plugins"] = trt_plugins_stub
        sys.modules["src.utils.helper"] = helper_stub
        sys.modules["src.utils.rprint"] = rprint_stub

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


validate_warping_trt = load_validate_warping_trt_module()


def test_fp32_thresholds_allow_slightly_higher_deformation_drift():
    assert validate_warping_trt.THRESHOLDS["fp32"]["occlusion_map"] == 1e-3
    assert validate_warping_trt.THRESHOLDS["fp32"]["deformation"] == 2e-3
    assert validate_warping_trt.THRESHOLDS["fp32"]["out"] == 3e-3


def test_named_outputs_maps_tuple_values_by_output_name():
    outputs = validate_warping_trt.named_outputs(["occlusion_map", "deformation"], ("a", "b"))

    assert outputs == {"occlusion_map": "a", "deformation": "b"}
