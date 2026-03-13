import importlib

from .engine_utils import (
    DEFAULT_ONNX_OPSET,
    EngineArtifactPaths,
    EngineManifest,
    backend_prefers_tensorrt,
    ensure_engine_artifact,
    get_ort_provider_bundle,
    resolve_engine_artifact,
)


_LAZY_IMPORTS = {
    "create_model_runner": (".factory", "create_model_runner"),
    "create_motion_generator_runner": (".factory", "create_motion_generator_runner"),
    "MotionGeneratorTensorRTRunner": (".runners", "MotionGeneratorTensorRTRunner"),
    "OrtTensorRTRunner": (".runners", "OrtTensorRTRunner"),
    "TensorRTRunner": (".runners", "TensorRTRunner"),
    "TorchMotionGeneratorRunner": (".runners", "TorchMotionGeneratorRunner"),
    "TorchRunner": (".runners", "TorchRunner"),
}

__all__ = [
    "DEFAULT_ONNX_OPSET",
    "EngineArtifactPaths",
    "EngineManifest",
    "MotionGeneratorTensorRTRunner",
    "OrtTensorRTRunner",
    "TensorRTRunner",
    "TorchMotionGeneratorRunner",
    "TorchRunner",
    "backend_prefers_tensorrt",
    "create_model_runner",
    "create_motion_generator_runner",
    "ensure_engine_artifact",
    "get_ort_provider_bundle",
    "resolve_engine_artifact",
]


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
