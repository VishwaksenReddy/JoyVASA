import argparse
import sys
from pathlib import Path

import onnx
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_trt_plugins import build_plugin_library
from export_onnx import export_liveportrait_stack
from src.config.inference_config import InferenceConfig
from src.runtime import ensure_engine_artifact, resolve_engine_artifact
from src.runtime.export_wrappers import WarpingModuleExportWrapper
from src.runtime.runners import TensorRTRunner
from src.runtime.trt_plugins import PLUGIN_NAMESPACE, PLUGIN_OP_TYPE
from src.utils.helper import load_model
from src.utils.rprint import rlog as log


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


THRESHOLDS = {
    "fp32": {
        "occlusion_map": 1e-3,
        "deformation": 2e-3,
        "out": 3e-3,
    },
    "fp16": {
        "occlusion_map": 5e-2,
        "deformation": 5e-2,
        "out": 5e-2,
    },
}


def named_outputs(output_names: list[str], values) -> dict[str, torch.Tensor]:
    if isinstance(values, dict):
        return values
    if isinstance(values, (tuple, list)):
        if len(values) != len(output_names):
            raise RuntimeError(f"Expected {len(output_names)} outputs, received {len(values)}.")
        return dict(zip(output_names, values))
    raise RuntimeError(f"Unsupported output container: {type(values).__name__}")


def count_custom_nodes(onnx_path: Path) -> int:
    graph = onnx.load(str(onnx_path))
    return sum(1 for node in graph.graph.node if node.op_type == PLUGIN_OP_TYPE and node.domain == PLUGIN_NAMESPACE)


def main():
    parser = argparse.ArgumentParser(description="Validate the JoyVASA TensorRT warping-module path.")
    parser.add_argument("--plugin-root", default=str(ROOT / "pretrained_weights" / "trt_artifacts" / "plugins"))
    parser.add_argument("--onnx-root", default=str(ROOT / "pretrained_weights" / "trt_artifacts" / "onnx"))
    parser.add_argument("--engine-root", default=str(ROOT / "pretrained_weights" / "trt_artifacts" / "engines"))
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to validate the TensorRT warping path.")

    cfg = InferenceConfig()
    cfg.trt_precision = args.precision
    cfg.trt_engine_root = args.engine_root
    model_config = yaml.load(open(cfg.models_config, "r"), Loader=yaml.SafeLoader)

    plugin_build = build_plugin_library(args.plugin_root, force=args.force)
    cfg.trt_plugin_library = plugin_build.library_path

    export_liveportrait_stack(
        "human",
        cfg,
        model_config,
        "cuda:0",
        animal=False,
        force=args.force,
        selected_models={"human_warping_module"},
        onnx_root=args.onnx_root,
        engine_root=args.engine_root,
    )

    artifact = resolve_engine_artifact(None, "human_warping_module", args.precision, onnx_root=args.onnx_root, engine_root=args.engine_root)
    custom_nodes = count_custom_nodes(Path(artifact.onnx_path))
    if custom_nodes != 2:
        raise RuntimeError(f"Expected 2 JoyVASA custom GridSample3D nodes, found {custom_nodes} in {artifact.onnx_path}.")

    ensure_engine_artifact(
        root=None,
        name="human_warping_module",
        precision=args.precision,
        source_paths=[cfg.checkpoint_W, cfg.models_config],
        force_rebuild=args.force,
        onnx_root=args.onnx_root,
        engine_root=args.engine_root,
        plugin_libraries=[plugin_build.library_path],
        plugin_build_id=plugin_build.build_id,
    )

    warping_model = WarpingModuleExportWrapper(load_model(cfg.checkpoint_W, model_config, "cuda:0", "warping_module")).eval()
    if args.precision == "fp16":
        warping_model = warping_model.half()

    torch.manual_seed(0)
    feature = torch.randn(1, 32, 16, 64, 64, device="cuda:0")
    kp_driving = torch.randn(1, model_config["model_params"]["motion_extractor_params"]["num_kp"], 3, device="cuda:0")
    kp_source = torch.randn_like(kp_driving)
    if args.precision == "fp16":
        feature = feature.half()
        kp_driving = kp_driving.half()
        kp_source = kp_source.half()

    with torch.no_grad():
        torch_outputs = named_outputs(warping_model.output_names, warping_model(feature, kp_driving, kp_source))

    trt_runner = TensorRTRunner(
        artifact.engine_path,
        device="cuda:0",
        name="human_warping_module",
        plugin_libraries=[plugin_build.library_path],
    )
    trt_outputs = named_outputs(warping_model.output_names, trt_runner(feature, kp_driving, kp_source))
    for key, value in trt_outputs.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite TensorRT output detected for {key}.")

    failures: list[str] = []
    for key in ["occlusion_map", "deformation", "out"]:
        drift = max_abs(torch_outputs[key], trt_outputs[key])
        log(f"{key} max_abs={drift:.6f}")
        threshold = THRESHOLDS[args.precision][key]
        if drift > threshold:
            failures.append(f"{key} drift {drift:.6f} exceeded threshold {threshold:.6f}")

    if failures:
        raise RuntimeError("; ".join(failures))

    log(f"Validated TensorRT warping module with plugin: {plugin_build.library_path}")


if __name__ == "__main__":
    main()
