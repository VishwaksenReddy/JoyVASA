import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.inference_config import InferenceConfig
from src.runtime import ensure_engine_artifact, resolve_engine_artifact
from src.runtime.trt_plugins import requires_plugin_library
from src.utils.rprint import rlog as log
from build_trt_plugins import PluginBuildResult, build_plugin_library


DEFAULT_CONFIG_PATH = ROOT / "tools" / "config.json"


def internal_specs(cfg: InferenceConfig) -> dict[str, dict]:
    return {
        "human_appearance_feature_extractor": {
            "source_paths": [cfg.checkpoint_F, cfg.models_config],
            "inputs": {"x": [1, 3, 256, 256]},
            "outputs": {"feature_3d": [1, 32, 16, 64, 64]},
        },
        "human_motion_extractor": {
            "source_paths": [cfg.checkpoint_M, cfg.models_config],
            "inputs": {"x": [1, 3, 256, 256]},
            "outputs": {},
        },
        "human_warping_module": {
            "source_paths": [cfg.checkpoint_W, cfg.models_config],
            "inputs": {},
            "outputs": {},
        },
        "human_spade_generator": {
            "source_paths": [cfg.checkpoint_G, cfg.models_config],
            "inputs": {"feature": [1, 256, 64, 64]},
            "outputs": {"image": [1, 3, 512, 512]},
        },
        "human_stitching": {
            "source_paths": [cfg.checkpoint_S, cfg.models_config],
            "inputs": {"feat": [1, 126]},
            "outputs": {"delta": [1, 65]},
        },
        "human_lip": {
            "source_paths": [cfg.checkpoint_S, cfg.models_config],
            "inputs": {"feat": [1, 65]},
            "outputs": {"delta": [1, 63]},
        },
        "human_eye": {
            "source_paths": [cfg.checkpoint_S, cfg.models_config],
            "inputs": {"feat": [1, 66]},
            "outputs": {"delta": [1, 63]},
        },
        "motion_audio_feature": {
            "source_paths": [cfg.checkpoint_MotionGenerator],
            "inputs": {"audio": [1, 64000]},
            "outputs": {},
        },
        "motion_denoiser": {
            "source_paths": [cfg.checkpoint_MotionGenerator],
            "inputs": {},
            "outputs": {},
        },
    }


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def resolve_output_path(raw_path: str, config_dir: Path) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return str(path)


def load_build_config(config_path: Path, supported_models: set[str]) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    directories = payload.get("directories", {})
    precision = payload.get("precision", {})
    build = payload.get("build", {})

    fp16_models = precision.get("fp16", [])
    fp32_models = precision.get("fp32", [])
    unknown = sorted((set(fp16_models) | set(fp32_models)) - supported_models)
    if unknown:
        raise ValueError(f"Unknown model names in config: {unknown}")

    return {
        "onnx_root": resolve_output_path(directories["onnx_root"], config_path.parent),
        "engine_root": resolve_output_path(directories["engine_root"], config_path.parent),
        "plugin_root": resolve_output_path(
            directories.get("plugin_root", "../pretrained_weights/trt_artifacts/plugins"),
            config_path.parent,
        ),
        "fp16_models": fp16_models,
        "fp32_models": fp32_models,
        "force_rebuild": bool(build.get("force_rebuild", False)),
    }


def ensure_internal_onnx(cfg: InferenceConfig, selected_models: set[str], onnx_root: str, engine_root: str, force: bool) -> None:
    missing = []
    for name in selected_models:
        artifact = resolve_engine_artifact(None, name, "fp16", onnx_root=onnx_root, engine_root=engine_root)
        if not Path(artifact.onnx_path).exists():
            missing.append(name)

    if not missing:
        log("All requested ONNX artifacts already exist; skipping export step.")
        return

    log(f"Missing ONNX artifacts detected: {missing}")
    try:
        import yaml
        from export_onnx import export_liveportrait_stack, export_motion_stack
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch-based ONNX export is required for missing artifacts, but the export dependencies are unavailable."
        ) from exc

    model_config = yaml.load(open(cfg.models_config, "r"), Loader=yaml.SafeLoader)
    export_liveportrait_stack(
        "human",
        cfg,
        model_config,
        "cuda:0",
        animal=False,
        force=force,
        selected_models=selected_models,
        onnx_root=onnx_root,
        engine_root=engine_root,
    )
    export_motion_stack(
        cfg,
        model_config,
        "cuda:0",
        force=force,
        selected_models=selected_models,
        onnx_root=onnx_root,
        engine_root=engine_root,
    )


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for the human JoyVASA stack from tools/config.json.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = InferenceConfig()
    specs = internal_specs(cfg)
    config_path = resolve_config_path(args.config)
    build_cfg = load_build_config(config_path, set(specs.keys()))
    force_rebuild = args.force or build_cfg["force_rebuild"]

    selected_models = set(build_cfg["fp16_models"]) | set(build_cfg["fp32_models"])
    if not selected_models:
        log("No models were selected for building in tools/config.json.")
        return

    plugin_build: PluginBuildResult | None = None
    if any(requires_plugin_library(name) for name in selected_models):
        plugin_build = build_plugin_library(build_cfg["plugin_root"], force=force_rebuild)
        log(f"Ready plugin library: {plugin_build.library_path}")

    ensure_internal_onnx(
        cfg=cfg,
        selected_models=selected_models,
        onnx_root=build_cfg["onnx_root"],
        engine_root=build_cfg["engine_root"],
        force=force_rebuild,
    )

    for precision, model_names in (("fp16", build_cfg["fp16_models"]), ("fp32", build_cfg["fp32_models"])):
        for name in model_names:
            spec = specs[name]
            try:
                artifact = ensure_engine_artifact(
                    root=None,
                    name=name,
                    precision=precision,
                    source_paths=spec["source_paths"],
                    inputs=spec["inputs"],
                    outputs=spec["outputs"],
                    force_rebuild=force_rebuild,
                    onnx_root=build_cfg["onnx_root"],
                    engine_root=build_cfg["engine_root"],
                    plugin_libraries=[plugin_build.library_path] if requires_plugin_library(name) and plugin_build else None,
                    plugin_build_id=plugin_build.build_id if requires_plugin_library(name) and plugin_build else "",
                )
                if Path(artifact.engine_path).exists():
                    log(f"Ready ({precision}): {artifact.engine_path}")
                else:
                    log(f"Skipped {name}: ONNX artifact not found.")
            except Exception as exc:
                log(f"Failed to build {name} ({precision}): {exc}")


if __name__ == "__main__":
    main()
