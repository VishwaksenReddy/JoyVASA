import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.inference_config import InferenceConfig
from src.runtime import DEFAULT_ONNX_OPSET, resolve_engine_artifact
from src.runtime.export_wrappers import (
    MotionAudioFeatureExportWrapper,
    MotionDenoiserExportWrapper,
    MotionExtractorExportWrapper,
    StitchingHeadExportWrapper,
    WarpingModuleExportWrapper,
)
from src.utils.helper import load_model
from src.utils.rprint import rlog as log


def export_module(module, onnx_path: Path, sample_inputs: dict[str, torch.Tensor], output_names: list[str], force: bool = False) -> bool:
    if onnx_path.exists() and not force:
        log(f"Skipping ONNX export because artifact already exists: {onnx_path}")
        return False
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            tuple(sample_inputs[name] for name in sample_inputs),
            str(onnx_path),
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=list(sample_inputs.keys()),
            output_names=output_names,
        )
    log(f"Exported ONNX: {onnx_path}")
    return True


def export_liveportrait_stack(
    prefix: str,
    cfg: InferenceConfig,
    model_config: dict,
    device: str,
    animal: bool,
    force: bool = False,
    selected_models: set[str] | None = None,
    onnx_root: str | None = None,
    engine_root: str | None = None,
) -> None:
    suffix = "_animal" if animal else ""
    checkpoint_map = {
        "appearance_feature_extractor": getattr(cfg, f"checkpoint_F{suffix}"),
        "motion_extractor": getattr(cfg, f"checkpoint_M{suffix}"),
        "warping_module": getattr(cfg, f"checkpoint_W{suffix}"),
        "spade_generator": getattr(cfg, f"checkpoint_G{suffix}"),
    }
    if not all(Path(path).exists() for path in checkpoint_map.values()):
        log(f"Skipping {prefix} exports because one or more checkpoints are missing.")
        return

    num_kp = model_config["model_params"]["motion_extractor_params"]["num_kp"]
    spade_upscale = model_config["model_params"]["spade_generator_params"].get("upscale", 1)
    spade_output = 256 * max(1, spade_upscale)

    appearance_name = f"{prefix}_appearance_feature_extractor"
    if selected_models is None or appearance_name in selected_models:
        appearance = load_model(checkpoint_map["appearance_feature_extractor"], model_config, device, "appearance_feature_extractor")
        export_module(
            appearance,
            Path(resolve_engine_artifact(cfg.trt_engine_root, appearance_name, cfg.trt_precision, onnx_root=onnx_root, engine_root=engine_root).onnx_path),
            {"x": torch.randn(1, 3, 256, 256, device=device)},
            ["feature_3d"],
            force=force,
        )

    motion_name = f"{prefix}_motion_extractor"
    if selected_models is None or motion_name in selected_models:
        motion = MotionExtractorExportWrapper(load_model(checkpoint_map["motion_extractor"], model_config, device, "motion_extractor"))
        export_module(
            motion,
            Path(resolve_engine_artifact(cfg.trt_engine_root, motion_name, cfg.trt_precision, onnx_root=onnx_root, engine_root=engine_root).onnx_path),
            {"x": torch.randn(1, 3, 256, 256, device=device)},
            motion.output_names,
            force=force,
        )

    warping_name = f"{prefix}_warping_module"
    if selected_models is None or warping_name in selected_models:
        warping = WarpingModuleExportWrapper(load_model(checkpoint_map["warping_module"], model_config, device, "warping_module"))
        export_module(
            warping,
            Path(resolve_engine_artifact(cfg.trt_engine_root, warping_name, cfg.trt_precision, onnx_root=onnx_root, engine_root=engine_root).onnx_path),
            {
                "feature_3d": torch.randn(1, 32, 16, 64, 64, device=device),
                "kp_driving": torch.randn(1, num_kp, 3, device=device),
                "kp_source": torch.randn(1, num_kp, 3, device=device),
            },
            warping.output_names,
            force=force,
        )

    spade_name = f"{prefix}_spade_generator"
    if selected_models is None or spade_name in selected_models:
        spade = load_model(checkpoint_map["spade_generator"], model_config, device, "spade_generator")
        export_module(
            spade,
            Path(resolve_engine_artifact(cfg.trt_engine_root, spade_name, cfg.trt_precision, onnx_root=onnx_root, engine_root=engine_root).onnx_path),
            {"feature": torch.randn(1, 256, 64, 64, device=device)},
            ["image"],
            force=force,
        )

    stitching_ckpt = getattr(cfg, f"checkpoint_S{suffix}")
    if Path(stitching_ckpt).exists():
        stitching_modules = load_model(stitching_ckpt, model_config, device, "stitching_retargeting_module")
        shapes = model_config["model_params"]["stitching_retargeting_module_params"]
        for head_name, head in stitching_modules.items():
            artifact_name = f"{prefix}_{head_name}"
            if selected_models is not None and artifact_name not in selected_models:
                continue
            export_module(
                StitchingHeadExportWrapper(head),
                Path(resolve_engine_artifact(cfg.trt_engine_root, artifact_name, cfg.trt_precision, onnx_root=onnx_root, engine_root=engine_root).onnx_path),
                {"feat": torch.randn(1, shapes[head_name]["input_size"], device=device)},
                ["delta"],
                force=force,
            )
    else:
        log(f"Skipping {prefix} stitching exports because {stitching_ckpt} is missing.")

    log(f"{prefix} stack exported; expected spade output size is {spade_output}px.")


def export_motion_stack(
    cfg: InferenceConfig,
    model_config: dict,
    device: str,
    force: bool = False,
    selected_models: set[str] | None = None,
    onnx_root: str | None = None,
    engine_root: str | None = None,
) -> None:
    if not Path(cfg.checkpoint_MotionGenerator).exists():
        log("Skipping motion exports because motion generator checkpoint is missing.")
        return

    needs_audio = selected_models is None or "motion_audio_feature" in selected_models
    needs_denoiser = selected_models is None or "motion_denoiser" in selected_models
    if not needs_audio and not needs_denoiser:
        return

    motion_model, motion_args = load_model(cfg.checkpoint_MotionGenerator, model_config, device, "motion_generator")
    if needs_audio:
        audio_wrapper = MotionAudioFeatureExportWrapper(motion_model)
        export_module(
            audio_wrapper,
            Path(resolve_engine_artifact(cfg.trt_engine_root, "motion_audio_feature", cfg.trt_precision, onnx_root=onnx_root, engine_root=engine_root).onnx_path),
            {"audio": torch.randn(1, 64000, device=device)},
            audio_wrapper.output_names,
            force=force,
        )

    if needs_denoiser:
        denoiser_wrapper = MotionDenoiserExportWrapper(motion_model)
        export_module(
            denoiser_wrapper,
            Path(resolve_engine_artifact(cfg.trt_engine_root, "motion_denoiser", cfg.trt_precision, onnx_root=onnx_root, engine_root=engine_root).onnx_path),
            {
                "motion_feat": torch.randn(1, motion_args.n_motions, motion_model.motion_feat_dim, device=device),
                "audio_feat": torch.randn(1, motion_args.n_motions, motion_model.feature_dim, device=device),
                "prev_motion_feat": torch.randn(1, motion_args.n_prev_motions, motion_model.motion_feat_dim, device=device),
                "prev_audio_feat": torch.randn(1, motion_args.n_prev_motions, motion_model.feature_dim, device=device),
                "step": torch.ones(1, dtype=torch.int32, device=device),
                "indicator": torch.ones(1, motion_args.n_motions, device=device),
            },
            denoiser_wrapper.output_names,
            force=force,
        )


def main():
    parser = argparse.ArgumentParser(description="Export JoyVASA modules to ONNX.")
    parser.add_argument("--scope", choices=["human", "animal", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--engine-root", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = InferenceConfig()
    if args.engine_root:
        cfg.trt_engine_root = args.engine_root
    model_config = yaml.load(open(cfg.models_config, "r"), Loader=yaml.SafeLoader)

    if args.scope in {"human", "all"}:
        export_liveportrait_stack("human", cfg, model_config, args.device, animal=False, force=args.force)
    if args.scope in {"animal", "all"}:
        export_liveportrait_stack("animal", cfg, model_config, args.device, animal=True, force=args.force)
    if args.scope in {"human", "all"}:
        export_motion_stack(cfg, model_config, args.device, force=args.force)


if __name__ == "__main__":
    main()
