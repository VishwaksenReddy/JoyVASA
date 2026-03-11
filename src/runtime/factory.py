import os.path as osp
from collections.abc import Callable
from typing import Any

from ..utils.rprint import rlog as log
from .engine_utils import backend_prefers_tensorrt, ensure_engine_artifact
from .runners import MotionGeneratorTensorRTRunner, TensorRTRunner, TorchMotionGeneratorRunner, TorchRunner


def create_model_runner(
    *,
    name: str,
    device: str,
    backend: str,
    precision: str,
    engine_root: str,
    force_rebuild: bool,
    loader: Callable[[], Any],
    source_paths: list[str],
    inputs: dict[str, list[int] | None] | None = None,
    outputs: dict[str, list[int] | None] | None = None,
):
    if backend_prefers_tensorrt(backend, device):
        try:
            artifact = ensure_engine_artifact(
                root=engine_root,
                name=name,
                precision=precision,
                source_paths=source_paths,
                inputs=inputs,
                outputs=outputs,
                force_rebuild=force_rebuild,
            )
            if osp.exists(artifact.engine_path):
                log(f"[TensorRT] Using engine for {name}: {artifact.engine_path}")
                return TensorRTRunner(artifact.engine_path, device=device, name=name)
            log(f"[TensorRT] ONNX/engine missing for {name}, falling back to PyTorch.")
        except Exception as exc:  # pragma: no cover - exercised in integration
            log(f"[TensorRT] Failed to initialize {name}: {exc}. Falling back to PyTorch.")

    return TorchRunner(loader(), name=name)


def create_motion_generator_runner(
    *,
    model,
    device: str,
    backend: str,
    precision: str,
    engine_root: str,
    force_rebuild: bool,
    source_paths: list[str],
):
    if backend_prefers_tensorrt(backend, device):
        try:
            audio_artifact = ensure_engine_artifact(
                root=engine_root,
                name="motion_audio_feature",
                precision=precision,
                source_paths=source_paths,
                inputs={"audio": [1, 64000]},
                outputs={"audio_feat": [1, model.n_motions, model.feature_dim]},
                force_rebuild=force_rebuild,
            )
            denoiser_artifact = ensure_engine_artifact(
                root=engine_root,
                name="motion_denoiser",
                precision=precision,
                source_paths=source_paths,
                inputs={
                    "motion_feat": [1, model.n_motions, model.motion_feat_dim],
                    "audio_feat": [1, model.n_motions, model.feature_dim],
                    "prev_motion_feat": [1, model.n_prev_motions, model.motion_feat_dim],
                    "prev_audio_feat": [1, model.n_prev_motions, model.feature_dim],
                    "step": [1],
                    "indicator": [1, model.n_motions],
                },
                outputs={"motion_feat_target": [1, model.n_prev_motions + model.n_motions, model.motion_feat_dim]},
                force_rebuild=force_rebuild,
            )
            if osp.exists(audio_artifact.engine_path) and osp.exists(denoiser_artifact.engine_path):
                log(f"[TensorRT] Using engines for motion generator from {engine_root}")
                return MotionGeneratorTensorRTRunner(
                    model=model,
                    audio_feature_runner=TensorRTRunner(audio_artifact.engine_path, device=device, name="motion_audio_feature"),
                    denoiser_runner=TensorRTRunner(denoiser_artifact.engine_path, device=device, name="motion_denoiser"),
                )
            log("[TensorRT] Motion generator engines are incomplete, falling back to PyTorch.")
        except Exception as exc:  # pragma: no cover - exercised in integration
            log(f"[TensorRT] Failed to initialize motion generator: {exc}. Falling back to PyTorch.")

    return TorchMotionGeneratorRunner(model)
