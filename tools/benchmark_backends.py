import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.argument_config import ArgumentConfig
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_wmg_wrapper import LivePortraitWrapper
from src.utils.io import load_image_rgb, resize_to_limit
from src.utils.rprint import rlog as log


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_run(fn, device: str, iterations: int):
    samples = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return result, float(np.mean(samples))


def tensor_max_abs(lhs, rhs):
    if isinstance(lhs, dict):
        return {key: tensor_max_abs(lhs[key], rhs[key]) for key in lhs}
    if isinstance(lhs, torch.Tensor):
        return float((lhs.float() - rhs.float()).abs().max().item())
    return float(np.max(np.abs(lhs - rhs)))


def build_wrapper(backend: str) -> LivePortraitWrapper:
    cfg = InferenceConfig()
    cfg.backend = backend
    cfg.trt_precision = "fp16"
    return LivePortraitWrapper(cfg)


def benchmark_human(reference: str, audio: str | None, iterations: int, include_motion: bool) -> None:
    torch_wrapper = build_wrapper("pytorch")
    trt_wrapper = build_wrapper("tensorrt")

    image = load_image_rgb(reference)
    image = resize_to_limit(image, torch_wrapper.inference_cfg.source_max_dim, torch_wrapper.inference_cfg.source_division)
    image = cv2.resize(image, (256, 256))
    source_torch = torch_wrapper.prepare_source(image)
    source_trt = trt_wrapper.prepare_source(image)

    feature_torch, feature_torch_time = timed_run(lambda: torch_wrapper.extract_feature_3d(source_torch), torch_wrapper.device, iterations)
    feature_trt, feature_trt_time = timed_run(lambda: trt_wrapper.extract_feature_3d(source_trt), trt_wrapper.device, iterations)
    kp_torch, kp_torch_time = timed_run(lambda: torch_wrapper.get_kp_info(source_torch), torch_wrapper.device, iterations)
    kp_trt, kp_trt_time = timed_run(lambda: trt_wrapper.get_kp_info(source_trt), trt_wrapper.device, iterations)

    x_s_torch = torch_wrapper.transform_keypoint(kp_torch)
    x_s_trt = trt_wrapper.transform_keypoint(kp_trt)
    render_torch, render_torch_time = timed_run(lambda: torch_wrapper.warp_decode(feature_torch, x_s_torch, x_s_torch), torch_wrapper.device, iterations)
    render_trt, render_trt_time = timed_run(lambda: trt_wrapper.warp_decode(feature_trt, x_s_trt, x_s_trt), trt_wrapper.device, iterations)

    log(f"feature_3d avg: pytorch={feature_torch_time:.4f}s tensorrt={feature_trt_time:.4f}s drift={tensor_max_abs(feature_torch, feature_trt)}")
    log(f"kp_info avg: pytorch={kp_torch_time:.4f}s tensorrt={kp_trt_time:.4f}s drift={tensor_max_abs(kp_torch, kp_trt)}")
    log(f"warp_decode avg: pytorch={render_torch_time:.4f}s tensorrt={render_trt_time:.4f}s drift={tensor_max_abs(render_torch, render_trt)}")
    log(f"Active backends pytorch={torch_wrapper.active_backends}")
    log(f"Active backends tensorrt={trt_wrapper.active_backends}")

    if include_motion and audio:
        args = ArgumentConfig(reference=reference, audio=audio, output_dir=str(ROOT / "animations"))
        _, motion_torch_time = timed_run(lambda: torch_wrapper.gen_motion_sequence(args), torch_wrapper.device, 1)
        _, motion_trt_time = timed_run(lambda: trt_wrapper.gen_motion_sequence(args), trt_wrapper.device, 1)
        log(f"motion_sequence avg: pytorch={motion_torch_time:.4f}s tensorrt={motion_trt_time:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark JoyVASA PyTorch vs TensorRT backends.")
    parser.add_argument("--reference", default=str(ROOT / "assets" / "examples" / "imgs" / "joyvasa_003.png"))
    parser.add_argument("--audio", default=str(ROOT / "assets" / "examples" / "audios" / "joyvasa_003.wav"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--include-motion", action="store_true")
    args = parser.parse_args()
    benchmark_human(args.reference, args.audio, args.iterations, args.include_motion)


if __name__ == "__main__":
    main()
