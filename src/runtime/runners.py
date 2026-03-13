from collections.abc import Callable
from typing import Any

import numpy as np
import torch

try:
    import onnxruntime
except ImportError:  # pragma: no cover - optional dependency
    onnxruntime = None

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - optional dependency
    trt = None

from .engine_utils import get_ort_provider_bundle
from .trt_plugins import load_plugin_libraries


def _torch_dtype_from_trt(dtype) -> torch.dtype:
    if trt is None:
        raise RuntimeError("TensorRT is not available.")
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    return mapping[dtype]


class TorchRunner:
    backend_label = "pytorch"

    def __init__(self, model: Callable[..., Any], name: str):
        self.model = model
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class TorchMotionGeneratorRunner:
    backend_label = "pytorch"

    def __init__(self, model):
        self.model = model
        self.name = "motion_generator"
        self.n_motions = model.n_motions
        self.n_prev_motions = model.n_prev_motions
        self.fps = model.fps

    @property
    def device(self):
        return self.model.device

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)


class TensorRTRunner:
    backend_label = "tensorrt"

    def __init__(self, engine_path: str, device: str, name: str, plugin_libraries: list[str] | None = None):
        if trt is None:
            raise RuntimeError("TensorRT Python package is not installed.")
        if not device.startswith("cuda"):
            raise RuntimeError("TensorRTRunner requires a CUDA device.")

        self.engine_path = engine_path
        self.device = torch.device(device)
        self.name = name
        self.plugin_libraries = plugin_libraries or []
        self.logger = trt.Logger(trt.Logger.WARNING)
        if self.plugin_libraries:
            load_plugin_libraries(self.plugin_libraries, logger=self.logger)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as handle:
            engine_bytes = handle.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {engine_path}")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)

    def _collect_inputs(self, *args, **kwargs) -> dict[str, Any]:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return dict(args[0])

        if len(args) > len(self.input_names):
            raise ValueError(f"{self.name} received {len(args)} positional inputs, expected at most {len(self.input_names)}.")

        raw_inputs = dict(zip(self.input_names, args))
        raw_inputs.update(kwargs)
        return raw_inputs

    def _prepare_inputs(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        raw_inputs = self._collect_inputs(*args, **kwargs)

        tensors: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            value = raw_inputs[name]
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device)
            else:
                tensor = torch.as_tensor(value, device=self.device)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            expected_dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            if -1 in tuple(self.engine.get_tensor_shape(name)):
                self.context.set_input_shape(name, tuple(tensor.shape))
            tensors[name] = tensor
        return tensors

    def __call__(self, *args, **kwargs):
        tensors = self._prepare_inputs(*args, **kwargs)
        outputs: dict[str, torch.Tensor] = {}

        for name, tensor in tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            output = torch.empty(shape, dtype=dtype, device=self.device)
            outputs[name] = output
            self.context.set_tensor_address(name, output.data_ptr())

        stream = torch.cuda.current_stream(device=self.device).cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError(f"TensorRT execution failed for {self.name}.")

        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs


class OrtTensorRTRunner:
    backend_label = "onnxruntime"

    def __init__(
        self,
        model_path: str,
        backend: str,
        device: str,
        device_id: int,
        precision: str,
        cache_root: str,
        name: str,
    ):
        if onnxruntime is None:
            raise RuntimeError("onnxruntime is not installed.")

        providers, provider_options, session_options = get_ort_provider_bundle(
            backend=backend,
            device=device,
            device_id=device_id,
            precision=precision,
            cache_root=cache_root,
        )
        opts = onnxruntime.SessionOptions()
        graph_level = session_options.get("graph_optimization_level", "ORT_ENABLE_ALL")
        opts.graph_optimization_level = getattr(onnxruntime.GraphOptimizationLevel, graph_level)
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=opts,
        )
        self.name = name
        self.model_path = model_path
        self.input_names = [item.name for item in self.session.get_inputs()]

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            feed_dict = dict(args[0])
        else:
            if len(args) > len(self.input_names):
                raise ValueError(f"{self.name} received {len(args)} positional inputs, expected at most {len(self.input_names)}.")
            feed_dict = dict(zip(self.input_names, args))
            feed_dict.update(kwargs)

        normalized: dict[str, np.ndarray] = {}
        for name in self.input_names:
            value = feed_dict[name]
            if isinstance(value, torch.Tensor):
                normalized[name] = value.detach().cpu().numpy()
            else:
                normalized[name] = np.asarray(value)
        return self.session.run(None, normalized)


class MotionGeneratorTensorRTRunner:
    backend_label = "tensorrt"

    def __init__(self, model, audio_feature_runner: TensorRTRunner, denoiser_runner: TensorRTRunner):
        self.name = "motion_generator"
        self.device = next(model.parameters()).device
        self.target = model.target
        self.motion_feat_dim = model.motion_feat_dim
        self.fps = model.fps
        self.n_motions = model.n_motions
        self.n_prev_motions = model.n_prev_motions
        self.cfg_mode = model.cfg_mode
        self.guiding_conditions = list(model.guiding_conditions)
        self.start_motion_feat = model.start_motion_feat.detach().clone().to(self.device)
        self.start_audio_feat = model.start_audio_feat.detach().clone().to(self.device)
        self.null_audio_feat = None
        if hasattr(model, "null_audio_feat"):
            self.null_audio_feat = model.null_audio_feat.detach().clone().to(self.device)
        self.diffusion_sched = model.diffusion_sched
        self.audio_feature_runner = audio_feature_runner
        self.denoiser_runner = denoiser_runner

    def extract_audio_feature(self, audio):
        return self.audio_feature_runner(audio)

    def _run_denoiser(
        self,
        motion_feat,
        audio_feat,
        prev_motion_feat,
        prev_audio_feat,
        step,
        indicator,
    ):
        feed_dict = {
            "motion_feat": motion_feat,
            "audio_feat": audio_feat,
            "prev_motion_feat": prev_motion_feat,
            "prev_audio_feat": prev_audio_feat,
            "step": step,
        }
        if "indicator" in self.denoiser_runner.input_names:
            feed_dict["indicator"] = indicator
        return self.denoiser_runner(feed_dict)

    def _run_denoiser_entries(
        self,
        motion_in,
        audio_feat_in,
        prev_motion_feat_in,
        prev_audio_feat_in,
        step_in,
        indicator_tensor,
        batch_size,
        n_entries,
    ):
        if n_entries == 1:
            return self._run_denoiser(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_tensor,
            )

        outputs = []
        for index in range(n_entries):
            start = index * batch_size
            end = start + batch_size
            outputs.append(
                self._run_denoiser(
                    motion_in[start:end],
                    audio_feat_in[start:end],
                    prev_motion_feat_in[start:end],
                    prev_audio_feat_in[start:end],
                    step_in[start:end],
                    indicator_tensor[start:end],
                )
            )
        return torch.cat(outputs, dim=0)

    def sample(
        self,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=None,
        cfg_mode=None,
        cfg_cond=None,
        cfg_scale=1.15,
        flexibility=0,
        dynamic_threshold=None,
        ret_traj=False,
    ):
        batch_size = audio_or_feat.shape[0]

        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        cfg_cond = [cond for cond in cfg_cond if cond in ["audio"]]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda item: ["audio"].index(item[0])))
        else:
            cfg_cond, cfg_scale = [], []

        if audio_or_feat.ndim == 2:
            audio_feat = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            audio_feat = audio_or_feat
        else:
            raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)
        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim), device=self.device)

        if "audio" in cfg_cond and self.null_audio_feat is not None:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        audio_feat_in = [audio_feat_null]
        for cond in cfg_cond:
            if cond == "audio":
                audio_feat_in.append(audio_feat)

        n_entries = len(audio_feat_in)
        audio_feat_in = torch.cat(audio_feat_in, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)
            if indicator_in is None:
                indicator_tensor = torch.zeros((motion_in.shape[0], self.n_motions), device=self.device)
            else:
                indicator_tensor = indicator_in

            results = self._run_denoiser_entries(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_tensor,
                batch_size=batch_size,
                n_entries=n_entries,
            )

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)
            target_theta = results[0][:, -self.n_motions:]
            for index in range(0, n_entries - 1):
                if cfg_mode == "independent":
                    target_theta += cfg_scale[index] * (results[index + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])
                elif cfg_mode == "incremental":
                    target_theta += cfg_scale[index] * (results[index + 1][:, -self.n_motions:] - results[index][:, -self.n_motions:])
                else:
                    raise NotImplementedError(f"Unknown cfg_mode {cfg_mode}")

            if self.target == "noise":
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == "sample":
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError(f"Unknown target type: {self.target}")

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        return traj[0], motion_at_T, audio_feat
