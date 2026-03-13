import json
import os
import os.path as osp
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .trt_plugins import file_sha256 as _file_sha256
from .trt_plugins import get_plugin_hashes, resolve_plugin_build_id


DEFAULT_ONNX_OPSET = 17


@dataclass(frozen=True)
class EngineArtifactPaths:
    name: str
    onnx_path: str
    engine_path: str
    manifest_path: str


@dataclass
class EngineManifest:
    name: str
    precision: str
    onnx_sha256: str
    source_sha256: dict[str, str]
    inputs: dict[str, list[int] | None]
    outputs: dict[str, list[int] | None]
    builder: str
    plugin_libraries: dict[str, str] = field(default_factory=dict)
    plugin_build_id: str = ""


def backend_prefers_tensorrt(backend: str, device: str) -> bool:
    return backend != "pytorch" and device.startswith("cuda")


def _ensure_parent(path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)


def file_sha256(path: str) -> str:
    return _file_sha256(path)


def resolve_engine_artifact(
    root: str | None,
    name: str,
    precision: str,
    onnx_root: str | None = None,
    engine_root: str | None = None,
) -> EngineArtifactPaths:
    base_root = root or engine_root or onnx_root
    if base_root is None:
        raise ValueError("At least one artifact root must be provided.")

    resolved_onnx_root = onnx_root or base_root
    resolved_engine_root = engine_root or base_root
    onnx_dir = osp.join(resolved_onnx_root, name)
    engine_dir = osp.join(resolved_engine_root, name)
    onnx_path = osp.join(onnx_dir, f"{name}.onnx")
    engine_path = osp.join(engine_dir, f"{name}.{precision}.engine")
    manifest_path = osp.join(engine_dir, f"{name}.{precision}.json")
    return EngineArtifactPaths(name=name, onnx_path=onnx_path, engine_path=engine_path, manifest_path=manifest_path)


def load_manifest(path: str) -> EngineManifest | None:
    if not osp.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EngineManifest(**payload)


def save_manifest(path: str, manifest: EngineManifest) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, sort_keys=True)


def _source_hashes(source_paths: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source_path in source_paths:
        if osp.exists(source_path):
            hashes[osp.abspath(source_path)] = file_sha256(source_path)
    return hashes


def _manifest_matches(
    manifest: EngineManifest | None,
    precision: str,
    onnx_path: str,
    source_paths: list[str],
    plugin_libraries: list[str] | None = None,
    plugin_build_id: str = "",
) -> bool:
    if manifest is None or manifest.precision != precision:
        return False
    if not osp.exists(onnx_path):
        return False
    if manifest.onnx_sha256 != file_sha256(onnx_path):
        return False
    if manifest.source_sha256 != _source_hashes(source_paths):
        return False
    if manifest.plugin_libraries != get_plugin_hashes(plugin_libraries):
        return False
    return manifest.plugin_build_id == (plugin_build_id or resolve_plugin_build_id(plugin_libraries))


def _detect_trtexec() -> str | None:
    discovered = shutil.which("trtexec")
    if discovered:
        return discovered

    for root in [
        os.environ.get("TENSORRT_ROOT"),
        os.environ.get("TensorRT_ROOT"),
    ]:
        if not root:
            continue
        candidate = Path(root) / "bin" / ("trtexec.exe" if os.name == "nt" else "trtexec")
        if candidate.exists():
            return str(candidate)
    return None


def _builder_label(trtexec_path: str | None) -> str:
    if trtexec_path is None:
        return "missing"
    try:
        result = subprocess.run(
            [trtexec_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    version_text = result.stdout.strip() or result.stderr.strip()
    return version_text or "unknown"


def build_engine_from_onnx(
    artifact: EngineArtifactPaths,
    precision: str,
    source_paths: list[str],
    inputs: dict[str, list[int] | None] | None = None,
    outputs: dict[str, list[int] | None] | None = None,
    force_rebuild: bool = False,
    plugin_libraries: list[str] | None = None,
    plugin_build_id: str = "",
) -> str:
    trtexec_path = _detect_trtexec()
    if trtexec_path is None:
        raise FileNotFoundError("TensorRT builder 'trtexec' was not found on PATH.")
    if not osp.exists(artifact.onnx_path):
        raise FileNotFoundError(f"ONNX artifact not found: {artifact.onnx_path}")
    for plugin_library in plugin_libraries or []:
        if not osp.exists(plugin_library):
            raise FileNotFoundError(f"TensorRT plugin library not found: {plugin_library}")

    manifest = load_manifest(artifact.manifest_path)
    resolved_plugin_build_id = plugin_build_id or resolve_plugin_build_id(plugin_libraries)
    if not force_rebuild and osp.exists(artifact.engine_path) and _manifest_matches(
        manifest,
        precision,
        artifact.onnx_path,
        source_paths,
        plugin_libraries=plugin_libraries,
        plugin_build_id=resolved_plugin_build_id,
    ):
        return artifact.engine_path

    _ensure_parent(artifact.engine_path)
    command = [
        trtexec_path,
        f"--onnx={artifact.onnx_path}",
        f"--saveEngine={artifact.engine_path}",
        "--skipInference",
        "--builderOptimizationLevel=3",
    ]
    if precision == "fp16":
        command.append("--fp16")
    for plugin_library in plugin_libraries or []:
        command.append(f"--dynamicPlugins={plugin_library}")
        command.append(f"--setPluginsToSerialize={plugin_library}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "TensorRT engine build failed.")

    save_manifest(
        artifact.manifest_path,
        EngineManifest(
            name=artifact.name,
            precision=precision,
            onnx_sha256=file_sha256(artifact.onnx_path),
            source_sha256=_source_hashes(source_paths),
            inputs=inputs or {},
            outputs=outputs or {},
            builder=_builder_label(trtexec_path),
            plugin_libraries=get_plugin_hashes(plugin_libraries),
            plugin_build_id=resolved_plugin_build_id,
        ),
    )
    return artifact.engine_path


def ensure_engine_artifact(
    root: str | None,
    name: str,
    precision: str,
    source_paths: list[str],
    inputs: dict[str, list[int] | None] | None = None,
    outputs: dict[str, list[int] | None] | None = None,
    force_rebuild: bool = False,
    onnx_root: str | None = None,
    engine_root: str | None = None,
    plugin_libraries: list[str] | None = None,
    plugin_build_id: str = "",
) -> EngineArtifactPaths:
    artifact = resolve_engine_artifact(root, name, precision, onnx_root=onnx_root, engine_root=engine_root)
    if osp.exists(artifact.onnx_path):
        build_engine_from_onnx(
            artifact,
            precision=precision,
            source_paths=source_paths,
            inputs=inputs,
            outputs=outputs,
            force_rebuild=force_rebuild,
            plugin_libraries=plugin_libraries,
            plugin_build_id=plugin_build_id,
        )
    return artifact


def get_ort_provider_bundle(
    backend: str,
    device: str,
    device_id: int,
    precision: str,
    cache_root: str,
    include_tensorrt: bool = True,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    os.makedirs(cache_root, exist_ok=True)

    providers: list[str]
    provider_options: list[dict[str, Any]]
    if include_tensorrt and backend_prefers_tensorrt(backend, device):
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {
                "device_id": device_id,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_root,
                "trt_fp16_enable": precision == "fp16",
                "trt_timing_cache_enable": True,
            },
            {"device_id": device_id},
            {},
        ]
    elif device.startswith("cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{"device_id": device_id}, {}]
    elif device == "mps":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{}, {}]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]

    session_options = {
        "graph_optimization_level": "ORT_ENABLE_ALL",
    }
    return providers, provider_options, session_options
