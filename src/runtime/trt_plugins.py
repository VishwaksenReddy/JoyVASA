import hashlib
import json
import os
import threading
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - optional dependency
    trt = None


PLUGIN_OP_TYPE = "JoyVASAGridSample3D"
PLUGIN_NAMESPACE = "com.joyvasa"
PLUGIN_VERSION = "1"
PLUGIN_LIBRARY_BASENAME = "joyvasa_trt_plugins"
PLUGIN_BUILD_INFO_SUFFIX = ".build.json"

_LOADED_PLUGIN_HANDLES: dict[str, object] = {}
_LOAD_LOCK = threading.Lock()


def plugin_library_filename() -> str:
    if os.name == "nt":
        return f"{PLUGIN_LIBRARY_BASENAME}.dll"
    return f"lib{PLUGIN_LIBRARY_BASENAME}.so"


def plugin_build_info_filename() -> str:
    return f"{PLUGIN_LIBRARY_BASENAME}{PLUGIN_BUILD_INFO_SUFFIX}"


def plugin_build_info_path(plugin_library: str) -> str:
    path = Path(plugin_library).resolve()
    return str(path.with_name(plugin_build_info_filename()))


def default_plugin_root(engine_root: str | None) -> str | None:
    if not engine_root:
        return None
    return str(Path(engine_root).resolve().parent / "plugins")


def resolve_plugin_library_path(configured_plugin_library: str | None, engine_root: str | None = None) -> str | None:
    if configured_plugin_library:
        candidate = Path(configured_plugin_library)
        if candidate.suffix.lower() not in {".dll", ".so", ".dylib"}:
            candidate = candidate / plugin_library_filename()
        if not candidate.is_absolute():
            search_roots = [Path.cwd()]
            if engine_root:
                search_roots.append(Path(engine_root).resolve().parent)
            for root in search_roots:
                resolved = (root / candidate).resolve()
                if resolved.exists():
                    return str(resolved)
            candidate = (search_roots[-1] / candidate).resolve()
        return str(candidate)

    root = default_plugin_root(engine_root)
    if root is None:
        return None
    return str((Path(root) / plugin_library_filename()).resolve())


def requires_plugin_library(model_name: str) -> bool:
    return model_name.endswith("_warping_module")


def resolve_model_plugin_libraries(
    model_name: str,
    engine_root: str | None,
    configured_plugin_library: str | None = None,
) -> list[str]:
    if not requires_plugin_library(model_name):
        return []
    library_path = resolve_plugin_library_path(configured_plugin_library, engine_root=engine_root)
    if library_path is None:
        return []
    return [library_path]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_plugin_hashes(plugin_libraries: list[str] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for plugin_library in plugin_libraries or []:
        resolved = str(Path(plugin_library).resolve())
        if Path(resolved).exists():
            hashes[resolved] = file_sha256(resolved)
    return hashes


def load_plugin_build_info(plugin_library: str) -> dict:
    info_path = Path(plugin_build_info_path(plugin_library))
    if not info_path.exists():
        return {}
    with open(info_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_plugin_build_id(plugin_libraries: list[str] | None) -> str:
    build_ids: list[str] = []
    for plugin_library in plugin_libraries or []:
        info = load_plugin_build_info(plugin_library)
        build_id = info.get("build_id")
        if build_id:
            build_ids.append(str(build_id))
        elif Path(plugin_library).exists():
            build_ids.append(file_sha256(plugin_library))
    return "|".join(build_ids)


def load_plugin_library(plugin_library: str, logger=None):
    if trt is None:
        raise RuntimeError("TensorRT Python package is not installed.")

    resolved = str(Path(plugin_library).resolve())
    if not Path(resolved).exists():
        raise FileNotFoundError(f"TensorRT plugin library not found: {resolved}")

    with _LOAD_LOCK:
        if resolved in _LOADED_PLUGIN_HANDLES:
            return _LOADED_PLUGIN_HANDLES[resolved]

        trt_logger = logger or trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")
        handle = trt.get_plugin_registry().load_library(resolved)
        if handle is None:
            raise RuntimeError(f"TensorRT failed to load plugin library: {resolved}")
        _LOADED_PLUGIN_HANDLES[resolved] = handle
        return handle


def load_plugin_libraries(plugin_libraries: list[str] | None, logger=None) -> list[object]:
    handles = []
    for plugin_library in plugin_libraries or []:
        handles.append(load_plugin_library(plugin_library, logger=logger))
    return handles
