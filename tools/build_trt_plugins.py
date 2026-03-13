import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.trt_plugins import (  # noqa: E402
    PLUGIN_LIBRARY_BASENAME,
    default_plugin_root,
    plugin_build_info_filename,
    plugin_library_filename,
)


PLUGIN_SOURCE_ROOT = ROOT / "native" / "trt_plugins" / "joyvasa_grid_sample3d"


@dataclass(frozen=True)
class PluginBuildResult:
    library_path: str
    build_info_path: str
    build_id: str


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_plugin_build_id(source_root: Path, include_dir: Path, library_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(include_dir.resolve()).encode("utf-8"))
    digest.update(str(library_dir.resolve()).encode("utf-8"))
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".cpp", ".cu", ".cuh", ".h", ".hpp", ".txt"} and path.name != "CMakeLists.txt":
            continue
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(_hash_file(path).encode("utf-8"))
    return digest.hexdigest()


def _candidate_tensorrt_roots(explicit_root: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in [explicit_root, os.environ.get("TENSORRT_ROOT"), os.environ.get("TensorRT_ROOT")]:
        if value:
            candidates.append(Path(value))

    trtexec_path = shutil.which("trtexec")
    if trtexec_path:
        trtexec = Path(trtexec_path).resolve()
        candidates.extend([trtexec.parent, trtexec.parent.parent])

    if os.name == "nt":
        candidates.extend(Path("C:/").glob("TensorRT*"))
        candidates.append(Path("C:/Program Files/TensorRT"))
    else:
        candidates.extend(
            [
                Path("/usr"),
                Path("/usr/local/TensorRT"),
                Path("/opt/tensorrt"),
                Path("/opt/nvidia/TensorRT"),
            ]
        )
    return [candidate for candidate in candidates if candidate.exists()]


def _candidate_cuda_roots(explicit_root: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in [
        explicit_root,
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDAToolkit_ROOT"),
    ]:
        if value:
            candidates.append(Path(value))

    for key, value in sorted(os.environ.items()):
        if key.startswith("CUDA_PATH_V") and value:
            candidates.append(Path(value))

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        candidates.append(Path(nvcc_path).resolve().parent.parent)

    if os.name == "nt":
        candidates.extend(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*"))
    else:
        candidates.extend([Path("/usr/local/cuda"), Path("/opt/cuda")])

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            unique.append(candidate)
    return unique


def _candidate_include_dirs(root: Path | None, explicit_include_dir: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in [explicit_include_dir, os.environ.get("TRT_INCLUDE_DIR"), os.environ.get("TENSORRT_INCLUDE_DIR")]:
        if value:
            candidates.append(Path(value))
    if root is not None:
        candidates.extend(
            [
                root / "include",
                root / "targets" / "x86_64-linux-gnu" / "include",
            ]
        )
    if os.name != "nt":
        candidates.extend([Path("/usr/include"), Path("/usr/include/x86_64-linux-gnu")])
    return [candidate for candidate in candidates if (candidate / "NvInferRuntime.h").exists()]


def _candidate_library_dirs(root: Path | None, explicit_library_dir: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in [explicit_library_dir, os.environ.get("TRT_LIB_DIR"), os.environ.get("TENSORRT_LIBRARY_DIR")]:
        if value:
            candidates.append(Path(value))
    if root is not None:
        candidates.extend(
            [
                root / "lib",
                root / "lib64",
                root / "lib" / "x64",
                root / "targets" / "x86_64-linux-gnu" / "lib",
            ]
        )
    if os.name != "nt":
        candidates.extend([Path("/usr/lib"), Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/local/lib")])
    if os.name == "nt":
        library_names = ["nvinfer.lib", "nvinfer_10.lib"]
    else:
        library_names = ["libnvinfer.so", "libnvinfer.so.10"]
    return [candidate for candidate in candidates if any((candidate / name).exists() for name in library_names)]


def resolve_tensorrt_sdk(
    tensorrt_root: str | None = None,
    tensorrt_include_dir: str | None = None,
    tensorrt_library_dir: str | None = None,
) -> tuple[Path, Path]:
    for root in _candidate_tensorrt_roots(tensorrt_root) + [None]:
        include_candidates = _candidate_include_dirs(root, tensorrt_include_dir)
        library_candidates = _candidate_library_dirs(root, tensorrt_library_dir)
        if include_candidates and library_candidates:
            return include_candidates[0].resolve(), library_candidates[0].resolve()
    raise FileNotFoundError(
        "TensorRT SDK headers/libraries were not found. Set TENSORRT_ROOT or pass "
        "--tensorrt-root/--tensorrt-include-dir/--tensorrt-library-dir."
    )


def resolve_cuda_sdk(cuda_root: str | None = None, cuda_compiler: str | None = None) -> tuple[Path, Path]:
    compiler_candidates: list[Path] = []
    for value in [cuda_compiler, os.environ.get("CMAKE_CUDA_COMPILER"), os.environ.get("CUDACXX")]:
        if value:
            compiler_candidates.append(Path(value))

    for root in _candidate_cuda_roots(cuda_root):
        compiler_candidates.append(root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"))

    for compiler in compiler_candidates:
        if not compiler.exists():
            continue
        resolved_compiler = compiler.resolve()
        resolved_root = resolved_compiler.parent.parent
        if (resolved_root / "include" / "cuda_runtime.h").exists():
            return resolved_root, resolved_compiler

    raise FileNotFoundError(
        "CUDA toolkit was not found. Set CUDA_PATH/CUDAToolkit_ROOT or pass --cuda-root/--cuda-compiler."
    )


def _write_build_info(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def resolve_cmake_executable() -> str:
    for candidate in [
        os.environ.get("CMAKE_EXECUTABLE"),
        shutil.which("cmake"),
    ]:
        if candidate:
            return candidate

    if os.name == "nt":
        common_candidates = [
            Path("C:/Program Files/CMake/bin/cmake.exe"),
            Path("C:/Program Files (x86)/CMake/bin/cmake.exe"),
        ]
        for candidate in common_candidates:
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "CMake executable was not found. Install CMake and add it to PATH, or set CMAKE_EXECUTABLE "
        "to the full path of cmake."
    )


def resolve_cmake_generator(explicit_generator: str | None = None) -> str | None:
    for candidate in [explicit_generator, os.environ.get("CMAKE_GENERATOR")]:
        if candidate:
            return candidate

    if os.name == "nt":
        return "Visual Studio 17 2022"
    return None


def _sanitize_build_dir_component(value: str) -> str:
    cleaned: list[str] = []
    pending_separator = False
    for char in value.lower():
        if char.isalnum():
            if pending_separator and cleaned:
                cleaned.append("-")
            cleaned.append(char)
            pending_separator = False
        else:
            pending_separator = True
    return "".join(cleaned) or "default"


def resolve_cmake_platform(generator: str | None, os_name: str | None = None) -> str | None:
    active_os_name = os_name or os.name
    if active_os_name == "nt" and generator and generator.startswith("Visual Studio"):
        return "x64"
    return None


def resolve_cmake_build_dir(plugin_root: Path, generator: str | None, os_name: str | None = None) -> Path:
    components = ["cmake-build"]
    if generator:
        components.append(_sanitize_build_dir_component(generator))
    platform = resolve_cmake_platform(generator, os_name=os_name)
    if platform:
        components.append(_sanitize_build_dir_component(platform))
    return plugin_root / "-".join(components)


def _is_cmake_cache_conflict(message: str) -> bool:
    lowered = message.lower()
    return "cmakecache.txt" in lowered and "used previously" in lowered


def _clear_cmake_build_dir(build_dir: Path) -> None:
    if build_dir.is_dir():
        shutil.rmtree(build_dir, ignore_errors=True)
        return
    try:
        build_dir.unlink()
    except FileNotFoundError:
        return


def _format_configure_error(message: str, resolved_cuda_root: Path) -> str:
    if "No CUDA toolset found" in message:
        message += (
            "\n\nDetected CUDA toolkit: "
            f"{resolved_cuda_root}\n"
            "CMake was invoked with an explicit CUDA toolset, but Visual Studio still did not expose a usable CUDA toolset. "
            "Run from a Visual Studio Developer PowerShell/Prompt or install the CUDA Visual Studio integration."
        )
    return message


def _format_subprocess_failure(
    step: str,
    command: list[str],
    result: subprocess.CompletedProcess,
    *,
    extra_hint: str | None = None,
) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    message = [f"{step} failed with exit code {result.returncode}.", f"Command: {' '.join(map(str, command))}"]
    if stdout:
        message.append(f"stdout:\n{stdout}")
    if stderr:
        message.append(f"stderr:\n{stderr}")
    if extra_hint:
        message.append(extra_hint)
    return "\n\n".join(message)


def _run_cmake_configure(configure_cmd: list[str], build_dir: Path, resolved_cuda_root: Path) -> None:
    configure = subprocess.run(configure_cmd, capture_output=True, text=True, check=False)
    if configure.returncode != 0:
        message = configure.stderr.strip() or configure.stdout.strip() or "CMake configure failed."
        if _is_cmake_cache_conflict(message):
            _clear_cmake_build_dir(build_dir)
            configure = subprocess.run(configure_cmd, capture_output=True, text=True, check=False)
            if configure.returncode != 0:
                message = configure.stderr.strip() or configure.stdout.strip() or "CMake configure failed."
            else:
                message = ""
        if configure.returncode != 0:
            extra_hint = None
            formatted = _format_configure_error(message, resolved_cuda_root)
            if formatted != message:
                extra_hint = formatted[len(message):].lstrip()
            raise RuntimeError(_format_subprocess_failure("CMake configure", configure_cmd, configure, extra_hint=extra_hint))


def build_plugin_library(
    plugin_root: str | Path,
    *,
    force: bool = False,
    tensorrt_root: str | None = None,
    tensorrt_include_dir: str | None = None,
    tensorrt_library_dir: str | None = None,
    cuda_root: str | None = None,
    cuda_compiler: str | None = None,
    generator: str | None = None,
) -> PluginBuildResult:
    plugin_root = Path(plugin_root).resolve()
    plugin_root.mkdir(parents=True, exist_ok=True)
    cmake_executable = resolve_cmake_executable()
    cmake_generator = resolve_cmake_generator(generator)
    include_dir, library_dir = resolve_tensorrt_sdk(
        tensorrt_root=tensorrt_root,
        tensorrt_include_dir=tensorrt_include_dir,
        tensorrt_library_dir=tensorrt_library_dir,
    )
    resolved_cuda_root, resolved_cuda_compiler = resolve_cuda_sdk(cuda_root=cuda_root, cuda_compiler=cuda_compiler)

    library_path = plugin_root / plugin_library_filename()
    build_info_path = plugin_root / plugin_build_info_filename()
    build_id = compute_plugin_build_id(PLUGIN_SOURCE_ROOT, include_dir, library_dir)

    if library_path.exists() and build_info_path.exists() and not force:
        with open(build_info_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("build_id") == build_id:
            return PluginBuildResult(str(library_path), str(build_info_path), build_id)

    build_dir = resolve_cmake_build_dir(plugin_root, cmake_generator)
    cmake_platform = resolve_cmake_platform(cmake_generator)
    configure_cmd = [
        cmake_executable,
        "-S",
        str(PLUGIN_SOURCE_ROOT),
        "-B",
        str(build_dir),
        f"-DTENSORRT_INCLUDE_DIR={include_dir}",
        f"-DTENSORRT_LIBRARY_DIR={library_dir}",
        f"-DTRT_PLUGIN_OUTPUT_DIR={plugin_root}",
        f"-DCUDAToolkit_ROOT={resolved_cuda_root}",
        f"-DCMAKE_CUDA_COMPILER={resolved_cuda_compiler}",
    ]
    if cmake_generator:
        configure_cmd.extend(["-G", cmake_generator])
    if os.name == "nt" and cmake_platform:
        configure_cmd.extend(["-A", cmake_platform, "-T", f"cuda={resolved_cuda_root}"])
    if os.name != "nt":
        configure_cmd.append("-DCMAKE_BUILD_TYPE=Release")

    _run_cmake_configure(configure_cmd, build_dir, resolved_cuda_root)

    build_cmd = [cmake_executable, "--build", str(build_dir), "--config", "Release", "--verbose"]
    build = subprocess.run(
        build_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        hint = None
        combined_output = "\n".join(part for part in [(build.stdout or "").strip(), (build.stderr or "").strip()] if part)
        if os.name == "nt" and "MSBuild version" in combined_output and "error " not in combined_output.lower():
            hint = (
                "MSBuild did not emit a compiler diagnostic here. Common causes are missing CUDA Visual Studio integration, "
                "an unsupported CUDA/MSVC pairing, or CUDA-specific compiler flags being passed incorrectly."
            )
        raise RuntimeError(_format_subprocess_failure("CMake build", build_cmd, build, extra_hint=hint))
    if not library_path.exists():
        raise FileNotFoundError(f"Built TensorRT plugin library was not found: {library_path}")

    _write_build_info(
        build_info_path,
        {
            "build_id": build_id,
            "include_dir": str(include_dir),
            "library_dir": str(library_dir),
            "library_path": str(library_path),
            "plugin_name": PLUGIN_LIBRARY_BASENAME,
            "source_root": str(PLUGIN_SOURCE_ROOT),
        },
    )
    return PluginBuildResult(str(library_path), str(build_info_path), build_id)


def main():
    parser = argparse.ArgumentParser(description="Build JoyVASA TensorRT plugin libraries.")
    parser.add_argument("--plugin-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tensorrt-root", default=None)
    parser.add_argument("--tensorrt-include-dir", default=None)
    parser.add_argument("--tensorrt-library-dir", default=None)
    parser.add_argument("--cuda-root", default=None)
    parser.add_argument("--cuda-compiler", default=None)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--engine-root", default=str(ROOT / "pretrained_weights" / "trt_artifacts" / "engines"))
    args = parser.parse_args()

    plugin_root = args.plugin_root or default_plugin_root(args.engine_root)
    if plugin_root is None:
        raise ValueError("Unable to infer plugin root. Pass --plugin-root explicitly.")

    result = build_plugin_library(
        plugin_root,
        force=args.force,
        tensorrt_root=args.tensorrt_root,
        tensorrt_include_dir=args.tensorrt_include_dir,
        tensorrt_library_dir=args.tensorrt_library_dir,
        cuda_root=args.cuda_root,
        cuda_compiler=args.cuda_compiler,
        generator=args.generator,
    )
    print(result.library_path)


if __name__ == "__main__":
    main()
