import importlib.util
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"


def load_build_trt_plugins_module():
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(ROOT / "src")]
    runtime_pkg = types.ModuleType("src.runtime")
    runtime_pkg.__path__ = [str(ROOT / "src" / "runtime")]
    trt_plugins_stub = types.ModuleType("src.runtime.trt_plugins")
    trt_plugins_stub.PLUGIN_LIBRARY_BASENAME = "joyvasa_trt_plugins"
    trt_plugins_stub.default_plugin_root = (
        lambda engine_root: str(Path(engine_root).resolve().parent / "plugins") if engine_root else None
    )
    trt_plugins_stub.plugin_build_info_filename = lambda: "joyvasa_trt_plugins.build.json"
    trt_plugins_stub.plugin_library_filename = lambda: "joyvasa_trt_plugins.dll"

    module_name = "test_build_trt_plugins_module"
    module_path = TOOLS_ROOT / "build_trt_plugins.py"
    previous_modules = {
        "src": sys.modules.get("src"),
        "src.runtime": sys.modules.get("src.runtime"),
        "src.runtime.trt_plugins": sys.modules.get("src.runtime.trt_plugins"),
        module_name: sys.modules.get(module_name),
    }

    try:
        sys.modules["src"] = src_pkg
        sys.modules["src.runtime"] = runtime_pkg
        sys.modules["src.runtime.trt_plugins"] = trt_plugins_stub

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


build_trt_plugins = load_build_trt_plugins_module()


def test_resolve_cmake_build_dir_scopes_visual_studio_platform():
    build_dir = build_trt_plugins.resolve_cmake_build_dir(
        Path("C:/tmp/plugins"),
        "Visual Studio 17 2022",
        os_name="nt",
    )

    assert build_dir.name == "cmake-build-visual-studio-17-2022-x64"


def test_run_cmake_configure_retries_after_cmake_cache_conflict(monkeypatch):
    configure_calls = []
    cleared_build_dirs = []

    def fake_run(cmd, capture_output, text, check):
        configure_calls.append(cmd)
        if len(configure_calls) == 1:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=(
                    "CMake Error: Error: generator platform: x64\n"
                    "Does not match the platform used previously:\n"
                    "Either remove the CMakeCache.txt file and CMakeFiles directory or choose a different binary directory."
                ),
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="configured", stderr="")

    monkeypatch.setattr(build_trt_plugins.subprocess, "run", fake_run)
    monkeypatch.setattr(build_trt_plugins, "_clear_cmake_build_dir", lambda build_dir: cleared_build_dirs.append(build_dir))

    build_dir = Path("C:/tmp/plugins/cmake-build-visual-studio-17-2022-x64")
    configure_cmd = ["cmake", "-S", "src", "-B", str(build_dir)]
    build_trt_plugins._run_cmake_configure(configure_cmd, build_dir, Path("C:/CUDA/v12.6"))

    assert len(configure_calls) == 2
    assert cleared_build_dirs == [build_dir]


def test_format_subprocess_failure_includes_command_output_and_hint():
    command = ["cmake", "--build", "cmake-build", "--config", "Release", "--verbose"]
    result = subprocess.CompletedProcess(
        command,
        1,
        stdout="MSBuild version 17.14.40+3e7442088 for .NET Framework",
        stderr="",
    )

    message = build_trt_plugins._format_subprocess_failure(
        "CMake build",
        command,
        result,
        extra_hint="A useful hint.",
    )

    assert "CMake build failed with exit code 1." in message
    assert "Command: cmake --build cmake-build --config Release --verbose" in message
    assert "MSBuild version 17.14.40+3e7442088 for .NET Framework" in message
    assert "A useful hint." in message
