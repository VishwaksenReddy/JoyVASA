import shutil
import uuid
from pathlib import Path

from src.runtime import trt_plugins


WORK_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


def make_temp_dir():
    WORK_ROOT.mkdir(exist_ok=True)
    path = WORK_ROOT / f"runtime_trt_plugins_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_resolve_model_plugin_libraries_defaults_to_sibling_plugins():
    temp_dir = make_temp_dir()
    engine_root = temp_dir / "engines"
    engine_root.mkdir()

    try:
        libraries = trt_plugins.resolve_model_plugin_libraries("human_warping_module", str(engine_root), None)

        assert libraries == [str((temp_dir / "plugins" / trt_plugins.plugin_library_filename()).resolve())]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_plugin_library_path_uses_configured_directory():
    temp_dir = make_temp_dir()
    plugin_dir = temp_dir / "custom_plugins"
    expected = plugin_dir / trt_plugins.plugin_library_filename()

    try:
        resolved = trt_plugins.resolve_plugin_library_path(str(plugin_dir), engine_root=str(temp_dir / "engines"))

        assert resolved == str(expected.resolve())
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_plugin_build_info_reads_sidecar():
    temp_dir = make_temp_dir()
    plugin_library = temp_dir / trt_plugins.plugin_library_filename()
    plugin_library.write_bytes(b"plugin")
    build_info = temp_dir / trt_plugins.plugin_build_info_filename()
    build_info.write_text('{"build_id":"abc123"}', encoding="utf-8")

    try:
        payload = trt_plugins.load_plugin_build_info(str(plugin_library))

        assert payload["build_id"] == "abc123"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_plugin_library_uses_registry_once(monkeypatch):
    temp_dir = make_temp_dir()
    plugin_library = temp_dir / trt_plugins.plugin_library_filename()
    plugin_library.write_bytes(b"plugin")

    try:
        loaded = []

        class FakeRegistry:
            def load_library(self, path):
                loaded.append(path)
                return f"handle:{path}"

        class FakeLogger:
            WARNING = 1

            def __init__(self, level):
                self.level = level

        class FakeTRT:
            Logger = FakeLogger

            @staticmethod
            def init_libnvinfer_plugins(logger, namespace):
                return None

            @staticmethod
            def get_plugin_registry():
                return FakeRegistry()

        monkeypatch.setattr(trt_plugins, "trt", FakeTRT)
        monkeypatch.setattr(trt_plugins, "_LOADED_PLUGIN_HANDLES", {})

        first = trt_plugins.load_plugin_library(str(plugin_library))
        second = trt_plugins.load_plugin_library(str(plugin_library))

        assert first == second
        assert loaded == [str(plugin_library.resolve())]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
