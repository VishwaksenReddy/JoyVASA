import importlib.util
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "runtime" / "engine_utils.py"
SPEC = importlib.util.spec_from_file_location("runtime_engine_utils", MODULE_PATH)
ENGINE_UTILS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ENGINE_UTILS)

EngineManifest = ENGINE_UTILS.EngineManifest
get_ort_provider_bundle = ENGINE_UTILS.get_ort_provider_bundle
resolve_engine_artifact = ENGINE_UTILS.resolve_engine_artifact


WORK_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


def make_temp_dir():
    WORK_ROOT.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=WORK_ROOT))


def test_resolve_engine_artifact_layout():
    temp_dir = make_temp_dir()
    artifact = resolve_engine_artifact(str(temp_dir), "human_motion_extractor", "fp16")
    assert artifact.onnx_path.endswith("human_motion_extractor.onnx")
    assert artifact.engine_path.endswith("human_motion_extractor.fp16.engine")
    assert artifact.manifest_path.endswith("human_motion_extractor.fp16.json")
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_manifest_roundtrip():
    manifest = EngineManifest(
        name="demo",
        precision="fp16",
        onnx_sha256="abc",
        source_sha256={"weights": "123"},
        inputs={"x": [1, 3, 256, 256]},
        outputs={"y": [1, 3, 256, 256]},
        builder="trtexec 10",
    )
    loaded = EngineManifest(**asdict(manifest))
    assert loaded == manifest


def test_get_ort_provider_bundle_prefers_tensorrt():
    temp_dir = make_temp_dir()
    providers, provider_options, session_options = get_ort_provider_bundle(
        backend="auto",
        device="cuda",
        device_id=0,
        precision="fp16",
        cache_root=str(temp_dir),
    )
    assert providers[0] == "TensorrtExecutionProvider"
    assert providers[1] == "CUDAExecutionProvider"
    assert provider_options[0]["trt_engine_cache_enable"] is True
    assert session_options["graph_optimization_level"] == "ORT_ENABLE_ALL"
    shutil.rmtree(temp_dir, ignore_errors=True)
