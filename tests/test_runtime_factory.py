import pytest


pytest.importorskip("torch")

from src.runtime.factory import create_model_runner
from src.runtime.runners import TorchRunner


class DummyCallable:
    def __call__(self, value):
        return value


def test_create_model_runner_falls_back_to_torch(tmp_path):
    loader_calls = []

    def loader():
        loader_calls.append(True)
        return DummyCallable()

    runner = create_model_runner(
        name="human_demo",
        device="cuda:0",
        backend="auto",
        precision="fp16",
        engine_root=str(tmp_path),
        force_rebuild=False,
        loader=loader,
        source_paths=[],
        inputs={"x": [1, 3, 256, 256]},
        outputs={"y": [1, 3, 256, 256]},
    )

    assert isinstance(runner, TorchRunner)
    assert loader_calls == [True]
    assert runner("ok") == "ok"
