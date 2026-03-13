import tempfile
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn.functional as F

from src.runtime.engine_utils import DEFAULT_ONNX_OPSET
from src.runtime.onnx_grid_sample3d import joyvasa_grid_sample3d_symbolic
from src.runtime.trt_plugins import PLUGIN_NAMESPACE, PLUGIN_OP_TYPE


class DoubleGridSampleModel(torch.nn.Module):
    def forward(self, feature, grid):
        first = F.grid_sample(feature, grid, align_corners=False)
        second = F.grid_sample(first, grid, align_corners=False)
        return second


@pytest.mark.skipif(not torch.__version__, reason="torch is required")
def test_grid_sample_5d_exports_to_custom_nodes():
    model = DoubleGridSampleModel().eval()
    feature = torch.randn(1, 2, 4, 4, 4)
    grid = torch.randn(1, 4, 4, 4, 3)

    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = Path(temp_dir) / "grid_sample3d.onnx"
        with joyvasa_grid_sample3d_symbolic(DEFAULT_ONNX_OPSET), torch.no_grad():
            torch.onnx.export(
                model,
                (feature, grid),
                str(onnx_path),
                opset_version=DEFAULT_ONNX_OPSET,
                input_names=["feature", "grid"],
                output_names=["out"],
            )

        graph = onnx.load(str(onnx_path))
        custom_nodes = [node for node in graph.graph.node if node.op_type == PLUGIN_OP_TYPE and node.domain == PLUGIN_NAMESPACE]
        assert len(custom_nodes) == 2
