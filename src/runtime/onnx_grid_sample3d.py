from contextlib import contextmanager

import torch
from torch.onnx import errors
from torch.onnx import symbolic_helper
from torch.onnx import symbolic_opset16

from .trt_plugins import PLUGIN_NAMESPACE, PLUGIN_OP_TYPE, PLUGIN_VERSION


def joyvasa_grid_sample3d_symbolic_fn(g, input, grid, mode_enum, padding_mode_enum, align_corners):
    rank = symbolic_helper._get_tensor_rank(input)
    if rank != 5:
        return symbolic_opset16.grid_sampler(g, input, grid, mode_enum, padding_mode_enum, align_corners)

    mode = symbolic_helper._parse_arg(mode_enum, "i")
    padding_mode = symbolic_helper._parse_arg(padding_mode_enum, "i")
    align = symbolic_helper._parse_arg(align_corners, "b")

    if mode != 0 or padding_mode != 0 or align:
        raise errors.OnnxExporterError(
            "Unsupported JoyVASA GridSample3D export configuration: only mode='bilinear', "
            "padding_mode='zeros', align_corners=False are supported."
        )

    return g.op(
        f"{PLUGIN_NAMESPACE}::{PLUGIN_OP_TYPE}",
        input,
        grid,
        mode_i=mode,
        padding_mode_i=padding_mode,
        align_corners_i=int(align),
        plugin_namespace_s=PLUGIN_NAMESPACE,
        plugin_version_s=PLUGIN_VERSION,
    )


@contextmanager
def joyvasa_grid_sample3d_symbolic(opset_version: int):
    torch.onnx.register_custom_op_symbolic("aten::grid_sampler", joyvasa_grid_sample3d_symbolic_fn, opset_version)
    try:
        yield
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::grid_sampler", opset_version)
