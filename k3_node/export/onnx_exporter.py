"""Turn-key ONNX exporter for K3-Node models and tasks."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import keras
from keras import ops


def _extract_model_and_inputs(
    model_or_task: Any,
    dummy_inputs: Optional[Any] = None,
) -> Tuple[Any, Tuple[Any, ...], List[Any]]:
    r"""Extracts the underlying neural network model and resolves dummy inputs / signatures."""
    # 1. Resolve model
    if hasattr(model_or_task, "model") and model_or_task.model is not None:
        raw_model = model_or_task.model
    else:
        raw_model = model_or_task

    # 2. Extract inputs from dummy_inputs if provided
    if dummy_inputs is not None:
        if hasattr(dummy_inputs, "x") and hasattr(dummy_inputs, "edge_index"):
            inputs = [dummy_inputs.x, dummy_inputs.edge_index]
            names = ["x", "edge_index"]
            if hasattr(dummy_inputs, "batch") and dummy_inputs.batch is not None:
                inputs.append(dummy_inputs.batch)
                names.append("batch")
            return raw_model, tuple(inputs), names
        elif hasattr(dummy_inputs, "z") and hasattr(dummy_inputs, "pos"):
            inputs = [dummy_inputs.z, dummy_inputs.pos]
            names = ["z", "pos"]
            if hasattr(dummy_inputs, "batch") and dummy_inputs.batch is not None:
                inputs.append(dummy_inputs.batch)
                names.append("batch")
            return raw_model, tuple(inputs), names
        elif isinstance(dummy_inputs, (tuple, list)):
            return raw_model, tuple(dummy_inputs), None
        elif isinstance(dummy_inputs, dict):
            return raw_model, (dummy_inputs,), None
        else:
            return raw_model, (dummy_inputs,), None

    # 3. Auto-infer dummy inputs from model hyperparameters
    cls_name = model_or_task.__class__.__name__
    in_channels = (
        getattr(model_or_task, "in_channels", None)
        or getattr(raw_model, "in_channels", None)
        or 16
    )

    if cls_name == "NodeClassifier":
        x = ops.zeros((4, in_channels), dtype="float32")
        edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
        return raw_model, (x, edge_index), ["x", "edge_index"]
    elif cls_name in ("GraphClassifier", "GraphRegressor"):
        x = ops.zeros((4, in_channels), dtype="float32")
        edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
        batch = ops.convert_to_tensor([0, 0, 1, 1], dtype="int64")
        if hasattr(raw_model, "num_graphs"):
            raw_model.num_graphs = 2
        return raw_model, (x, edge_index, batch), ["x", "edge_index", "batch"]
    elif cls_name == "LinkPredictor":
        x = ops.zeros((4, in_channels), dtype="float32")
        edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
        label_idx = ops.convert_to_tensor([[0, 1], [1, 2]], dtype="int64")
        return raw_model, ((x, edge_index), label_idx), ["edge_tuple", "edge_label_index"]
    elif cls_name in ("SchNet", "DimeNet", "DimeNetPlusPlus", "ViSNet", "GNNFF"):
        z = ops.convert_to_tensor([1, 6, 8, 1], dtype="int32")
        pos = ops.zeros((4, 3), dtype="float32")
        return raw_model, (z, pos), ["z", "pos"]
    else:
        # Default standard GNN: (x, edge_index)
        x = ops.zeros((4, in_channels), dtype="float32")
        edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
        return raw_model, (x, edge_index), ["x", "edge_index"]


def export_onnx(
    model_or_task: Any,
    output_path: Union[str, Path],
    dummy_inputs: Optional[Any] = None,
    opset: int = 17,
    dynamic_axes: bool = True,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Path:
    r"""Exports a K3-Node GNN model or task to high-performance ONNX format.

    Supports arbitrary Graph Neural Networks (GCN, GAT, GraphSAGE, GIN, SchNet,
    materials models, and task estimators) with dynamic graph sizing (varying numbers
    of nodes and edges).

    Args:
        model_or_task: A K3-Node task instance (e.g. `NodeClassifier`, `GraphClassifier`)
            or model instance (e.g. `GCN`, `SchNet`, `CHGNet`).
        output_path: Target path for the `.onnx` file.
        dummy_inputs: Optional sample input data (e.g., PyG `Data` object, tuple of tensors).
            If `None`, automatically generated based on model topology.
        opset: ONNX operator set version. (default: `17`)
        dynamic_axes: Whether node and edge dimensions should be dynamic. (default: `True`)
        input_names: Optional custom names for input tensors.
        output_names: Optional custom names for output tensors.
        verbose: Whether to print verbose export progress. (default: `False`)

    Returns:
        Path object pointing to the generated `.onnx` file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    from k3_node.export.cross_backend import is_tensorflow_backend, export_via_tf_subprocess

    if not is_tensorflow_backend():
        return export_via_tf_subprocess(
            exporter_type="onnx",
            model_or_task=model_or_task,
            output_path=output_path,
            dummy_inputs=dummy_inputs,
            opset=opset,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
            verbose=verbose,
        )

    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "The `tf2onnx` and `tensorflow` packages are required for ONNX export. "
            "Install them via `pip install tf2onnx tensorflow onnx`."
        )

    model, inputs, inferred_names = _extract_model_and_inputs(model_or_task, dummy_inputs)
    names = input_names or inferred_names or [f"input_{i}" for i in range(len(inputs))]

    # Build input signature with dynamic axes if requested
    signature = []
    for inp, name in zip(inputs, names):
        inp_np = ops.convert_to_numpy(inp)
        dtype = tf.as_dtype(inp_np.dtype)
        if dynamic_axes:
            if inp_np.ndim == 2 and inp_np.shape[0] == 2 and (
                np.issubdtype(inp_np.dtype, np.integer) or "edge" in name.lower()
            ):
                # Edge index: (2, num_edges) -> dynamic num_edges
                shape = (2, None)
            elif inp_np.ndim == 2:
                # Node feature: (num_nodes, in_channels) -> dynamic num_nodes
                shape = (None, inp_np.shape[1])
            elif inp_np.ndim == 1:
                # Vector (batch or z): (num_nodes,) -> dynamic num_nodes
                shape = (None,)
            else:
                shape = tuple(None if i == 0 else s for i, s in enumerate(inp_np.shape))
        else:
            shape = inp_np.shape
        signature.append(tf.TensorSpec(shape=shape, dtype=dtype, name=name))

    # Define trace function
    @tf.function(input_signature=signature)
    def forward_fn(*tensors):
        return model(*tensors)

    if verbose:
        print(f"Exporting model to ONNX with input signature: {signature}")

    # Convert using tf2onnx
    model_proto, _ = tf2onnx.convert.from_function(
        forward_fn,
        input_signature=signature,
        output_path=str(out_file),
        opset=opset,
    )

    # Validate ONNX graph
    try:
        import onnx
        onnx_model = onnx.load(str(out_file))
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        if verbose:
            print(f"Warning: ONNX validation check returned: {e}")

    return out_file
