"""Turn-key TensorFlow Lite / LiteRT exporter for K3-Node models and tasks."""

from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
import numpy as np

import keras
from keras import ops

from k3_node.export.onnx_exporter import _extract_model_and_inputs


def export_tflite(
    model_or_task: Any,
    output_path: Union[str, Path],
    dummy_inputs: Optional[Any] = None,
    quantization: Optional[str] = None,
    representative_dataset: Optional[Callable[[], Generator[List[np.ndarray], None, None]]] = None,
    verbose: bool = False,
) -> Path:
    r"""Exports a K3-Node GNN model or task to an optimized TensorFlow Lite flatbuffer.

    Supports float32, float16 (FP16), and dynamic range INT8 quantization for
    low-latency deployment on edge, mobile, and embedded hardware.

    Args:
        model_or_task: A K3-Node task instance or model instance.
        output_path: Target path for the `.tflite` file.
        dummy_inputs: Optional sample input data (e.g., PyG `Data` object, tuple of tensors).
        quantization: Quantization strategy:
            - `None` or `"fp32"`: Full 32-bit floating point precision.
            - `"fp16"`: 16-bit floating point quantization (~2x smaller & faster on GPU/NPU).
            - `"int8_dynamic"`: Dynamic range integer quantization (~4x smaller with 8-bit weights).
            - `"int8_full"`: Full 8-bit integer quantization (requires `representative_dataset`).
        representative_dataset: Generator of representative calibration inputs for `"int8_full"`.
        verbose: Whether to print verbose progress. (default: `False`)

    Returns:
        Path object pointing to the generated `.tflite` file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    from k3_node.export.cross_backend import is_tensorflow_backend, export_via_tf_subprocess

    if not is_tensorflow_backend():
        return export_via_tf_subprocess(
            exporter_type="tflite",
            model_or_task=model_or_task,
            output_path=output_path,
            dummy_inputs=dummy_inputs,
            quantization=quantization,
            verbose=verbose,
        )

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for TFLite export. Install it via `pip install tensorflow`."
        )

    model, inputs, inferred_names = _extract_model_and_inputs(model_or_task, dummy_inputs)

    # Build concrete function with exact input shapes
    specs = []
    for inp in inputs:
        inp_np = ops.convert_to_numpy(inp)
        specs.append(tf.TensorSpec(shape=inp_np.shape, dtype=tf.as_dtype(inp_np.dtype)))

    concrete_func = tf.function(lambda *args: model(*args)).get_concrete_function(*specs)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # Apply quantization options
    if quantization in ("fp16", "float16"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        if verbose:
            print("Applying FP16 quantization to TFLite model.")
    elif quantization in ("int8_dynamic", "int8"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if verbose:
            print("Applying dynamic range INT8 quantization to TFLite model.")
    elif quantization == "int8_full":
        if representative_dataset is None:
            raise ValueError(
                "Full INT8 quantization (`quantization='int8_full'`) requires a "
                "`representative_dataset` generator to calibrate activation ranges."
            )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        if verbose:
            print("Applying full integer INT8 quantization with calibration dataset.")

    # Convert model to TFLite flatbuffer
    tflite_model_bytes = converter.convert()

    with open(out_file, "wb") as f:
        f.write(tflite_model_bytes)

    if verbose:
        print(f"Exported TFLite model ({len(tflite_model_bytes)} bytes) to {out_file}")

    return out_file
