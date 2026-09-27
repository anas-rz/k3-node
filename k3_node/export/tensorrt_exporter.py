"""Turn-key TensorRT engine exporter and Triton config generator for K3-Node."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from k3_node.export.onnx_exporter import export_onnx


def export_tensorrt(
    model_or_task_or_onnx: Any,
    output_path: Union[str, Path],
    dummy_inputs: Optional[Any] = None,
    precision: str = "fp16",
    workspace_gb: int = 1,
    min_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    opt_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    max_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    verbose: bool = False,
) -> Path:
    r"""Compiles a K3-Node model or ONNX file into an ultra-low latency NVIDIA TensorRT engine.

    Uses the Python `tensorrt` API if available on the GPU host, or invokes NVIDIA's
    `trtexec` binary directly.

    Args:
        model_or_task_or_onnx: K3-Node model/task, or an existing `.onnx` file path.
        output_path: Target path for the compiled `.engine` or `.plan` binary.
        dummy_inputs: Optional input sample to determine shapes and topology.
        precision: Precision mode: `"fp32"`, `"fp16"`, or `"int8"`. (default: `"fp16"`)
        workspace_gb: Max GPU memory in GB allocated for TensorRT engine building. (default: `1`)
        min_shapes: Optional minimum dynamic shapes dictionary (e.g. `{'x': (1, 16)}`).
        opt_shapes: Optional optimal dynamic shapes dictionary (e.g. `{'x': (100, 16)}`).
        max_shapes: Optional maximum dynamic shapes dictionary (e.g. `{'x': (10000, 16)}`).
        verbose: Whether to log engine building progress. (default: `False`)

    Returns:
        Path object pointing to the compiled TensorRT `.engine` file.
    """
    engine_file = Path(output_path)
    engine_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve ONNX file
    if isinstance(model_or_task_or_onnx, (str, Path)) and str(model_or_task_or_onnx).endswith(".onnx"):
        onnx_path = Path(model_or_task_or_onnx)
    else:
        temp_onnx = engine_file.with_suffix(".onnx")
        export_onnx(model_or_task_or_onnx, temp_onnx, dummy_inputs=dummy_inputs, verbose=verbose)
        onnx_path = temp_onnx

    # 2. Check for TensorRT Python API
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                error_msgs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"TensorRT failed to parse ONNX graph:\n{error_msgs}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

        if precision in ("fp16", "float16") and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision in ("int8",) and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        # Optimization profile for dynamic shapes if provided
        if min_shapes and opt_shapes and max_shapes:
            profile = builder.create_optimization_profile()
            for name in min_shapes:
                profile.set_shape(name, min_shapes[name], opt_shapes[name], max_shapes[name])
            config.add_optimization_profile(profile)

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("Failed to build TensorRT serialized network.")

        with open(engine_file, "wb") as f:
            f.write(plan)

        if verbose:
            print(f"TensorRT engine compiled successfully via Python API: {engine_file}")
        return engine_file

    except ImportError:
        # Fall back to trtexec binary if available
        trtexec_bin = shutil.which("trtexec")
        if trtexec_bin:
            cmd = [
                trtexec_bin,
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_file}",
            ]
            if precision in ("fp16", "float16"):
                cmd.append("--fp16")
            elif precision in ("int8",):
                cmd.append("--int8")

            if min_shapes and opt_shapes and max_shapes:
                min_str = ",".join(f"{k}:{'x'.join(map(str, v))}" for k, v in min_shapes.items())
                opt_str = ",".join(f"{k}:{'x'.join(map(str, v))}" for k, v in opt_shapes.items())
                max_str = ",".join(f"{k}:{'x'.join(map(str, v))}" for k, v in max_shapes.items())
                cmd.extend([
                    f"--minShapes={min_str}",
                    f"--optShapes={opt_str}",
                    f"--maxShapes={max_str}",
                ])

            if verbose:
                print(f"Compiling TensorRT engine via trtexec: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return engine_file
        else:
            raise ImportError(
                "Neither NVIDIA `tensorrt` Python package nor `trtexec` binary was found on this system.\n"
                "To compile TensorRT engines, please ensure you are running on an NVIDIA GPU system with:\n"
                "  pip install tensorrt\n"
                "or install the NVIDIA TensorRT SDK (providing `trtexec`).\n"
                f"Your ONNX model was successfully exported to: {onnx_path}\n"
                f"You can compile it later on any GPU server with:\n"
                f"  trtexec --onnx={onnx_path} --saveEngine={engine_file} --fp16"
            )


def generate_triton_config(
    model_name: str,
    output_dir: Union[str, Path],
    in_channels: int,
    out_channels: int,
    backend: str = "onnxruntime",
    max_batch_size: int = 0,
) -> Path:
    r"""Generates a complete Triton Inference Server model repository directory structure and `config.pbtxt`.

    Args:
        model_name: Name of the model in Triton (e.g., `'cora_gcn'`).
        output_dir: Root path for the model repository folder.
        in_channels: Number of node input features.
        out_channels: Number of output features or classes.
        backend: Backend engine (`'onnxruntime'` or `'tensorrt_plan'`).
        max_batch_size: Maximum batch size (default: `0` for dynamic graph sizing).

    Returns:
        Path to the generated `config.pbtxt`.
    """
    model_dir = Path(output_dir) / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    platform = "onnxruntime_onnx" if backend == "onnxruntime" else "tensorrt_plan"

    config_text = f"""name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "x"
    data_type: TYPE_FP32
    dims: [ -1, {in_channels} ]
  }},
  {{
    name: "edge_index"
    data_type: TYPE_INT64
    dims: [ 2, -1 ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1, {out_channels} ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 1000
}}
"""
    config_path = model_dir / "config.pbtxt"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_text)

    return config_path
