"""Cross-backend export bridge for PyTorch and JAX Keras backends."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Union
import numpy as np

import keras
from keras import ops


def is_tensorflow_backend() -> bool:
    r"""Checks whether the currently active Keras backend is TensorFlow."""
    return keras.backend.backend() == "tensorflow"


def export_via_tf_subprocess(
    exporter_type: str,  # "onnx" or "tflite"
    model_or_task: Any,
    output_path: Union[str, Path],
    dummy_inputs: Optional[Any] = None,
    **kwargs: Any,
) -> Path:
    r"""Bridges model export to a TensorFlow worker subprocess when running under PyTorch or JAX.

    Args:
        exporter_type: "onnx" or "tflite".
        model_or_task: Model or task instance.
        output_path: Target path for the exported model file.
        dummy_inputs: Optional dummy inputs.
        **kwargs: Extra arguments forwarded to the exporter.

    Returns:
        Path to the exported model file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ensure model is built before saving weights
        raw_model = getattr(model_or_task, "model", None) or model_or_task
        if hasattr(raw_model, "built") and not raw_model.built:
            from k3_node.export.onnx_exporter import _extract_model_and_inputs
            try:
                _, extracted_inputs, _ = _extract_model_and_inputs(model_or_task, dummy_inputs)
                raw_model(*extracted_inputs)
            except Exception:
                pass

        # 2. Save model or task weights & config
        if hasattr(model_or_task, "save_pretrained"):
            model_or_task.save_pretrained(str(model_dir))
            cls_module = model_or_task.__class__.__module__
            cls_name = model_or_task.__class__.__name__
        elif hasattr(model_or_task, "model") and hasattr(model_or_task.model, "save_pretrained"):
            model_or_task.model.save_pretrained(str(model_dir))
            cls_module = model_or_task.model.__class__.__module__
            cls_name = model_or_task.model.__class__.__name__
        else:
            raise TypeError(
                f"Model or task of type {type(model_or_task)} must support `save_pretrained` "
                "for cross-backend export."
            )

        # 2. Serialize dummy inputs if provided
        dummy_path = tmp_path / "dummy.npz"
        dummy_meta_path = tmp_path / "dummy_meta.json"
        has_dummy = False

        if dummy_inputs is not None:
            arrays: Dict[str, np.ndarray] = {}
            meta: Dict[str, Any] = {}

            if hasattr(dummy_inputs, "x") and hasattr(dummy_inputs, "edge_index"):
                meta["type"] = "data"
                arrays["x"] = np.asarray(ops.convert_to_numpy(dummy_inputs.x))
                arrays["edge_index"] = np.asarray(ops.convert_to_numpy(dummy_inputs.edge_index))
                if hasattr(dummy_inputs, "batch") and dummy_inputs.batch is not None:
                    arrays["batch"] = np.asarray(ops.convert_to_numpy(dummy_inputs.batch))
            elif hasattr(dummy_inputs, "z") and hasattr(dummy_inputs, "pos"):
                meta["type"] = "data"
                arrays["z"] = np.asarray(ops.convert_to_numpy(dummy_inputs.z))
                arrays["pos"] = np.asarray(ops.convert_to_numpy(dummy_inputs.pos))
                if hasattr(dummy_inputs, "batch") and dummy_inputs.batch is not None:
                    arrays["batch"] = np.asarray(ops.convert_to_numpy(dummy_inputs.batch))
            elif isinstance(dummy_inputs, (tuple, list)):
                meta["type"] = "tuple"
                for i, elem in enumerate(dummy_inputs):
                    arrays[f"arr_{i}"] = np.asarray(ops.convert_to_numpy(elem))
            elif isinstance(dummy_inputs, dict):
                meta["type"] = "dict"
                for k, v in dummy_inputs.items():
                    if v is not None:
                        arrays[str(k)] = np.asarray(ops.convert_to_numpy(v))
            else:
                meta["type"] = "single"
                arrays["arr_0"] = np.asarray(ops.convert_to_numpy(dummy_inputs))

            np.savez(str(dummy_path), **arrays)
            with open(dummy_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            has_dummy = True

        # 3. Build worker command
        worker_code = f"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import importlib
import json
from pathlib import Path
import numpy as np

import k3_node
from k3_node.data import Data
from k3_node.export.onnx_exporter import export_onnx
from k3_node.export.tflite_exporter import export_tflite

# Load model / task
mod = importlib.import_module("{cls_module}")
cls = getattr(mod, "{cls_name}")
model_or_task = cls.from_pretrained(r"{model_dir}")

# Load dummy inputs
dummy = None
has_dummy = {has_dummy}
if has_dummy:
    with open(r"{dummy_meta_path}", "r") as f:
        meta = json.load(f)
    npz = np.load(r"{dummy_path}")
    t = meta["type"]
    if t == "data":
        kwargs = {{k: npz[k] for k in npz.files}}
        dummy = Data(**kwargs)
    elif t == "tuple":
        dummy = tuple(npz[f"arr_{{i}}"] for i in range(len(npz.files)))
    elif t == "dict":
        dummy = {{k: npz[k] for k in npz.files}}
    else:
        dummy = npz["arr_0"]

extra_kwargs = json.loads(r'''{json.dumps(kwargs)}''')
if "{exporter_type}" == "onnx":
    export_onnx(model_or_task, r"{out_file}", dummy_inputs=dummy, **extra_kwargs)
elif "{exporter_type}" == "tflite":
    export_tflite(model_or_task, r"{out_file}", dummy_inputs=dummy, **extra_kwargs)
"""

        env = dict(os.environ, KERAS_BACKEND="tensorflow")
        res = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True,
            text=True,
            env=env,
        )

        if res.returncode != 0:
            raise RuntimeError(
                f"Cross-backend export to {exporter_type.upper()} failed with exit code {res.returncode}:\n"
                f"STDOUT: {res.stdout}\n"
                f"STDERR: {res.stderr}"
            )

    return out_file
