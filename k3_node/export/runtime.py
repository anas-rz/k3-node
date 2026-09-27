"""Lightweight inference runtime engines for serving ONNX and TFLite models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class ONNXModel:
    r"""High-performance serving wrapper for exported ONNX GNN models.

    Requires only `onnxruntime` and `numpy`. Completely decoupled from Keras,
    PyTorch, and TensorFlow for lightweight production microservices.

    Example:
        ```python
        from k3_node.export import ONNXModel
        model = ONNXModel("cora_gcn.onnx")
        preds = model.predict(graph_data)
        ```
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        providers: Optional[List[str]] = None,
        session_options: Optional[Any] = None,
    ):
        r"""Initializes the ONNX runtime inference session.

        Args:
            model_path: Path to the `.onnx` model file.
            providers: Execution providers list (e.g. `['CUDAExecutionProvider', 'CPUExecutionProvider']`).
                If `None`, automatically picks the fastest available provider.
            session_options: Optional custom ONNX Runtime SessionOptions.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "The `onnxruntime` package is required to load and serve ONNX models. "
                "Install it via `pip install onnxruntime` (or `onnxruntime-gpu` for CUDA/TensorRT)."
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")

        if providers is None:
            available = ort.get_available_providers()
            # Prioritize TensorRT, CUDA, CoreML, DirectML, CPU
            priority = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in priority if p in available]

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict(self, data: Any = None, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Runs low-latency inference on graph data.

        Accepts:
            - PyG / K3-Node `Data` object
            - Dictionary of tensors
            - Positional numpy arrays

        Args:
            data: Input graph Data, dictionary, or array.
            *args: Additional positional inputs.

        Returns:
            Numpy array containing model predictions or logits.
        """
        feed_dict = {}

        # 1. PyG/K3 Data object
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            feed_dict = self._match_inputs({
                "x": np.asarray(data.x, dtype=np.float32),
                "edge_index": np.asarray(data.edge_index, dtype=np.int64),
                "batch": np.asarray(getattr(data, "batch", None), dtype=np.int64) if getattr(data, "batch", None) is not None else None,
            })
        elif hasattr(data, "z") and hasattr(data, "pos"):
            feed_dict = self._match_inputs({
                "z": np.asarray(data.z, dtype=np.int32),
                "pos": np.asarray(data.pos, dtype=np.float32),
                "batch": np.asarray(getattr(data, "batch", None), dtype=np.int32) if getattr(data, "batch", None) is not None else None,
            })
        elif isinstance(data, dict):
            feed_dict = self._match_inputs(data)
        elif isinstance(data, (tuple, list)):
            for name, val in zip(self.input_names, data):
                feed_dict[name] = np.asarray(val)
        elif data is not None:
            all_args = [data, *args]
            for name, val in zip(self.input_names, all_args):
                feed_dict[name] = np.asarray(val)

        outputs = self.session.run(self.output_names, feed_dict)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _match_inputs(self, named_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        matched = {}
        valid_inputs = {k: np.asarray(v) for k, v in named_inputs.items() if v is not None}
        lower_inputs = {k.lower(): v for k, v in valid_inputs.items()}

        used_keys = set()
        session_inputs = self.session.get_inputs()

        # Step 1: Match exact or known semantic names
        for sess_inp in session_inputs:
            name = sess_inp.name
            clean = name.split(":")[0].lower()

            target_val = None
            matched_key = None

            if clean in lower_inputs:
                target_val = lower_inputs[clean]
                matched_key = clean
            elif "edge" in clean and "edge_index" in lower_inputs:
                target_val = lower_inputs["edge_index"]
                matched_key = "edge_index"
            elif clean == "x" and "x" in lower_inputs:
                target_val = lower_inputs["x"]
                matched_key = "x"
            elif clean in ("batch", "batch_idx") and "batch" in lower_inputs:
                target_val = lower_inputs["batch"]
                matched_key = "batch"
            elif clean in ("z", "atomic_numbers") and "z" in lower_inputs:
                target_val = lower_inputs["z"]
                matched_key = "z"
            elif clean in ("pos", "positions", "coord") and "pos" in lower_inputs:
                target_val = lower_inputs["pos"]
                matched_key = "pos"

            if target_val is not None:
                matched[name] = target_val
                used_keys.add(matched_key)

        # Step 2: Positional fallback for remaining inputs
        unmatched_session = [inp for inp in session_inputs if inp.name not in matched]
        unmatched_keys = [k for k in valid_inputs.keys() if k.lower() not in used_keys]

        if unmatched_session and unmatched_keys:
            for sess_inp, k in zip(unmatched_session, unmatched_keys):
                matched[sess_inp.name] = valid_inputs[k]

        # Step 3: Align data types with expected ONNX session tensor types
        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        for sess_inp in session_inputs:
            if sess_inp.name in matched:
                expected_np_type = type_map.get(sess_inp.type)
                if expected_np_type and matched[sess_inp.name].dtype != expected_np_type:
                    matched[sess_inp.name] = matched[sess_inp.name].astype(expected_np_type)

        return matched


class TFLiteModel:
    r"""Lightweight serving wrapper for TensorFlow Lite flatbuffer GNN models.

    Requires only standard `tensorflow` or `tflite_runtime`. Ideal for mobile,
    Raspberry Pi, and edge embedded devices.

    Example:
        ```python
        from k3_node.export import TFLiteModel
        model = TFLiteModel("model.tflite")
        preds = model.predict(graph_data)
        ```
    """

    def __init__(self, model_path: Union[str, Path]):
        r"""Initializes the TFLite interpreter.

        Args:
            model_path: Path to the `.tflite` model file.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model file not found at: {self.model_path}")

        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
            except ImportError:
                raise ImportError(
                    "Either `tensorflow` or `tflite_runtime` is required to run TFLite models. "
                    "Install via `pip install tflite-runtime` or `pip install tensorflow`."
                )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, data: Any = None, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Runs inference using the TFLite interpreter.

        Args:
            data: Input graph Data, dict, or numpy array.
            *args: Additional positional inputs.

        Returns:
            Numpy array containing prediction results.
        """
        inputs = []
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            inputs = [np.asarray(data.x, dtype=np.float32), np.asarray(data.edge_index, dtype=np.int64)]
            if hasattr(data, "batch") and data.batch is not None:
                inputs.append(np.asarray(data.batch, dtype=np.int64))
        elif hasattr(data, "z") and hasattr(data, "pos"):
            inputs = [np.asarray(data.z, dtype=np.int32), np.asarray(data.pos, dtype=np.float32)]
            if hasattr(data, "batch") and data.batch is not None:
                inputs.append(np.asarray(data.batch, dtype=np.int32))
        elif isinstance(data, (tuple, list)):
            inputs = [np.asarray(x) for x in data]
        elif isinstance(data, dict):
            inputs = [np.asarray(v) for v in data.values()]
        elif data is not None:
            inputs = [np.asarray(data)] + [np.asarray(a) for a in args]

        # Feed tensors
        for detail, inp in zip(self.input_details, inputs):
            # Cast dtype to match interpreter expectation
            target_dtype = detail["dtype"]
            inp_cast = inp.astype(target_dtype) if inp.dtype != target_dtype else inp
            self.interpreter.set_tensor(detail["index"], inp_cast)

        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(d["index"]) for d in self.output_details]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
