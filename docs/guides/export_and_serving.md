# Model Export & Production Serving Guide

**K3-Node** provides turn-key, zero-friction export of Graph Neural Networks to **ONNX**, **TensorRT**, and **TensorFlow Lite (LiteRT)** flatbuffers, accompanied by decoupled, ultra-lightweight inference runtimes (`ONNXModel`, `TFLiteModel`) and enterprise serving utilities.

Whether you trained your model using **PyTorch**, **JAX**, or **TensorFlow**, K3-Node handles export seamlessly across backends.

---

## Overview of Export Formats

| Format | Target Hardware / Platform | Key Benefits | Runtime |
| :--- | :--- | :--- | :--- |
| **ONNX** | Cloud Servers, CPUs, GPUs (NVIDIA, AMD, Intel) | Cross-platform standard, dynamic graph sizing, hardware acceleration | `ONNXModel` (`onnxruntime`) |
| **TensorRT** | NVIDIA GPUs (Data Center, Jetson) | Maximum throughput, kernel auto-tuning, ultra-low latency FP16/INT8 | TensorRT Engine / Triton |
| **TFLite / LiteRT** | Edge, Mobile, Embedded, IoT, Raspberry Pi | Tiny flatbuffer binary, low memory footprint, INT8 dynamic quantization | `TFLiteModel` (`tflite-runtime`) |
| **Triton** | Enterprise Microservice Clusters | Dynamic batching, concurrent model execution, gRPC/HTTP endpoints | NVIDIA Triton Server |

---

## 1. ONNX Export & Low-Latency Serving

### Exporting to ONNX

Export any K3-Node model or high-level task estimator directly using `export_onnx()` or the built-in `.export_onnx()` method:

```python
from k3_node.models import GCN
from k3_node.datasets import Planetoid
from k3_node.export import export_onnx

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

# Initialize and build model
model = GCN(in_channels=dataset.num_features, hidden_channels=32, out_channels=dataset.num_classes)

# Export directly to ONNX (dynamic node & edge axes enabled by default)
onnx_path = model.export_onnx("cora_gcn.onnx", dummy_inputs=data)
print(f"Model exported to: {onnx_path}")
```

#### Dynamic Graph Sizing

By default, `dynamic_axes=True`. This ensures that your exported ONNX model can serve graphs with **any number of nodes and edges** at runtime without needing recompilation:

```python
# Node features shape: (None, in_channels)
# Edge index shape: (2, None)
# Predictions shape: (None, out_channels)
```

### Lightweight Serving with `ONNXModel`

In production, you don't need heavy deep learning frameworks like PyTorch or TensorFlow installed in your serving image. K3-Node's `ONNXModel` requires only `onnxruntime` and `numpy`:

```python
from k3_node.export import ONNXModel
from k3_node.data import Data

# Load exported model into inference engine
runtime = ONNXModel("cora_gcn.onnx")

# Run inference on any PyG/K3-Node Data object, dict, or tensors
preds = runtime.predict(data)
print("Predicted class probabilities shape:", preds.shape)
```

#### Hardware Acceleration (GPU / TensorRT / DirectML)

`ONNXModel` automatically detects and uses the fastest available execution provider on your system:
- **CUDA**: `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`
- **TensorRT**: `providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]`
- **CPU**: Default fallback for lightweight CPU deployments.

```python
# Custom execution provider configuration
gpu_runtime = ONNXModel("cora_gcn.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

---

## 2. TensorFlow Lite (LiteRT) Export for Mobile & Edge

For IoT devices, Raspberry Pi, Android/iOS, and embedded robotics, export your graph models to optimized `.tflite` flatbuffers.

### Exporting with Quantization

```python
from k3_node.models import GCN
from k3_node.export import export_tflite

model = GCN(in_channels=16, hidden_channels=32, out_channels=4)

# 1. Standard FP32 flatbuffer
model.export_tflite("model_fp32.tflite")

# 2. FP16 Quantization (~2x smaller file size, fast GPU/NPU execution)
model.export_tflite("model_fp16.tflite", quantization="fp16")

# 3. Dynamic Range INT8 Quantization (~4x smaller file size, 8-bit weights)
model.export_tflite("model_int8.tflite", quantization="int8_dynamic")
```

### Serving with `TFLiteModel`

Serve the flatbuffer with zero external framework dependencies using `tflite-runtime` or standard `tensorflow`:

```python
from k3_node.export import TFLiteModel

tflite_runtime = TFLiteModel("model_int8.tflite")
predictions = tflite_runtime.predict(data)
```

---

## 3. High-Throughput TensorRT Compilation

For maximum inference throughput on NVIDIA hardware (e.g. A100, H100, RTX 4090, or Jetson Orin), compile your model directly into a TensorRT execution engine (`.engine`):

```python
from k3_node.models import GCN
from k3_node.export import export_tensorrt

model = GCN(in_channels=16, hidden_channels=64, out_channels=7)

# Turn-key compilation: converts to ONNX and builds TensorRT engine with FP16 kernels
engine_path = model.export_tensorrt(
    "model.engine",
    fp16=True,
    min_shapes={"x": (1, 16), "edge_index": (2, 1)},
    opt_shapes={"x": (1000, 16), "edge_index": (2, 5000)},
    max_shapes={"x": (50000, 16), "edge_index": (2, 250000)},
)
```

---

## 4. Production Microservice with FastAPI

Here is a complete, production-ready asynchronous REST microservice for serving graph predictions with sub-millisecond overhead:

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from k3_node.export import ONNXModel
from k3_node.data import Data

app = FastAPI(title="K3-Node GNN Inference Service", version="1.0.0")

# Load model once at server startup
model = ONNXModel("cora_gcn.onnx")

class GraphRequest(BaseModel):
    x: List[List[float]]        # (num_nodes, in_channels)
    edge_index: List[List[int]] # (2, num_edges)

@app.post("/predict")
async def predict_graph(graph_req: GraphRequest):
    try:
        x = np.array(graph_req.x, dtype=np.float32)
        edge_index = np.array(graph_req.edge_index, dtype=np.int64)
        
        data = Data(x=x, edge_index=edge_index)
        logits = model.predict(data)
        
        # Softmax & class predictions
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        classes = np.argmax(probs, axis=-1).tolist()
        
        return {
            "num_nodes": len(classes),
            "predicted_classes": classes,
            "probabilities": probs.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

Run with Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 5. Enterprise Serving with NVIDIA Triton Inference Server

K3-Node provides automated configuration generation for **NVIDIA Triton Inference Server**, creating standard versioned model repositories:

```python
from k3_node.export import generate_triton_config

# Generate versioned repository and Triton config.pbtxt
triton_dir = generate_triton_config(
    model_or_path="cora_gcn.onnx",
    model_name="cora_gcn",
    repository_root="./triton_models",
    max_batch_size=0,  # Dynamic graph input
    backend="onnxruntime",
    version=1,
)
```

This generates the standard Triton repository structure:
```
triton_models/
└── cora_gcn/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### Launching Triton via Docker

```bash
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/triton_models:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

Query the server using the Triton client library:
```python
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
inputs = [
    httpclient.InferInput("x", [num_nodes, in_channels], "FP32"),
    httpclient.InferInput("edge_index", [2, num_edges], "INT64"),
]
inputs[0].set_data_from_numpy(data.x.numpy())
inputs[1].set_data_from_numpy(data.edge_index.numpy())

results = client.infer(model_name="cora_gcn", inputs=inputs)
output = results.as_numpy("output_0")
```

---

## 6. Multi-Backend Transparency

In K3-Node, model export is truly multi-backend. You can train your model with **PyTorch**, **JAX**, or **TensorFlow**:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or "jax", "tensorflow"

from k3_node.tasks import NodeClassifier
from k3_node.datasets import Planetoid

data = Planetoid(root="/tmp/Cora", name="Cora")[0]

# Train with PyTorch
clf = NodeClassifier(backbone="gcn", in_channels=1433, out_channels=7)
clf.fit(data, epochs=20)

# Export directly to ONNX or TFLite - K3-Node handles cross-backend compilation automatically!
clf.export_onnx("cora.onnx")
clf.export_tflite("cora.tflite", quantization="fp16")
```
