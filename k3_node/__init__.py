"""
`k3_node` is a library for building multibackend graph neural networks. 
Built upon Keras 3.0 the models can be trained using TensorFlow, PyTorch, 
or JAX.

To install the package, run:
    
```bash
pip install k3-node
```
        
```python
# in your code
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # or 'torch' or 'jax'

import k3_node as k3
```
"""
__version__ = "1.0.0"

import warnings

# Index tensors are requested as int64; without `jax_enable_x64`, JAX stores them as int32
# (which is all graphs of this size need) and would warn on every conversion.
warnings.filterwarnings("ignore", message="Explicitly requested dtype int64", category=UserWarning)

from k3_node import data
from k3_node.data import Data, Batch
from k3_node import datasets
from k3_node import io
from k3_node import layers
from k3_node import loader
from k3_node import transforms
from k3_node import models
from k3_node import applications

from k3_node import metrics
from k3_node import tasks
from k3_node.tasks import (
    NodeClassifier,
    NodeRegressor,
    GraphClassifier,
    GraphRegressor,
    LinkPredictor,
)

from k3_node import etl
from k3_node.etl import (
    TableToGraph,
    TabularToGraph,
    table_to_graph,
    RelationalToGraph,
    relational_to_graph,
)

from k3_node import hub
from k3_node.hub import (
    from_pretrained,
    push_to_hub,
    save_pretrained,
    load_dataset_from_hub,
    push_dataset_to_hub,
)

from k3_node import export
from k3_node.export import (
    export_onnx,
    export_tflite,
    export_tensorrt,
    generate_triton_config,
    ONNXModel,
    TFLiteModel,
)

from k3_node import rag
from k3_node.layers import kge

__all__ = [
    "data",
    "Data",
    "Batch",
    "datasets",
    "io",
    "layers",
    "loader",
    "transforms",
    "models",
    "applications",
    "metrics",
    "tasks",
    "NodeClassifier",
    "NodeRegressor",
    "GraphClassifier",
    "GraphRegressor",
    "LinkPredictor",
    "etl",
    "TableToGraph",
    "TabularToGraph",
    "table_to_graph",
    "RelationalToGraph",
    "relational_to_graph",
    "hub",
    "from_pretrained",
    "push_to_hub",
    "save_pretrained",
    "load_dataset_from_hub",
    "push_dataset_to_hub",
    "export",
    "export_onnx",
    "export_tflite",
    "export_tensorrt",
    "generate_triton_config",
    "ONNXModel",
    "TFLiteModel",
    "rag",
    "kge",
]

