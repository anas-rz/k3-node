"""
`k3_node` is a library for building multibackend graph neural networks. 
Built upon Keras 3.0 the models can be trained using TensorFlow, PyTorch, 
or JAX.

To install the package, run:
    
```bash
git clone https://github.com/anas-rz/k3-node.git # bash
```
        
```python
# in your code
import sys
sys.path.append('k3-node')

import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # or 'torch' or 'jax'

from k3_node import ...
```
"""

from k3_node import data
from k3_node.data import Data, Batch
from k3_node import datasets
from k3_node import io
from k3_node import layers
from k3_node import loader
from k3_node import transforms
from k3_node import models
from k3_node import materials
from k3_node import bio
from k3_node import chemistry

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
    "materials",
    "bio",
    "chemistry",
]

