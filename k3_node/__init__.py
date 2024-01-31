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
