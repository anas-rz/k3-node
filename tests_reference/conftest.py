import os

# PyTorch Geometric (PyG) parity tests run using the PyTorch backend
# to ensure direct tensor compatibility and numerical parity with PyG.
os.environ["KERAS_BACKEND"] = "torch"

