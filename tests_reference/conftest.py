import os
import sys
import os.path as osp

# PyTorch Geometric (PyG) parity tests run using the PyTorch backend
# to ensure direct tensor compatibility and numerical parity with PyG.
os.environ["KERAS_BACKEND"] = "torch"

repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
pyg_path = osp.join(repo_root, "pytorch_geometric")
if osp.exists(pyg_path) and pyg_path not in sys.path:
    sys.path.insert(0, pyg_path)

if "xxhash" not in sys.modules:
    try:
        import xxhash
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["xxhash"] = MagicMock()

