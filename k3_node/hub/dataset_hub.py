"""Hugging Face Hub integration for Graph Datasets in K3-Node."""

import os
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
from keras import ops

from k3_node.data import Data, Batch


def _to_numpy_dict(data: Data) -> dict:
    r"""Converts all tensor attributes of a Data object to numpy arrays."""
    arrays = {}
    for key, val in data.items():
        if val is not None:
            try:
                arrays[key] = ops.convert_to_numpy(val)
            except Exception:
                arrays[key] = np.array(val)
    return arrays


def save_graph_dataset(
    dataset: Union[Data, List[Data]],
    filepath: Union[str, Path],
) -> Path:
    r"""Saves a single graph (Data) or a list of graphs (List[Data])
    into a compressed multi-backend `.npz` file.

    Args:
        dataset: A :class:`k3_node.data.Data` object or list of :class:`Data` objects.
        filepath: Output `.npz` file path.

    Returns:
        Path of the saved file.
    """
    path = Path(filepath)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, (list, tuple)):
        arrays = {
            "_k3_node_is_list": np.array(True),
            "_k3_node_num_graphs": np.array(len(dataset)),
        }
        for idx, g in enumerate(dataset):
            g_dict = _to_numpy_dict(g)
            for k, v in g_dict.items():
                arrays[f"g{idx}_{k}"] = v
    elif isinstance(dataset, Data):
        arrays = _to_numpy_dict(dataset)
        arrays["_k3_node_is_list"] = np.array(False)
    else:
        raise TypeError(f"Unsupported dataset type '{type(dataset).__name__}'. Expected Data or List[Data].")

    np.savez_compressed(str(path), **arrays)
    return path


def load_graph_dataset(filepath: Union[str, Path]) -> Union[Data, List[Data]]:
    r"""Loads a graph or collection of graphs from a `.npz` file.

    Args:
        filepath: Path to the `.npz` dataset file.

    Returns:
        A :class:`Data` object or a list of :class:`Data` objects.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file '{path}' does not exist.")

    with np.load(str(path), allow_pickle=True) as data_dict:
        is_list = bool(data_dict.get("_k3_node_is_list", False))

        if is_list:
            num_graphs = int(data_dict["_k3_node_num_graphs"])
            graphs = []
            for idx in range(num_graphs):
                prefix = f"g{idx}_"
                g_dict = {
                    k[len(prefix):]: v
                    for k, v in data_dict.items()
                    if k.startswith(prefix)
                }
                graphs.append(Data(**g_dict))
            return graphs
        else:
            filtered = {k: v for k, v in data_dict.items() if not k.startswith("_k3_node_")}
            return Data(**filtered)


def generate_dataset_card(
    dataset: Union[Data, List[Data]],
    repo_id: str,
    description: Optional[str] = None,
    license: str = "mit",
) -> str:
    r"""Generates a standard Hugging Face Dataset Card (README.md)."""
    is_collection = isinstance(dataset, (list, tuple))
    num_graphs = len(dataset) if is_collection else 1
    sample = dataset[0] if is_collection else dataset

    card = f"""---
language:
- en
license: {license}
tags:
- graph-machine-learning
- gnn
- k3-node
- graph-dataset
size_categories:
- {'10K<n<100K' if num_graphs > 10000 else '1K<n<10K' if num_graphs > 1000 else 'n<1K'}
---

# {repo_id}

{description or f"Graph dataset consisting of {num_graphs} graph(s) formatted for [**K3-Node**](https://github.com/anas-rz/k3-node)."}

## Dataset Summary

- **Total Graphs**: `{num_graphs}`
- **Sample Node Features**: `{sample.x.shape[-1] if hasattr(sample, 'x') and sample.x is not None else 'None'}`
- **Sample Edge Count**: `{sample.edge_index.shape[-1] if hasattr(sample, 'edge_index') and sample.edge_index is not None else 'None'}`
- **Attributes**: `{list(sample.keys()) if hasattr(sample, 'keys') else 'N/A'}`

## Usage

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or "jax", "tensorflow"

from k3_node.hub import load_dataset_from_hub

# Download and load directly from Hugging Face Hub
data = load_dataset_from_hub("{repo_id}")
print(data)
```
"""
    return card.strip() + "\n"


def push_dataset_to_hub(
    dataset: Union[Data, List[Data]],
    repo_id: str,
    token: Optional[Union[str, bool]] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    description: Optional[str] = None,
    license: str = "mit",
) -> str:
    r"""Pushes a graph dataset to the Hugging Face Hub under a dataset repository.

    Args:
        dataset: A :class:`k3_node.data.Data` object or list of :class:`Data` objects.
        repo_id: Hugging Face dataset repository ID (e.g. ``"username/my-graph-data"``).
        token: Optional authentication token.
        private: Whether the repository should be private. (default: ``False``)
        commit_message: Commit message for the upload.
        description: Optional dataset description.
        license: License tag. (default: ``"mit"``)

    Returns:
        URL of the dataset on Hugging Face Hub.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "The `huggingface_hub` package is required to push datasets to Hugging Face Hub. "
            "Install it via `pip install huggingface_hub`."
        )

    import tempfile

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        save_graph_dataset(dataset, tmp_dir_path / "graph_data.npz")

        card_md = generate_dataset_card(
            dataset=dataset,
            repo_id=repo_id,
            description=description,
            license=license,
        )
        with open(tmp_dir_path / "README.md", "w", encoding="utf-8") as f:
            f.write(card_md)

        api.upload_folder(
            folder_path=str(tmp_dir_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message or "Upload K3-Node graph dataset",
        )

    return f"https://huggingface.co/datasets/{repo_id}"


def load_dataset_from_hub(
    repo_id: str,
    filename: str = "graph_data.npz",
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Union[Data, List[Data]]:
    r"""Downloads and loads a graph dataset from the Hugging Face Hub.

    Args:
        repo_id: Hugging Face dataset repository ID (e.g. ``"username/my-graph-data"``).
        filename: Name of the dataset file in the repo. (default: ``"graph_data.npz"``)
        token: Optional authentication token.
        cache_dir: Optional cache directory.

    Returns:
        Restored :class:`Data` or :class:`List[Data]` object.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "The `huggingface_hub` package is required to load datasets from Hugging Face Hub. "
            "Install it via `pip install huggingface_hub`."
        )

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        cache_dir=cache_dir,
    )
    return load_graph_dataset(downloaded)
