"""Model card generator for Hugging Face Hub integration."""

from typing import Any, Dict, Optional


def generate_model_card(
    task_type: str,
    backbone: str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    dataset_name: Optional[str] = None,
    repo_id: Optional[str] = None,
    license: str = "mit",
) -> str:
    r"""Generates a standard Hugging Face Model Card with YAML frontmatter
    and markdown documentation for a K3-Node GNN model.

    Args:
        task_type: Name of the task or model (e.g. 'NodeClassifier', 'SchNet').
        backbone: Name of the backbone architecture (e.g. 'gcn', 'schnet', 'chgnet').
        config: Dictionary containing model architecture and training hyperparameters.
        metrics: Optional dictionary of evaluation metrics (e.g. {'accuracy': 0.82}).
        dataset_name: Optional name of the dataset the model was trained on.
        repo_id: Optional repository ID on Hugging Face Hub.
        license: Open-source license tag. (default: 'mit')

    Returns:
        Formatted markdown string representing the README.md model card.
    """
    model_class = config.get("model_class", task_type)
    task_slug = task_type.lower().replace("classifier", "-classification").replace("regressor", "-regression")
    title = repo_id if repo_id else f"{backbone.upper()} {task_type}"
    dataset_str = f" on `{dataset_name}`" if dataset_name else ""

    # YAML Frontmatter
    frontmatter = f"""---
language:
- en
license: {license}
library_name: keras
tags:
- graph-machine-learning
- gnn
- k3-node
- keras-3
- multi-backend
- {task_slug}
pipeline_tag: graph-ml
---
"""

    # Model Details
    content = f"""# {title}

This is a **{task_type}** Graph Neural Network model{dataset_str} built with [**K3-Node**](https://github.com/anas-rz/k3-node) and **Keras 3**.

It runs natively and seamlessly across **PyTorch**, **JAX**, and **TensorFlow** backends.

## Model Details

- **Model / Task**: `{task_type}`
- **Architecture**: `{backbone}`
- **Library**: `k3-node` (Keras 3)
- **Input Channels**: `{config.get('in_channels', 'Auto')}`
- **Hidden Channels**: `{config.get('hidden_channels', 64)}`
- **Output / Classes**: `{config.get('num_classes') or config.get('out_channels', 'Auto')}`
- **Number of Layers**: `{config.get('num_layers', 2)}`
- **Dropout**: `{config.get('dropout', 0.0)}`
"""

    # Optional metrics
    if metrics:
        content += "\n## Evaluation Metrics\n\n| Metric | Value |\n| :--- | :--- |\n"
        for k, v in metrics.items():
            if isinstance(v, float):
                content += f"| {k} | {v:.4f} |\n"
            else:
                content += f"| {k} | {v} |\n"

    # Usage code snippet
    target_repo = repo_id or "username/model-repo"
    is_task = task_type in ("NodeClassifier", "GraphClassifier", "GraphRegressor", "LinkPredictor")

    if is_task:
        usage_code = f"""from k3_node.tasks import {task_type}

# Load the pretrained model directly from Hugging Face Hub
model = {task_type}.from_pretrained("{target_repo}")

# Run predictions on your graph data
predictions = model.predict(data)"""
    else:
        usage_code = f"""import k3_node as k3

# Load pre-trained weights with one line
model = k3.models.{model_class}.from_pretrained("{target_repo}")

# Run predictions directly on graph or molecular data
predictions = model.predict(data)"""

    content += f"""
## Usage

Install `k3-node` with your preferred backend (PyTorch, JAX, or TensorFlow):

```bash
pip install k3-node huggingface_hub
```

### Loading & Inference

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or "jax", "tensorflow"

{usage_code}
```

## Framework & Citation

This model was trained with **K3-Node**, the multi-backend Graph Neural Network framework built on Keras 3.

```bibtex
@software{{k3_node,
  author = {{Muhammad Anas Raza}},
  title = {{K3-Node: Multi-Backend Graph Neural Networks on Keras 3}},
  year = {{2026}},
  url = {{https://github.com/anas-rz/k3-node}}
}}
```
"""

    return frontmatter + "\n" + content.strip() + "\n"
