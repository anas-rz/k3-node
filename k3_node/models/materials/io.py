"""Checkpoint downloading and pretrained weight loading utilities for MatGL models."""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
import torch
import keras

logger = logging.getLogger(__name__)

HF_MATGL_ORG = "materialyze"

KNOWN_PRETRAINED_MODELS = [
    "CHGNet-PES-MatPES-PBE-2025.2.10",
    "CHGNet-PES-MatPES-r2SCAN-2025.2.10",
    "M3GNet-Eform-MP-2018.6.1",
    "M3GNet-Eform-MP-2019.4.1",
    "M3GNet-PES-ANI-1x-Subset",
    "M3GNet-PES-MatPES-PBE-2025.2",
    "M3GNet-PES-MatPES-r2SCAN-2025.2",
    "MEGNet-BandGap-mfi-MP-2019.4.1",
    "MEGNet-Eform-MP-2018.6.1",
    "QET-PES-MatPES-PBE-2025.2",
    "QET-PES-MatPES-r2SCAN-2025.2",
    "QET-PES-MatQ",
    "SO3Net-PES-ANI-1x-Subset",
    "TensorNet-PES-ANI-1x-Subset",
    "TensorNet-PES-MatPES-PBE-2025.2",
    "TensorNet-PES-MatPES-PBE-2025.2-m",
    "TensorNet-PES-MatPES-r2SCAN-2025.2",
    "TensorNet-PES-MatPES-r2SCAN-2025.2-m",
]


def get_available_pretrained_models() -> List[str]:
    """Return list of available pretrained materials models."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        names = []
        for model_info in api.list_models(author=HF_MATGL_ORG):
            repo_id = str(getattr(model_info, "id", "") or getattr(model_info, "modelId", "") or "")
            if "/" in repo_id:
                names.append(repo_id.split("/", 1)[1])
        if names:
            return sorted(names)
    except Exception as e:
        logger.debug("Hugging Face API listing unavailable, returning static list: %s", e)
    return sorted(KNOWN_PRETRAINED_MODELS)


def download_matgl_checkpoint(
    name_or_repo: str,
    folder: str = "checkpoints",
    log: bool = True,
) -> Dict[str, str]:
    """Download matgl model files (model.json, state.pt) from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    repo_id = name_or_repo if "/" in name_or_repo else f"{HF_MATGL_ORG}/{name_or_repo}"
    if log:
        print(f"Downloading checkpoint files for {repo_id}...")

    os.makedirs(folder, exist_ok=True)
    f_json = hf_hub_download(repo_id=repo_id, filename="model.json", cache_dir=folder)
    f_state = hf_hub_download(repo_id=repo_id, filename="state.pt", cache_dir=folder)

    return {
        "model.json": f_json,
        "state.pt": f_state,
        "repo_id": repo_id,
    }


def _assign_dense_weights(keras_layer, weight_tensor, bias_tensor=None):
    """Assign PyTorch Linear weight/bias to Keras Dense layer."""
    w_np = weight_tensor.detach().cpu().numpy()
    if len(w_np.shape) == 2:
        # PyTorch is [out_features, in_features], Keras is [in_features, out_features]
        w_np = np.transpose(w_np, (1, 0))
    weights = [w_np]
    if bias_tensor is not None:
        b_np = bias_tensor.detach().cpu().numpy()
        weights.append(b_np)
    keras_layer.set_weights(weights)


def load_matgl_weights(
    model: keras.Model,
    state_dict_or_path: Union[str, Dict[str, torch.Tensor]],
    log: bool = False,
) -> int:
    """Load PyTorch checkpoint weights into multi-backend Keras 3 MatGL model."""
    if isinstance(state_dict_or_path, (str, Path)):
        state = torch.load(state_dict_or_path, map_location="cpu", weights_only=True)
    else:
        state = state_dict_or_path

    # Strip potential 'model.' prefix
    cleaned_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned_state[k[6:]] = v
        else:
            cleaned_state[k] = v

    loaded_count = 0

    # 1. Bond expansion centers/width if present
    if hasattr(model, "bond_expansion") and hasattr(model.bond_expansion, "rbf"):
        rbf = model.bond_expansion.rbf
        for k in ("bond_expansion.rbf.centers", "bond_expansion.centers"):
            if k in cleaned_state and hasattr(rbf, "centers"):
                rbf.centers.assign(cleaned_state[k].detach().cpu().numpy())
                loaded_count += 1
        for k in ("bond_expansion.rbf.width", "bond_expansion.width"):
            if k in cleaned_state and hasattr(rbf, "width"):
                rbf.width.assign(cleaned_state[k].detach().cpu().numpy())
                loaded_count += 1

    # 2. Embedding block
    if hasattr(model, "embedding"):
        emb = model.embedding
        if hasattr(emb, "layer_node_embedding") and "embedding.layer_node_embedding.weight" in cleaned_state:
            w = cleaned_state["embedding.layer_node_embedding.weight"].detach().cpu().numpy()
            emb.layer_node_embedding.set_weights([w])
            loaded_count += 1
        if hasattr(emb, "emb") and "embedding.emb.weight" in cleaned_state:
            w = cleaned_state["embedding.emb.weight"].detach().cpu().numpy()
            emb.emb.set_weights([w])
            loaded_count += 1

    # 3. Traverse model sublayers and match with state keys
    for name, sublayer in model.__dict__.items():
        if isinstance(sublayer, keras.layers.Layer):
            # Check for direct linear weight matches
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"
            if w_key in cleaned_state:
                _assign_dense_weights(sublayer, cleaned_state[w_key], cleaned_state.get(b_key))
                loaded_count += 1

    if log:
        print(f"Loaded {loaded_count} weight tensors into {model.__class__.__name__}")

    return loaded_count


def load_model(name_or_path: str, **kwargs) -> keras.Model:
    """Convenience factory to download/load and instantiate any MatGL model."""
    from .megnet import MEGNet
    from .m3gnet import M3GNet
    from .tensornet import TensorNet
    from .chgnet import CHGNet
    from .so3net import SO3Net
    from .grace import GRACE
    from .qet import QET
    from .wrappers import TransformedTargetModel

    if os.path.exists(name_or_path) and os.path.isdir(name_or_path):
        f_json = os.path.join(name_or_path, "model.json")
        f_state = os.path.join(name_or_path, "state.pt")
    else:
        files = download_matgl_checkpoint(name_or_path, **kwargs)
        f_json = files["model.json"]
        f_state = files["state.pt"]

    with open(f_json) as f:
        meta = json.load(f)

    cls_name = meta.get("@class")
    init_kwargs = meta.get("kwargs", {})

    # Handle TransformedTargetModel
    if cls_name == "TransformedTargetModel":
        inner_info = init_kwargs.get("model", {})
        inner_cls = inner_info.get("@class", "MEGNet")
        inner_args = inner_info.get("init_args", {})
        transformer_info = init_kwargs.get("target_transformer", {})
        mean = float(transformer_info.get("mean", 0.0))
        std = float(transformer_info.get("std", 1.0))

        if inner_cls == "MEGNet":
            base_model = MEGNet(**{k: v for k, v in inner_args.items() if k in (
                "dim_node_embedding", "dim_edge_embedding", "dim_state_embedding",
                "nblocks", "cutoff"
            )})
        elif inner_cls == "M3GNet":
            base_model = M3GNet(**{k: v for k, v in inner_args.items() if k in (
                "dim_node_embedding", "dim_edge_embedding", "nblocks", "cutoff", "threebody_cutoff"
            )})
        elif inner_cls == "TensorNet":
            base_model = TensorNet(**{k: v for k, v in inner_args.items() if k in (
                "units", "nblocks", "num_rbf", "cutoff"
            )})
        elif inner_cls == "CHGNet":
            base_model = CHGNet(**{k: v for k, v in inner_args.items() if k in (
                "dim_atom_embedding", "dim_bond_embedding", "cutoff", "threebody_cutoff"
            )})
        else:
            base_model = MEGNet()

        load_matgl_weights(base_model, f_state)
        return TransformedTargetModel(model=base_model, mean=mean, std=std)

    elif cls_name == "MEGNet":
        model = MEGNet()
    elif cls_name == "M3GNet":
        model = M3GNet()
    elif cls_name == "TensorNet":
        model = TensorNet()
    elif cls_name == "CHGNet":
        model = CHGNet()
    elif cls_name == "SO3Net":
        model = SO3Net()
    elif cls_name == "GRACE":
        model = GRACE()
    elif cls_name == "QET":
        model = QET()
    else:
        model = MEGNet()

    load_matgl_weights(model, f_state)
    return model

