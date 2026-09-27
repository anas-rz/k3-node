"""Hugging Face Hub integration mixin and methods for K3-Node models and tasks."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union
import inspect
import numpy as np

from k3_node.hub.model_card import generate_model_card

T = TypeVar("T", bound="K3NodeHubMixin")


class K3NodeHubMixin:
    r"""Mixin class providing seamless saving, loading, and publishing of
    K3-Node GNN models and tasks to/from the Hugging Face Hub.

    Methods:
        save_pretrained: Saves model weights, config, and Model Card to a local directory.
        from_pretrained: Loads a model from a local folder or Hugging Face Hub repository.
        push_to_hub: Automatically saves and pushes the model to a Hugging Face Hub repo.
        predict: Runs inference on graph data (PyG Data, molecular structures, dicts, or tensors).
    """

    def _get_config(self) -> Dict[str, Any]:
        r"""Extracts serializable architecture and task hyperparameters."""
        config: Dict[str, Any] = {
            "model_class": self.__class__.__name__,
        }
        # Include task_type for tasks
        if (
            hasattr(self, "backbone")
            or self.__class__.__name__.endswith("Classifier")
            or self.__class__.__name__.endswith("Regressor")
            or self.__class__.__name__.endswith("Predictor")
        ):
            config["task_type"] = self.__class__.__name__

        for attr in [
            "backbone",
            "in_channels",
            "hidden_channels",
            "out_channels",
            "num_classes",
            "num_layers",
            "num_filters",
            "num_interactions",
            "num_gaussians",
            "cutoff",
            "max_num_neighbors",
            "readout",
            "dipole",
            "mean",
            "std",
            "units",
            "nblocks",
            "dim_atom_embedding",
            "dim_bond_embedding",
            "dim_angle_embedding",
            "num_blocks",
            "pooling",
            "decoder",
            "loss_name",
            "act",
            "multi_label",
        ]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    if isinstance(val, (int, float, str, bool, list, dict)):
                        config[attr] = val
                    elif isinstance(val, tuple):
                        config[attr] = list(val)
                    elif hasattr(val, "__name__"):
                        config[attr] = val.__name__

        # Handle dropout specifically
        if hasattr(self, "dropout_p") and isinstance(self.dropout_p, (int, float)):
            config["dropout"] = float(self.dropout_p)
        elif hasattr(self, "dropout") and isinstance(self.dropout, (int, float)):
            config["dropout"] = float(self.dropout)

        # Signature inspection for any remaining constructor arguments
        try:
            sig = inspect.signature(self.__class__.__init__)
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "args", "kwargs", "name"):
                    continue
                if param_name not in config and hasattr(self, param_name):
                    val = getattr(self, param_name)
                    if isinstance(val, (int, float, str, bool, list, dict)):
                        config[param_name] = val
                    elif isinstance(val, tuple):
                        config[param_name] = list(val)
                    elif hasattr(val, "__name__"):
                        config[param_name] = val.__name__
        except Exception:
            pass

        if hasattr(self, "backbone_kwargs") and isinstance(self.backbone_kwargs, dict):
            config["backbone_kwargs"] = self.backbone_kwargs

        return config

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        dataset_name: Optional[str] = None,
        repo_id: Optional[str] = None,
        license: str = "mit",
        **kwargs,
    ) -> Path:
        r"""Saves model weights, config.json, and README.md (Model Card) to disk.

        Args:
            save_directory: Directory path to save model files in.
            config: Optional custom configuration dictionary.
            metrics: Optional evaluation metrics dictionary to include in Model Card.
            dataset_name: Optional dataset name for the Model Card.
            repo_id: Optional Hugging Face repository ID.
            license: License identifier. (default: ``"mit"``)

        Returns:
            Path object of the saved directory.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Config
        final_config = config or self._get_config()
        config_path = save_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=2)

        # 2. Weights
        weights_path = save_dir / "model.weights.h5"
        if getattr(self, "model", None) is not None and hasattr(self.model, "save_weights"):
            self.model.save_weights(str(weights_path))
        elif hasattr(self, "save_weights"):
            self.save_weights(str(weights_path))
        else:
            raise RuntimeError(f"Cannot save weights for model of type '{self.__class__.__name__}'.")

        # 3. Model Card (README.md)
        task_type = final_config.get("task_type", self.__class__.__name__)
        backbone_str = str(final_config.get("backbone", final_config.get("model_class", "gnn")))
        card_content = generate_model_card(
            task_type=task_type,
            backbone=backbone_str,
            config=final_config,
            metrics=metrics,
            dataset_name=dataset_name,
            repo_id=repo_id,
            license=license,
        )
        readme_path = save_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        return save_dir

    def predict(self, data: Any = None, *args: Any, **kwargs: Any) -> Any:
        r"""Infers predictions on graph or molecular data.

        Supports PyG / K3-Node ``Data`` objects (extracting ``(z, pos, batch)``
        for molecular models or ``(x, edge_index, ...)`` for standard GNNs),
        dictionaries, tuples of tensors, or direct positional tensors.

        Args:
            data: Input graph or molecule Data, dict, or tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Model prediction tensor or array.
        """
        # If this is a BaseTask wrapping a model with its own task predict logic:
        if hasattr(self, "_task_predict"):
            return self._task_predict(data, *args, **kwargs)

        # 1. Molecular data (z, pos, batch)
        if hasattr(data, "z") and hasattr(data, "pos"):
            batch = getattr(data, "batch", None)
            return self(data.z, data.pos, batch=batch, training=False, **kwargs)

        # 2. Graph data with node features & edges (x, edge_index, ...)
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            edge_weight = getattr(data, "edge_weight", None)
            edge_attr = getattr(data, "edge_attr", None)
            batch = getattr(data, "batch", None)
            call_kwargs = {}
            try:
                sig = inspect.signature(self.call if hasattr(self, "call") else self.__call__)
                params = sig.parameters
                if "edge_weight" in params and edge_weight is not None:
                    call_kwargs["edge_weight"] = edge_weight
                elif "edge_attr" in params and edge_attr is not None:
                    call_kwargs["edge_attr"] = edge_attr
                if "batch" in params and batch is not None:
                    call_kwargs["batch"] = batch
            except Exception:
                pass
            call_kwargs.update(kwargs)
            return self(data.x, data.edge_index, training=False, **call_kwargs)

        # 3. Dictionary input (e.g., Materials models CHGNet, MEGNet, M3GNet)
        if isinstance(data, dict):
            if "z" in data and "pos" in data:
                return self(data["z"], data["pos"], batch=data.get("batch"), training=False, **kwargs)
            elif "x" in data and "edge_index" in data:
                return self(data["x"], data["edge_index"], training=False, **kwargs)
            else:
                return self(data, training=False, **kwargs)

        # 4. Tuple or list of inputs
        if isinstance(data, (tuple, list)):
            return self(*data, training=False, **kwargs)

        # 5. Direct arguments
        if data is not None and len(args) > 0:
            return self(data, *args, training=False, **kwargs)
        elif data is not None:
            return self(data, training=False, **kwargs)
        else:
            return self(*args, training=False, **kwargs)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        repo_id_or_path: Union[str, Path],
        revision: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **model_kwargs,
    ) -> T:
        r"""Loads a pretrained K3-Node task or model from a local folder or Hugging Face Hub.

        Args:
            repo_id_or_path: Local directory path or Hugging Face repo ID (e.g. ``"k3-node/schnet-qm9"``).
            revision: Specific git revision/branch on Hugging Face Hub.
            token: Hugging Face authentication token.
            cache_dir: Cache directory for downloaded Hub files.
            **model_kwargs: Overrides for configuration parameters.

        Returns:
            Restored and initialized model or task instance with loaded weights.
        """
        repo_path = Path(repo_id_or_path)

        if repo_path.is_dir():
            config_path = repo_path / "config.json"
            weights_path = repo_path / "model.weights.h5"
            if not config_path.exists():
                raise FileNotFoundError(f"config.json not found in local directory '{repo_path}'.")
            if not weights_path.exists():
                raise FileNotFoundError(f"model.weights.h5 not found in local directory '{repo_path}'.")
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ImportError(
                    "The `huggingface_hub` package is required to load models from Hugging Face Hub. "
                    "Install it via `pip install huggingface_hub`."
                )

            repo_id = str(repo_id_or_path)
            config_file = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )
            weights_file = hf_hub_download(
                repo_id=repo_id,
                filename="model.weights.h5",
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )
            config_path = Path(config_file)
            weights_path = Path(weights_file)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Merge any user overrides
        config.update(model_kwargs)

        # Resolve target class
        target_cls = cls
        if target_cls.__name__ in ("BaseTask", "K3NodeHubMixin"):
            if "task_type" in config:
                from k3_node import tasks
                target_cls = getattr(tasks, config["task_type"], None)
            if target_cls is None or target_cls.__name__ in ("BaseTask", "K3NodeHubMixin"):
                model_class = config.get("model_class")
                if model_class:
                    from k3_node import models
                    target_cls = getattr(models, model_class, cls)

        # Extract arguments compatible with target_cls.__init__
        sig = inspect.signature(target_cls.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}

        init_kwargs = {}
        for k, v in config.items():
            if k in accepted_params and v is not None:
                init_kwargs[k] = v

        if "backbone_kwargs" in config and "backbone_kwargs" not in accepted_params:
            init_kwargs.update(config["backbone_kwargs"])

        # Instantiate model or task
        instance = target_cls(**init_kwargs)

        # Initialize model topology if task
        if hasattr(instance, "_init_model"):
            instance._init_model(None)

        # Build variables so weights can be loaded
        _build_model_if_needed(instance, config)

        # Load weights
        if hasattr(instance, "model") and instance.model is not None and hasattr(instance.model, "load_weights"):
            instance.model.load_weights(str(weights_path))
            instance._is_compiled = True
        elif hasattr(instance, "load_weights"):
            instance.load_weights(str(weights_path))
        else:
            raise RuntimeError(f"Instance '{instance}' does not support load_weights.")

        return instance

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[Union[str, bool]] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        dataset_name: Optional[str] = None,
        license: str = "mit",
        **kwargs,
    ) -> str:
        r"""Saves the model and pushes it directly to the Hugging Face Hub.

        Args:
            repo_id: Hugging Face repo ID in format ``"username/model_name"`` or ``"org/model_name"``.
            token: Optional Hugging Face auth token. If not passed, uses cached credentials.
            private: Whether the repository should be private. (default: ``False``)
            commit_message: Optional commit message for the upload.
            metrics: Optional dictionary of evaluation metrics to document in the Model Card.
            dataset_name: Optional dataset name for the Model Card.
            license: License tag. (default: ``"mit"``)

        Returns:
            Web URL of the repository on Hugging Face Hub.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "The `huggingface_hub` package is required to push models to Hugging Face Hub. "
                "Install it via `pip install huggingface_hub`."
            )

        import tempfile

        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(
                tmpdir,
                metrics=metrics,
                dataset_name=dataset_name,
                repo_id=repo_id,
                license=license,
                **kwargs,
            )
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message or "Upload K3-Node GNN model with weights and model card",
            )

        return f"https://huggingface.co/{repo_id}"


def _build_model_if_needed(instance: Any, config: Dict[str, Any]) -> None:
    r"""Builds/initializes weights for tasks and models so weights can be loaded."""
    cls_name = instance.__class__.__name__

    # Task estimators
    if cls_name == "NodeClassifier":
        in_c = getattr(instance, "in_channels", None) or 16
        dummy_x = np.zeros((2, in_c), dtype="float32")
        dummy_edge = np.zeros((2, 1), dtype="int64")
        instance.model((dummy_x, dummy_edge))
        return
    elif cls_name in ("GraphClassifier", "GraphRegressor"):
        in_c = getattr(instance, "in_channels", None) or 16
        dummy_x = np.zeros((2, in_c), dtype="float32")
        dummy_edge = np.zeros((2, 1), dtype="int64")
        dummy_batch = np.zeros((2,), dtype="int64")
        if hasattr(instance.model, "num_graphs"):
            instance.model.num_graphs = 1
        instance.model((dummy_x, dummy_edge, dummy_batch))
        return
    elif cls_name == "LinkPredictor":
        in_c = getattr(instance, "in_channels", None) or 16
        dummy_x = np.zeros((2, in_c), dtype="float32")
        dummy_edge = np.zeros((2, 1), dtype="int64")
        dummy_label_idx = np.zeros((2, 1), dtype="int64")
        instance.model(((dummy_x, dummy_edge), dummy_label_idx))
        return
    elif hasattr(instance, "model") and instance.model is not None:
        if not getattr(instance.model, "built", False):
            try:
                instance.model.build(None)
            except Exception:
                pass
        return

    # Direct models
    # Molecular 3D models (SchNet, DimeNet, DimeNetPlusPlus, ViSNet)
    if cls_name in ("SchNet", "DimeNet", "DimeNetPlusPlus", "ViSNet", "GNNFF"):
        try:
            import keras.ops as ops
            z = ops.convert_to_tensor([1, 6], dtype="int32")
            pos = ops.convert_to_tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32")
            instance(z, pos)
            return
        except Exception:
            pass

    # Materials models (CHGNet, MEGNet, M3GNet, TensorNet, SO3Net)
    if cls_name in ("CHGNet", "MEGNet", "M3GNet", "TensorNet", "SO3Net"):
        try:
            crystal = {
                "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.2, 0.8], [1.5, 1.5, 1.0]], dtype=np.float32),
                "edge_index": np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=np.int32),
                "line_edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32),
                "node_type": np.array([6, 8, 1, 6], dtype=np.int32),
                "batch": np.array([0, 0, 0, 0], dtype=np.int32),
                "state_attr": np.array([[0.0, 0.0]], dtype=np.float32),
            }
            instance(crystal)
            return
        except Exception:
            pass

    # Standard GNNs (GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN, BasicGNN)
    in_c = getattr(instance, "in_channels", None) or config.get("in_channels") or 16
    try:
        import keras.ops as ops
        dummy_x = ops.zeros((2, in_c), dtype="float32")
        dummy_edge = ops.zeros((2, 1), dtype="int64")
        instance(dummy_x, dummy_edge)
        return
    except Exception:
        pass

    # Generic fallback
    if hasattr(instance, "build"):
        try:
            instance.build(None)
        except Exception:
            pass


# Standalone functional API wrappers
def save_pretrained(
    model_or_task: Any,
    save_directory: Union[str, Path],
    **kwargs,
) -> Path:
    r"""Saves a model or task to disk in Hugging Face Hub format."""
    if hasattr(model_or_task, "save_pretrained"):
        return model_or_task.save_pretrained(save_directory, **kwargs)
    raise TypeError(f"Object of type '{type(model_or_task).__name__}' does not support save_pretrained.")


def from_pretrained(
    repo_id_or_path: Union[str, Path],
    task_cls: Optional[Type[Any]] = None,
    **kwargs,
) -> Any:
    r"""Loads a model or task from a local directory or Hugging Face Hub."""
    loader_cls = task_cls or K3NodeHubMixin
    return loader_cls.from_pretrained(repo_id_or_path, **kwargs)


def push_to_hub(
    model_or_task: Any,
    repo_id: str,
    **kwargs,
) -> str:
    r"""Pushes a model or task directly to the Hugging Face Hub."""
    if hasattr(model_or_task, "push_to_hub"):
        return model_or_task.push_to_hub(repo_id, **kwargs)
    raise TypeError(f"Object of type '{type(model_or_task).__name__}' does not support push_to_hub.")
