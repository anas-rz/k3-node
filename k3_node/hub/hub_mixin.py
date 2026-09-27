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
    """

    def _get_config(self) -> Dict[str, Any]:
        r"""Extracts serializable architecture and task hyperparameters."""
        config: Dict[str, Any] = {
            "task_type": self.__class__.__name__,
        }
        for attr in [
            "backbone",
            "in_channels",
            "hidden_channels",
            "out_channels",
            "num_classes",
            "num_layers",
            "pooling",
            "decoder",
            "loss_name",
            "dropout",
            "multi_label",
        ]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    if hasattr(val, "__class__") and not isinstance(val, (int, float, str, bool, list, dict)):
                        val = val.__class__.__name__
                    config[attr] = val

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
        if getattr(self, "model", None) is not None:
            weights_path = save_dir / "model.weights.h5"
            self.model.save_weights(str(weights_path))
        else:
            raise RuntimeError("Cannot save an uninitialized model. Initialize or train the model first.")

        # 3. Model Card (README.md)
        task_type = final_config.get("task_type", self.__class__.__name__)
        backbone_str = str(final_config.get("backbone", "gnn"))
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
            repo_id_or_path: Local directory path or Hugging Face repo ID (e.g. ``"org/cora-gcn"``).
            revision: Specific git revision/branch on Hugging Face Hub.
            token: Hugging Face authentication token.
            cache_dir: Cache directory for downloaded Hub files.
            **model_kwargs: Overrides for configuration parameters.

        Returns:
            Restored and initialized task instance with loaded weights.
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

        # Resolve target task class
        target_cls = cls
        if target_cls.__name__ in ("BaseTask", "K3NodeHubMixin"):
            task_type = config.get("task_type", "NodeClassifier")
            from k3_node import tasks
            target_cls = getattr(tasks, task_type, cls)

        # Extract arguments compatible with target_cls.__init__
        sig = inspect.signature(target_cls.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}

        init_kwargs = {}
        for k, v in config.items():
            if k in accepted_params:
                init_kwargs[k] = v

        if "backbone_kwargs" in config and "backbone_kwargs" not in accepted_params:
            init_kwargs.update(config["backbone_kwargs"])

        # Instantiate task
        instance = target_cls(**init_kwargs)

        # Initialize model topology
        if hasattr(instance, "_init_model"):
            instance._init_model(None)

        # Build variables with dummy forward pass so weights can be loaded
        in_c = getattr(instance, "in_channels", None) or 16
        cls_name = instance.__class__.__name__

        if cls_name == "NodeClassifier":
            dummy_x = np.zeros((2, in_c), dtype="float32")
            dummy_edge = np.zeros((2, 1), dtype="int64")
            instance.model((dummy_x, dummy_edge))
        elif cls_name in ("GraphClassifier", "GraphRegressor"):
            dummy_x = np.zeros((2, in_c), dtype="float32")
            dummy_edge = np.zeros((2, 1), dtype="int64")
            dummy_batch = np.zeros((2,), dtype="int64")
            if hasattr(instance.model, "num_graphs"):
                instance.model.num_graphs = 1
            instance.model((dummy_x, dummy_edge, dummy_batch))
        elif cls_name == "LinkPredictor":
            dummy_x = np.zeros((2, in_c), dtype="float32")
            dummy_edge = np.zeros((2, 1), dtype="int64")
            dummy_label_idx = np.zeros((2, 1), dtype="int64")
            instance.model(((dummy_x, dummy_edge), dummy_label_idx))
        elif hasattr(instance, "model") and instance.model is not None and not instance.model.built:
            try:
                instance.model.build(None)
            except Exception:
                pass

        # Load weights
        instance.model.load_weights(str(weights_path))
        instance._is_compiled = True
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
