r"""Captum model interpretability integration.

Note: Captum is a PyTorch-specific model interpretability library.
In k3-node (a multi-backend Keras 3 graph library supporting TensorFlow,
PyTorch, and JAX), Captum integrations are provided for PyTorch backend
compatibility or documented as PyTorch-exclusive.
"""

from typing import Optional, Union, Any


def to_captum_model(
    model: Any,
    mask_type: str = "edge",
    output_idx: Optional[int] = None,
    metadata: Optional[Any] = None,
):
    r"""Converts a model into a Captum-compatible module.

    .. note::
        Captum is a PyTorch-exclusive library. This function requires
        PyTorch backend and torch.nn.Module models.
    """
    raise NotImplementedError(
        "Captum integration is PyTorch-specific and requires native torch.nn.Module."
    )


def to_captum_input(
    x: Any,
    edge_index: Any,
    mask_type: str = "edge",
    *args,
    **kwargs,
):
    r"""Converts graph inputs into Captum-compatible inputs."""
    raise NotImplementedError(
        "Captum integration is PyTorch-specific and requires native torch.nn.Module."
    )


def captum_output_to_dicts(
    attributions: Any,
    mask_type: str = "edge",
    *args,
    **kwargs,
):
    r"""Converts Captum attributions into dictionaries."""
    raise NotImplementedError(
        "Captum integration is PyTorch-specific and requires native torch.nn.Module."
    )

