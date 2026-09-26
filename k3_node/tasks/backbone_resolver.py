"""Backbone resolver mapping string identifiers to K3-Node models."""

from typing import Any, Dict, Optional, Union
import keras

from k3_node import models


BACKBONE_REGISTRY = {
    "gcn": models.GCN,
    "gat": models.GAT,
    "sage": models.GraphSAGE,
    "graphsage": models.GraphSAGE,
    "gin": models.GIN,
    "pna": models.PNA,
    "edge_cnn": models.EdgeCNN,
    "edgecnn": models.EdgeCNN,
    "mlp": models.MLP,
    "linkx": models.LINKX,
    "pmlp": models.PMLP,
    "sgformer": models.SGFormer,
    "polynormer": models.Polynormer,
    "schnet": models.SchNet,
    "dimenet": models.DimeNet,
    "dimenet++": models.DimeNetPlusPlus,
    "dimenetplusplus": models.DimeNetPlusPlus,
    "attentive_fp": models.AttentiveFP,
    "attentivefp": models.AttentiveFP,
    "graph_unet": models.GraphUNet,
}


def resolve_backbone(
    backbone: Union[str, keras.Model, Any],
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    num_layers: int = 2,
    dropout: float = 0.5,
    **kwargs,
) -> keras.Model:
    r"""Resolves a backbone string or instance into a compiled/callable Keras model."""
    if isinstance(backbone, keras.Model):
        return backbone

    if not isinstance(backbone, str):
        raise TypeError(
            f"Expected backbone to be a string name or keras.Model instance, got {type(backbone)}"
        )

    name = backbone.lower().strip()
    if name not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{backbone}'. Available backbones: {available}")

    cls = BACKBONE_REGISTRY[name]

    # Handle models with specialized constructors
    if name in ("schnet",):
        return cls(hidden_channels=hidden_channels, num_filters=hidden_channels, num_interactions=num_layers, **kwargs)
    elif name in ("dimenet", "dimenet++", "dimenetplusplus"):
        return cls(hidden_channels=hidden_channels, out_channels=out_channels, num_blocks=num_layers, **kwargs)
    elif name in ("attentive_fp", "attentivefp"):
        edge_dim = kwargs.pop("edge_dim", in_channels)
        num_timesteps = kwargs.pop("num_timesteps", 2)
        return cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
            **kwargs,
        )
    elif name in ("mlp",):
        channel_list = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        return cls(channel_list=channel_list, dropout=dropout, **kwargs)
    elif name in ("linkx",):
        num_nodes = kwargs.pop("num_nodes", 1000)
        return cls(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
    elif name in ("sgformer",):
        return cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
    elif name in ("polynormer",):
        return cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
    elif name in ("graph_unet",):
        return cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=num_layers,
            **kwargs,
        )
    else:
        # Standard BasicGNN (GCN, GAT, SAGE, GIN, PNA, EdgeCNN)
        return cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            **kwargs,
        )
