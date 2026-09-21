import math
from typing import List, Optional, Union
from keras import layers, ops
import numpy as np

from .base import Aggregation
from .utils import MultiheadAttentionBlock


class PatchTransformerAggregation(Aggregation):
    r"""Performs patch transformer aggregation in which the elements to
    aggregate are processed by multi-head attention blocks across patches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        hidden_channels: int,
        num_transformer_blocks: int = 1,
        heads: int = 1,
        dropout: float = 0.0,
        aggr: Union[str, List[str]] = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_channels = hidden_channels
        self.aggrs = [aggr] if isinstance(aggr, str) else list(aggr)

        self.lin = layers.Dense(hidden_channels)
        self.pad_projector = layers.Dense(hidden_channels)
        self.blocks = [
            MultiheadAttentionBlock(
                channels=hidden_channels,
                heads=heads,
                layer_norm=True,
                dropout=dropout,
            )
            for _ in range(num_transformer_blocks)
        ]
        self.fc = layers.Dense(out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.pad_projector.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.fc.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
        training: bool = False,
        **kwargs,
    ):
        if max_num_elements is None:
            from k3_node.layers.conv.utils import is_tracing
            if is_tracing(x) or is_tracing(index):
                if hasattr(x, "shape") and x.shape[0] is not None:
                    max_num_elements = int(x.shape[0])
                else:
                    max_num_elements = 16
            elif ptr is not None:
                ptr_np = ops.convert_to_numpy(ptr)
                count = ptr_np[1:] - ptr_np[:-1]
                max_num_elements = int(np.max(count)) if len(count) > 0 else 1
            else:
                idx_np = ops.convert_to_numpy(index).astype(np.int64)
                counts = np.bincount(idx_np)
                max_num_elements = int(np.max(counts)) if len(counts) > 0 else 1

        # Ensure max_num_elements is a multiple of patch_size
        num_patches = max(math.ceil(max_num_elements / self.patch_size), 1)
        target_elements = num_patches * self.patch_size

        x_dense, _ = self.to_dense_batch(
            x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
            max_num_elements=target_elements,
        )

        B = ops.shape(x_dense)[0]
        x_proj = self.lin(x_dense)  # [B, target_elements, hidden_channels]

        # Reshape to patches: [B, num_patches, patch_size * hidden_channels]
        x_patches = ops.reshape(x_proj, (B, num_patches, self.patch_size * self.hidden_channels))
        x_patches = self.pad_projector(x_patches)  # [B, num_patches, hidden_channels]

        # Process through transformer blocks
        for block in self.blocks:
            x_patches = block(x_patches, x_patches, training=training)

        outs = []
        for aggr_mode in self.aggrs:
            if aggr_mode == "mean":
                outs.append(ops.mean(x_patches, axis=1))
            elif aggr_mode == "sum":
                outs.append(ops.sum(x_patches, axis=1))
            elif aggr_mode == "max":
                outs.append(ops.max(x_patches, axis=1))
            elif aggr_mode == "min":
                outs.append(ops.min(x_patches, axis=1))
            elif aggr_mode == "var":
                outs.append(ops.var(x_patches, axis=1))
            elif aggr_mode == "std":
                outs.append(ops.std(x_patches, axis=1))

        combined = ops.concatenate(outs, axis=-1) if len(outs) > 1 else outs[0]
        return self.fc(combined)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, patch_size={self.patch_size})"
        )

