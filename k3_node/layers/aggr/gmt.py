from typing import Optional
from keras import ops

from .base import Aggregation
from .utils import PoolingByMultiheadAttention, SetAttentionBlock


class GraphMultisetTransformer(Aggregation):
    r"""The Graph Multiset Transformer pooling operator from the
    `"Accurate Learning of Graph Representations
    with Graph Multiset Pooling" <https://arxiv.org/abs/2102.11533>`_ paper.
    """

    def __init__(
        self,
        channels: int,
        k: int,
        num_encoder_blocks: int = 1,
        heads: int = 1,
        layer_norm: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.k = k
        self.heads = heads
        self.layer_norm = layer_norm
        self.dropout = dropout

        self.pma1 = PoolingByMultiheadAttention(channels, k, heads, layer_norm, dropout)
        self.encoders = [
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_encoder_blocks)
        ]
        self.pma2 = PoolingByMultiheadAttention(channels, 1, heads, layer_norm, dropout)

    def reset_parameters(self):
        self.pma1.reset_parameters()
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.pma2.reset_parameters()

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
        x_dense, mask = self.to_dense_batch(
            x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
            max_num_elements=max_num_elements,
        )

        x_dense = self.pma1(x_dense, mask=mask, training=training)

        for encoder in self.encoders:
            x_dense = encoder(x_dense, training=training)

        x_dense = self.pma2(x_dense, training=training)

        # Output shape: [B, channels]
        return ops.squeeze(x_dense, axis=1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, k={self.k}, "
            f"heads={self.heads}, layer_norm={self.layer_norm}, "
            f"dropout={self.dropout})"
        )

