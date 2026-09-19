from typing import Optional
from keras import layers, ops

from .base import Aggregation
from .utils import PoolingByMultiheadAttention, SetAttentionBlock


class SetTransformerAggregation(Aggregation):
    r"""Performs "Set Transformer" aggregation in which the elements to
    aggregate are processed by multi-head attention blocks, as described in
    the `"Graph Neural Networks with Adaptive Readouts"
    <https://arxiv.org/abs/2211.04952>`_ paper.
    """

    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 1,
        num_decoder_blocks: int = 1,
        heads: int = 1,
        concat: bool = True,
        layer_norm: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_seed_points = num_seed_points
        self.heads = heads
        self.concat = concat
        self.layer_norm = layer_norm
        self.dropout = dropout

        self.encoders = [
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_encoder_blocks)
        ]
        self.pma = PoolingByMultiheadAttention(
            channels, num_seed_points, heads, layer_norm, dropout
        )
        self.decoders = [
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_decoder_blocks)
        ]

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.pma.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()

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

        for encoder in self.encoders:
            x_dense = encoder(x_dense, mask=mask, training=training)

        x_dense = self.pma(x_dense, mask=mask, training=training)

        for decoder in self.decoders:
            x_dense = decoder(x_dense, training=training)

        # Handle NaNs if any
        x_dense = ops.where(ops.isnan(x_dense), 0.0, x_dense)

        if self.concat:
            B = ops.shape(x_dense)[0]
            return ops.reshape(x_dense, (B, self.num_seed_points * self.channels))
        else:
            return ops.mean(x_dense, axis=1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"num_seed_points={self.num_seed_points}, "
            f"heads={self.heads}, layer_norm={self.layer_norm}, "
            f"dropout={self.dropout})"
        )

