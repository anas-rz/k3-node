from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import (
    add_self_loops,
    degree,
    extend_mask_for_self_loops,
    remove_self_loops_masked,
)
from k3_node.ops.segment import segment_sum


class FeaStConv(MessagePassing):
    r"""The (fault-tolerant) feature-steered graph convolution operator from
    the `"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis"
    <https://arxiv.org/abs/1706.05206>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import FeaStConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges

        layer = FeaStConv(in_channels=8, out_channels=16, heads=2)
        out = layer(x, edge_index)
        print(tuple(out.shape))  # (10, 16)
        ```
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.use_bias = bias

        self.lin = Dense(heads * out_channels, use_bias=False)
        self.u = self.add_weight(
            shape=(in_channels, heads),
            initializer="glorot_uniform",
            name="u",
        )
        self.c = self.add_weight(
            shape=(heads,),
            initializer="zeros",
            name="c",
        )

        if bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        self.lin.build((None, self.in_channels))
        self.built = True

    def call(self, inputs, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                x, edge_index = inputs
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        num_nodes = ops.shape(x_dst)[0]
        keep_mask = None
        if self.add_self_loops:
            edge_index, _, keep_mask = remove_self_loops_masked(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            keep_mask = extend_mask_for_self_loops(keep_mask, num_nodes)

        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            keep_mask=keep_mask,
            size=(ops.shape(x_src)[0], num_nodes),
        )
        if keep_mask is not None and self.aggr == "mean":
            # Masked messages are zero but still counted by the mean; rescale to the kept count.
            col = ops.cast(edge_index[1], "int32")
            count_all = degree(col, num_nodes=num_nodes, dtype=out.dtype)
            count_kept = segment_sum(ops.cast(keep_mask, out.dtype), col, num_segments=num_nodes)
            out = out * ops.expand_dims(count_all / ops.maximum(count_kept, 1.0), -1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_i, x_j, keep_mask=None):
        q = ops.matmul(x_j - x_i, self.u) + self.c
        q = ops.softmax(q, axis=-1)
        x_j_mapped = ops.reshape(self.lin(x_j), (-1, self.heads, self.out_channels))
        msg = ops.sum(x_j_mapped * ops.expand_dims(q, -1), axis=1)
        # For max/min a duplicated self-loop message is harmless, so only sum/mean need masking.
        if keep_mask is not None and self.aggr in ("add", "sum", "mean"):
            msg = msg * ops.expand_dims(ops.cast(keep_mask, msg.dtype), -1)
        return msg

