from typing import Dict, List, Optional, Tuple, Union
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import softmax


def semantic_group(xs: List[any], q: any, k_lin: keras.layers.Layer) -> Tuple[Optional[any], Optional[any]]:
    if len(xs) == 0:
        return None, None
    num_edge_types = len(xs)
    out = ops.stack(xs, axis=0)  # (E, N, C)
    if ops.size(out) == 0:
        return ops.reshape(out, (0, ops.shape(out)[-1])), None

    k_proj = ops.tanh(k_lin(out))  # (E, N, C)
    mean_k = ops.mean(k_proj, axis=1)  # (E, C)
    attn_score = ops.sum(q * mean_k, axis=-1)  # (E,)
    if num_edge_types == 1:
        attn = ops.ones_like(attn_score)
    else:
        attn = ops.softmax(attn_score, axis=0)  # (E,)

    attn_expanded = ops.reshape(attn, (num_edge_types, 1, 1))
    out = ops.sum(attn_expanded * out, axis=0)  # (N, C)
    return out, attn


class HANConv(MessagePassing):
    r"""The Heterogeneous Graph Attention Operator from the
    `"Heterogeneous Graph Attention Network" <https://arxiv.org/abs/1903.07293>`_ paper.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every node type.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): Node types and edge types.
        heads (int, optional): Number of multi-head-attentions. (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative slope. (default: :obj:`0.2`)
    """

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        heads: int = 1,
        negative_slope: float = 0.2,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata

        self.k_lin = keras.layers.Dense(out_channels, use_bias=True)
        self.proj = {
            node_type: keras.layers.Dense(out_channels, use_bias=True)
            for node_type, ch in self.in_channels.items()
        }

    def build(self, input_shape=None):
        H, D = self.heads, self.out_channels // self.heads

        self.q = self.add_weight(
            shape=(1, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
            name="q",
        )

        self.lin_src = {}
        self.lin_dst = {}
        for edge_type in self.metadata[1]:
            key = "__".join(edge_type)
            self.lin_src[key] = self.add_weight(
                shape=(1, H, D),
                initializer="glorot_uniform",
                trainable=True,
                name=f"lin_src_{key}",
            )
            self.lin_dst[key] = self.add_weight(
                shape=(1, H, D),
                initializer="glorot_uniform",
                trainable=True,
                name=f"lin_dst_{key}",
            )

        for nt, layer in self.proj.items():
            if not layer.built:
                layer.build((None, self.in_channels[nt]))
        if not self.k_lin.built:
            self.k_lin.build((None, self.out_channels))

        super().build(input_shape)

    def call(
        self,
        x_dict: Dict[str, any],
        edge_index_dict: Dict[Tuple[str, str, str], any],
        return_semantic_attention_weights: bool = False,
    ):
        if not self.built:
            self.build()

        H, D = self.heads, self.out_channels // self.heads
        x_node_dict = {}
        out_dict = {nt: [] for nt in self.metadata[0]}

        for node_type, x in x_dict.items():
            proj_x = self.proj[node_type](x)
            x_node_dict[node_type] = ops.reshape(proj_x, (-1, H, D))

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            key = "__".join(edge_type)
            lin_src = self.lin_src[key]
            lin_dst = self.lin_dst[key]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]

            alpha_src = ops.sum(x_src * lin_src, axis=-1)  # (N_src, H)
            alpha_dst = ops.sum(x_dst * lin_dst, axis=-1)  # (N_dst, H)

            out = self.propagate(
                edge_index,
                x=(x_src, x_dst),
                alpha=(alpha_src, alpha_dst),
            )
            out = ops.relu(out)
            out_dict[dst_type].append(out)

        result = {}
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, attn = semantic_group(outs, self.q, self.k_lin)
            result[node_type] = out
            semantic_attn_dict[node_type] = attn

        if return_semantic_attention_weights:
            return result, semantic_attn_dict
        return result

    def message(self, x_j, alpha_i, alpha_j, index=None):
        alpha = alpha_j + alpha_i
        alpha = ops.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index=index)
        out = x_j * ops.expand_dims(alpha, axis=-1)
        return ops.reshape(out, (-1, self.out_channels))

