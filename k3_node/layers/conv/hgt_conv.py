import math
from typing import Dict, List, Optional, Tuple, Union
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import softmax
from k3_node.layers.dense.linear import HeteroDictLinear, HeteroLinear


class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every node type.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): Node types and edge types.
        heads (int, optional): Number of multi-head-attentions. (default: :obj:`1`)
    """

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' ({out_channels}) must be divisible by 'heads' ({heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {edge_type: i for i, edge_type in enumerate(metadata[1])}
        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels, self.out_channels * 3)
        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels, types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)

    def build(self, input_shape=None):
        self.skip = {}
        for nt in self.node_types:
            self.skip[nt] = self.add_weight(
                shape=(1,),
                initializer="ones",
                trainable=True,
                name=f"skip_{nt}",
            )
        self.p_rel = {}
        for et in self.edge_types:
            key = "__".join(et)
            self.p_rel[key] = self.add_weight(
                shape=(1, self.heads),
                initializer="ones",
                trainable=True,
                name=f"p_rel_{key}",
            )
        super().build(input_shape)

    def _cat(self, x_dict: Dict[str, any]) -> Tuple[any, Dict[str, int]]:
        cumsum = 0
        outs = []
        offset = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += ops.shape(x)[0]
        return ops.concatenate(outs, axis=0), offset

    def _construct_src_node_feat(
        self,
        k_dict: Dict[str, any],
        v_dict: Dict[str, any],
        edge_index_dict: Dict[Tuple[str, str, str], any],
    ):
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        ks = []
        vs = []
        type_list = []
        offset = {}

        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = ops.shape(k_dict[src])[0]
            offset[edge_type] = cumsum
            cumsum += N

            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = (
                ops.repeat(ops.reshape(ops.arange(H, dtype="int32"), (-1, 1)), N, axis=1)
                * num_edge_types
                + edge_type_offset
            )

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks_cat = ops.reshape(ops.transpose(ops.concatenate(ks, axis=0), (1, 0, 2)), (-1, D))
        vs_cat = ops.reshape(ops.transpose(ops.concatenate(vs, axis=0), (1, 0, 2)), (-1, D))
        type_vec_cat = ops.reshape(ops.concatenate(type_list, axis=1), (-1,))

        k = self.k_rel(ks_cat, type_vec_cat)
        k = ops.transpose(ops.reshape(k, (H, -1, D)), (1, 0, 2))

        v = self.v_rel(vs_cat, type_vec_cat)
        v = ops.transpose(ops.reshape(v, (H, -1, D)), (1, 0, 2))

        return k, v, offset

    def call(
        self,
        x_dict: Dict[str, any],
        edge_index_dict: Dict[Tuple[str, str, str], any],
    ) -> Dict[str, any]:
        if not self.built:
            self.build()

        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = ops.split(val, 3, axis=1)
            k_dict[key] = ops.reshape(k, (-1, H, D))
            q_dict[key] = ops.reshape(q, (-1, H, D))
            v_dict[key] = ops.reshape(v, (-1, H, D))

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(k_dict, v_dict, edge_index_dict)

        # Build concatenated bipartite edge index and edge attributes
        edge_indices = []
        edge_attrs = []
        for edge_type, s_offset in src_offset.items():
            e_idx = edge_index_dict[edge_type]
            d_offset = dst_offset[edge_type[-1]]

            row = e_idx[0] + s_offset
            col = e_idx[1] + d_offset
            edge_indices.append(ops.stack([row, col], axis=0))

            p_val = self.p_rel["__".join(edge_type)]
            num_e = ops.shape(e_idx)[1]
            edge_attrs.append(ops.repeat(p_val, num_e, axis=0))

        edge_index = ops.concatenate(edge_indices, axis=1)
        edge_attr = ops.concatenate(edge_attrs, axis=0)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        for node_type, start_offset in dst_offset.items():
            count = ops.shape(q_dict[node_type])[0]
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset : start_offset + count]

        transformed = {}
        for k_nt, v_nt in out_dict.items():
            transformed[k_nt] = ops.gelu(v_nt) if v_nt is not None else v_nt
        a_dict = self.out_lin(transformed)

        result = {}
        for node_type, o in out_dict.items():
            res = a_dict[node_type]
            if ops.shape(res)[-1] == ops.shape(x_dict[node_type])[-1]:
                alpha = ops.sigmoid(self.skip[node_type])
                res = alpha * res + (1.0 - alpha) * x_dict[node_type]
            result[node_type] = res

        return result

    def message(self, k_j, q_i, v_j, edge_attr, index=None):
        alpha = ops.sum(q_i * k_j, axis=-1) * edge_attr
        alpha = alpha / math.sqrt(ops.shape(q_i)[-1])
        alpha = softmax(alpha, index=index)
        out = v_j * ops.expand_dims(alpha, axis=-1)
        return ops.reshape(out, (-1, self.out_channels))

