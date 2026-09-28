from typing import Optional, Union, Tuple

import numpy as np
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import scatter


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the
    `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import RGCNConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges
        edge_type = np.random.randint(0, 3, size=(30,))  # relation type of each edge

        layer = RGCNConv(in_channels=8, out_channels=16, num_relations=3)
        out = layer(x, edge_index, edge_type)
        print(tuple(out.shape))  # (10, 16)
        ```
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = "mean",
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.root_weight = root_weight
        self.is_sorted = is_sorted
        self.use_bias = bias

        if isinstance(in_channels, int):
            self.in_channels_l = in_channels
            self.in_channels_r = in_channels
        else:
            self.in_channels_l, self.in_channels_r = in_channels

        if num_bases is not None:
            self.weight = self.add_weight(
                shape=(num_bases, self.in_channels_l, out_channels),
                initializer="glorot_uniform",
                name="weight",
            )
            self.comp = self.add_weight(
                shape=(num_relations, num_bases),
                initializer="glorot_uniform",
                name="comp",
            )
        elif num_blocks is not None:
            assert (
                self.in_channels_l % num_blocks == 0
                and out_channels % num_blocks == 0
            ), "in_channels and out_channels must be divisible by num_blocks"
            self.weight = self.add_weight(
                shape=(
                    num_relations,
                    num_blocks,
                    self.in_channels_l // num_blocks,
                    out_channels // num_blocks,
                ),
                initializer="glorot_uniform",
                name="weight",
            )
        else:
            self.weight = self.add_weight(
                shape=(num_relations, self.in_channels_l, out_channels),
                initializer="glorot_uniform",
                name="weight",
            )

        if root_weight:
            self.root = self.add_weight(
                shape=(self.in_channels_r, out_channels),
                initializer="glorot_uniform",
                name="root",
            )
        else:
            self.root = None

        if bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        self.built = True

    def _get_weight(self):
        if self.num_bases is not None:
            # comp @ weight.reshape(num_bases, -1) -> (num_relations, in_channels, out_channels)
            w_flat = ops.reshape(self.weight, (self.num_bases, -1))
            w = ops.matmul(self.comp, w_flat)
            return ops.reshape(
                w, (self.num_relations, self.in_channels_l, self.out_channels)
            )
        return self.weight

    def call(self, inputs, edge_index=None, edge_type=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, edge_index, edge_type = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if isinstance(x, (list, tuple)):
            x_l, x_r = x
        else:
            x_l = x_r = x

        edge_index = ops.cast(edge_index, "int32")
        edge_type = ops.cast(edge_type, "int32")
        # Featureless nodes (x=None): node i's input is the one-hot vector e_i, so in_channels is
        # the number of nodes and x @ W reduces to looking up row i of W.
        size = (
            self.in_channels_l if x_l is None else ops.shape(x_l)[0],
            self.in_channels_r if x_r is None else ops.shape(x_r)[0],
        )

        # Either transform every source node once per relation and look up each edge's message
        # ([relations * nodes, out] values), or multiply every edge with its relation's weight
        # ([edges, in, out] values). Featureless nodes always use the lookup.
        self._node_messages = None
        if x_l is None or self._prefer_node_transform(x_l, edge_index):
            self._node_messages = self._relation_features(x_l)
            self._num_sources = size[0]
            x_l = None

        out = self.propagate(
            edge_index, x=x_l, edge_type=edge_type, size=size
        )
        self._node_messages = None

        if self.root is not None:
            out = out + (self.root if x_r is None else ops.matmul(x_r, self.root))

        if self.bias is not None:
            out = out + self.bias

        return out

    def _prefer_node_transform(self, x, edge_index) -> bool:
        num_nodes, num_edges = x.shape[0], edge_index.shape[1]
        if not isinstance(num_nodes, int) or not isinstance(num_edges, int):
            return True
        if _is_concrete(edge_index):
            # Eagerly, edges are grouped by relation instead (see `_grouped_message`), which costs
            # one [in, out] product per edge instead of one per node and relation.
            return self.num_relations * num_nodes < num_edges
        in_per_block = self.in_channels_l // (self.num_blocks or 1)
        return self.num_relations * num_nodes <= num_edges * in_per_block

    def _relation_features(self, x):
        """Returns ``x @ W_r`` for every relation ``r`` and node, flattened to ``[R * N, out]``."""
        weight = self._get_weight()
        if x is None:  # featureless: node i's input is the one-hot e_i, so x @ W_r is row i of W_r
            if self.num_blocks is not None:
                raise ValueError("Block decomposition does not support featureless nodes (x=None).")
            h = weight
        elif self.num_blocks is not None:
            x_b = ops.reshape(x, (-1, self.num_blocks, self.in_channels_l // self.num_blocks))
            h = ops.einsum("nbi,rbio->rnbo", x_b, weight)
        else:
            h = ops.einsum("ni,rio->rno", x, weight)
        return ops.reshape(h, (-1, self.out_channels))

    def message(self, x_j, edge_type):
        if x_j is None:  # look up the pre-transformed source node of every edge
            return ops.take(self._node_messages, edge_type * self._num_sources + self.index_sources, axis=0)
        if _is_concrete(edge_type):
            return self._grouped_message(x_j, edge_type)
        weight = self._get_weight()
        if self.num_blocks is not None:
            w_r = ops.take(weight, edge_type, axis=0)  # (E, num_blocks, in_b, out_b)
            x_j_b = ops.reshape(
                x_j,
                (-1, self.num_blocks, 1, self.in_channels_l // self.num_blocks),
            )
            msg = ops.matmul(x_j_b, w_r)
            return ops.reshape(msg, (-1, self.out_channels))
        else:
            w_r = ops.take(weight, edge_type, axis=0)  # (E, in_channels, out_channels)
            x_j_exp = ops.expand_dims(x_j, 1)
            msg = ops.squeeze(ops.matmul(x_j_exp, w_r), 1)
            return msg

    def _grouped_message(self, x_j, edge_type):
        """Messages of edges grouped by relation, as in PyG: the edges of relation ``r`` are
        multiplied with ``W_r`` together (``[edges, out]`` memory, no per-edge weight matrices)."""
        weight = self._get_weight()
        edge_type = np.asarray(ops.convert_to_numpy(edge_type)).astype(np.int64)
        order = np.argsort(edge_type, kind="stable")
        counts = np.bincount(edge_type, minlength=self.num_relations)
        outs, start = [], 0
        for r in np.nonzero(counts)[0]:
            x_r = ops.take(x_j, order[start:start + counts[r]], axis=0)
            start += counts[r]
            if self.num_blocks is not None:
                x_r = ops.reshape(x_r, (-1, self.num_blocks, self.in_channels_l // self.num_blocks))
                out = ops.reshape(ops.einsum("ebi,bio->ebo", x_r, weight[int(r)]), (-1, self.out_channels))
            else:
                out = ops.matmul(x_r, weight[int(r)])
            outs.append(out)
        if not outs:
            return ops.zeros((0, self.out_channels), dtype=x_j.dtype)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return ops.take(ops.concatenate(outs, axis=0), inverse, axis=0)

    def aggregate(self, inputs, edge_index=None, index=None, edge_type=None, dim_size=None, **kwargs):
        if index is None and edge_index is not None:
            index = edge_index[1]
        if self.aggr == "mean" and edge_type is not None and index is not None:
            one_hot = ops.one_hot(edge_type, self.num_relations)
            norm = scatter(one_hot, index, dim=0, dim_size=dim_size, reduce="sum")
            norm_per_edge = ops.take(norm, index, axis=0)
            edge_type_expanded = ops.expand_dims(edge_type, -1)
            norm_val = ops.take_along_axis(norm_per_edge, edge_type_expanded, axis=1)
            norm_val = ops.maximum(norm_val, 1.0)
            inputs = inputs / ops.cast(norm_val, inputs.dtype)
            return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
        return super().aggregate(inputs, edge_index=edge_index, index=index, dim_size=dim_size, **kwargs)


def _is_concrete(t) -> bool:
    """Whether ``t`` has values on the host now (not traced, symbolic or on torch's meta device)."""
    from k3_node.layers.conv.utils import is_tracing
    from k3_node.ops.host import _in_shape_inference, _is_meta

    return t is not None and not is_tracing(t) and not _in_shape_inference() and not _is_meta(t)


class FastRGCNConv(RGCNConv):
    r"""See :class:`RGCNConv`.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import FastRGCNConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges
        edge_type = np.random.randint(0, 3, size=(30,))  # relation type of each edge

        layer = FastRGCNConv(in_channels=8, out_channels=16, num_relations=3)
        out = layer(x, edge_index, edge_type)
        print(tuple(out.shape))  # (10, 16)
        ```
    """


class CuGraphRGCNConv(RGCNConv):
    r"""Fallback for CuGraphRGCNConv.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import CuGraphRGCNConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges
        edge_type = np.random.randint(0, 3, size=(30,))  # relation type of each edge

        layer = CuGraphRGCNConv(in_channels=8, out_channels=16, num_relations=3)
        out = layer(x, edge_index, edge_type)
        print(tuple(out.shape))  # (10, 16)
        ```
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_bases=num_bases,
            aggr=aggr,
            root_weight=root_weight,
            bias=bias,
            **kwargs,
        )
