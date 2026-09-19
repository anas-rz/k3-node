from typing import Optional, Union, Tuple

from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import scatter


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the
    `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
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

        num_nodes = ops.shape(x_r)[0]
        out = ops.zeros((num_nodes, self.out_channels), dtype=x_r.dtype)
        weight = self._get_weight()

        edge_index = ops.cast(edge_index, "int32")
        if edge_type is not None:
            edge_type = ops.cast(edge_type, "int32")

        if edge_type is None:
            raise ValueError("RGCNConv requires edge_type tensor")

        # Iterate over relations
        for r in range(self.num_relations):
            mask = ops.equal(edge_type, r)
            where_mask = ops.where(mask)
            idx = where_mask[0] if isinstance(where_mask, (list, tuple)) else where_mask
            idx = ops.reshape(idx, (-1,))
            idx = ops.cast(idx, "int32")
            if ops.shape(idx)[0] == 0:
                continue
            edge_index_r = ops.take(edge_index, idx, axis=1)

            if self.num_blocks is not None:
                # Block-diagonal
                h = self.propagate(edge_index_r, x=x_l, size=(ops.shape(x_l)[0], num_nodes))
                # h: (N, in_channels) -> (N, num_blocks, in_block)
                h = ops.reshape(
                    h,
                    (
                        -1,
                        self.num_blocks,
                        self.in_channels_l // self.num_blocks,
                    ),
                )
                # weight[r]: (num_blocks, in_block, out_block)
                # einsum 'nbc,bcd->nbd'
                h_out = ops.einsum("nbc,bcd->nbd", h, weight[r])
                out = out + ops.reshape(h_out, (-1, self.out_channels))
            else:
                h = self.propagate(edge_index_r, x=x_l, size=(ops.shape(x_l)[0], num_nodes))
                out = out + ops.matmul(h, weight[r])

        if self.root is not None:
            out = out + ops.matmul(x_r, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j):
        return x_j


class FastRGCNConv(RGCNConv):
    r"""See :class:`RGCNConv`."""
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
        num_nodes = ops.shape(x_r)[0]
        size = (ops.shape(x_l)[0], num_nodes)

        out = self.propagate(
            edge_index, x=x_l, edge_type=edge_type, size=size
        )

        if self.root is not None:
            out = out + ops.matmul(x_r, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_type):
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
            # x_j: (E, in_channels) -> (E, 1, in_channels) @ (E, in_channels, out_channels) -> (E, out_channels)
            x_j_exp = ops.expand_dims(x_j, 1)
            msg = ops.squeeze(ops.matmul(x_j_exp, w_r), 1)
            return msg

    def aggregate(self, inputs, edge_index=None, index=None, edge_type=None, dim_size=None, **kwargs):
        if index is None and edge_index is not None:
            index = edge_index[1]
        if self.aggr == "mean" and edge_type is not None and index is not None:
            # Normalization per relation
            one_hot = ops.one_hot(edge_type, self.num_relations)
            norm = scatter(one_hot, index, dim=0, dim_size=dim_size, reduce="sum")
            norm_per_edge = ops.take(norm, index, axis=0)
            edge_type_expanded = ops.expand_dims(edge_type, -1)
            norm_val = ops.take_along_axis(norm_per_edge, edge_type_expanded, axis=1)
            norm_val = ops.maximum(norm_val, 1.0)
            inputs = inputs / ops.cast(norm_val, inputs.dtype)
            return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
        return super().aggregate(inputs, edge_index=edge_index, index=index, dim_size=dim_size, **kwargs)


class CuGraphRGCNConv(RGCNConv):
    r"""Fallback for CuGraphRGCNConv."""
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
