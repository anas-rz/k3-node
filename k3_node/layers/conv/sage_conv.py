from typing import Optional, Union, Tuple, List

from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.aggr.base import Aggregation


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        aggr: The aggregation scheme to use (``"mean"``, ``"max"``,
            ``"lstm"``, etc.). (default: ``"mean"``)
        normalize: If set to :obj:`True`, output features will be
            :math:`\ell_2`-normalized. (default: :obj:`False`)
        root_weight: If set to :obj:`False`, the layer will not add
            the transformed root node features. (default: :obj:`True`)
        project: If set to :obj:`True`, the layer will apply a linear
            transformation followed by an activation to source node features.
            (default: :obj:`False`)
        bias: If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int], None],
        out_channels: Optional[int] = None,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        if out_channels is None:
            # Backward compatibility: SAGEConv(out_channels, ...)
            out_channels = in_channels
            in_channels = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.use_bias = bias

        super().__init__(aggr=aggr, **kwargs)

        if self.project:
            self.lin_proj = layers.Dense(
                in_channels[0] if isinstance(in_channels, (tuple, list)) else in_channels,
                activation="relu",
                use_bias=True,
            )
        else:
            self.lin_proj = None

        self.lin_l = layers.Dense(out_channels, use_bias=bias)
        if self.root_weight:
            self.lin_r = layers.Dense(out_channels, use_bias=False)
        else:
            self.lin_r = None

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            in_channels_l = input_shape[0][-1]
            in_channels_r = input_shape[1][-1] if len(input_shape) > 1 and input_shape[1] is not None else in_channels_l
        else:
            in_channels_l = input_shape[-1]
            in_channels_r = input_shape[-1]

        self.lin_l.build((None, self.out_channels if self.project else in_channels_l))
        if self.lin_r is not None:
            self.lin_r.build((None, in_channels_r))
        if self.lin_proj is not None:
            self.lin_proj.build((None, in_channels_l))
        self.built = True

    def call(self, x, edge_index=None, size=None, **kwargs):
        # Handle legacy calling: conv(x, adj) where adj is [N, N]
        shape = getattr(edge_index, "shape", None)
        if (
            shape is not None
            and len(shape) == 2
            and shape[0] is not None
            and shape[1] is not None
            and shape[0] > 2
            and shape[0] == shape[1]
        ):
            where_adj = ops.where(edge_index != 0)
            where_adj = where_adj if not isinstance(where_adj, list) else where_adj
            edge_index = ops.stack([where_adj[0], where_adj[1]], axis=0)

        # Handle legacy calling: conv((x, adj))
        if edge_index is None and isinstance(x, (tuple, list)) and len(x) == 2:
            arg0, arg1 = x[0], x[1]
            s1 = getattr(arg1, "shape", None)
            if (
                s1 is not None
                and len(s1) == 2
                and s1[0] is not None
                and s1[1] is not None
                and s1[0] > 2
                and s1[0] == s1[1]
            ):
                where_adj = ops.where(arg1 != 0)
                where_adj = where_adj if not isinstance(where_adj, list) else where_adj
                edge_index = ops.stack([where_adj[0], where_adj[1]], axis=0)
                x = arg0
            elif s1 is not None and len(s1) >= 1 and s1[0] == 2:
                edge_index = arg1
                x = arg0

        if not isinstance(x, (tuple, list)):
            x_src = x
            x_dst = x
        else:
            x_src, x_dst = x[0], x[1]

        if self.project and self.lin_proj is not None:
            x_src = self.lin_proj(x_src)

        out = self.propagate(edge_index, x=(x_src, x_dst), size=size)
        out = self.lin_l(out)

        if self.root_weight and self.lin_r is not None and x_dst is not None:
            out = out + self.lin_r(x_dst)

        if self.normalize:
            norm = ops.sqrt(ops.sum(ops.square(out), axis=-1, keepdims=True) + 1e-12)
            out = out / norm

        return out

    def message(self, x_j):
        return x_j
