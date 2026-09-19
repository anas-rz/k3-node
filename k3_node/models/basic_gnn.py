import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
from keras import ops

from k3_node.layers.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from k3_node.models.mlp import MLP, _normalization_resolver
from k3_node.models.jumping_knowledge import JumpingKnowledge


class BasicGNN(keras.layers.Layer):
    r"""An abstract base class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    supports_edge_weight: bool = False
    supports_edge_attr: bool = False
    supports_norm_batch: bool = False

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.dropout = keras.layers.Dropout(rate=dropout) if dropout > 0 else None

        if isinstance(act, str):
            self.act = keras.activations.get(act)
        else:
            self.act = act

        self.jk_mode = jk
        self.act_first = act_first
        self.norm_query = norm
        self.norm_kwargs = norm_kwargs or {}

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = []
        curr_in = in_channels
        if num_layers > 1:
            self.convs.append(self.init_conv(curr_in, hidden_channels, **kwargs))
            if isinstance(curr_in, (tuple, list)):
                curr_in = (hidden_channels, hidden_channels)
            else:
                curr_in = hidden_channels

        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(curr_in, hidden_channels, **kwargs))
            if isinstance(curr_in, (tuple, list)):
                curr_in = (hidden_channels, hidden_channels)
            else:
                curr_in = hidden_channels

        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(self.init_conv(curr_in, out_channels, **kwargs))
        else:
            self.convs.append(self.init_conv(curr_in, hidden_channels, **kwargs))

        self.norms = []
        self.supports_norm_batch = False

        for _ in range(num_layers - 1):
            if norm is not None:
                norm_layer = _normalization_resolver(norm, hidden_channels, **self.norm_kwargs)
                self.norms.append(norm_layer)
                if hasattr(norm_layer, "call"):
                    sig = inspect.signature(norm_layer.call).parameters
                    self.supports_norm_batch = "batch" in sig
            else:
                self.norms.append(None)

        if jk is not None:
            if norm is not None:
                self.norms.append(_normalization_resolver(norm, hidden_channels, **self.norm_kwargs))
            else:
                self.norms.append(None)
        else:
            self.norms.append(None)

        if jk is not None and jk != "last":
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == "cat":
                jk_in = num_layers * hidden_channels
            else:
                jk_in = hidden_channels
            self.lin = keras.layers.Dense(self.out_channels)

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for norm in self.norms:
            if norm is not None and hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if hasattr(self, "jk") and hasattr(self.jk, "reset_parameters"):
            self.jk.reset_parameters()
        if hasattr(self, "lin") and hasattr(self.lin, "reset_parameters"):
            self.lin.reset_parameters()

    def call(
        self,
        x,
        edge_index,
        edge_weight=None,
        edge_attr=None,
        batch=None,
        batch_size=None,
        training=None,
    ):
        xs: List = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if norm is not None:
                    if self.supports_norm_batch and batch is not None:
                        x = norm(x, batch=batch)
                    else:
                        x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                if self.dropout is not None:
                    x = self.dropout(x, training=training)
                if hasattr(self, "jk"):
                    xs.append(x)

        if hasattr(self, "jk"):
            x = self.jk(xs)
        if hasattr(self, "lin"):
            x = self.lin(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~k3_node.layers.conv.GCNConv` operator for message passing.
    """
    supports_edge_weight: bool = True
    supports_edge_attr: bool = False

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~k3_node.layers.conv.SAGEConv` operator for message passing.
    """
    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~k3_node.layers.conv.GINConv` operator for message passing.
    """
    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm_query,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)


class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~k3_node.layers.conv.GATConv` or
    :class:`~k3_node.layers.conv.GATv2Conv` operator for message passing.
    """
    supports_edge_weight: bool = False
    supports_edge_attr: bool = True

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout_p, **kwargs)


class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~k3_node.layers.conv.PNAConv` operator for message passing.
    """
    supports_edge_weight: bool = False
    supports_edge_attr: bool = True

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)


class EdgeCNN(BasicGNN):
    r"""The Graph Neural Network from the `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper, using the
    :class:`~k3_node.layers.conv.EdgeConv` operator for message passing.
    """
    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        mlp = MLP(
            [2 * in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm_query,
            norm_kwargs=self.norm_kwargs,
        )
        return EdgeConv(mlp, **kwargs)

