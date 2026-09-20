import inspect
from typing import Any, List, Optional, Tuple, Union
from keras import layers, ops

from k3_node.utils import (
    is_layer_kwarg,
    deserialize_kwarg,
    serialize_kwarg,
    is_keras_kwarg,
    deserialize_scatter,
    serialize_scatter,
)
from k3_node.ops import get_source_target
from k3_node.layers.aggr.resolver import aggregation_resolver
from k3_node.layers.aggr.base import Aggregation
from k3_node.layers.aggr.basic import SumAggregation


class MessagePassing(layers.Layer):
    r"""Base class for creating Message Passing Neural Networks (MPNNs).

    Args:
        aggr: The aggregation scheme to use, such as ``"add"``, ``"sum"``,
            ``"mean"``, ``"min"``, ``"max"``, ``"mul"``, or an instance of
            :class:`~k3_node.layers.aggr.Aggregation`. (default: ``"add"``)
        flow: The direction of message passing (``"source_to_target"`` or
            ``"target_to_source"``). (default: ``"source_to_target"``)
        node_dim: The axis along which to index node features. (default: ``-2``)
        decomposed_layers: Number of decomposed layers for memory-efficient
            aggregation. (default: ``1``)
    """

    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation, None] = "add",
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
        **kwargs,
    ):
        # Support legacy aggregate arg
        if "aggregate" in kwargs:
            aggr = kwargs.pop("aggregate")

        # Extract and set layer kwargs for backwards compatibility
        self.kwargs_keys = []
        for key in list(kwargs.keys()):
            if is_layer_kwarg(key):
                attr = kwargs.pop(key)
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.aggr = aggr
        self.flow = flow
        self.node_dim = node_dim
        self.decomposed_layers = decomposed_layers

        if flow not in ["source_to_target", "target_to_source"]:
            raise ValueError(f"Flow {flow} must be 'source_to_target' or 'target_to_source'")

        # Legacy scatter operator support
        try:
            scatter_name = "sum" if aggr in ("add", "sum") else aggr
            self.agg = deserialize_scatter(scatter_name)
        except Exception:
            self.agg = ops.segment_sum

        if aggr is None:
            self.aggr_module = None
        elif isinstance(aggr, Aggregation):
            self.aggr_module = aggr
        elif isinstance(aggr, str):
            aggr_name = "sum" if aggr == "add" else aggr
            try:
                self.aggr_module = aggregation_resolver(aggr_name)
            except Exception:
                self.aggr_module = SumAggregation()
        elif isinstance(aggr, (list, tuple)):
            from k3_node.layers.aggr.multi import MultiAggregation
            self.aggr_module = MultiAggregation(aggr)
        else:
            self.aggr_module = SumAggregation()

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters

    def build(self, input_shape=None):
        self.built = True

    @staticmethod
    def get_inputs(inputs):
        if len(inputs) == 3:
            x, a, e = inputs
            if hasattr(e, "shape") and e.shape is not None:
                assert len(e.shape) in (2, 3), "E must have rank 2 or 3"
        elif len(inputs) == 2:
            x, a = inputs
            e = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (X, A, E), got {}.".format(len(inputs))
            )
        if hasattr(a, "shape") and a.shape is not None:
            assert len(a.shape) == 2, "A must have rank 2"
        return x, a, e

    def get_targets(self, x):
        return ops.take(x, self.index_targets, axis=self.node_dim)

    def get_sources(self, x):
        return ops.take(x, self.index_sources, axis=self.node_dim)

    def get_kwargs(self, x, a, e, signature, kwargs):
        output = {}
        for k in signature.keys():
            if k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a":
                output[k] = a
            elif k == "e":
                output[k] = e
            elif k in kwargs:
                output[k] = kwargs[k]
            elif signature[k].default is inspect.Parameter.empty:
                pass
            else:
                pass
        return output

    def _get_dim_size(self, kwargs, i, size=None):
        if size is not None and size[1] is not None:
            return size[1]
        x = kwargs.get("x", None)
        if x is not None:
            if isinstance(x, (tuple, list)):
                target_x = x[1] if x[1] is not None else x[0]
                if target_x is not None:
                    if hasattr(target_x, "shape") and target_x.shape[self.node_dim] is not None:
                        return int(target_x.shape[self.node_dim])
                    return ops.shape(target_x)[self.node_dim]
            else:
                if hasattr(x, "shape") and x.shape[self.node_dim] is not None:
                    return int(x.shape[self.node_dim])
                return ops.shape(x)[self.node_dim]
        for k, val in kwargs.items():
            if k in ("edge_index", "edge_attr", "edge_weight", "ptr"):
                continue
            if hasattr(val, "shape") and len(val.shape) >= 2:
                if val.shape[self.node_dim] is not None:
                    return int(val.shape[self.node_dim])
                return ops.shape(val)[self.node_dim]
        if i is not None:
            from k3_node.layers.conv.utils import is_tracing
            if is_tracing(i):
                return 0
            if hasattr(i, "shape") and len(i.shape) > 0 and i.shape[0] == 0:
                return 0
            try:
                return int(ops.max(i)) + 1
            except Exception:
                return 0
        return 0

    def propagate(self, *args, **kwargs: Any):
        r"""The initial call to start propagating messages."""
        # Detect legacy Spektral-style call: propagate(x, a, e=None, **kwargs)
        is_spektral = False
        if len(args) >= 2:
            arg0, arg1 = args[0], args[1]
            shape0 = arg0.shape if hasattr(arg0, "shape") and arg0.shape is not None else ()
            shape1 = arg1.shape if hasattr(arg1, "shape") and arg1.shape is not None else ()
            if len(shape1) == 2 and shape1[0] is not None and shape1[0] == shape1[1]:
                is_spektral = True
            elif len(shape0) >= 2 and shape0[0] is not None and shape0[0] != 2 and not isinstance(arg1, (tuple, list, type(None))):
                if len(shape1) == 2 and shape1[0] is not None and shape1[0] != 2:
                    is_spektral = True

        if is_spektral:
            x, a = args[0], args[1]
            e = args[2] if len(args) >= 3 else kwargs.get("e", None)
            self.n_nodes = x.shape[-2] if hasattr(x, "shape") and x.shape[-2] is not None else ops.shape(x)[-2]
            self.index_sources, self.index_targets = get_source_target(a)

            # Call legacy message
            msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
            messages = self.message(**msg_kwargs)

            # Call legacy aggregate
            agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
            embeddings = self.aggregate(messages, **agg_kwargs)

            # Call legacy update
            upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
            output = self.update(embeddings, **upd_kwargs)
            return output

        # Standard PyG MessagePassing propagate(edge_index, size=None, **kwargs)
        if len(args) >= 1:
            edge_index = args[0]
            size = args[1] if len(args) >= 2 else kwargs.pop("size", None)
        else:
            edge_index = kwargs.pop("edge_index")
            size = kwargs.pop("size", None)

        edge_index = ops.convert_to_tensor(edge_index)

        # Handle dense adjacency [N, N] passed as edge_index
        e_shape = getattr(edge_index, "shape", None)
        if e_shape is not None and len(e_shape) == 2 and e_shape[0] is not None and e_shape[1] is not None and e_shape[0] > 2 and e_shape[0] == e_shape[1]:
            where_adj = ops.where(edge_index != 0)
            where_adj = where_adj if not isinstance(where_adj, list) else where_adj
            edge_index = ops.stack([where_adj[0], where_adj[1]], axis=0)

        if self.flow == "source_to_target":
            i = edge_index[1]  # Target
            j = edge_index[0]  # Source
        else:
            i = edge_index[0]
            j = edge_index[1]

        i = ops.cast(i, "int32")
        j = ops.cast(j, "int32")
        self.index_targets = i
        self.index_sources = j

        dim_size = size[1] if size is not None and size[1] is not None else self._get_dim_size(kwargs, i, size)
        self.n_nodes = dim_size

        # Construct message arguments
        msg_kwargs = {}
        for param_name in self.msg_signature.keys():
            if param_name in kwargs:
                msg_kwargs[param_name] = kwargs[param_name]
            elif param_name.endswith("_i"):
                root = param_name[:-2]
                if root in kwargs:
                    val = kwargs[root]
                    val = val[1] if isinstance(val, (tuple, list)) else val
                    msg_kwargs[param_name] = ops.take(val, i, axis=self.node_dim) if val is not None else None
            elif param_name.endswith("_j"):
                root = param_name[:-2]
                if root in kwargs:
                    val = kwargs[root]
                    val = val[0] if isinstance(val, (tuple, list)) else val
                    msg_kwargs[param_name] = ops.take(val, j, axis=self.node_dim) if val is not None else None
            elif param_name == "index":
                msg_kwargs["index"] = i
            elif param_name in ("dim_size", "size_i"):
                msg_kwargs[param_name] = dim_size
            elif param_name == "size_j":
                msg_kwargs["size_j"] = size[0] if size is not None and size[0] is not None else dim_size
            elif param_name == "edge_index":
                msg_kwargs["edge_index"] = edge_index

        out = self.message(**msg_kwargs)

        # Aggregate
        agg_kwargs = {}
        for param_name in self.agg_signature.keys():
            if param_name in kwargs and param_name not in ["inputs", "index", "ptr", "dim_size"]:
                agg_kwargs[param_name] = kwargs[param_name]

        out = self.aggregate(
            out,
            index=i,
            ptr=kwargs.get("ptr", None),
            dim_size=dim_size,
            **agg_kwargs,
        )

        # Update
        upd_kwargs = {}
        for param_name in self.upd_signature.keys():
            if param_name in kwargs and param_name not in ["inputs", "aggr_out", "embeddings"]:
                upd_kwargs[param_name] = kwargs[param_name]
            elif param_name.endswith("_i"):
                root = param_name[:-2]
                if root in kwargs:
                    val = kwargs[root]
                    val = val[1] if isinstance(val, (tuple, list)) else val
                    upd_kwargs[param_name] = val

        out = self.update(out, **upd_kwargs)
        return out

    def message(self, x=None, x_j=None, **kwargs):
        r"""Constructs messages from node :math:`j` to node :math:`i`."""
        if x_j is not None:
            return x_j
        if x is not None and hasattr(self, "index_sources"):
            return self.get_sources(x)
        return x_j

    def aggregate(
        self,
        inputs=None,
        index=None,
        ptr: Optional[Any] = None,
        dim_size: Optional[int] = None,
        **kwargs,
    ):
        r"""Aggregates messages from neighbors as given by :obj:`index`."""
        # Check if called as Spektral: aggregate(messages, **kwargs)
        if inputs is not None and index is None and hasattr(self, "index_targets"):
            index = self.index_targets
            dim_size = self.n_nodes
            if hasattr(self, "agg") and callable(self.agg):
                return self.agg(inputs, index, dim_size)

        if self.aggr_module is not None:
            return self.aggr_module(
                inputs, index=index, ptr=ptr, dim_size=dim_size, dim=self.node_dim
            )
        return ops.segment_sum(inputs, index, num_segments=dim_size)

    def update(self, embeddings=None, **kwargs):
        r"""Updates node embeddings."""
        return embeddings

    def edge_updater(
        self,
        edge_index,
        size: Optional[Tuple[Optional[int], Optional[int]]] = None,
        **kwargs: Any,
    ):
        r"""Computes or updates edge-level representations."""
        edge_index = ops.convert_to_tensor(edge_index)
        if self.flow == "source_to_target":
            i = edge_index[1]
            j = edge_index[0]
        else:
            i = edge_index[0]
            j = edge_index[1]

        i = ops.cast(i, "int32")
        j = ops.cast(j, "int32")

        dim_size = self._get_dim_size(kwargs, i, size)
        edge_update_params = inspect.signature(self.edge_update).parameters

        upd_kwargs = {}
        for param_name in edge_update_params.keys():
            if param_name in kwargs:
                upd_kwargs[param_name] = kwargs[param_name]
            elif param_name.endswith("_i"):
                root = param_name[:-2]
                if root in kwargs:
                    val = kwargs[root]
                    val = val[1] if isinstance(val, (tuple, list)) else val
                    upd_kwargs[param_name] = ops.take(val, i, axis=self.node_dim) if val is not None else None
            elif param_name.endswith("_j"):
                root = param_name[:-2]
                if root in kwargs:
                    val = kwargs[root]
                    val = val[0] if isinstance(val, (tuple, list)) else val
                    upd_kwargs[param_name] = ops.take(val, j, axis=self.node_dim) if val is not None else None
            elif param_name == "index":
                upd_kwargs["index"] = i
            elif param_name == "dim_size":
                upd_kwargs["dim_size"] = dim_size
            elif param_name == "edge_index":
                upd_kwargs["edge_index"] = edge_index

        return self.edge_update(**upd_kwargs)

    def edge_update(self, **kwargs):
        r"""Computes or updates edge attributes."""
        raise NotImplementedError

    def call(self, inputs, edge_index=None, **kwargs):
        r"""Default call handler supporting both (x, edge_index) and legacy (inputs,) tuples."""
        if edge_index is None and isinstance(inputs, (tuple, list)):
            x, a, e = self.get_inputs(inputs)
            return self.propagate(x, a, e, **kwargs)
        if edge_index is not None:
            return self.propagate(edge_index, x=inputs, **kwargs)
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not implement call() with inputs={inputs}, edge_index={edge_index}"
        )
