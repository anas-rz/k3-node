from dataclasses import dataclass
from typing import Optional
from keras import layers, ops

from ..select.base import SelectOutput


@dataclass
class ConnectOutput:
    r"""The output of the :class:`Connect` method, which holds the coarsened
    graph structure, and optional pooled edge features and batch vectors.

    Args:
        edge_index: The edge indices of the coarsened graph.
        edge_attr: The pooled edge features of the coarsened graph. (default: None)
        batch: The pooled batch vector of the coarsened graph. (default: None)
    """
    edge_index: any
    edge_attr: Optional[any] = None
    batch: Optional[any] = None

    def __post_init__(self):
        shape_edge = getattr(self.edge_index, "shape", None)
        if shape_edge is not None:
            if len(shape_edge) != 2:
                raise ValueError(
                    f"Expected 'edge_index' to be two-dimensional "
                    f"(got {len(shape_edge)} dimensions)"
                )
            if shape_edge[0] is not None and shape_edge[0] != 2:
                raise ValueError(
                    f"Expected 'edge_index' to have size '2' in the first dimension "
                    f"(got '{shape_edge[0]}')"
                )
        if self.edge_attr is not None:
            shape_attr = getattr(self.edge_attr, "shape", None)
            if (
                shape_edge is not None
                and shape_attr is not None
                and len(shape_edge) == 2
                and len(shape_attr) >= 1
                and shape_edge[1] is not None
                and shape_attr[0] is not None
                and shape_attr[0] != shape_edge[1]
            ):
                raise ValueError(
                    f"Expected 'edge_index' and 'edge_attr' to hold the same number "
                    f"of edges (got {shape_edge[1]} and {shape_attr[0]} edges)"
                )


try:
    import jax
    from jax.tree_util import register_pytree_node

    register_pytree_node(
        ConnectOutput,
        lambda c: ((c.edge_index, c.edge_attr, c.batch), ()),
        lambda aux, children: ConnectOutput(children[0], children[1], children[2]),
    )
except Exception:
    pass


class Connect(layers.Layer):
    r"""An abstract base class for implementing custom edge connection
    operators as described in the `"Understanding Pooling in Graph Neural
    Networks" <https://arxiv.org/abs/1905.05178>`_ paper.
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def __call__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], SelectOutput):
            return self.call(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def call(
        self,
        select_output: SelectOutput,
        edge_index,
        edge_attr: Optional[any] = None,
        batch: Optional[any] = None,
    ) -> ConnectOutput:
        raise NotImplementedError

    @staticmethod
    def get_pooled_batch(
        select_output: SelectOutput,
        batch: Optional[any],
    ) -> Optional[any]:
        r"""Returns the batch vector of the coarsened graph."""
        if batch is None:
            return None
        return ops.take(batch, select_output.node_index, axis=0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
