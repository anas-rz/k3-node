from dataclasses import dataclass
from typing import Optional
from keras import layers, ops


@dataclass
class SelectOutput:
    r"""The output of the :class:`Select` method, which holds an assignment
    from selected nodes to their respective cluster(s).

    Args:
        node_index: The indices of the selected nodes.
        num_nodes: The number of nodes.
        cluster_index: The indices of the clusters each node in
            :obj:`node_index` is assigned to.
        num_clusters: The number of clusters.
        weight (optional): A weight vector, denoting the strength
            of the assignment of a node to its cluster. (default: :obj:`None`)
    """
    node_index: any
    num_nodes: int
    cluster_index: any
    num_clusters: int
    weight: Optional[any] = None

    def __post_init__(self):
        shape_node = getattr(self.node_index, "shape", None)
        shape_cluster = getattr(self.cluster_index, "shape", None)
        if shape_node is not None and len(shape_node) != 1:
            raise ValueError(
                f"Expected 'node_index' to be one-dimensional "
                f"(got {len(shape_node)} dimensions)"
            )
        if shape_cluster is not None and len(shape_cluster) != 1:
            raise ValueError(
                f"Expected 'cluster_index' to be one-dimensional "
                f"(got {len(shape_cluster)} dimensions)"
            )
        if (
            shape_node is not None
            and shape_cluster is not None
            and len(shape_node) > 0
            and len(shape_cluster) > 0
            and shape_node[0] is not None
            and shape_cluster[0] is not None
            and shape_node[0] != shape_cluster[0]
        ):
            raise ValueError(
                f"Expected 'node_index' and 'cluster_index' to hold the same "
                f"number of values (got {shape_node[0]} and "
                f"{shape_cluster[0]} values)"
            )
        if self.weight is not None:
            shape_weight = getattr(self.weight, "shape", None)
            if shape_weight is not None and len(shape_weight) != 1:
                raise ValueError(
                    f"Expected 'weight' vector to be one-dimensional "
                    f"(got {len(shape_weight)} dimensions)"
                )
            if (
                shape_weight is not None
                and shape_node is not None
                and len(shape_weight) > 0
                and len(shape_node) > 0
                and shape_weight[0] is not None
                and shape_node[0] is not None
                and shape_weight[0] != shape_node[0]
            ):
                raise ValueError(
                    f"Expected 'weight' to hold {shape_node[0]} "
                    f"values (got {shape_weight[0]} values)"
                )


try:
    import jax
    from jax.tree_util import register_pytree_node

    register_pytree_node(
        SelectOutput,
        lambda s: (
            (s.node_index, s.cluster_index, s.weight),
            (s.num_nodes, s.num_clusters),
        ),
        lambda aux, children: SelectOutput(
            children[0], aux[0], children[1], aux[1], children[2]
        ),
    )
except Exception:
    pass


class Select(layers.Layer):
    r"""An abstract base class for implementing custom node selections as
    described in the `"Understanding Pooling in Graph Neural Networks"
    <https://arxiv.org/abs/1905.05178>`_ paper, which maps the nodes of an
    input graph to supernodes in the coarsened graph.
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def call(self, *args, **kwargs) -> SelectOutput:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

