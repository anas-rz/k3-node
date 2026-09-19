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
        if len(ops.shape(self.node_index)) != 1:
            raise ValueError(
                f"Expected 'node_index' to be one-dimensional "
                f"(got {len(ops.shape(self.node_index))} dimensions)"
            )
        if len(ops.shape(self.cluster_index)) != 1:
            raise ValueError(
                f"Expected 'cluster_index' to be one-dimensional "
                f"(got {len(ops.shape(self.cluster_index))} dimensions)"
            )
        if ops.shape(self.node_index)[0] != ops.shape(self.cluster_index)[0]:
            raise ValueError(
                f"Expected 'node_index' and 'cluster_index' to hold the same "
                f"number of values (got {ops.shape(self.node_index)[0]} and "
                f"{ops.shape(self.cluster_index)[0]} values)"
            )
        if self.weight is not None:
            if len(ops.shape(self.weight)) != 1:
                raise ValueError(
                    f"Expected 'weight' vector to be one-dimensional "
                    f"(got {len(ops.shape(self.weight))} dimensions)"
                )
            if ops.shape(self.weight)[0] != ops.shape(self.node_index)[0]:
                raise ValueError(
                    f"Expected 'weight' to hold {ops.shape(self.node_index)[0]} "
                    f"values (got {ops.shape(self.weight)[0]} values)"
                )


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

