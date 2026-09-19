import math

import keras
from keras import ops

from k3_node.layers.kge.base import KGEModel, margin_ranking_loss, normalize


class TransE(KGEModel):
    r"""The TransE model from the `"Translating Embeddings for Modeling
    Multi-Relational Data" <https://proceedings.neurips.cc/paper/2013/file/
    1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>`_ paper.

    :class:`TransE` models relations as a translation from head to tail
    entities such that

    .. math::
        \mathbf{e}_h + \mathbf{e}_r \approx \mathbf{e}_t,

    resulting in the scoring function:

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h + \mathbf{e}_r - \mathbf{e}_t \|}_p

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        p_norm (float, optional): The order embedding and distance
            normalization. (default: :obj:`1.0`)
        sparse (bool, optional): Kept for API compatibility. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
        **kwargs,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse, **kwargs)

        self.p_norm = p_norm
        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        bound = 6.0 / math.sqrt(self.hidden_channels)
        uniform = keras.initializers.RandomUniform(-bound, bound)
        self.node_emb.embeddings.assign(uniform(ops.shape(self.node_emb.embeddings)))
        self.rel_emb.embeddings.assign(uniform(ops.shape(self.rel_emb.embeddings)))
        self.rel_emb.embeddings.assign(normalize(self.rel_emb.embeddings, p=self.p_norm, axis=-1))

    def call(self, head_index, rel_type, tail_index):
        head_index = ops.cast(head_index, "int32")
        rel_type = ops.cast(rel_type, "int32")
        tail_index = ops.cast(tail_index, "int32")

        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        head = normalize(head, p=self.p_norm, axis=-1)
        tail = normalize(tail, p=self.p_norm, axis=-1)

        # Calculate *negative* TransE norm:
        diff = (head + rel) - tail
        return -ops.power(ops.sum(ops.power(ops.abs(diff), self.p_norm), axis=-1), 1.0 / self.p_norm)

    def loss(self, head_index, rel_type, tail_index):
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return margin_ranking_loss(pos_score, neg_score, margin=self.margin)
