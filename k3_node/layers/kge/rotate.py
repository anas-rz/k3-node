import math

import keras
from keras import ops

from k3_node.layers.kge.base import KGEModel, binary_cross_entropy_with_logits


class RotatE(KGEModel):
    r"""The RotatE model from the `"RotatE: Knowledge Graph Embedding by
    Relational Rotation in Complex Space" <https://arxiv.org/abs/
    1902.10197>`_ paper.

    :class:`RotatE` models relations as a rotation in complex space
    from head to tail such that

    .. math::
        \mathbf{e}_t = \mathbf{e}_h \circ \mathbf{e}_r,

    resulting in the scoring function

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h \circ \mathbf{e}_r - \mathbf{e}_t \|}_p

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        sparse (bool, optional): Kept for API compatibility. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
        **kwargs,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse, **kwargs)

        self.margin = margin
        self.node_emb_im = keras.layers.Embedding(num_nodes, hidden_channels)
        self.node_emb_im.build((None,))

        self.reset_parameters()

    def reset_parameters(self):
        glorot = keras.initializers.GlorotUniform()
        self.node_emb.embeddings.assign(glorot(ops.shape(self.node_emb.embeddings)))
        self.node_emb_im.embeddings.assign(glorot(ops.shape(self.node_emb_im.embeddings)))
        uniform = keras.initializers.RandomUniform(0, 2 * math.pi)
        self.rel_emb.embeddings.assign(uniform(ops.shape(self.rel_emb.embeddings)))

    def call(self, head_index, rel_type, tail_index):
        head_index = ops.cast(head_index, "int32")
        rel_type = ops.cast(rel_type, "int32")
        tail_index = ops.cast(tail_index, "int32")

        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = ops.cos(rel_theta), ops.sin(rel_theta)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = ops.stack([re_score, im_score], axis=2)
        score = ops.sqrt(ops.sum(ops.square(complex_score), axis=(1, 2)))

        return self.margin - score

    def loss(self, head_index, rel_type, tail_index):
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = ops.concatenate([pos_score, neg_score], axis=0)

        pos_target = ops.ones_like(pos_score)
        neg_target = ops.zeros_like(neg_score)
        target = ops.concatenate([pos_target, neg_target], axis=0)

        return binary_cross_entropy_with_logits(scores, target)
