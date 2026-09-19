import keras
from keras import ops

from k3_node.layers.kge.base import KGEModel, binary_cross_entropy_with_logits


def triple_dot(x, y, z):
    return ops.sum(x * y * z, axis=-1)


class ComplEx(KGEModel):
    r"""The ComplEx model from the `"Complex Embeddings for Simple Link
    Prediction" <https://arxiv.org/abs/1606.06357>`_ paper.

    :class:`ComplEx` models relations as complex-valued bilinear mappings
    between head and tail entities using the Hermetian dot product.
    The entities and relations are embedded in different dimensional spaces,
    resulting in the scoring function:

    .. math::
        d(h, r, t) = Re(< \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t>)

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): Kept for API compatibility. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        **kwargs,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse, **kwargs)

        self.node_emb_im = keras.layers.Embedding(num_nodes, hidden_channels)
        self.rel_emb_im = keras.layers.Embedding(num_relations, hidden_channels)
        self.node_emb_im.build((None,))
        self.rel_emb_im.build((None,))

        self.reset_parameters()

    def reset_parameters(self):
        glorot = keras.initializers.GlorotUniform()
        self.node_emb.embeddings.assign(glorot(ops.shape(self.node_emb.embeddings)))
        self.node_emb_im.embeddings.assign(glorot(ops.shape(self.node_emb_im.embeddings)))
        self.rel_emb.embeddings.assign(glorot(ops.shape(self.rel_emb.embeddings)))
        self.rel_emb_im.embeddings.assign(glorot(ops.shape(self.rel_emb_im.embeddings)))

    def call(self, head_index, rel_type, tail_index):
        head_index = ops.cast(head_index, "int32")
        rel_type = ops.cast(rel_type, "int32")
        tail_index = ops.cast(tail_index, "int32")

        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        rel_re = self.rel_emb(rel_type)
        rel_im = self.rel_emb_im(rel_type)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        return (
            triple_dot(head_re, rel_re, tail_re)
            + triple_dot(head_im, rel_re, tail_im)
            + triple_dot(head_re, rel_im, tail_im)
            - triple_dot(head_im, rel_im, tail_re)
        )

    def loss(self, head_index, rel_type, tail_index):
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = ops.concatenate([pos_score, neg_score], axis=0)

        pos_target = ops.ones_like(pos_score)
        neg_target = ops.zeros_like(neg_score)
        target = ops.concatenate([pos_target, neg_target], axis=0)

        return binary_cross_entropy_with_logits(scores, target)
