import keras
from keras import ops

from k3_node.layers.kge.base import KGEModel, margin_ranking_loss


class DistMult(KGEModel):
    r"""The DistMult model from the `"Embedding Entities and Relations for
    Learning and Inference in Knowledge Bases"
    <https://arxiv.org/abs/1412.6575>`_ paper.

    :class:`DistMult` models relations as diagonal matrices, which simplifies
    the bi-linear interaction between the head and tail entities to the score
    function:

    .. math::
        d(h, r, t) = < \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t >

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        sparse (bool, optional): Kept for API compatibility. (default: :obj:`False`)

    Example:
        ```python
        import numpy as np
        from k3_node.layers import DistMult

        head = np.random.randint(0, 20, size=(10,))  # 10 (head, relation, tail) triples
        rel = np.random.randint(0, 5, size=(10,))
        tail = np.random.randint(0, 20, size=(10,))

        model = DistMult(num_nodes=20, num_relations=5, hidden_channels=8)
        score = model(head, rel, tail)  # plausibility score of every triple
        print(tuple(score.shape))  # (10,)
        loss = model.loss(head, rel, tail)  # training loss against randomly corrupted triples
        print(tuple(loss.shape))  # (): a scalar
        ```
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

        self.reset_parameters()

    def reset_parameters(self):
        # A new initializer per tensor: a reused unseeded Keras 3 initializer returns the same values on every call.
        glorot = lambda shape: keras.initializers.GlorotUniform()(shape)
        self.node_emb.embeddings.assign(glorot(ops.shape(self.node_emb.embeddings)))
        self.rel_emb.embeddings.assign(glorot(ops.shape(self.rel_emb.embeddings)))

    def call(self, head_index, rel_type, tail_index):
        head_index = ops.cast(head_index, "int32")
        rel_type = ops.cast(rel_type, "int32")
        tail_index = ops.cast(tail_index, "int32")

        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        return ops.sum(head * rel * tail, axis=-1)

    def loss(self, head_index, rel_type, tail_index):
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return margin_ranking_loss(pos_score, neg_score, margin=self.margin)
