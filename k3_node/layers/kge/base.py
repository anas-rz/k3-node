from typing import Tuple

import numpy as np
import keras
from keras import ops

from k3_node.layers.kge.loader import KGTripletLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable


def normalize(x, p: float = 2.0, axis: int = -1, eps: float = 1e-12):
    r"""Lp-normalizes `x` along `axis`, mirroring `torch.nn.functional.normalize`."""
    norm = ops.power(ops.sum(ops.power(ops.abs(x), p), axis=axis, keepdims=True), 1.0 / p)
    return x / ops.maximum(norm, eps)


def margin_ranking_loss(pos_score, neg_score, margin: float = 1.0):
    r"""Mirrors `torch.nn.functional.margin_ranking_loss` with `target=1`."""
    return ops.mean(ops.relu(margin - pos_score + neg_score))


def binary_cross_entropy_with_logits(logits, target):
    r"""Numerically stable sigmoid cross-entropy, mirroring
    `torch.nn.functional.binary_cross_entropy_with_logits`."""
    zeros = ops.zeros_like(logits)
    cond = logits >= zeros
    relu_logits = ops.where(cond, logits, zeros)
    neg_abs_logits = ops.where(cond, -logits, logits)
    loss = relu_logits - logits * target + ops.log(1.0 + ops.exp(neg_abs_logits))
    return ops.mean(loss)


class KGEModel(keras.layers.Layer):
    r"""An abstract base class for implementing custom KGE models.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): Kept for API compatibility with PyG; has no
            effect since Keras optimizers do not distinguish sparse
            embedding gradients the way PyTorch does. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels
        self.sparse = sparse

        self.node_emb = keras.layers.Embedding(num_nodes, hidden_channels)
        self.rel_emb = keras.layers.Embedding(num_relations, hidden_channels)
        self.node_emb.build((None,))
        self.rel_emb.build((None,))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_emb.embeddings.assign(
            self.node_emb.embeddings_initializer(ops.shape(self.node_emb.embeddings))
        )
        self.rel_emb.embeddings.assign(
            self.rel_emb.embeddings_initializer(ops.shape(self.rel_emb.embeddings))
        )

    def call(self, head_index, rel_type, tail_index):
        r"""Returns the score for the given triplet.

        Args:
            head_index: The head indices.
            rel_type: The relation type.
            tail_index: The tail indices.
        """
        raise NotImplementedError

    def loss(self, head_index, rel_type, tail_index):
        r"""Returns the loss value for the given triplet."""
        raise NotImplementedError

    def loader(self, head_index, rel_type, tail_index, **kwargs):
        r"""Returns a mini-batch loader that samples a subset of triplets.

        Args:
            head_index: The head indices.
            rel_type: The relation type.
            tail_index: The tail indices.
            **kwargs (optional): Additional arguments of
                :class:`k3_node.layers.kge.KGTripletLoader`, such as
                `batch_size`, `shuffle` or `drop_last`.
        """
        return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)

    def test(
        self,
        head_index,
        rel_type,
        tail_index,
        batch_size: int,
        k: int = 10,
        log: bool = True,
    ) -> Tuple[float, float, float]:
        r"""Evaluates the model quality by computing Mean Rank, MRR and
        Hits@:math:`k` across all possible tail entities.

        Args:
            head_index: The head indices.
            rel_type: The relation type.
            tail_index: The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The :math:`k` in Hits @ :math:`k`.
                (default: :obj:`10`)
            log (bool, optional): If set to :obj:`False`, will not print a
                progress bar to the console. (default: :obj:`True`)
        """
        head_index = ops.convert_to_numpy(head_index)
        rel_type = ops.convert_to_numpy(rel_type)
        tail_index = ops.convert_to_numpy(tail_index)

        arange = range(head_index.shape[0])
        arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            h, r, t = int(head_index[i]), int(rel_type[i]), int(tail_index[i])

            scores = []
            tail_indices = np.arange(self.num_nodes)
            for start in range(0, self.num_nodes, batch_size):
                ts = tail_indices[start:start + batch_size]
                hs = np.full_like(ts, h)
                rs = np.full_like(ts, r)
                out = self(
                    ops.convert_to_tensor(hs),
                    ops.convert_to_tensor(rs),
                    ops.convert_to_tensor(ts),
                )
                scores.append(ops.convert_to_numpy(out))
            scores = np.concatenate(scores)
            rank = int(np.nonzero(np.argsort(-scores) == t)[0][0])

            mean_ranks.append(rank)
            reciprocal_ranks.append(1.0 / (rank + 1))
            hits_at_k.append(rank < k)

        mean_rank = float(np.mean(mean_ranks))
        mrr = float(np.mean(reciprocal_ranks))
        hits_at_k = float(np.mean(hits_at_k))

        return mean_rank, mrr, hits_at_k

    def random_sample(
        self,
        head_index,
        rel_type,
        tail_index,
    ):
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index: The head indices.
            rel_type: The relation type.
            tail_index: The tail indices.
        """
        num_triplets = ops.shape(head_index)[0]
        num_negatives = num_triplets // 2

        rnd_index = keras.random.randint(
            ops.shape(head_index), 0, self.num_nodes, dtype="int32"
        )
        rnd_index = ops.cast(rnd_index, head_index.dtype)

        head_index = ops.concatenate(
            [rnd_index[:num_negatives], head_index[num_negatives:]], axis=0
        )
        tail_index = ops.concatenate(
            [tail_index[:num_negatives], rnd_index[num_negatives:]], axis=0
        )

        return head_index, rel_type, tail_index

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.num_nodes}, "
            f"num_relations={self.num_relations}, "
            f"hidden_channels={self.hidden_channels})"
        )

    __str__ = __repr__
