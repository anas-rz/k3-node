from typing import Optional
import numpy as np
import keras
from keras import ops


class Node2Vec(keras.layers.Layer):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    Args:
        edge_index: The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1.0`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy. (default: :obj:`1.0`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
    """
    def __init__(
        self,
        edge_index,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        edge_index_np = ops.convert_to_numpy(edge_index).astype(np.int64)
        if num_nodes is None:
            num_nodes = int(edge_index_np.max()) + 1 if edge_index_np.size > 0 else 0

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.EPS = 1e-15

        # Build adjacency list for fast random walks
        self.adj = [[] for _ in range(num_nodes)]
        if edge_index_np.size > 0:
            for src, dst in zip(edge_index_np[0], edge_index_np[1]):
                self.adj[src].append(int(dst))

        self.embedding = keras.layers.Embedding(num_nodes, embedding_dim)

    def reset_parameters(self):
        if self.embedding.built:
            self.embedding.embeddings.assign(
                keras.initializers.GlorotUniform()(self.embedding.embeddings.shape)
            )

    def call(self, batch: Optional[any] = None):
        """Returns the embeddings for the nodes in :obj:`batch`."""
        if batch is None:
            batch = ops.arange(self.num_nodes, dtype="int64")
        return self.embedding(batch)

    def pos_sample(self, batch):
        batch_np = ops.convert_to_numpy(batch).astype(np.int64)
        repeated = np.repeat(batch_np, self.walks_per_node)

        # Sample random walks
        all_walks = []
        for node in repeated:
            walk = [int(node)]
            for _ in range(self.walk_length):
                cur = walk[-1]
                nbrs = self.adj[cur]
                if len(nbrs) > 0:
                    walk.append(nbrs[np.random.randint(len(nbrs))])
                else:
                    walk.append(cur)
            all_walks.append(walk)

        rw = np.array(all_walks, dtype=np.int64)
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        out = np.concatenate(walks, axis=0) if len(walks) > 0 else rw
        return ops.convert_to_tensor(out, dtype="int64")

    def neg_sample(self, batch):
        batch_np = ops.convert_to_numpy(batch).astype(np.int64)
        repeated = np.repeat(
            batch_np, self.walks_per_node * self.num_negative_samples
        )
        rand_rest = np.random.randint(
            0, max(self.num_nodes, 1), size=(len(repeated), self.walk_length)
        )
        rw = np.concatenate([repeated[:, None], rand_rest], axis=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        out = np.concatenate(walks, axis=0) if len(walks) > 0 else rw
        return ops.convert_to_tensor(out, dtype="int64")

    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss
        start = pos_rw[:, 0]
        rest = pos_rw[:, 1:]
        pos_b = ops.shape(pos_rw)[0]

        h_start = ops.reshape(self.embedding(start), (pos_b, 1, self.embedding_dim))
        h_rest = ops.reshape(
            self.embedding(ops.reshape(rest, (-1,))),
            (pos_b, -1, self.embedding_dim),
        )

        out = ops.reshape(ops.sum(h_start * h_rest, axis=-1), (-1,))
        pos_loss = -ops.mean(ops.log(ops.sigmoid(out) + self.EPS))

        # Negative loss
        start = neg_rw[:, 0]
        rest = neg_rw[:, 1:]
        neg_b = ops.shape(neg_rw)[0]

        h_start = ops.reshape(self.embedding(start), (neg_b, 1, self.embedding_dim))
        h_rest = ops.reshape(
            self.embedding(ops.reshape(rest, (-1,))),
            (neg_b, -1, self.embedding_dim),
        )

        out = ops.reshape(ops.sum(h_start * h_rest, axis=-1), (-1,))
        neg_loss = -ops.mean(ops.log(1.0 - ops.sigmoid(out) + self.EPS))

        return pos_loss + neg_loss
