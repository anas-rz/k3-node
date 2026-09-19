from typing import Dict, List, Optional, Tuple
import numpy as np
import keras
from keras import ops

EdgeType = Tuple[str, str, str]
NodeType = str


class MetaPath2Vec(keras.layers.Layer):
    r"""The MetaPath2Vec model from the `"metapath2vec: Scalable Representation
    Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/
    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper where random walks based
    on a given :obj:`metapath` are sampled in a heterogeneous graph, and node
    embeddings are learned via negative sampling optimization.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], Tensor]): Dictionary
            holding edge indices for each edge type.
        embedding_dim (int): The size of each embedding vector.
        metapath (List[Tuple[str, str, str]]): The sequence of edge types
            denoting the metapath.
        walk_length (int): The walk length.
        context_size (int): The context size considered for positive samples.
        walks_per_node (int, optional): The number of walks to sample for each node.
            (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples.
            (default: :obj:`1`)
        num_nodes_dict (Dict[str, int], optional): The number of nodes for each
            node type. (default: :obj:`None`)
    """
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, any],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                e_np = ops.convert_to_numpy(edge_index)
                key = keys[0]
                N = int(e_np[0].max() + 1) if e_np.size > 0 else 0
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(e_np[1].max() + 1) if e_np.size > 0 else 0
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        # Build adjacency dictionaries
        self.adj_dict = {}
        for keys, edge_index in edge_index_dict.items():
            src_type, _, dst_type = keys
            num_src = num_nodes_dict[src_type]
            adj = [[] for _ in range(num_src)]
            e_np = ops.convert_to_numpy(edge_index).astype(np.int64)
            if e_np.size > 0:
                for s, d in zip(e_np[0], e_np[1]):
                    adj[s].append(int(d))
            self.adj_dict[keys] = adj

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError(
                    "Found invalid metapath. Ensure that the destination node "
                    "type matches with the source node type across all "
                    "consecutive edge types."
                )

        assert walk_length + 1 >= context_size

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict
        self.EPS = 1e-15

        types = sorted(list({x[0] for x in metapath} | {x[-1] for x in metapath}))

        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath] * int(
            (walk_length / len(metapath)) + 1
        )
        offset = offset[: walk_length + 1]
        self.offset = np.array(offset, dtype=np.int64)

        self.dummy_idx = count
        self.embedding = keras.layers.Embedding(count + 1, embedding_dim)

    def reset_parameters(self):
        if self.embedding.built:
            self.embedding.embeddings.assign(
                keras.initializers.GlorotUniform()(self.embedding.embeddings.shape)
            )

    def __call__(self, node_type: str, batch: Optional[any] = None, **kwargs):
        return self.call(node_type, batch)

    def forward(self, node_type: str, batch: Optional[any] = None, **kwargs):
        return self.call(node_type, batch)

    def call(self, node_type: str, batch: Optional[any] = None):
        r"""Returns the embeddings for the nodes in :obj:`batch` of type
        :obj:`node_type`.
        """
        start = self.start[node_type]
        end = self.end[node_type]
        if batch is None:
            batch = ops.arange(start, end, dtype="int64")
        else:
            batch = ops.cast(batch, "int64") + start
        return self.embedding(batch)

    def _pos_sample(self, batch):
        batch_np = ops.convert_to_numpy(batch).astype(np.int64)
        repeated = np.repeat(batch_np, self.walks_per_node)

        all_walks = []
        for node in repeated:
            walk = [int(node)]
            for i in range(self.walk_length):
                edge_type = self.metapath[i % len(self.metapath)]
                cur = walk[-1]
                nbrs = (
                    self.adj_dict[edge_type][cur]
                    if cur < len(self.adj_dict[edge_type])
                    else []
                )
                if len(nbrs) > 0:
                    walk.append(nbrs[np.random.randint(len(nbrs))])
                else:
                    walk.append(self.dummy_idx)
            all_walks.append(walk)

        rw = np.array(all_walks, dtype=np.int64)
        rw = rw + self.offset[None, :]
        rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        out = np.concatenate(walks, axis=0) if len(walks) > 0 else rw
        return ops.convert_to_tensor(out, dtype="int64")

    def _neg_sample(self, batch):
        batch_np = ops.convert_to_numpy(batch).astype(np.int64)
        repeated = np.repeat(
            batch_np, self.walks_per_node * self.num_negative_samples
        )

        rws = [repeated]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            num_nodes = self.num_nodes_dict[keys[-1]]
            rand = np.random.randint(0, max(num_nodes, 1), size=len(repeated))
            rws.append(rand)

        rw = np.stack(rws, axis=-1)
        rw = rw + self.offset[None, :]

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
