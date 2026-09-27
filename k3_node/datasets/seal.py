from typing import Optional

import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset


class SEALDataset(InMemoryDataset):
    r"""Enclosing subgraphs for link prediction, as in SEAL
    (`"Link Prediction Based on Graph Neural Networks" <https://arxiv.org/abs/1802.09691>`_).

    Every link to classify becomes a small graph: the ``num_hops``-hop neighborhood of its two
    nodes, without the link itself. Nodes are described only by their double-radius labels
    (their distances to the two nodes, one-hot encoded as ``x``); ``y`` is 1 for a true link and
    0 for a non-edge. This turns link prediction into graph classification.

    Args:
        data (Data): One split from :class:`~k3_node.transforms.RandomLinkSplit`: message passing
            edges in ``edge_index`` and the links in ``edge_label_index`` / ``edge_label`` (or
            ``pos_edge_label_index`` / ``neg_edge_label_index``).
        num_hops (int): Size of the neighborhood around each link. (default: ``2``)
        num_labels (int, optional): Size of the one-hot node labels. Use the training set's
            ``num_labels`` for the validation and test sets. (default: the largest label + 1)

    Example:
        ```python
        import numpy as np
        from k3_node.data import Data
        from k3_node.datasets import SEALDataset
        from k3_node.transforms import RandomLinkSplit

        data = Data(edge_index=np.random.randint(0, 30, size=(2, 120)), num_nodes=30)
        train_data, val_data, test_data = RandomLinkSplit(num_val=0.1, num_test=0.1)(data)
        train_dataset = SEALDataset(train_data, num_hops=2)
        print(len(train_dataset) == train_data.edge_label_index.shape[1])  # True: one graph per link
        ```
    """

    def __init__(self, data, num_hops: int = 2, num_labels: Optional[int] = None):
        super().__init__(None)
        from k3_node.utils.graph import drnl_node_labeling, k_hop_subgraph

        links, labels = self._links(data)
        edge_index = np.asarray(ops.convert_to_numpy(data.edge_index)).astype(np.int64)
        graphs = []
        for (src, dst), y in zip(links.T, labels):
            nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], num_hops, edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
            s, d = (int(m) for m in mapping)
            keep = ~(((sub_edge_index[0] == s) & (sub_edge_index[1] == d))
                     | ((sub_edge_index[0] == d) & (sub_edge_index[1] == s)))
            sub_edge_index = sub_edge_index[:, keep]  # hide the link to predict
            z = drnl_node_labeling(sub_edge_index, s, d, num_nodes=len(nodes))
            graphs.append((z, sub_edge_index, y))

        self.num_labels = num_labels or max(int(z.max()) for z, _, _ in graphs) + 1
        self.data, self.slices = self.collate([
            Data(x=np.eye(self.num_labels, dtype=np.float32)[np.minimum(z, self.num_labels - 1)],
                 edge_index=e, y=np.array([y], dtype=np.float32))
            for z, e, y in graphs
        ])

    @staticmethod
    def _links(data):
        def arr(x):
            return np.asarray(ops.convert_to_numpy(x))

        if getattr(data, "edge_label_index", None) is not None:
            return arr(data.edge_label_index).astype(np.int64), arr(data.edge_label)
        pos = arr(data.pos_edge_label_index).astype(np.int64)
        neg = getattr(data, "neg_edge_label_index", None)
        neg = np.zeros((2, 0), np.int64) if neg is None else arr(neg).astype(np.int64)
        return np.concatenate([pos, neg], axis=1), np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
