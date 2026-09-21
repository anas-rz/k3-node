import json
import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
from keras import ops

from k3_node.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from k3_node.layers.conv.utils import remove_self_loops


class PPI(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`k3_node.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`k3_node.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`k3_node.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 20
          - ~2,245.3
          - ~61,318.4
          - 50
          - 121
    """

    url = "https://data.dgl.ai/dataset/ppi.zip"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert split.lower() in ["train", "val", "valid", "test"], f"Invalid split '{split}'"
        self.split = "val" if split.lower() == "valid" else split.lower()

        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )

        if self.split == "train":
            self.load(self.processed_paths[0])
        elif self.split == "val":
            self.load(self.processed_paths[1])
        elif self.split == "test":
            self.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> List[str]:
        splits = ["train", "valid", "test"]
        files = ["feats.npy", "graph_id.npy", "graph.json", "labels.npy"]
        return [f"{split}_{name}" for split, name in product(splits, files)]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt"]

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        if osp.exists(path):
            os.unlink(path)

    def process(self) -> None:
        try:
            import networkx as nx
            from networkx.readwrite import json_graph
            has_networkx = True
        except ImportError:
            has_networkx = False

        for s, split in enumerate(["train", "valid", "test"]):
            path = osp.join(self.raw_dir, f"{split}_graph.json")
            with open(path, "r", encoding="utf-8") as f:
                graph_json = json.load(f)

            if has_networkx:
                try:
                    G = nx.DiGraph(json_graph.node_link_graph(graph_json, edges="links"))
                except TypeError:
                    G = nx.DiGraph(json_graph.node_link_graph(graph_json))
            else:
                links = graph_json.get("links", [])
                src_all = np.array([link["source"] for link in links], dtype=np.int64)
                dst_all = np.array([link["target"] for link in links], dtype=np.int64)

            x_np = np.load(osp.join(self.raw_dir, f"{split}_feats.npy"))
            y_np = np.load(osp.join(self.raw_dir, f"{split}_labels.npy"))

            data_list = []
            path = osp.join(self.raw_dir, f"{split}_graph_id.npy")
            idx = np.load(path)
            idx = idx - idx.min()

            for i in range(int(idx.max()) + 1):
                mask = (idx == i)
                node_indices = np.where(mask)[0]

                if has_networkx:
                    G_s = G.subgraph(node_indices.tolist())
                    edges = list(G_s.edges)
                    if len(edges) > 0:
                        edge_index = np.array(edges, dtype=np.int64).T
                        edge_index = edge_index - edge_index.min()
                    else:
                        edge_index = np.zeros((2, 0), dtype=np.int64)
                else:
                    min_node, max_node = node_indices.min(), node_indices.max()
                    edge_mask = (
                        (src_all >= min_node)
                        & (src_all <= max_node)
                        & (dst_all >= min_node)
                        & (dst_all <= max_node)
                    )
                    sub_src = src_all[edge_mask] - min_node
                    sub_dst = dst_all[edge_mask] - min_node
                    edge_index = np.stack([sub_src, sub_dst], axis=0)

                edge_index = ops.convert_to_tensor(edge_index, dtype="int64")
                edge_index, _ = remove_self_loops(edge_index)

                x = ops.convert_to_tensor(x_np[mask], dtype="float32")
                y = ops.convert_to_tensor(y_np[mask], dtype="float32")

                data = Data(edge_index=edge_index, x=x, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, self.processed_paths[s])

    @property
    def num_classes(self) -> int:
        data = self.get(0)
        y = getattr(data, "y", None)
        if y is not None and len(y.shape) > 1:
            return y.shape[-1]
        return super().num_classes

