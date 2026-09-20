import os.path as osp
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs
from k3_node.utils.graph import coalesce


class Actor(InMemoryDataset):
    r"""The actor-only induced subgraph of the film-director-actor-writer network
    used in the "Geom-GCN: Geometric Graph Convolutional Networks" paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): Transform function.
        pre_transform (callable, optional): Pre-transform function.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "out1_node_feature_label.txt",
            "out1_graph_edges.txt",
            "film_split_0.6_0.2_0.npz",
            "film_split_0.6_0.2_1.npz",
            "film_split_0.6_0.2_2.npz",
            "film_split_0.6_0.2_3.npz",
            "film_split_0.6_0.2_4.npz",
            "film_split_0.6_0.2_5.npz",
            "film_split_0.6_0.2_6.npz",
            "film_split_0.6_0.2_7.npz",
            "film_split_0.6_0.2_8.npz",
            "film_split_0.6_0.2_9.npz",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for f in self.raw_file_names[:2]:
            fs.cp(f"{self.url}/new_data/film/{f}", self.raw_dir)
        for f in self.raw_file_names[2:]:
            fs.cp(f"{self.url}/splits/{f}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], "r") as f:
            lines = f.read().split("\n")[1:-1]
        xs = [[float(v) for v in r.split("\t")[1].split(",")] for r in lines]
        ys = [int(r.split("\t")[2]) for r in lines]

        x = np.array(xs, dtype=np.float32)
        y = np.array(ys, dtype=np.int64)

        with open(self.raw_paths[1], "r") as f:
            lines = f.read().split("\n")[1:-1]
        edges = [[int(v) for v in r.split("\t")] for r in lines]
        edge_index = np.array(edges, dtype=np.int64).T
        edge_index_t, _ = coalesce(edge_index, num_nodes=x.shape[0])
        edge_index = ops.convert_to_numpy(edge_index_t)

        train_masks, val_masks, test_masks = [], [], []
        for filepath in self.raw_paths[2:]:
            masks = np.load(filepath)
            train_masks.append(masks["train_mask"])
            val_masks.append(masks["val_mask"])
            test_masks.append(masks["test_mask"])

        train_mask = np.stack(train_masks, axis=1)
        val_mask = np.stack(val_masks, axis=1)
        test_mask = np.stack(test_masks, axis=1)

        data = Data(
            x=ops.convert_to_tensor(x, dtype="float32"),
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            y=ops.convert_to_tensor(y, dtype="int64"),
            train_mask=ops.convert_to_tensor(train_mask, dtype="bool"),
            val_mask=ops.convert_to_tensor(val_mask, dtype="bool"),
            test_mask=ops.convert_to_tensor(test_mask, dtype="bool"),
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

