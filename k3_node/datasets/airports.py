import os.path as osp
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs
from k3_node.utils.graph import coalesce


class Airports(InMemoryDataset):
    r"""The Airports dataset from the "struc2vec: Learning Node Representations from Structural Identity" paper."""

    edge_url = "https://github.com/leoribeiro/struc2vec/raw/master/graph/{}-airports.edgelist"
    label_url = "https://github.com/leoribeiro/struc2vec/raw/master/graph/labels-{}-airports.txt"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.name = name.lower()
        assert self.name in ["usa", "brazil", "europe"]
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return [f"{self.name}-airports.edgelist", f"labels-{self.name}-airports.txt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        fs.cp(self.edge_url.format(self.name), self.raw_dir)
        fs.cp(self.label_url.format(self.name), self.raw_dir)

    def process(self):
        index_map, ys = {}, []
        with open(self.raw_paths[1]) as f:
            rows = f.read().split("\n")[1:-1]
            for i, row in enumerate(rows):
                idx, label = row.split()
                index_map[int(idx)] = i
                ys.append(int(label))
        y = np.array(ys, dtype=np.int64)
        x = np.eye(len(ys), dtype=np.float32)

        edge_indices = []
        with open(self.raw_paths[0]) as f:
            rows = f.read().split("\n")[:-1]
            for row in rows:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_index_t, _ = coalesce(edge_index, num_nodes=len(ys))
        edge_index = ops.convert_to_numpy(edge_index_t)

        data = Data(
            x=ops.convert_to_tensor(x, dtype="float32"),
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            y=ops.convert_to_tensor(y, dtype="int64"),
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}Airports()"

