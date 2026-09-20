import os
import os.path as osp
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs


class PolBlogs(InMemoryDataset):
    r"""The Political Blogs dataset containing 1,490 vertices and 19,025 edges.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): Transform function.
        pre_transform (callable, optional): Pre-transform function.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = "https://netset.telecom-paris.fr/datasets/polblogs.tar.gz"

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
        return ["adjacency.tsv", "labels.tsv"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        tar_path = osp.join(self.raw_dir, "polblogs.tar.gz")
        fs.cp(self.url, tar_path, extract=True)
        if osp.exists(tar_path):
            fs.rm(tar_path)

    def process(self):
        adj = np.loadtxt(self.raw_paths[0], delimiter="\t", usecols=(0, 1), dtype=np.int64)
        edge_index = adj.T

        y = np.loadtxt(self.raw_paths[1], delimiter="\t", usecols=(1,), dtype=np.int64)

        data = Data(
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            y=ops.convert_to_tensor(y, dtype="int64"),
            num_nodes=int(y.shape[0]),
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

