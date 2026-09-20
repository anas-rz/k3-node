import json
import warnings
from itertools import chain
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs
from k3_node.transforms.utils import to_undirected


class WikiCS(InMemoryDataset):
    r"""The semi-supervised Wikipedia-based dataset from the
    "Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks" paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): Transform function.
        pre_transform (callable, optional): Pre-transform function.
        is_undirected (bool, optional): Whether the graph is undirected. (default: True)
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = "https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        is_undirected: Optional[bool] = None,
        force_reload: bool = False,
    ):
        if is_undirected is None:
            is_undirected = True
        self.is_undirected = is_undirected
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["data.json"]

    @property
    def processed_file_names(self) -> str:
        return "data_undirected.pt" if self.is_undirected else "data.pt"

    def download(self):
        for name in self.raw_file_names:
            fs.cp(f"{self.url}/{name}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0]) as f:
            data = json.load(f)

        x = np.array(data["features"], dtype=np.float32)
        y = np.array(data["labels"], dtype=np.int64)

        edges = [[(i, j) for j in js] for i, js in enumerate(data["links"])]
        edges = list(chain(*edges))
        edge_index = np.array(edges, dtype=np.int64).T
        if self.is_undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.shape[0])

        train_mask = np.array(data["train_masks"], dtype=bool).T
        val_mask = np.array(data["val_masks"], dtype=bool).T
        test_mask = np.array(data["test_mask"], dtype=bool)
        stopping_mask = np.array(data["stopping_masks"], dtype=bool).T

        data_obj = Data(
            x=ops.convert_to_tensor(x, dtype="float32"),
            y=ops.convert_to_tensor(y, dtype="int64"),
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            train_mask=ops.convert_to_tensor(train_mask, dtype="bool"),
            val_mask=ops.convert_to_tensor(val_mask, dtype="bool"),
            test_mask=ops.convert_to_tensor(test_mask, dtype="bool"),
            stopping_mask=ops.convert_to_tensor(stopping_mask, dtype="bool"),
        )

        if self.pre_transform is not None:
            data_obj = self.pre_transform(data_obj)

        self.save([data_obj], self.processed_paths[0])

