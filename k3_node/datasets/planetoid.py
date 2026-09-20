import os.path as osp
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import InMemoryDataset
from k3_node.io import fs, read_planetoid_data


class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    "Revisiting Semi-Supervised Learning with Graph Embeddings" paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset ("Cora", "CiteSeer", "PubMed").
        split (str, optional): The type of dataset split ("public", "full", "geom-gcn", "random").
            (default: "public")
        num_train_per_class (int, optional): The number of training samples per class for "random" split.
            (default: 20)
        num_val (int, optional): The number of validation samples for "random" split. (default: 500)
        num_test (int, optional): The number of test samples for "random" split. (default: 1000)
        transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
        force_reload (bool, optional): Whether to re-process the dataset. (default: False)
    """

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    geom_gcn_url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.name = name
        self.split = split.lower()
        assert self.split in ["public", "full", "geom-gcn", "random"]

        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

        if self.split == "full":
            data = self.get(0)
            val_m = ops.convert_to_numpy(data.val_mask)
            test_m = ops.convert_to_numpy(data.test_mask)
            train_mask = np.ones(data.num_nodes, dtype=bool)
            train_mask[val_m | test_m] = False
            data.train_mask = ops.convert_to_tensor(train_mask, dtype="bool")
            self.data, self.slices = self.collate([data])

        elif self.split == "random":
            data = self.get(0)
            num_nodes = data.num_nodes
            y_np = ops.convert_to_numpy(data.y)
            train_mask = np.zeros(num_nodes, dtype=bool)
            for c in range(self.num_classes):
                idx = np.where(y_np == c)[0]
                perm = np.random.permutation(len(idx))
                train_mask[idx[perm[:num_train_per_class]]] = True

            remaining = np.where(~train_mask)[0]
            remaining = remaining[np.random.permutation(len(remaining))]

            val_mask = np.zeros(num_nodes, dtype=bool)
            val_mask[remaining[:num_val]] = True

            test_mask = np.zeros(num_nodes, dtype=bool)
            test_mask[remaining[num_val : num_val + num_test]] = True

            data.train_mask = ops.convert_to_tensor(train_mask, dtype="bool")
            data.val_mask = ops.convert_to_tensor(val_mask, dtype="bool")
            data.test_mask = ops.convert_to_tensor(test_mask, dtype="bool")

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "raw")
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "processed")
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name.lower()}.{name}" for name in names]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for name in self.raw_file_names:
            fs.cp(f"{self.url}/{name}", self.raw_dir)
        if self.split == "geom-gcn":
            for i in range(10):
                url = f"{self.geom_gcn_url}/splits/{self.name.lower()}"
                fs.cp(f"{url}_split_0.6_0.2_{i}.npz", self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == "geom-gcn":
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f"{self.name.lower()}_split_0.6_0.2_{i}.npz"
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(splits["train_mask"])
                val_masks.append(splits["val_mask"])
                test_masks.append(splits["test_mask"])
            data.train_mask = ops.convert_to_tensor(np.stack(train_masks, axis=1), dtype="bool")
            data.val_mask = ops.convert_to_tensor(np.stack(val_masks, axis=1), dtype="bool")
            data.test_mask = ops.convert_to_tensor(np.stack(test_masks, axis=1), dtype="bool")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"

