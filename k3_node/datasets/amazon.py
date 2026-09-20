import os.path as osp
from typing import Callable, Optional

from k3_node.data import InMemoryDataset
from k3_node.io import fs, read_npz


class Amazon(InMemoryDataset):
    r"""The Amazon Computers and Amazon Photo networks from the
    "Pitfalls of Graph Neural Network Evaluation" paper.
    Nodes represent goods and edges represent that two goods are frequently
    bought together.
    Given product reviews as bag-of-words node features, the task is to
    map goods to their respective product category.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset ("Computers", "Photo").
        transform (callable, optional): Transform function.
        pre_transform (callable, optional): Pre-transform function.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.name = name.lower()
        assert self.name in ["computers", "photo"]
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), "processed")

    @property
    def raw_file_names(self) -> str:
        return f"amazon_electronics_{self.name.lower()}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        fs.cp(f"{self.url}{self.raw_file_names}", self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0], to_undirected=True)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.name.capitalize()}()"

