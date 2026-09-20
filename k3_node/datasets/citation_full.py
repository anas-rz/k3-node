import os.path as osp
from typing import Callable, Optional

from k3_node.data import InMemoryDataset
from k3_node.io import fs, read_npz


class CitationFull(InMemoryDataset):
    r"""The full citation network datasets from the
    "Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" paper.
    Nodes represent documents and edges represent citation links.
    Datasets include "Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed".

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset ("Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed").
        transform (callable, optional): Transform function.
        pre_transform (callable, optional): Pre-transform function.
        to_undirected (bool, optional): Whether the original graph is converted to undirected.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = "https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        to_undirected: bool = True,
        force_reload: bool = False,
    ):
        self.name = name.lower()
        self.to_undirected = to_undirected
        assert self.name in ["cora", "cora_ml", "citeseer", "dblp", "pubmed"]
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        suffix = "undirected" if self.to_undirected else "directed"
        return f"data_{suffix}.pt"

    def download(self):
        fs.cp(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0], to_undirected=self.to_undirected)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}Full()"


class CoraFull(CitationFull):
    r"""Alias for CitationFull with name="Cora"."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, "cora", transform, pre_transform)

