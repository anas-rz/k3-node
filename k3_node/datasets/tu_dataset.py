import os
import os.path as osp
import pickle
from typing import Callable, List, Optional
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs, read_tu_data


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, e.g., "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the TU Dortmund University.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset.
        transform (callable, optional): Transform function for Data objects.
        pre_transform (callable, optional): Pre-transform function.
        pre_filter (callable, optional): Pre-filter function.
        force_reload (bool, optional): Whether to re-process the dataset.
        use_node_attr (bool, optional): Whether to include continuous node attributes.
        use_edge_attr (bool, optional): Whether to include continuous edge attributes.
        cleaned (bool, optional): Whether to use cleaned dataset version.
    """

    url = "https://www.chrsmrrs.com/graphkerneldatasets"
    cleaned_url = "https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        self.load(self.processed_paths[0])

        if self._data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            if num_node_attributes > 0:
                self._data.x = self._data.x[:, num_node_attributes:]
        if self._data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            if num_edge_attrs > 0:
                self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        name = f"raw{'_cleaned' if self.cleaned else ''}"
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f"processed{'_cleaned' if self.cleaned else ''}"
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes.get("num_node_labels", 0)

    @property
    def num_node_attributes(self) -> int:
        return self.sizes.get("num_node_attributes", 0)

    @property
    def num_edge_labels(self) -> int:
        return self.sizes.get("num_edge_labels", 0)

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes.get("num_edge_attributes", 0)

    @property
    def raw_file_names(self) -> List[str]:
        names = ["A", "graph_indicator"]
        return [f"{self.name}_{name}.txt" for name in names]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        fs.cp(f"{url}/{self.name}.zip", self.raw_dir, extract=True)
        inner_dir = osp.join(self.raw_dir, self.name)
        if osp.isdir(inner_dir):
            for filename in os.listdir(inner_dir):
                src = osp.join(inner_dir, filename)
                dst = osp.join(self.raw_dir, filename)
                fs.cp(src, dst)
            fs.rm(inner_dir)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.data, self.slices = self.collate(data_list)
            self._data_list = None

        os.makedirs(osp.dirname(self.processed_paths[0]), exist_ok=True)
        saved = False
        if self.processed_paths[0].endswith((".pt", ".pth")):
            try:
                import torch
                data_dict = self._data.to_dict() if hasattr(self._data, "to_dict") else dict(self._data)
                torch.save((data_dict, self.slices, sizes), self.processed_paths[0])
                saved = True
            except Exception:
                pass
        if not saved:
            with open(self.processed_paths[0], "wb") as f:
                pickle.dump((self._data, self.slices, sizes), f)

    def __repr__(self) -> str:
        return f"{self.name}({len(self)})"

