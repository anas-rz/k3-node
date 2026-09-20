import os
import os.path as osp
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs


class EmailEUCore(InMemoryDataset):
    r"""An e-mail communication network of a large European research institution."""

    urls = [
        "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
    ]

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
        return ["email-Eu-core.txt", "email-Eu-core-department-labels.txt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for url in self.urls:
            filename = osp.basename(url)
            gz_path = osp.join(self.raw_dir, filename)
            fs.cp(url, gz_path, extract=True)
            if osp.exists(gz_path):
                fs.rm(gz_path)

    def process(self):
        edge_index = np.loadtxt(self.raw_paths[0], dtype=np.int64).T
        labels = np.loadtxt(self.raw_paths[1], dtype=np.int64)
        y = labels[:, 1]

        data = Data(
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            y=ops.convert_to_tensor(y, dtype="int64"),
            num_nodes=int(y.shape[0]),
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

