import os.path as osp
from typing import Callable, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs


class Twitch(InMemoryDataset):
    r"""The Twitch Gamer networks."""

    url = "https://graphmining.ai/datasets/ptg/twitch"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.name = name.upper()
        assert self.name in ["DE", "EN", "ES", "FR", "PT", "RU"]
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
        return "data.pt"

    def download(self):
        fs.cp(f"{self.url}/{self.name}.npz", self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], allow_pickle=True)
        x = data["features"].astype(np.float32)
        y = data["target"].astype(np.int64)
        edge_index = data["edges"].astype(np.int64).T

        data_obj = Data(
            x=ops.convert_to_tensor(x, dtype="float32"),
            y=ops.convert_to_tensor(y, dtype="int64"),
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
        )

        if self.pre_transform is not None:
            data_obj = self.pre_transform(data_obj)

        self.save([data_obj], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.name}()"

