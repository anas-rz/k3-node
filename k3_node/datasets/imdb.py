import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import HeteroData, InMemoryDataset
from k3_node.io import fs


class IMDB(InMemoryDataset):
    r"""A subset of the Internet Movie Database (IMDB) containing three types of entities:
    movies, actors, and directors.
    """

    url = "https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=1"

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
            "adjM.npz",
            "features_0.npz",
            "features_1.npz",
            "features_2.npz",
            "labels.npy",
            "train_val_test_idx.npz",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        zip_path = osp.join(self.raw_dir, "IMDB_processed.zip")
        fs.cp(self.url, zip_path, extract=True)
        if osp.exists(zip_path):
            fs.rm(zip_path)

    def process(self):
        import scipy.sparse as sp

        data = HeteroData()
        node_types = ["movie", "director", "actor"]

        for i, node_type in enumerate(node_types):
            x = sp.load_npz(osp.join(self.raw_dir, f"features_{i}.npz"))
            data[node_type].x = ops.convert_to_tensor(
                np.array(x.todense(), dtype=np.float32), dtype="float32"
            )

        y = np.load(osp.join(self.raw_dir, "labels.npy"))
        data["movie"].y = ops.convert_to_tensor(y.astype(np.int64), dtype="int64")

        split = np.load(osp.join(self.raw_dir, "train_val_test_idx.npz"))
        for name in ["train", "val", "test"]:
            idx = split[f"{name}_idx"]
            mask = np.zeros(data["movie"].num_nodes, dtype=bool)
            mask[idx] = True
            data["movie"][f"{name}_mask"] = ops.convert_to_tensor(mask, dtype="bool")

        s = {}
        N_m = data["movie"].num_nodes
        N_d = data["director"].num_nodes
        N_a = data["actor"].num_nodes
        s["movie"] = (0, N_m)
        s["director"] = (N_m, N_m + N_d)
        s["actor"] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_dir, "adjM.npz"))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0] : s[src][1], s[dst][0] : s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = np.array(A_sub.row, dtype=np.int64)
                col = np.array(A_sub.col, dtype=np.int64)
                edge_index = np.stack([row, col], axis=0)
                data[src, dst].edge_index = ops.convert_to_tensor(edge_index, dtype="int64")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

