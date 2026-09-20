import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import HeteroData, InMemoryDataset
from k3_node.io import fs


class DBLP(InMemoryDataset):
    r"""A subset of the DBLP computer science bibliography website containing
    four types of entities: authors, papers, terms, and conferences.
    """

    url = "https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1"

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
            "features_2.npy",
            "labels.npy",
            "node_types.npy",
            "train_val_test_idx.npz",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        zip_path = osp.join(self.raw_dir, "DBLP_processed.zip")
        fs.cp(self.url, zip_path, extract=True)
        if osp.exists(zip_path):
            fs.rm(zip_path)

    def process(self):
        import scipy.sparse as sp

        data = HeteroData()
        node_types = ["author", "paper", "term", "conference"]

        for i, node_type in enumerate(node_types[:2]):
            feat_path = osp.join(self.raw_dir, f"features_{i}.npz")
            x = sp.load_npz(feat_path)
            data[node_type].x = ops.convert_to_tensor(
                np.array(x.todense(), dtype=np.float32), dtype="float32"
            )

        x2 = np.load(osp.join(self.raw_dir, "features_2.npy"))
        data["term"].x = ops.convert_to_tensor(x2.astype(np.float32), dtype="float32")

        node_type_idx = np.load(osp.join(self.raw_dir, "node_types.npy"))
        data["conference"].num_nodes = int((node_type_idx == 3).sum())

        y = np.load(osp.join(self.raw_dir, "labels.npy"))
        data["author"].y = ops.convert_to_tensor(y.astype(np.int64), dtype="int64")

        split = np.load(osp.join(self.raw_dir, "train_val_test_idx.npz"))
        for name in ["train", "val", "test"]:
            idx = split[f"{name}_idx"]
            mask = np.zeros(data["author"].num_nodes, dtype=bool)
            mask[idx] = True
            data["author"][f"{name}_mask"] = ops.convert_to_tensor(mask, dtype="bool")

        s = {}
        N_a = data["author"].num_nodes
        N_p = data["paper"].num_nodes
        N_t = data["term"].num_nodes
        N_c = data["conference"].num_nodes
        s["author"] = (0, N_a)
        s["paper"] = (N_a, N_a + N_p)
        s["term"] = (N_a + N_p, N_a + N_p + N_t)
        s["conference"] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

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

