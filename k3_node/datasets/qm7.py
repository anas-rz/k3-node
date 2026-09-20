from typing import Callable, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs


class QM7b(InMemoryDataset):
    r"""The QM7b dataset consisting of 7,211 molecules with 14 regression targets."""

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return "qm7b.mat"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        fs.cp(self.url, self.raw_dir)

    def process(self):
        from scipy.io import loadmat

        data = loadmat(self.raw_paths[0])
        coulomb_matrix = data["X"]
        target = data["T"].astype(np.float32)

        data_list = []
        for i in range(target.shape[0]):
            nz = np.nonzero(coulomb_matrix[i])
            edge_index = np.stack([nz[0], nz[1]], axis=0).astype(np.int64)
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]].astype(np.float32)
            y = target[i].reshape(1, -1)
            num_nodes = int(np.max(edge_index)) + 1 if edge_index.size > 0 else 0

            d = Data(
                edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
                edge_attr=ops.convert_to_tensor(edge_attr, dtype="float32"),
                y=ops.convert_to_tensor(y, dtype="float32"),
                num_nodes=num_nodes,
            )
            data_list.append(d)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

