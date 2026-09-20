import pickle
from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs


class BA2MotifDataset(InMemoryDataset):
    r"""The synthetic BA-2motifs graph classification dataset for evaluating
    explainability algorithms, containing 1000 random Barabasi-Albert graphs.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): Transform function.
        pre_transform (callable, optional): Pre-transform function.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = "https://github.com/flyingdoog/PGExplainer/raw/master/dataset"
    filename = "BA-2motif.pkl"

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
    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        fs.cp(f"{self.url}/{self.filename}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], "rb") as f:
            adj, x, y = pickle.load(f)

        adj_np = np.array(adj)
        x_np = np.array(x, dtype=np.float32)
        y_np = np.array(y)

        data_list: List[Data] = []
        for i in range(x_np.shape[0]):
            nz = np.nonzero(adj_np[i])
            edge_index = np.stack([nz[0], nz[1]], axis=0).astype(np.int64)
            xi = x_np[i]
            y_nz = np.nonzero(y_np[i])[0]
            yi = int(y_nz[0]) if len(y_nz) > 0 else 0

            data = Data(
                x=ops.convert_to_tensor(xi, dtype="float32"),
                edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
                y=ops.convert_to_tensor(np.array([yi], dtype=np.int64), dtype="int64"),
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

