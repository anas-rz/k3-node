import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
from keras import ops

from k3_node.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from k3_node.utils import coalesce


class Reddit(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`k3_node.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`k3_node.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`k3_node.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 232,965
          - 114,615,892
          - 602
          - 41
    """

    url = "https://data.dgl.ai/dataset/reddit.zip"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["reddit_data.npz", "reddit_graph.npz"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    @property
    def num_classes(self) -> int:
        return 41

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        if osp.exists(path):
            os.unlink(path)

    def process(self) -> None:
        import scipy.sparse as sp

        data = np.load(osp.join(self.raw_dir, "reddit_data.npz"))
        x = ops.convert_to_tensor(data["feature"], dtype="float32")
        y = ops.convert_to_tensor(data["label"], dtype="int64")
        split = data["node_types"]

        adj = sp.load_npz(osp.join(self.raw_dir, "reddit_graph.npz"))
        if not hasattr(adj, "row"):
            adj = adj.tocoo()
        row = adj.row.astype(np.int64)
        col = adj.col.astype(np.int64)
        edge_index = np.stack([row, col], axis=0)
        edge_index = ops.convert_to_tensor(edge_index, dtype="int64")
        edge_index, _ = coalesce(edge_index, num_nodes=int(data["feature"].shape[0]))

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = ops.convert_to_tensor(split == 1, dtype="bool")
        data.val_mask = ops.convert_to_tensor(split == 2, dtype="bool")
        data.test_mask = ops.convert_to_tensor(split == 3, dtype="bool")

        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

