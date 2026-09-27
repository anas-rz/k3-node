import datetime
import os
from typing import Callable, Optional

import numpy as np

from k3_node.data import Data, InMemoryDataset
from k3_node.data.download import download_url
from k3_node.data.extract import extract_gz


class BitcoinOTC(InMemoryDataset):
    r"""The Bitcoin-OTC dataset from the `"EvolveGCN: Evolving Graph Convolutional Networks for
    Dynamic Graphs" <https://arxiv.org/abs/1902.10191>`_ paper: 138 who-trusts-whom networks of
    consecutive time steps among 6,005 users. ``edge_attr`` holds the trust rating of every edge,
    from -10 (total distrust) to +10 (total trust).

    Args:
        root (str): Root directory where the dataset should be saved.
        edge_window_size (int, optional): The number of time steps an edge stays in the graph
            sequence after it was created. (default: ``10``)
        transform (callable, optional): A function applied to each graph when it is accessed.
        pre_transform (callable, optional): A function applied to each graph before saving.
        force_reload (bool, optional): Whether to re-process the dataset. (default: ``False``)
    """

    url = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"

    def __init__(self, root: str, edge_window_size: int = 10, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return "soc-sign-bitcoinotc.csv"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    @property
    def num_nodes(self) -> int:
        return int(np.max(np.asarray(self._data.edge_index))) + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0]) as f:
            lines = [line.split(",") for line in f.read().split("\n")[:-1]]
        edge_index = np.array([[int(line[0]), int(line[1])] for line in lines], dtype=np.int64)
        edge_index = (edge_index - edge_index.min()).T
        num_nodes = int(edge_index.max()) + 1
        edge_attr = np.array([int(line[2]) for line in lines], dtype=np.int64)
        stamps = [datetime.datetime.fromtimestamp(int(float(line[3]))) for line in lines]

        offset = datetime.timedelta(days=13.8)  # results in 138 time steps
        graph_indices, factor = [], 1
        for t in stamps:
            factor = factor if t < stamps[0] + factor * offset else factor + 1
            graph_indices.append(factor - 1)
        graph_idx = np.array(graph_indices, dtype=np.int64)

        data_list = []
        for i in range(int(graph_idx.max()) + 1):
            mask = (graph_idx > (i - self.edge_window_size)) & (graph_idx <= i)
            data_list.append(Data(edge_index=edge_index[:, mask], edge_attr=edge_attr[mask], num_nodes=num_nodes))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])
