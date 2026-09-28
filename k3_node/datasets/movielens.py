import os
import os.path as osp
from typing import Callable, Optional

import numpy as np

from k3_node.data import Data, InMemoryDataset
from k3_node.data.download import download_url
from k3_node.data.extract import extract_zip


class MovieLens100K(InMemoryDataset):
    r"""The MovieLens 100K ratings (943 users, 1,682 movies, 100,000 ratings) as a user-movie graph
    for recommendation, a small stand-in for PyG's ``AmazonBook``.

    Every rating counts as an interaction. The graph is homogeneous: users are nodes
    ``0 .. num_users - 1`` and movies ``num_users .. num_nodes - 1``. ``edge_index`` holds the
    training interactions in both directions, ``edge_label_index`` the test interactions
    (user, movie), from the official 80/20 split ``u1.base`` / ``u1.test``.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function applied to the graph when it is accessed.
        force_reload (bool, optional): Whether to re-process the dataset. (default: ``False``)
    """

    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    num_users, num_items = 943, 1682

    def __init__(self, root: str, transform: Optional[Callable] = None, force_reload: bool = False):
        super().__init__(root, transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [osp.join("ml-100k", "u1.base"), osp.join("ml-100k", "u1.test")]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def _read(self, path):
        ratings = np.loadtxt(path, dtype=np.int64)[:, :2] - 1  # user id, movie id (1-based)
        return np.stack([ratings[:, 0], ratings[:, 1] + self.num_users])

    def process(self):
        train, test = self._read(self.raw_paths[0]), self._read(self.raw_paths[1])
        data = Data(edge_index=np.concatenate([train, train[::-1]], axis=1), edge_label_index=test,
                    num_nodes=self.num_users + self.num_items)
        self.save([data], self.processed_paths[0])
