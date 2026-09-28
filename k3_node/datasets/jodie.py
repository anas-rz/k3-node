import os
import os.path as osp

import numpy as np


class JODIEDataset:
    r"""The temporal interaction datasets of `"JODIE: Predicting Dynamic Embedding Trajectory in
    Temporal Interaction Networks" <https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf>`_:
    ``"wikipedia"``, ``"reddit"``, ``"mooc"`` and ``"lastfm"``.

    ``dataset[0]`` is a :class:`~k3_node.data.TemporalData` event stream: user ``src`` interacts
    with item ``dst`` (numbered after the users) at time ``t``, with features ``msg`` and label
    ``y``. MOOC (7,144 nodes, 411,749 events, 4 features) is the smallest download (40 MB).

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset.
    """

    url = "https://snap.stanford.edu/jodie/{}.csv"
    names = ["wikipedia", "reddit", "mooc", "lastfm"]

    def __init__(self, root: str, name: str):
        self.name = name.lower()
        assert self.name in self.names
        folder = osp.join(root, self.name)
        cache = osp.join(folder, "processed.npz")
        if not osp.exists(cache):
            os.makedirs(folder, exist_ok=True)
            csv = osp.join(folder, f"{self.name}.csv")
            if not osp.exists(csv):
                from k3_node.data.download import download_url

                download_url(self.url.format(self.name), folder)
            import pandas as pd

            df = pd.read_csv(csv, skiprows=1, header=None)
            src = df.iloc[:, 0].values.astype(np.int64)
            dst = df.iloc[:, 1].values.astype(np.int64) + int(src.max()) + 1
            np.savez(cache, src=src, dst=dst, t=df.iloc[:, 2].values.astype(np.int64),
                     y=df.iloc[:, 3].values.astype(np.int64), msg=df.iloc[:, 4:].values.astype(np.float32))
        self._arrays = dict(np.load(cache))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        from k3_node.data import TemporalData

        if idx != 0:
            raise IndexError(idx)
        return TemporalData(**self._arrays)

    def __repr__(self):
        return f"JODIEDataset({self.name})"
