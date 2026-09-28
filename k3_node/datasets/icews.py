from typing import Callable, List, Optional

import numpy as np

from k3_node.data import Data, InMemoryDataset
from k3_node.data.download import download_url


class ICEWS18(InMemoryDataset):
    r"""The ICEWS18 temporal knowledge graph (Integrated Crisis Early Warning System, events from
    1/1/2018 to 10/31/2018 at a daily resolution), used by RE-Net: every graph is one event with
    subject ``sub``, relation ``rel``, object ``obj`` and day ``t``.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): ``"train"``, ``"val"`` or ``"test"``. (default: ``"train"``)
        transform (callable, optional): A function applied to each event when it is accessed.
        pre_transform (callable, optional): A function applied to the events in time order before
            saving, e.g. :meth:`~k3_node.models.RENet.pre_transform`.
        force_reload (bool, optional): Whether to re-process the dataset. (default: ``False``)
    """

    url = 'https://github.com/INK-USC/RE-Net/raw/master/data/ICEWS18'
    splits = [0, 373018, 419013, 468558]
    num_nodes, num_rels = 23033, 256

    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[['train', 'val', 'test'].index(split)])

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{name}.txt' for name in ['train', 'valid', 'test']]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self):
        events = np.concatenate([np.loadtxt(path, delimiter='\t', usecols=range(4), dtype=np.int64)
                                 for path in self.raw_paths])
        events[:, 3] //= 24  # hours -> days
        events -= events.min(axis=0, keepdims=True)
        data_list = []
        for sub, rel, obj, t in events.tolist():
            data = Data(sub=sub, rel=rel, obj=obj, t=t)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        s = self.splits
        for i in range(3):
            self.save(data_list[s[i]:s[i + 1]], self.processed_paths[i])
