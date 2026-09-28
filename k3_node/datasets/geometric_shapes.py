import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np

from k3_node.data import InMemoryDataset
from k3_node.data.download import download_url
from k3_node.data.extract import extract_zip
from k3_node.io.off import read_off


class GeometricShapes(InMemoryDataset):
    r"""Synthetic meshes of 40 geometric shapes such as cubes, spheres or pyramids (one training and
    one test mesh per shape), as in PyG.

    The graphs hold mesh faces (``face``) and vertex positions (``pos``) but no edges. Use
    :class:`~k3_node.transforms.FaceToEdge` to turn a mesh into a graph, or
    :class:`~k3_node.transforms.SamplePoints` to sample a point cloud from its surface.

    Args:
        root (str): Root directory where the dataset should be saved.
        train (bool, optional): Loads the training meshes if ``True``, else the test meshes.
        transform (callable, optional): A function applied to each graph when it is accessed.
        pre_transform (callable, optional): A function applied to each graph before saving.
        pre_filter (callable, optional): A function deciding which graphs to keep.
        force_reload (bool, optional): Whether to re-process the dataset. (default: ``False``)
    """

    url = 'https://github.com/Yannick-S/geometric_shapes/raw/master/raw.zip'

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0] if train else self.processed_paths[1])

    @property
    def raw_file_names(self) -> str:
        return '2d_circle'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)

    def process(self):
        self.save(self._process_set('train'), self.processed_paths[0])
        self.save(self._process_set('test'), self.processed_paths[1])

    def _process_set(self, split: str):
        categories = sorted(x.split(os.sep)[-2] for x in glob.glob(osp.join(self.raw_dir, '*', '')))
        data_list = []
        for target, category in enumerate(categories):
            for path in sorted(glob.glob(osp.join(self.raw_dir, category, split, '*.off'))):
                data = read_off(path)
                data.pos = data.pos - data.pos.mean(axis=0, keepdims=True)
                data.y = np.array([target], dtype=np.int64)
                data_list.append(data)
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        return data_list
