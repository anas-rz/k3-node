import copy
import os
import os.path as osp
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from keras import ops

from k3_node.data.data import BaseData
from k3_node.data.storage import is_tensor_like


class Dataset:
    """Dataset base class for creating graph datasets."""

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self.force_reload = force_reload
        self._indices: Optional[Sequence] = None

        if self.has_download:
            self._download()

        if self.has_process:
            self._process()

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return []

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root or "", "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root or "", "processed")

    @property
    def raw_paths(self) -> List[str]:
        files = self.raw_file_names
        if isinstance(files, str):
            files = [files]
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        files = self.processed_file_names
        if isinstance(files, str):
            files = [files]
        return [osp.join(self.processed_dir, f) for f in files]

    @property
    def has_download(self) -> bool:
        return "download" in self.__class__.__dict__

    @property
    def has_process(self) -> bool:
        return "process" in self.__class__.__dict__

    def download(self):
        pass

    def process(self):
        pass

    def _download(self):
        if all(osp.exists(p) for p in self.raw_paths if p):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def _process(self):
        if not self.force_reload and len(self.processed_paths) > 0 and all(osp.exists(p) for p in self.processed_paths):
            return
        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

    def len(self) -> int:
        raise NotImplementedError

    def get(self, idx: int) -> BaseData:
        raise NotImplementedError

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    def __len__(self) -> int:
        return len(self.indices())

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, (int, np.integer)):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)

    def index_select(self, idx: Any) -> "Dataset":
        indices = list(self.indices())
        if isinstance(idx, slice):
            indices = indices[idx]
        elif is_tensor_like(idx):
            idx_np = ops.convert_to_numpy(idx)
            if idx_np.dtype == bool:
                indices = [indices[i] for i in np.where(idx_np)[0]]
            else:
                indices = [indices[int(i)] for i in idx_np]
        elif isinstance(idx, Sequence):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(f"Invalid index type {type(idx)}")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def __iter__(self) -> Iterator[BaseData]:
        for i in range(len(self)):
            yield self[i]

    @property
    def num_node_features(self) -> int:
        data = self[0]
        return getattr(data, "num_node_features", 0)

    @property
    def num_features(self) -> int:
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        data = self[0]
        return getattr(data, "num_edge_features", 0)

    @property
    def num_classes(self) -> int:
        y_list = [d.y for d in self if hasattr(d, "y") and d.y is not None]
        if len(y_list) == 0:
            return 0
        y = ops.convert_to_numpy(ops.concatenate(y_list, axis=0)) if len(y_list) > 1 else ops.convert_to_numpy(y_list[0])
        if np.issubdtype(y.dtype, np.integer):
            return int(np.max(y)) + 1
        return len(np.unique(y))

