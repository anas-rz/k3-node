import copy
import os
import os.path as osp
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from k3_node.data.collate import collate
from k3_node.data.data import BaseData, Data
from k3_node.data.dataset import Dataset
from k3_node.data.separate import separate


class InMemoryDataset(Dataset):
    """Dataset base class for in-memory graph collections."""

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self._data: Optional[BaseData] = None
        self.slices: Optional[Dict[str, Any]] = None
        self._data_list: Optional[List[BaseData]] = None

    @property
    def data(self) -> Optional[BaseData]:
        return self._data

    @data.setter
    def data(self, value: Optional[BaseData]):
        self._data = value

    def len(self) -> int:
        if self._data_list is not None:
            return len(self._data_list)
        if self.slices is None:
            return 1 if self._data is not None else 0
        for key, value in self.slices.items():
            if isinstance(value, dict):
                for _, sub_val in value.items():
                    return len(sub_val) - 1
            return len(value) - 1
        return 0

    def get(self, idx: int) -> BaseData:
        if self._data_list is not None:
            return self._data_list[idx]
        if self._data is None:
            raise RuntimeError("Dataset does not contain data. Call 'load' first.")
        if self.slices is None:
            if idx == 0:
                return copy.copy(self._data)
            raise IndexError(f"Index {idx} out of bounds for single graph dataset.")
        return separate(self._data.__class__, self._data, idx, self.slices)

    @classmethod
    def collate(cls, data_list: List[BaseData]) -> Tuple[BaseData, Optional[Dict[str, Any]]]:
        if len(data_list) == 1:
            return data_list[0], None
        base_cls = data_list[0].__class__
        data, slices, _ = collate(
            base_cls,
            data_list,
            increment=False,
            add_batch=False,
        )
        return data, slices

    def save(self, data_list: List[BaseData], path: str):
        os.makedirs(osp.dirname(path), exist_ok=True)
        data, slices = self.collate(data_list)
        with open(path, "wb") as f:
            pickle.dump((data, slices), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, tuple) and len(obj) == 2:
            self._data, self.slices = obj
        elif isinstance(obj, list):
            self._data_list = obj
        else:
            self._data = obj

