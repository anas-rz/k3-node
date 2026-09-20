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
        self.sizes: Dict[str, Any] = {}
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
        if path.endswith((".pt", ".pth")):
            try:
                import torch
                data_dict = data.to_dict() if hasattr(data, "to_dict") else dict(data)
                torch.save((data_dict, slices), path)
                return
            except Exception:
                pass
        with open(path, "wb") as f:
            pickle.dump((data, slices), f)

    def load(self, path: str):
        obj = None
        if path.endswith((".pt", ".pth")):
            try:
                import torch
                obj = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                pass

        if obj is None:
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
            except Exception:
                try:
                    import torch
                    obj = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:
                    pass

        if obj is None:
            if hasattr(self, "process") and callable(self.process):
                self.process()
                try:
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                except Exception:
                    try:
                        import torch
                        obj = torch.load(path, map_location="cpu", weights_only=False)
                    except Exception:
                        pass
            if obj is None:
                raise RuntimeError(f"Cannot load dataset from {path}")

        if isinstance(obj, tuple):
            if len(obj) == 2:
                data, self.slices = obj
            elif len(obj) == 3:
                data, self.slices, extra = obj
                if isinstance(extra, dict):
                    self.sizes = extra
            elif len(obj) >= 4:
                data, self.slices = obj[0], obj[1]
                if isinstance(obj[2], dict):
                    self.sizes = obj[2]
            else:
                data = obj[0]
            if isinstance(data, dict):
                data = Data(**data)
            self._data = data
        elif isinstance(obj, list):
            self._data_list = obj
        elif isinstance(obj, dict):
            self._data = Data(**obj)
        else:
            self._data = obj

        if self._data is not None and hasattr(self._data, "to_backend"):
            try:
                self._data.to_backend()
            except Exception:
                pass

        if isinstance(self.slices, dict):
            from k3_node.data.storage import is_tensor_like, to_numpy
            self.slices = {k: to_numpy(v) if is_tensor_like(v) else v for k, v in self.slices.items()}

