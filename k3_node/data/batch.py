from typing import Any, List, Optional, Union

from keras import ops

from k3_node.data.collate import collate
from k3_node.data.data import BaseData, Data
from k3_node.data.hetero_data import HeteroData
from k3_node.data.separate import separate


class Batch(Data):
    """A data object describing a batch of graphs as one big (disconnected) graph."""

    def __init__(self, _base_cls=Data, **kwargs):
        self._base_cls = _base_cls
        if issubclass(_base_cls, HeteroData):
            self.__class__ = HeteroBatch
            HeteroBatch.__init__(self, **kwargs)
        else:
            super().__init__(**kwargs)

    @classmethod
    def from_data_list(
        cls,
        data_list: List[BaseData],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> "Batch":
        if len(data_list) == 0:
            raise ValueError("Cannot batch empty data_list")

        base_cls = data_list[0].__class__
        batch_cls = HeteroBatch if issubclass(base_cls, HeteroData) else Batch

        batch, slice_dict, inc_dict = collate(
            batch_cls,
            data_list=data_list,
            increment=True,
            add_batch=True,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict
        batch._base_cls = base_cls
        return batch

    @property
    def num_graphs(self) -> int:
        if hasattr(self, "_num_graphs"):
            return self._num_graphs
        if "batch" in self and self.batch is not None:
            return int(ops.convert_to_numpy(self.batch).max()) + 1 if self.batch.shape[0] > 0 else 0
        if "ptr" in self and self.ptr is not None:
            return len(self.ptr) - 1
        return 1

    def __getitem__(self, idx: Union[int, slice, str]):
        if isinstance(idx, str):
            return super().__getitem__(idx)
        if isinstance(idx, int):
            if idx < 0:
                idx = self.num_graphs + idx
            if idx < 0 or idx >= self.num_graphs:
                raise IndexError(f"Index {idx} out of bounds for batch of {self.num_graphs} graphs")
            return separate(self._base_cls, self, idx, self._slice_dict, self._inc_dict)
        raise NotImplementedError("Batch slicing with slice is not supported yet")

    def to_data_list(self) -> List[BaseData]:
        return [self[i] for i in range(self.num_graphs)]


class HeteroBatch(HeteroData):
    """A batch object for heterogeneous graphs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_cls = HeteroData

    @property
    def num_graphs(self) -> int:
        if hasattr(self, "_num_graphs"):
            return self._num_graphs
        for store in self.node_stores:
            if "batch" in store and store.batch is not None:
                return int(ops.convert_to_numpy(store.batch).max()) + 1 if store.batch.shape[0] > 0 else 0
        return 1

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, int):
            if idx < 0:
                idx = self.num_graphs + idx
            if idx < 0 or idx >= self.num_graphs:
                raise IndexError(f"Index {idx} out of bounds for batch of {self.num_graphs} graphs")
            return separate(self._base_cls, self, idx, self._slice_dict, self._inc_dict)
        return super().__getitem__(idx)

    def to_data_list(self) -> List[BaseData]:
        return [self[i] for i in range(self.num_graphs)]

