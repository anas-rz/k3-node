from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Union

import numpy as np

try:
    import torch
    import torch.utils.data
    from torch.utils.data.dataloader import default_collate
except ImportError:
    torch = None
    default_collate = None

from k3_node.data import Batch
from k3_node.data.data import BaseData
from k3_node.data.dataset import Dataset


class Collater:
    r"""Collates a list of graph data objects or primitives into a mini-batch."""
    def __init__(
        self,
        dataset: Optional[Union[Dataset, Sequence[BaseData]]] = None,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif torch is not None and isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, np.ndarray):
            return np.stack(batch, axis=0)
        elif hasattr(elem, '__array__') and not isinstance(elem, (str, bytes)):
            import keras
            np_batch = np.stack([np.asarray(x) for x in batch], axis=0)
            return keras.ops.convert_to_tensor(np_batch)
        elif isinstance(elem, float):
            if torch is not None:
                return torch.tensor(batch, dtype=torch.float)
            return np.array(batch, dtype=np.float32)
        elif isinstance(elem, int):
            if torch is not None:
                return torch.tensor(batch, dtype=torch.long)
            return np.array(batch, dtype=np.int64)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


BaseDataLoader = torch.utils.data.DataLoader if torch is not None else object


class DataLoader(BaseDataLoader):
    r"""A data loader which merges data objects from a
    :class:`k3_node.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~k3_node.data.Data` or
    :class:`~k3_node.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs.pop('collate_fn', None)

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        if torch is not None:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=Collater(dataset, follow_batch, exclude_keys),
                **kwargs,
            )
        else:
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = Collater(dataset, follow_batch, exclude_keys)

