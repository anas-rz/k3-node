from typing import Any, Iterator, List, Optional, Tuple, Union

try:
    import torch
    from torch import Tensor
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    Tensor = type(None)
    BaseDataLoader = object

from k3_node.data import Data, HeteroData
from k3_node.loader.base import DataLoaderIterator
from k3_node.loader.utils import infer_filter_per_worker


class ZipLoader(BaseDataLoader):
    r"""A loader that returns a tuple of data objects by sampling from multiple
    loader instances.

    Args:
        loaders (List[Any]): The loader instances.
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returned data in each worker's subprocess. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        loaders: List[Any],
        filter_per_worker: Optional[bool] = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            first_data = getattr(loaders[0], 'data', None)
            filter_per_worker = infer_filter_per_worker(first_data) if first_data is not None else True

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        for loader in loaders:
            if not callable(getattr(loader, 'collate_fn', None)):
                raise ValueError(f"'{loader.__class__.__name__}' does not have a 'collate_fn' method")
            if not callable(getattr(loader, 'filter_fn', None)):
                raise ValueError(f"'{loader.__class__.__name__}' does not have a 'filter_fn' method")
            loader.filter_per_worker = filter_per_worker

        lens = []
        for loader in loaders:
            if hasattr(loader, 'dataset'):
                lens.append(len(loader.dataset))
            elif hasattr(loader, '__len__'):
                lens.append(len(loader))
            else:
                lens.append(0)

        iterator = range(min(lens) if lens else 0)

        self.loaders = loaders
        self.filter_per_worker = filter_per_worker

        if torch is not None:
            super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)
        else:
            self.dataset = iterator
            self.collate_fn = self.collate_fn

    def __call__(self, index: Union[Tensor, List[int]]) -> Union[Tuple[Data, ...], Tuple[HeteroData, ...]]:
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: List[int]) -> Tuple[Any, ...]:
        if torch is not None and not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)
        return tuple(loader.collate_fn(index) for loader in self.loaders)

    def filter_fn(self, outs: Tuple[Any, ...]) -> Tuple[Union[Data, HeteroData], ...]:
        return tuple(loader.filter_fn(v) for loader, v in zip(self.loaders, outs))

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(loaders={self.loaders})'

