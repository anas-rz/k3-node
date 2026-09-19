from typing import Any, Callable

try:
    from torch.utils.data.dataloader import (
        _BaseDataLoaderIter,
        _MultiProcessingDataLoaderIter,
    )
except ImportError:
    _BaseDataLoaderIter = object
    _MultiProcessingDataLoaderIter = object


class DataLoaderIterator:
    r"""A data loader iterator extended by a post transformation function
    :meth:`transform_fn`.
    """
    def __init__(self, iterator: Any, transform_fn: Callable):
        self.iterator = iterator
        self.transform_fn = transform_fn

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        if hasattr(self.iterator, '_reset'):
            self.iterator._reset(loader, first_iter)

    def __len__(self) -> int:
        return len(self.iterator)

    def __next__(self) -> Any:
        return self.transform_fn(next(self.iterator))

    def __del__(self) -> Any:
        if isinstance(self.iterator, _MultiProcessingDataLoaderIter):
            self.iterator.__del__()

