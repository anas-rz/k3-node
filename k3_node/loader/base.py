from typing import Any, Callable

try:
    import torch
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    class BaseDataLoader:
        r"""Fallback BaseDataLoader when PyTorch is not installed."""
        def __init__(self, dataset=None, batch_size=1, shuffle=False, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            collate_fn = getattr(self, 'collate_fn', None) or (lambda x: x)
            dataset = getattr(self, 'dataset', [])
            batch_size = getattr(self, 'batch_size', 1)
            shuffle = getattr(self, 'shuffle', False)
            indices = list(range(len(dataset)))
            if shuffle:
                import random
                random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = [dataset[idx] for idx in batch_indices]
                yield collate_fn(batch)

        def __len__(self):
            dataset = getattr(self, 'dataset', [])
            batch_size = getattr(self, 'batch_size', 1)
            return (len(dataset) + batch_size - 1) // batch_size if len(dataset) > 0 else 0

try:
    from torch.utils.data.dataloader import (
        _BaseDataLoaderIter,
        _MultiProcessingDataLoaderIter,
    )
except ImportError:
    class _BaseDataLoaderIter:
        pass

    class _MultiProcessingDataLoaderIter:
        pass


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
