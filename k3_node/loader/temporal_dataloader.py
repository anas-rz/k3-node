from typing import List

import numpy as np

try:
    import torch
    import torch.utils.data
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    BaseDataLoader = object

from k3_node.data import TemporalData


class TemporalDataLoader(BaseDataLoader):
    r"""A data loader which merges successive events of a
    :class:`k3_node.data.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The :obj:`~k3_node.data.TemporalData` from which to load.
        batch_size (int, optional): How many samples per batch to load. (default: :obj:`1`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            destination nodes to the number of positive destination nodes. (default: :obj:`0.0`)
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        data: TemporalData,
        batch_size: int = 1,
        neg_sampling_ratio: float = 0.0,
        **kwargs,
    ):
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        kwargs.pop('shuffle', None)

        self.data = data
        self.events_per_batch = batch_size
        self.neg_sampling_ratio = neg_sampling_ratio

        if neg_sampling_ratio > 0:
            dst = data.dst
            if torch is not None and isinstance(dst, torch.Tensor):
                self.min_dst = int(dst.min())
                self.max_dst = int(dst.max())
            else:
                self.min_dst = int(np.min(np.asarray(dst)))
                self.max_dst = int(np.max(np.asarray(dst)))

        if kwargs.get('drop_last', False) and len(data) % batch_size != 0:
            arange = list(range(0, len(data) - batch_size, batch_size))
        else:
            arange = list(range(0, len(data), batch_size))

        if torch is not None:
            super().__init__(arange, 1, shuffle=False, collate_fn=self, **kwargs)
        else:
            self.dataset = arange
            self.batch_size = 1
            self.shuffle = False
            self.collate_fn = self

    def __call__(self, arange: List[int]) -> TemporalData:
        start = arange[0]
        end = start + self.events_per_batch
        batch = self.data[start:end]

        is_torch = torch is not None and isinstance(batch.dst, torch.Tensor)

        n_ids = [batch.src, batch.dst]

        if self.neg_sampling_ratio > 0:
            num_neg = round(self.neg_sampling_ratio * (batch.dst.size(0) if is_torch else len(batch.dst)))
            if is_torch:
                batch.neg_dst = torch.randint(
                    low=self.min_dst,
                    high=self.max_dst + 1,
                    size=(num_neg,),
                    dtype=batch.dst.dtype,
                    device=batch.dst.device,
                )
            else:
                batch.neg_dst = np.random.randint(
                    low=self.min_dst,
                    high=self.max_dst + 1,
                    size=(num_neg,),
                    dtype=np.int64,
                )
            n_ids.append(batch.neg_dst)

        if is_torch:
            batch.n_id = torch.cat(n_ids, dim=0).unique()
        else:
            batch.n_id = np.unique(np.concatenate([np.asarray(x) for x in n_ids], axis=0))

        return batch

