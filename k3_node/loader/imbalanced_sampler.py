from typing import Any, List, Optional, Union

import numpy as np

try:
    import torch
    from torch import Tensor
    BaseWeightedRandomSampler = torch.utils.data.WeightedRandomSampler
except ImportError:
    torch = None
    Tensor = type(None)
    BaseWeightedRandomSampler = object

from k3_node.data import Data, Dataset, InMemoryDataset


class ImbalancedSampler(BaseWeightedRandomSampler):
    r"""A weighted random sampler that randomly samples elements according to class distribution.

    Args:
        dataset (Dataset or Data or Tensor): The dataset or class distribution from which to sample.
        input_nodes (Tensor, optional): The indices of nodes used by the corresponding loader. (default: :obj:`None`)
        num_samples (int, optional): The number of samples to draw for a single epoch. (default: :obj:`None`)
    """
    def __init__(
        self,
        dataset: Union[Dataset, Data, List[Data], Any],
        input_nodes: Optional[Any] = None,
        num_samples: Optional[int] = None,
    ):
        if isinstance(dataset, Data):
            y = dataset.y
            if hasattr(y, 'view'):
                y = y.view(-1)
            else:
                y = np.asarray(y).reshape(-1)
            if input_nodes is not None:
                y = y[input_nodes]

        elif torch is not None and isinstance(dataset, Tensor):
            y = dataset.view(-1)
            if input_nodes is not None:
                y = y[input_nodes]

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.y
            if hasattr(y, 'view'):
                y = y.view(-1)
            else:
                y = np.asarray(y).reshape(-1)

        elif isinstance(dataset, (list, tuple)):
            ys = [data.y for data in dataset]
            if torch is not None and isinstance(ys[0], Tensor):
                y = torch.cat(ys, dim=0).view(-1)
            else:
                y = np.concatenate([np.asarray(x).reshape(-1) for x in ys], axis=0)
        else:
            y = np.asarray(dataset).reshape(-1)

        if torch is not None and not isinstance(y, Tensor):
            y = torch.as_tensor(y, dtype=torch.long)

        num_samples = (y.numel() if hasattr(y, 'numel') else len(y)) if num_samples is None else num_samples

        if torch is not None and isinstance(y, Tensor):
            bincount = y.bincount().float()
            class_weight = 1.0 / bincount
            weight = class_weight[y]
            super().__init__(weight, num_samples, replacement=True)
        else:
            classes, counts = np.unique(y, return_counts=True)
            class_weight = {c: 1.0 / count for c, count in zip(classes, counts)}
            weight = np.array([class_weight[int(val)] for val in y], dtype=np.float64)
            weight = weight / weight.sum()
            self.weight = weight
            self.num_samples = num_samples
            self.replacement = True

    def __iter__(self):
        if torch is not None and hasattr(super(), '__iter__'):
            return super().__iter__()
        indices = np.random.choice(len(self.weight), size=self.num_samples, replace=True, p=self.weight)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
