from typing import Iterator, List, Optional

import numpy as np

try:
    import torch
    import torch.utils.data.sampler
    BaseSampler = torch.utils.data.sampler.Sampler
except ImportError:
    torch = None
    BaseSampler = object

from k3_node.data import Dataset


class DynamicBatchSampler(BaseSampler):
    r"""Dynamically adds samples to a mini-batch up to a maximum size (either
    based on number of nodes or number of edges).

    Args:
        dataset (Dataset): Dataset to sample from.
        max_num (int): Size of mini-batch to aim for in number of nodes or edges.
        mode (str, optional): :obj:`"node"` or :obj:`"edge"` to measure batch size. (default: :obj:`"node"`)
        shuffle (bool, optional): If set to :obj:`True`, will have the data reshuffled at every epoch. (default: :obj:`False`)
        skip_too_big (bool, optional): If set to :obj:`True`, skip samples which cannot fit in a batch by itself. (default: :obj:`False`)
        num_steps (int, optional): The number of mini-batches to draw for a single epoch. (default: :obj:`None`)
    """
    def __init__(
        self,
        dataset: Dataset,
        max_num: int,
        mode: str = 'node',
        shuffle: bool = False,
        skip_too_big: bool = False,
        num_steps: Optional[int] = None,
    ):
        if max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num})")
        if mode not in ['node', 'edge']:
            raise ValueError(f"`mode` choice should be either 'node' or 'edge' (got '{mode}')")

        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        self.max_steps = num_steps or len(dataset)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            if torch is not None:
                indices = torch.randperm(len(self.dataset)).tolist()
            else:
                indices = np.random.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        samples: List[int] = []
        current_num: int = 0
        num_steps: int = 0
        num_processed: int = 0

        while num_processed < len(self.dataset) and num_steps < self.max_steps:
            for i in indices[num_processed:]:
                data = self.dataset[i]
                num = data.num_nodes if self.mode == 'node' else data.num_edges

                if current_num + num > self.max_num:
                    if current_num == 0:
                        if self.skip_too_big:
                            num_processed += 1
                            continue
                    else:
                        break

                samples.append(i)
                num_processed += 1
                current_num += num

            yield samples
            samples = []
            current_num = 0
            num_steps += 1

    def __len__(self) -> int:
        if self.num_steps is None:
            raise ValueError(
                f"The length of '{self.__class__.__name__}' is undefined since the number of steps per epoch "
                f"is ambiguous. Either specify `num_steps` or use a static batch sampler."
            )
        return self.num_steps

