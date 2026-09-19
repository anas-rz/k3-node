from typing import List, Union

import numpy as np

try:
    import torch
    import torch.utils.data
    from torch.utils.data.dataloader import default_collate
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    default_collate = None
    BaseDataLoader = object

from k3_node.data import Batch, Data, Dataset


def collate_fn(data_list: List[Data]) -> Batch:
    batch = Batch()
    for key in data_list[0].keys():
        vals = [data[key] for data in data_list]
        if default_collate is not None and isinstance(vals[0], torch.Tensor):
            batch[key] = default_collate(vals)
        elif isinstance(vals[0], np.ndarray):
            batch[key] = np.stack(vals, axis=0)
        elif hasattr(vals[0], '__array__'):
            import keras
            batch[key] = keras.ops.stack(vals, axis=0)
        else:
            batch[key] = vals
    return batch


class DenseDataLoader(BaseDataLoader):
    r"""A data loader which batches data objects from a
    :class:`k3_node.data.Dataset` to a :class:`k3_node.data.Batch`
    object by stacking all attributes in a new dimension.
    """
    def __init__(
        self,
        dataset: Union[Dataset, List[Data]],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        kwargs.pop('collate_fn', None)

        if torch is not None:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                **kwargs,
            )
        else:
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn

