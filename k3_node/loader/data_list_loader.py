from typing import List, Union

try:
    import torch
    import torch.utils.data
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    BaseDataLoader = object

from k3_node.data import Dataset
from k3_node.data.data import BaseData


def collate_fn(data_list):
    return data_list


class DataListLoader(BaseDataLoader):
    r"""A data loader which batches data objects from a
    :class:`k3_node.data.Dataset` to a Python list without batch collation.
    """
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
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

