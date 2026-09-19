import math

import numpy as np
from keras import ops


class KGTripletLoader:
    r"""A minimal, framework-agnostic batching iterator over knowledge-graph
    triplets, mirroring :class:`torch.utils.data.DataLoader` usage in
    :meth:`k3_node.layers.kge.KGEModel.loader`.

    Args:
        head_index: The head indices.
        rel_type: The relation type.
        tail_index: The tail indices.
        batch_size (int, optional): The batch size. (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, shuffles the
            triplets at every epoch. (default: :obj:`False`)
        drop_last (bool, optional): If set to :obj:`True`, drops the last
            incomplete batch. (default: :obj:`False`)
    """
    def __init__(self, head_index, rel_type, tail_index, batch_size: int = 1,
                shuffle: bool = False, drop_last: bool = False, **kwargs):
        self.head_index = ops.convert_to_numpy(head_index)
        self.rel_type = ops.convert_to_numpy(rel_type)
        self.tail_index = ops.convert_to_numpy(tail_index)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_triplets = self.head_index.shape[0]

    def __len__(self):
        if self.drop_last:
            return self.num_triplets // self.batch_size
        return math.ceil(self.num_triplets / self.batch_size)

    def __iter__(self):
        indices = np.arange(self.num_triplets)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_triplets, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if self.drop_last and batch_idx.shape[0] < self.batch_size:
                continue
            yield (
                ops.convert_to_tensor(self.head_index[batch_idx]),
                ops.convert_to_tensor(self.rel_type[batch_idx]),
                ops.convert_to_tensor(self.tail_index[batch_idx]),
            )
