from typing import List, Optional, Union

import keras
from keras import ops


class GroupAddRev(keras.layers.Layer):
    r"""The Grouped Reversible GNN module from the `"Graph Neural Networks with
    1000 Layers" <https://arxiv.org/abs/2106.07476>`_ paper.

    Args:
        conv (keras.layers.Layer or List[keras.layers.Layer]): A seed GNN layer
            or list of GNN layers.
        split_dim (int, optional): The dimension across which to split groups.
            (default: :obj:`-1`)
        num_groups (int, optional): The number of groups. (default: :obj:`None`)
    """
    def __init__(
        self,
        conv: Union[keras.layers.Layer, List[keras.layers.Layer]],
        split_dim: int = -1,
        num_groups: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_dim = split_dim

        if isinstance(conv, (list, tuple)):
            self.convs = list(conv)
        else:
            assert num_groups is not None, "Please specify 'num_groups'"
            self.convs = [conv]
            # Since Keras layers might not be easily deepcopied with unbuilt state,
            # we allow passing a list or cloning via layer config if possible
            for _ in range(num_groups - 1):
                try:
                    cloned = conv.__class__.from_config(conv.get_config())
                except Exception:
                    cloned = copy.deepcopy(conv)
                self.convs.append(cloned)

        if len(self.convs) < 2:
            raise ValueError(f"The number of groups should not be smaller than '2' (got '{self.num_groups}')")

    @property
    def num_groups(self) -> int:
        return len(self.convs)

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

    def call(self, x, edge_index, *args):
        xs = ops.split(x, self.num_groups, axis=self.split_dim)

        ys = []
        y_in = xs[1]
        for item in xs[2:]:
            y_in = y_in + item

        for i in range(self.num_groups):
            conv_out = self.convs[i](y_in, edge_index, *args)
            y_in = xs[i] + conv_out
            ys.append(y_in)

        return ops.concatenate(ys, axis=self.split_dim)

    def inverse(self, y, edge_index, *args):
        ys = ops.split(y, self.num_groups, axis=self.split_dim)

        xs = []
        for i in range(self.num_groups - 1, -1, -1):
            if i != 0:
                y_in = ys[i - 1]
            else:
                y_in = xs[0]
                for item in xs[1:]:
                    y_in = y_in + item
            conv_out = self.convs[i](y_in, edge_index, *args)
            x_i = ys[i] - conv_out
            xs.append(x_i)

        return ops.concatenate(xs[::-1], axis=self.split_dim)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.convs[0]}, '
                f'num_groups={self.num_groups})')

