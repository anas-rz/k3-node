from typing import Callable, List, Optional, Union

import keras
from keras import ops
import numpy as np
import scipy.sparse as sp

from k3_node.layers.conv import GCNConv
from k3_node.layers.pool import TopKPooling


class GraphUNet(keras.layers.Layer):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (str or Callable, optional): The nonlinearity to use.
            (default: :obj:`"relu"`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        if isinstance(pool_ratios, (int, float)):
            self.pool_ratios = [float(pool_ratios)] * depth
        else:
            self.pool_ratios = list(pool_ratios)

        if isinstance(act, str):
            self.act = keras.activations.get(act)
        else:
            self.act = act
        self.sum_res = sum_res

        channels = hidden_channels
        self.down_convs = []
        self.pools = []
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, ratio=self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels_up = channels if sum_res else 2 * channels
        self.up_convs = []
        for _ in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels_up, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels_up, out_channels, improved=True))

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def augment_adj(self, edge_index, edge_weight, num_nodes: int):
        row = ops.convert_to_numpy(edge_index[0])
        col = ops.convert_to_numpy(edge_index[1])
        if edge_weight is not None:
            val = ops.convert_to_numpy(edge_weight)
        else:
            val = np.ones(row.shape[0], dtype=np.float32)

        # Remove self-loops
        mask = row != col
        row, col, val = row[mask], col[mask], val[mask]

        # Add self-loops
        diag = np.arange(num_nodes)
        row = np.concatenate([row, diag])
        col = np.concatenate([col, diag])
        val = np.concatenate([val, np.ones(num_nodes, dtype=np.float32)])

        adj = sp.csr_matrix((val, (row, col)), shape=(num_nodes, num_nodes))
        adj2 = (adj @ adj).tocoo()

        r2, c2, v2 = adj2.row, adj2.col, adj2.data
        mask2 = r2 != c2
        r2, c2, v2 = r2[mask2], c2[mask2], v2[mask2]

        new_edge_index = ops.convert_to_tensor(np.stack([r2, c2], axis=0), dtype=edge_index.dtype)
        new_edge_weight = ops.convert_to_tensor(v2, dtype="float32")
        return new_edge_index, new_edge_weight

    def call(self, x, edge_index, batch=None, edge_weight=None):
        num_nodes = ops.shape(x)[0]
        if batch is None:
            batch = ops.zeros((num_nodes,), dtype="int32")

        if edge_weight is None:
            edge_weight = ops.ones((ops.shape(edge_index)[1],), dtype=x.dtype)

        x = self.down_convs[0](x, edge_index, edge_weight=edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            curr_nodes = ops.shape(x)[0]
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, curr_nodes)
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_attr=edge_weight, batch=batch
            )

            x = self.down_convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs.append(x)
                edge_indices.append(edge_index)
                edge_weights.append(edge_weight)
            perms.append(perm)

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = ops.scatter_update(ops.zeros_like(res), ops.expand_dims(perm, -1), x)
            x = res + up if self.sum_res else ops.concatenate([res, up], axis=-1)

            x = self.up_convs[i](x, edge_index, edge_weight=edge_weight)
            if i < self.depth - 1:
                x = self.act(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
