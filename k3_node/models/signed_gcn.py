from typing import Optional, Tuple

import keras
from keras import ops
import numpy as np

from k3_node.layers.conv import SignedConv


class SignedGCN(keras.layers.Layer):
    r"""The signed graph convolutional network model from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of layers.
        lamb (float, optional): Balances the contributions of the overall
            objective. (default: :obj:`5`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        lamb: float = 5.0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb

        self.conv1 = SignedConv(in_channels, hidden_channels // 2, first_aggr=True, bias=bias)
        self.convs = [
            SignedConv(hidden_channels // 2, hidden_channels // 2, first_aggr=False, bias=bias)
            for _ in range(num_layers - 1)
        ]
        self.lin = keras.layers.Dense(3, use_bias=bias)

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin.built:
            self.lin.kernel.assign(keras.initializers.GlorotUniform()(self.lin.kernel.shape))
            if self.lin.bias is not None:
                self.lin.bias.assign(ops.zeros(self.lin.bias.shape))

    @staticmethod
    def split_edges(edge_index, test_ratio: float = 0.2):
        num_edges = ops.shape(edge_index)[1]
        perm = np.random.permutation(num_edges)
        num_test = int(test_ratio * num_edges)
        test_mask = np.zeros(num_edges, dtype=bool)
        test_mask[perm[:num_test]] = True

        edge_index_np = ops.convert_to_numpy(edge_index)
        train_edge_index = ops.convert_to_tensor(edge_index_np[:, ~test_mask])
        test_edge_index = ops.convert_to_tensor(edge_index_np[:, test_mask])
        return train_edge_index, test_edge_index

    def call(self, x, pos_edge_index, neg_edge_index):
        z = ops.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = ops.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z, edge_index):
        value = ops.concatenate([
            ops.take(z, edge_index[0], axis=0),
            ops.take(z, edge_index[1], axis=0),
        ], axis=-1)
        value = self.lin(value)
        return ops.log_softmax(value, axis=-1)

    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        from k3_node.models.utils import negative_sampling

        edge_index = ops.concatenate([pos_edge_index, neg_edge_index], axis=1)
        none_edge_index = negative_sampling(edge_index, ops.shape(z)[0])

        nll_loss = 0.0
        pos_pred = self.discriminate(z, pos_edge_index)
        nll_loss = nll_loss - ops.mean(pos_pred[:, 0])

        neg_pred = self.discriminate(z, neg_edge_index)
        nll_loss = nll_loss - ops.mean(neg_pred[:, 1])

        none_pred = self.discriminate(z, none_edge_index)
        nll_loss = nll_loss - ops.mean(none_pred[:, 2])

        return nll_loss / 3.0

    def pos_embedding_loss(self, z, pos_edge_index):
        from k3_node.models.utils import structured_negative_sampling

        i, j, k = structured_negative_sampling(pos_edge_index, ops.shape(z)[0])
        out = (
            ops.sum((ops.take(z, i, axis=0) - ops.take(z, j, axis=0)) ** 2, axis=1)
            - ops.sum((ops.take(z, i, axis=0) - ops.take(z, k, axis=0)) ** 2, axis=1)
        )
        return ops.mean(ops.maximum(out, 0.0))

    def neg_embedding_loss(self, z, neg_edge_index):
        from k3_node.models.utils import structured_negative_sampling

        i, j, k = structured_negative_sampling(neg_edge_index, ops.shape(z)[0])
        out = (
            ops.sum((ops.take(z, i, axis=0) - ops.take(z, k, axis=0)) ** 2, axis=1)
            - ops.sum((ops.take(z, i, axis=0) - ops.take(z, j, axis=0)) ** 2, axis=1)
        )
        return ops.mean(ops.maximum(out, 0.0))

    def loss(self, z, pos_edge_index, neg_edge_index):
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def test(self, z, pos_edge_index, neg_edge_index):
        from sklearn.metrics import f1_score, roc_auc_score

        pos_p = ops.argmax(self.discriminate(z, pos_edge_index)[:, :2], axis=1)
        neg_p = ops.argmax(self.discriminate(z, neg_edge_index)[:, :2], axis=1)
        pred = 1 - ops.convert_to_numpy(ops.concatenate([pos_p, neg_p], axis=0))
        y = np.concatenate([
            np.ones(ops.shape(pos_p)[0], dtype=np.int64),
            np.zeros(ops.shape(neg_p)[0], dtype=np.int64),
        ])
        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average="binary") if pred.sum() > 0 else 0.0
        return auc, f1

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, num_layers={self.num_layers})')

