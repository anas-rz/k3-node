import numpy as np
from keras import ops

from k3_node.models import DeepGraphInfomax


def test_infomax():
    model = DeepGraphInfomax(
        hidden_channels=16,
        encoder=lambda x: x,
        summary=lambda z, *args: ops.mean(z, axis=0),
        corruption=lambda x: x + 1,
    )
    assert str(model) == "DeepGraphInfomax(16)"

    x = ops.ones((20, 16))

    pos_z, neg_z, summary = model(x)
    assert ops.shape(pos_z) == (20, 16)
    assert ops.shape(neg_z) == (20, 16)
    assert ops.shape(summary) == (16,)

    loss = model.loss(pos_z, neg_z, summary)
    assert float(ops.convert_to_numpy(loss)) >= 0

    acc = model.test(
        train_z=ops.ones((20, 16)),
        train_y=ops.convert_to_tensor(np.random.randint(0, 10, (20,))),
        test_z=ops.ones((20, 16)),
        test_y=ops.convert_to_tensor(np.random.randint(0, 10, (20,))),
    )
    assert 0 <= acc <= 1


def test_infomax_predefined_model():
    from k3_node.layers.conv import GCNConv

    class Encoder:
        def __init__(self):
            self.conv1 = GCNConv(16, 16)
            self.conv2 = GCNConv(16, 16)

        def __call__(self, x, edge_index, edge_weight=None):
            x = ops.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
            return self.conv2(x, edge_index, edge_weight=edge_weight)

    def corruption(x, edge_index, edge_weight):
        perm = np.random.permutation(ops.shape(x)[0])
        return ops.take(x, ops.convert_to_tensor(perm), axis=0), edge_index, edge_weight

    model = DeepGraphInfomax(
        hidden_channels=16,
        encoder=Encoder(),
        summary=lambda z, *args, **kwargs: ops.sigmoid(ops.mean(z, axis=0)),
        corruption=corruption,
    )

    x = ops.convert_to_tensor(np.random.randn(4, 16).astype("float32"))
    edge_index = ops.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]], dtype="int64")
    edge_weight = ops.convert_to_tensor(np.random.rand(edge_index.shape[1]).astype("float32"))

    pos_z, neg_z, summary = model(x, edge_index, edge_weight=edge_weight)
    assert ops.shape(pos_z) == (4, 16)
    assert ops.shape(neg_z) == (4, 16)
    assert ops.shape(summary) == (16,)

    loss = model.loss(pos_z, neg_z, summary)
    assert float(ops.convert_to_numpy(loss)) >= 0
