import numpy as np
import keras
from keras import ops

from k3_node.data import Data
from k3_node.layers import GCNConv
from k3_node.loader import FullGraphDataset


def _graph(num_nodes=30, num_classes=3):
    rng = np.random.default_rng(0)
    y = rng.integers(0, num_classes, num_nodes)
    x = (np.eye(num_classes)[y] + 0.3 * rng.standard_normal((num_nodes, num_classes))).astype("float32")
    edge_index = rng.integers(0, num_nodes, (2, 90)).astype("int32")
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[:10] = True
    return Data(x=x, edge_index=edge_index, y=y.astype("int32"), train_mask=train_mask, test_mask=~train_mask)


class SmallGCN(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(3, 3)

    def call(self, inputs):
        x, edge_index = inputs
        return self.conv(x, edge_index)


def test_fit_and_evaluate_on_full_graph():
    data = _graph()
    model = SmallGCN()
    model.compile(keras.optimizers.Adam(0.05), keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(FullGraphDataset(data, mask="train_mask"), epochs=30, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]
    model.evaluate(FullGraphDataset(data, mask="test_mask"), verbose=0)
    preds = model.predict(FullGraphDataset(data, target=None), verbose=0)
    assert preds.shape == (30, 3)


def test_normalized_mask_gives_mean_over_masked_nodes():
    data = _graph()
    model = SmallGCN()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    reported = model.evaluate(FullGraphDataset(data, mask="train_mask"), verbose=0)
    per_node = ops.convert_to_numpy(loss_fn(data.y, model((data.x, data.edge_index))))
    np.testing.assert_allclose(reported, per_node[: 10].mean(), rtol=1e-5)
