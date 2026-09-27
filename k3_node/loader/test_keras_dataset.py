import numpy as np
import keras
from keras import ops

from k3_node.data import Data
from k3_node.layers import GCNConv
from k3_node.loader import DataLoader, FullGraphDataset, NeighborLoader
from k3_node.loader.keras_dataset import to_keras_batch


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

    def call(self, data):
        return self.conv(data.x, data.edge_index)


def test_fit_and_evaluate_on_full_graph():
    data = _graph()
    model = SmallGCN()
    model.compile(keras.optimizers.Adam(0.05), keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(FullGraphDataset(data, mask="train_mask"), epochs=30, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]
    model.evaluate(FullGraphDataset(data, mask="test_mask"), verbose=0)
    preds = model.predict(FullGraphDataset(data), verbose=0)
    assert preds.shape == (30, 3)


def test_normalized_mask_gives_mean_over_masked_nodes():
    data = _graph()
    model = SmallGCN()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    reported = model.evaluate(FullGraphDataset(data, mask="train_mask"), verbose=0)
    per_node = ops.convert_to_numpy(loss_fn(data.y, model(to_keras_batch(data)[0])))
    np.testing.assert_allclose(reported, per_node[: 10].mean(), rtol=1e-5)


def test_graph_loader_works_with_fit_evaluate_predict():
    from k3_node.datasets import FakeDataset
    from k3_node.layers import GCNConv, global_mean_pool

    class GraphClassifier(keras.Model):
        def __init__(self):
            super().__init__()
            self.conv = GCNConv(8, 16)
            self.head = keras.layers.Dense(3)

        def call(self, data):
            x = self.conv(data.x, data.edge_index)
            return self.head(global_mean_pool(x, data.batch, data.num_graphs))

    dataset = FakeDataset(num_graphs=20, avg_num_nodes=8, num_channels=8, num_classes=3)
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    model = GraphClassifier()
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    history = model.fit(loader, epochs=2, verbose=0)  # compiled on TF/JAX: num_graphs must be static
    assert len(history.history["loss"]) == 2
    model.evaluate(DataLoader(dataset, batch_size=6), verbose=0)
    assert model.predict(DataLoader(dataset, batch_size=6), verbose=0).shape == (20, 3)
    # Iterating the loader directly still yields Batch objects
    batch = next(iter(loader))
    assert hasattr(batch, "edge_index") and batch.num_graphs == 6


def test_neighbor_loader_weights_seed_nodes_only():
    data = _graph()
    loader = NeighborLoader(data, num_neighbors=[3], batch_size=4, input_nodes=np.arange(10))
    inputs, y, weight = loader[0]
    assert y.shape[0] == inputs.x.shape[0]
    assert (weight[:4] > 0).all() and (weight[4:] == 0).all()
