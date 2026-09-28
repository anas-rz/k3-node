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


def test_full_graph_dataset_index_split():
    import numpy as np
    from k3_node.data import Data
    from k3_node.loader import FullGraphDataset

    data = Data(x=np.ones((5, 2), "float32"), edge_index=np.array([[0, 1], [1, 2]]),
                train_idx=np.array([3, 1]), train_y=np.array([2, 1]), num_nodes=5)
    inputs, y, weight = FullGraphDataset(data, index="train_idx", target="train_y")[0]
    assert "train_idx" not in inputs._fields and "train_y" not in inputs._fields
    np.testing.assert_array_equal(y, [0, 1, 0, 2, 0])
    np.testing.assert_allclose(weight, [0, 2.5, 0, 2.5, 0])  # mean over the 2 indexed nodes


def test_full_graph_dataset_fresh_negatives_every_epoch():
    import numpy as np
    from k3_node.data import Data
    from k3_node.loader import FullGraphDataset

    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    data = Data(x=np.ones((50, 2), "float32"), edge_index=edge_index, edge_label_index=edge_index,
                edge_label=np.ones(4, "float32"), num_nodes=50)
    dataset = FullGraphDataset(data, neg_sampling_ratio=2.0)
    (inputs1, y1), (inputs2, _) = dataset[0], dataset[0]
    assert inputs1.edge_label_index.shape == (2, 12)
    np.testing.assert_array_equal(y1, [1] * 4 + [0] * 8)
    assert not np.array_equal(inputs1.edge_label_index[:, 4:], inputs2.edge_label_index[:, 4:])


def test_link_neighbor_loader_uses_local_ids():
    import numpy as np
    from k3_node.data import Data
    from k3_node.loader import LinkNeighborLoader

    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 200, size=(2, 600))
    data = Data(x=np.arange(200, dtype="float32")[:, None], edge_index=edge_index, num_nodes=200)
    loader = LinkNeighborLoader(data, num_neighbors=[5], batch_size=16, neg_sampling_ratio=1.0)
    batch = next(iter(loader))
    eli = np.asarray(batch.edge_label_index)
    n_id = np.asarray(batch.n_id)
    assert eli.max() < n_id.shape[0]  # local ids into the sampled subgraph
    label = np.asarray(batch.edge_label)
    pos = n_id[eli[:, label == 1]]  # back to global ids: must be real edges
    real = set(map(tuple, edge_index.T.tolist()))
    assert all(tuple(e) in real for e in pos.T.tolist())


def test_loader_with_mask():
    import numpy as np
    from k3_node.data import Data
    from k3_node.loader import ClusterData, ClusterLoader

    rng = np.random.default_rng(0)
    data = Data(x=rng.random((60, 3)).astype("float32"), edge_index=rng.integers(0, 60, (2, 200)),
                y=rng.integers(0, 3, 60), train_mask=np.arange(60) < 30, num_nodes=60)
    loader = ClusterLoader(ClusterData(data, num_parts=4), batch_size=2).with_mask("train_mask")
    inputs, y, weight = loader[0]
    assert weight.shape == y.shape and "train_mask" not in inputs._fields
