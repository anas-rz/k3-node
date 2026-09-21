import os
import os.path as osp
import pickle
import tempfile
import numpy as np
import pytest
import scipy.sparse as sp
from keras import ops

import k3_node.datasets as datasets
import k3_node.io as io


def to_np(x):
    if x is None:
        return None
    return ops.convert_to_numpy(x)


def test_karate_club():
    dataset = datasets.KarateClub()
    assert len(dataset) == 1
    assert repr(dataset) == "KarateClub()"
    assert dataset.num_classes == 4
    assert dataset.num_features == 34
    assert dataset.num_node_features == 34

    data = dataset[0]
    assert data.num_nodes == 34
    assert data.num_edges == 156
    assert data.x.shape == (34, 34)
    assert data.edge_index.shape == (2, 156)
    assert data.y.shape == (34,)
    assert data.train_mask.shape == (34,)

    # Exactly 4 train nodes (1 per class)
    assert int(np.sum(to_np(data.train_mask))) == 4
    assert data.is_undirected() is True
    assert data.has_self_loops() is False


def test_fake_dataset_node():
    dataset = datasets.FakeDataset(
        num_graphs=1,
        avg_num_nodes=50,
        avg_degree=5.0,
        num_channels=16,
        edge_dim=4,
        num_classes=5,
        task="node",
    )
    assert len(dataset) == 1
    assert repr(dataset) == "FakeDataset()"
    data = dataset[0]
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.x.shape[-1] == 16
    assert data.edge_attr.shape[-1] == 4
    assert data.y.shape == (data.num_nodes,)


def test_fake_dataset_graph():
    dataset = datasets.FakeDataset(
        num_graphs=3,
        avg_num_nodes=30,
        avg_degree=4.0,
        num_channels=8,
        num_classes=2,
        task="graph",
    )
    assert len(dataset) == 3
    assert repr(dataset) == "FakeDataset(3)"
    for i in range(3):
        data = dataset[i]
        assert data.num_nodes > 0
        assert data.x.shape[-1] == 8
        assert data.y.shape == (1,)


def test_fake_hetero_dataset():
    dataset = datasets.FakeHeteroDataset(
        num_graphs=1,
        num_node_types=2,
        num_edge_types=2,
        avg_num_nodes=30,
        avg_degree=3.0,
        avg_num_channels=8,
        num_classes=3,
        task="node",
    )
    assert len(dataset) == 1
    data = dataset[0]
    assert len(data.node_types) == 2
    assert len(data.edge_types) == 2

    for nt in data.node_types:
        assert data[nt].num_nodes > 0
        assert data[nt].x.shape[-1] > 0
        assert data[nt].y.shape == (data[nt].num_nodes,)

    for et in data.edge_types:
        assert data[et].edge_index.shape[0] == 2


def test_ba_shapes():
    dataset = datasets.BAShapes(connection_distribution="random")
    assert len(dataset) == 1
    data = dataset[0]
    assert data.num_nodes == 700
    assert data.x.shape == (700, 10)
    assert data.expl_mask.shape == (700,)
    assert int(to_np(data.y).max()) == 3


def test_sbm_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = datasets.StochasticBlockModelDataset(
            root=tmp_dir,
            block_sizes=[10, 10],
            edge_probs=[[0.5, 0.05], [0.05, 0.5]],
            num_graphs=2,
            num_channels=4,
        )
        assert len(dataset) == 2
        d0 = dataset[0]
        assert d0.num_nodes == 20
        assert d0.x.shape == (20, 4)
        assert d0.y.shape == (20,)


def test_explainer_dataset():
    dataset = datasets.ExplainerDataset(
        graph_generator="ba",
        motif_generator="house",
        num_motifs=3,
        num_graphs=2,
        graph_generator_kwargs={"num_nodes": 30, "num_edges": 2},
    )
    assert len(dataset) == 2
    for i in range(2):
        d = dataset[i]
        assert d.edge_index.shape[0] == 2
        assert d.node_mask.shape[0] == d.num_nodes
        assert d.edge_mask.shape[0] == d.edge_index.shape[1]


def test_read_npz():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = osp.join(tmp_dir, "test.npz")
        x_dense = (np.random.rand(8, 4) > 0.5).astype(np.float32)
        x_csr = sp.csr_matrix(x_dense)

        adj_dense = np.zeros((8, 8), dtype=np.float32)
        adj_dense[0, 1] = adj_dense[1, 0] = 1.0
        adj_csr = sp.csr_matrix(adj_dense)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.savez(
            path,
            attr_data=x_csr.data,
            attr_indices=x_csr.indices,
            attr_indptr=x_csr.indptr,
            attr_shape=x_csr.shape,
            adj_data=adj_csr.data,
            adj_indices=adj_csr.indices,
            adj_indptr=adj_csr.indptr,
            adj_shape=adj_csr.shape,
            labels=labels,
        )

        data = io.read_npz(path, to_undirected=True)
        assert data.num_nodes == 8
        assert data.x.shape == (8, 4)
        assert data.edge_index.shape[0] == 2
        assert data.y.shape == (8,)


def test_read_planetoid():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = "cora"
        num_train = 10
        num_val = 500
        num_test = 20
        num_nodes = num_train + num_val + num_test

        x = sp.csr_matrix(np.random.randn(num_train, 6).astype(np.float32))
        y = np.eye(2)[np.random.randint(0, 2, num_train)]
        tx = sp.csr_matrix(np.random.randn(num_test, 6).astype(np.float32))
        ty = np.eye(2)[np.random.randint(0, 2, num_test)]
        allx = sp.csr_matrix(np.random.randn(num_train + num_val, 6).astype(np.float32))
        ally = np.eye(2)[np.random.randint(0, 2, num_train + num_val)]
        graph = {i: [(i + 1) % num_nodes] for i in range(num_nodes)}
        test_index = np.arange(num_train + num_val, num_nodes, dtype=np.int64)

        for name, obj in [
            ("x", x),
            ("tx", tx),
            ("allx", allx),
            ("y", y),
            ("ty", ty),
            ("ally", ally),
            ("graph", graph),
        ]:
            with open(osp.join(tmp_dir, f"ind.{prefix}.{name}"), "wb") as f:
                pickle.dump(obj, f, protocol=2)

        np.savetxt(osp.join(tmp_dir, f"ind.{prefix}.test.index"), test_index, fmt="%d")

        data = io.read_planetoid_data(tmp_dir, prefix)
        assert data.num_nodes == num_nodes
        assert data.x.shape == (num_nodes, 6)
        assert data.y.shape == (num_nodes,)
        assert data.train_mask.shape == (num_nodes,)
        assert data.test_mask.shape == (num_nodes,)


def test_read_tu():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = "MUTAG"
        np.savetxt(osp.join(tmp_dir, f"{prefix}_graph_indicator.txt"), [1, 1, 2, 2], fmt="%d")
        np.savetxt(
            osp.join(tmp_dir, f"{prefix}_A.txt"),
            [[1, 2], [2, 1], [3, 4], [4, 3]],
            fmt="%d",
            delimiter=", ",
        )
        np.savetxt(osp.join(tmp_dir, f"{prefix}_node_labels.txt"), [0, 1, 0, 1], fmt="%d")
        np.savetxt(osp.join(tmp_dir, f"{prefix}_graph_labels.txt"), [1, 0], fmt="%d")

        data, slices, sizes = io.read_tu_data(tmp_dir, prefix)
        assert data.num_nodes == 4
        assert "edge_index" in slices
        assert "x" in slices
        assert "y" in slices
        assert sizes["num_node_labels"] == 2


def test_ppi():
    import json
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = osp.join(tmp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        for split in ["train", "valid", "test"]:
            graph = {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
                "links": [
                    {"source": 0, "target": 1},
                    {"source": 1, "target": 0},
                    {"source": 2, "target": 3},
                    {"source": 3, "target": 2},
                ],
            }
            with open(osp.join(raw_dir, f"{split}_graph.json"), "w") as f:
                json.dump(graph, f)

            np.save(osp.join(raw_dir, f"{split}_feats.npy"), np.ones((4, 50), dtype=np.float32))
            np.save(osp.join(raw_dir, f"{split}_labels.npy"), np.ones((4, 121), dtype=np.float32))
            np.save(osp.join(raw_dir, f"{split}_graph_id.npy"), np.array([1, 1, 2, 2], dtype=np.int64))

        dataset = datasets.PPI(root=tmp_dir, split="train")
        assert len(dataset) == 2
        assert dataset.num_features == 50
        assert dataset.num_classes == 121

        data0 = dataset[0]
        assert data0.num_nodes == 2
        assert data0.x.shape == (2, 50)
        assert data0.y.shape == (2, 121)
        assert data0.edge_index.shape == (2, 2)

        val_dataset = datasets.PPI(root=tmp_dir, split="val")
        assert len(val_dataset) == 2

        with pytest.raises(AssertionError):
            datasets.PPI(root=tmp_dir, split="unknown")


def test_reddit():
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = osp.join(tmp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        num_nodes = 10
        num_features = 602
        features = np.random.randn(num_nodes, num_features).astype(np.float32)
        labels = np.random.randint(0, 41, size=num_nodes, dtype=np.int64)
        node_types = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)

        np.savez(
            osp.join(raw_dir, "reddit_data.npz"),
            feature=features,
            label=labels,
            node_types=node_types,
        )

        row = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        col = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8])
        data_arr = np.ones(10, dtype=np.float32)
        adj_coo = sp.coo_matrix((data_arr, (row, col)), shape=(num_nodes, num_nodes))
        sp.save_npz(osp.join(raw_dir, "reddit_graph.npz"), adj_coo)

        dataset = datasets.Reddit(root=tmp_dir)
        assert len(dataset) == 1
        assert dataset.num_features == 602
        assert dataset.num_classes == 41

        data = dataset[0]
        assert data.num_nodes == 10
        assert data.x.shape == (10, 602)
        assert data.y.shape == (10,)
        assert data.edge_index.shape[0] == 2
        assert data.train_mask.shape == (10,)
        assert data.val_mask.shape == (10,)
        assert data.test_mask.shape == (10,)
        assert int(ops.sum(ops.cast(data.train_mask, "int32"))) == 4
        assert int(ops.sum(ops.cast(data.val_mask, "int32"))) == 3
        assert int(ops.sum(ops.cast(data.test_mask, "int32"))) == 3

