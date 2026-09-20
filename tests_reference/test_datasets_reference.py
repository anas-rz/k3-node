import os
import os.path as osp
import pickle
import tempfile
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from keras import ops

import torch_geometric.datasets as pyg_datasets
import torch_geometric.io as pyg_io
import torch_geometric.data as pyg_data

import k3_node.datasets as k3_datasets
import k3_node.io as k3_io
import k3_node.data as k3_data


def to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return ops.convert_to_numpy(x)


# ---------------------------------------------------------------------------
# 1. KarateClub Parity Test
# ---------------------------------------------------------------------------
def test_reference_karate_club_parity():
    pyg_dataset = pyg_datasets.KarateClub()
    k3_dataset = k3_datasets.KarateClub()

    assert len(pyg_dataset) == len(k3_dataset) == 1
    assert repr(pyg_dataset) == repr(k3_dataset) == "KarateClub()"
    assert pyg_dataset.num_classes == k3_dataset.num_classes == 4
    assert pyg_dataset.num_features == k3_dataset.num_features == 34
    assert pyg_dataset.num_node_features == k3_dataset.num_node_features == 34

    pyg_data_obj = pyg_dataset[0]
    k3_data_obj = k3_dataset[0]

    # Verify nodes, edges, features, classes count
    assert pyg_data_obj.num_nodes == k3_data_obj.num_nodes == 34
    assert pyg_data_obj.num_edges == k3_data_obj.num_edges == 156

    # Verify node features x (34, 34 identity matrix)
    assert np.allclose(to_np(pyg_data_obj.x), to_np(k3_data_obj.x))

    # Verify edge indices (2, 156)
    assert np.array_equal(to_np(pyg_data_obj.edge_index), to_np(k3_data_obj.edge_index))

    # Verify labels y (34,)
    assert np.array_equal(to_np(pyg_data_obj.y), to_np(k3_data_obj.y))

    # Verify train_mask (34,)
    assert np.array_equal(to_np(pyg_data_obj.train_mask), to_np(k3_data_obj.train_mask))

    # Parity of graph properties
    assert pyg_data_obj.is_undirected() == k3_data_obj.is_undirected() is True
    assert pyg_data_obj.has_self_loops() == k3_data_obj.has_self_loops() is False
    assert pyg_data_obj.has_isolated_nodes() == k3_data_obj.has_isolated_nodes() is False


# ---------------------------------------------------------------------------
# 2. FakeDataset & FakeHeteroDataset Parity Test
# ---------------------------------------------------------------------------
def test_reference_fake_dataset_parity():
    # Node classification task
    pyg_ds_node = pyg_datasets.FakeDataset(
        num_graphs=1,
        avg_num_nodes=50,
        avg_degree=5.0,
        num_channels=16,
        edge_dim=4,
        num_classes=5,
        task="node",
    )
    k3_ds_node = k3_datasets.FakeDataset(
        num_graphs=1,
        avg_num_nodes=50,
        avg_degree=5.0,
        num_channels=16,
        edge_dim=4,
        num_classes=5,
        task="node",
    )

    assert len(pyg_ds_node) == len(k3_ds_node) == 1
    assert repr(pyg_ds_node) == repr(k3_ds_node) == "FakeDataset()"

    k3_d = k3_ds_node[0]
    assert k3_d.num_nodes > 0
    assert k3_d.num_edges > 0
    assert k3_d.x.shape[-1] == 16
    assert k3_d.edge_attr.shape[-1] == 4
    assert k3_d.y.shape == (k3_d.num_nodes,)

    # Graph classification task with multiple graphs
    pyg_ds_graph = pyg_datasets.FakeDataset(
        num_graphs=5,
        avg_num_nodes=30,
        avg_degree=4.0,
        num_channels=8,
        edge_dim=0,
        num_classes=3,
        task="graph",
    )
    k3_ds_graph = k3_datasets.FakeDataset(
        num_graphs=5,
        avg_num_nodes=30,
        avg_degree=4.0,
        num_channels=8,
        edge_dim=0,
        num_classes=3,
        task="graph",
    )

    assert len(pyg_ds_graph) == len(k3_ds_graph) == 5
    assert repr(pyg_ds_graph) == repr(k3_ds_graph) == "FakeDataset(5)"

    for i in range(5):
        gi = k3_ds_graph[i]
        assert gi.x.shape[-1] == 8
        assert gi.y.shape == (1,)
        assert int(to_np(gi.y)[0]) in range(3)


def test_reference_fake_hetero_dataset_parity():
    pyg_ds = pyg_datasets.FakeHeteroDataset(
        num_graphs=1,
        num_node_types=2,
        num_edge_types=3,
        avg_num_nodes=40,
        avg_degree=4.0,
        avg_num_channels=16,
        edge_dim=2,
        num_classes=4,
        task="node",
    )
    k3_ds = k3_datasets.FakeHeteroDataset(
        num_graphs=1,
        num_node_types=2,
        num_edge_types=3,
        avg_num_nodes=40,
        avg_degree=4.0,
        avg_num_channels=16,
        edge_dim=2,
        num_classes=4,
        task="node",
    )

    assert len(pyg_ds) == len(k3_ds) == 1
    assert repr(pyg_ds) == repr(k3_ds) == "FakeHeteroDataset()"

    k3_hd = k3_ds[0]
    assert len(k3_hd.node_types) == 2
    assert len(k3_hd.edge_types) == 3

    for nt in k3_hd.node_types:
        assert k3_hd[nt].num_nodes > 0
        assert k3_hd[nt].x.shape[0] == k3_hd[nt].num_nodes
        assert k3_hd[nt].y.shape == (k3_hd[nt].num_nodes,)

    for et in k3_hd.edge_types:
        assert k3_hd[et].edge_index.shape[0] == 2
        assert k3_hd[et].edge_attr.shape[-1] == 2


# ---------------------------------------------------------------------------
# 3. BAShapes Parity Test
# ---------------------------------------------------------------------------
def test_reference_ba_shapes_parity():
    k3_dataset = k3_datasets.BAShapes(connection_distribution="random")
    assert len(k3_dataset) == 1
    data = k3_dataset[0]

    # BAShapes contains 300 base BA nodes + 80 houses * 5 nodes = 700 nodes
    assert data.num_nodes == 700
    assert data.x.shape == (700, 10)
    assert np.allclose(to_np(data.x), 1.0)
    assert data.expl_mask.shape == (700,)
    # House node labels range from 0 to 3
    assert int(to_np(data.y).max()) == 3


# ---------------------------------------------------------------------------
# 4. BA2Motif Dataset Mock I/O Parity Test
# ---------------------------------------------------------------------------
def test_reference_ba2motif_parity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock BA-2motif.pkl
        num_graphs = 4
        num_nodes = 10
        adjs = np.zeros((num_graphs, num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_graphs):
            # simple cycle
            for n in range(num_nodes):
                adjs[i, n, (n + 1) % num_nodes] = 1.0
                adjs[i, (n + 1) % num_nodes, n] = 1.0

        xs = np.ones((num_graphs, num_nodes, 10), dtype=np.float32)
        ys = np.zeros((num_graphs, 2), dtype=np.int64)
        ys[0, 0] = 1
        ys[1, 1] = 1
        ys[2, 0] = 1
        ys[3, 1] = 1

        raw_dir = osp.join(tmp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        pkl_path = osp.join(raw_dir, "BA-2motif.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump((adjs, xs, ys), f)

        # Process with PyG
        pyg_ds = pyg_datasets.BA2MotifDataset(root=tmp_dir)
        # Process with k3_node
        tmp_dir_k3 = tempfile.mkdtemp()
        raw_dir_k3 = osp.join(tmp_dir_k3, "raw")
        os.makedirs(raw_dir_k3, exist_ok=True)
        with open(osp.join(raw_dir_k3, "BA-2motif.pkl"), "wb") as f:
            pickle.dump((adjs, xs, ys), f)

        k3_ds = k3_datasets.BA2MotifDataset(root=tmp_dir_k3)

        assert len(pyg_ds) == len(k3_ds) == num_graphs
        for i in range(num_graphs):
            pyg_d = pyg_ds[i]
            k3_d = k3_ds[i]
            assert np.allclose(to_np(pyg_d.x), to_np(k3_d.x))
            assert np.array_equal(to_np(pyg_d.edge_index), to_np(k3_d.edge_index))
            assert int(to_np(pyg_d.y).item() if to_np(pyg_d.y).ndim == 0 else to_np(pyg_d.y)[0]) == int(to_np(k3_d.y)[0])


# ---------------------------------------------------------------------------
# 5. read_npz Parity Test (Amazon / Coauthor / CitationFull)
# ---------------------------------------------------------------------------
def test_reference_read_npz_parity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        npz_path = osp.join(tmp_dir, "test_graph.npz")

        # Create sparse node features (10 nodes, 5 features)
        x_dense = (np.random.rand(10, 5) > 0.6).astype(np.float32)
        x_csr = sp.csr_matrix(x_dense)

        # Create sparse adjacency (10 nodes, 10 edges)
        adj_dense = np.zeros((10, 10), dtype=np.float32)
        edges = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5]])
        adj_dense[edges[0], edges[1]] = 1.0
        adj_csr = sp.csr_matrix(adj_dense)

        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.savez(
            npz_path,
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

        pyg_data_obj = pyg_io.read_npz(npz_path, to_undirected=True)
        k3_data_obj = k3_io.read_npz(npz_path, to_undirected=True)

        assert np.allclose(to_np(pyg_data_obj.x), to_np(k3_data_obj.x))
        assert np.array_equal(to_np(pyg_data_obj.edge_index), to_np(k3_data_obj.edge_index))
        assert np.array_equal(to_np(pyg_data_obj.y), to_np(k3_data_obj.y))
        assert pyg_data_obj.num_nodes == k3_data_obj.num_nodes == 10


# ---------------------------------------------------------------------------
# 6. Planetoid I/O Parity Test (read_planetoid_data)
# ---------------------------------------------------------------------------
def test_reference_planetoid_io_parity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = "cora"
        num_train = 20
        num_val = 500
        num_test = 100
        num_nodes = num_train + num_val + num_test
        num_feats = 8
        num_classes = 3

        # Create dummy data for planetoid
        x = sp.csr_matrix(np.random.randn(num_train, num_feats).astype(np.float32))
        y = np.eye(num_classes)[np.random.randint(0, num_classes, num_train)]

        tx = sp.csr_matrix(np.random.randn(num_test, num_feats).astype(np.float32))
        ty = np.eye(num_classes)[np.random.randint(0, num_classes, num_test)]

        allx = sp.csr_matrix(np.random.randn(num_train + num_val, num_feats).astype(np.float32))
        ally = np.eye(num_classes)[np.random.randint(0, num_classes, num_train + num_val)]

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

        pyg_data_obj = pyg_io.read_planetoid_data(tmp_dir, prefix)
        k3_data_obj = k3_io.read_planetoid_data(tmp_dir, prefix)

        assert np.allclose(to_np(pyg_data_obj.x), to_np(k3_data_obj.x))
        assert np.array_equal(to_np(pyg_data_obj.edge_index), to_np(k3_data_obj.edge_index))
        assert np.array_equal(to_np(pyg_data_obj.y), to_np(k3_data_obj.y))
        assert np.array_equal(to_np(pyg_data_obj.train_mask), to_np(k3_data_obj.train_mask))
        assert np.array_equal(to_np(pyg_data_obj.test_mask), to_np(k3_data_obj.test_mask))


# ---------------------------------------------------------------------------
# 7. TU Dataset I/O Parity Test (read_tu_data)
# ---------------------------------------------------------------------------
def test_reference_tu_data_io_parity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = "MUTAG"
        # 2 graphs: Graph 1 has 3 nodes (1, 2, 3), Graph 2 has 2 nodes (4, 5)
        # Graph indicator (1-indexed)
        np.savetxt(osp.join(tmp_dir, f"{prefix}_graph_indicator.txt"), [1, 1, 1, 2, 2], fmt="%d")
        # Adjacency (1-indexed)
        np.savetxt(
            osp.join(tmp_dir, f"{prefix}_A.txt"),
            [[1, 2], [2, 1], [2, 3], [3, 2], [4, 5], [5, 4]],
            fmt="%d",
            delimiter=", ",
        )
        # Node labels (1-indexed)
        np.savetxt(osp.join(tmp_dir, f"{prefix}_node_labels.txt"), [0, 1, 0, 1, 0], fmt="%d")
        # Graph labels
        np.savetxt(osp.join(tmp_dir, f"{prefix}_graph_labels.txt"), [1, -1], fmt="%d")

        pyg_data_obj, pyg_slices, pyg_sizes = pyg_io.read_tu_data(tmp_dir, prefix)
        k3_data_obj, k3_slices, k3_sizes = k3_io.read_tu_data(tmp_dir, prefix)

        assert np.allclose(to_np(pyg_data_obj.x), to_np(k3_data_obj.x))
        assert np.array_equal(to_np(pyg_data_obj.edge_index), to_np(k3_data_obj.edge_index))
        assert np.array_equal(to_np(pyg_data_obj.y), to_np(k3_data_obj.y))

        assert np.array_equal(to_np(pyg_slices["edge_index"]), to_np(k3_slices["edge_index"]))
        assert np.array_equal(to_np(pyg_slices["x"]), to_np(k3_slices["x"]))
        assert np.array_equal(to_np(pyg_slices["y"]), to_np(k3_slices["y"]))
        assert pyg_sizes == k3_sizes


# ---------------------------------------------------------------------------
# 8. ExplainerDataset Parity Test
# ---------------------------------------------------------------------------
def test_reference_explainer_dataset_parity():
    # Test ExplainerDataset generation
    k3_expl = k3_datasets.ExplainerDataset(
        graph_generator="ba",
        motif_generator="house",
        num_motifs=5,
        num_graphs=2,
        graph_generator_kwargs={"num_nodes": 50, "num_edges": 3},
    )

    assert len(k3_expl) == 2
    for i in range(2):
        d = k3_expl[i]
        assert hasattr(d, "edge_index")
        assert hasattr(d, "y")
        assert hasattr(d, "node_mask")
        assert hasattr(d, "edge_mask")
        assert d.edge_index.shape[0] == 2
        assert d.node_mask.shape[0] == d.num_nodes
        assert d.edge_mask.shape[0] == d.edge_index.shape[1]
