"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node pooling layers produce numerically
equivalent outputs to torch_geometric.nn.pool reference implementations.
These tests are intended for local validation and documentation, and are
not run as part of the default GitHub Actions test suites.
"""

import numpy as np
import torch
from keras import ops

# PyG references
from torch_geometric.nn.pool import (
    global_add_pool as pyg_global_add_pool,
    global_mean_pool as pyg_global_mean_pool,
    global_max_pool as pyg_global_max_pool,
    TopKPooling as PyGTopKPooling,
    SAGPooling as PyGSAGPooling,
    EdgePooling as PyGEdgePooling,
    ClusterPooling as PyGClusterPooling,
    MemPooling as PyGMemPooling,
    max_pool_x as pyg_max_pool_x,
    avg_pool_x as pyg_avg_pool_x,
    L2KNNIndex as PyGL2KNNIndex,
    MIPSKNNIndex as PyGMIPSKNNIndex,
)
from torch_geometric.nn.pool.select.topk import (
    topk as pyg_topk,
    SelectTopK as PyGSelectTopK,
)
from torch_geometric.nn.pool.connect.filter_edges import (
    filter_adj as pyg_filter_adj,
)
from torch_geometric.nn.pool.consecutive import (
    consecutive_cluster as pyg_consecutive_cluster,
)
from torch_geometric.nn.pool.pool import (
    pool_edge as pyg_pool_edge,
    pool_batch as pyg_pool_batch,
    pool_pos as pyg_pool_pos,
)

# k3-node implementations
from k3_node.layers.pool import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    topk,
    SelectTopK,
    filter_adj,
    consecutive_cluster,
    pool_edge,
    pool_batch,
    pool_pos,
    max_pool_x,
    avg_pool_x,
    TopKPooling,
    SAGPooling,
    EdgePooling,
    ClusterPooling,
    MemPooling,
    L2KNNIndex,
    MIPSKNNIndex,
    knn,
    knn_graph,
)


def test_reference_glob_pool():
    np.random.seed(42)
    x = np.random.randn(7, 16).astype(np.float32)
    batch = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)

    # Without batch
    pyg_add = pyg_global_add_pool(torch.from_numpy(x), None).detach().numpy()
    k3_add = ops.convert_to_numpy(global_add_pool(ops.convert_to_tensor(x), None))
    assert np.allclose(pyg_add, k3_add, atol=1e-5)

    pyg_mean = pyg_global_mean_pool(torch.from_numpy(x), None).detach().numpy()
    k3_mean = ops.convert_to_numpy(global_mean_pool(ops.convert_to_tensor(x), None))
    assert np.allclose(pyg_mean, k3_mean, atol=1e-5)

    pyg_max = pyg_global_max_pool(torch.from_numpy(x), None).detach().numpy()
    k3_max = ops.convert_to_numpy(global_max_pool(ops.convert_to_tensor(x), None))
    assert np.allclose(pyg_max, k3_max, atol=1e-5)

    # With batch
    pyg_add_b = pyg_global_add_pool(torch.from_numpy(x), torch.from_numpy(batch)).detach().numpy()
    k3_add_b = ops.convert_to_numpy(global_add_pool(ops.convert_to_tensor(x), ops.convert_to_tensor(batch)))
    assert np.allclose(pyg_add_b, k3_add_b, atol=1e-5)

    pyg_mean_b = pyg_global_mean_pool(torch.from_numpy(x), torch.from_numpy(batch)).detach().numpy()
    k3_mean_b = ops.convert_to_numpy(global_mean_pool(ops.convert_to_tensor(x), ops.convert_to_tensor(batch)))
    assert np.allclose(pyg_mean_b, k3_mean_b, atol=1e-5)

    pyg_max_b = pyg_global_max_pool(torch.from_numpy(x), torch.from_numpy(batch)).detach().numpy()
    k3_max_b = ops.convert_to_numpy(global_max_pool(ops.convert_to_tensor(x), ops.convert_to_tensor(batch)))
    assert np.allclose(pyg_max_b, k3_max_b, atol=1e-5)


def test_reference_consecutive_cluster():
    src = np.array([2, 5, 2, 7, 5], dtype=np.int64)
    pyg_inv, pyg_perm = pyg_consecutive_cluster(torch.from_numpy(src))
    k3_inv, k3_perm = consecutive_cluster(ops.convert_to_tensor(src))

    assert np.array_equal(pyg_inv.numpy(), ops.convert_to_numpy(k3_inv))
    assert np.array_equal(pyg_perm.numpy(), ops.convert_to_numpy(k3_perm))


def test_reference_filter_adj():
    edge_index = np.array([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 3, 0, 2, 1, 3, 0, 2],
    ], dtype=np.int64)
    edge_attr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    perm = np.array([2, 3], dtype=np.int64)

    pyg_ei, pyg_ea = pyg_filter_adj(torch.from_numpy(edge_index), torch.from_numpy(edge_attr), torch.from_numpy(perm))
    k3_ei, k3_ea = filter_adj(ops.convert_to_tensor(edge_index), ops.convert_to_tensor(edge_attr), ops.convert_to_tensor(perm))

    assert np.array_equal(pyg_ei.numpy(), ops.convert_to_numpy(k3_ei))
    assert np.allclose(pyg_ea.numpy(), ops.convert_to_numpy(k3_ea))


def test_reference_topk():
    batch = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    x = np.array([1.2, 0.5, 2.3, 0.1, 0.8, 3.1, 1.5], dtype=np.float32)

    # Ratio
    pyg_perm_r = pyg_topk(torch.from_numpy(x), 0.5, torch.from_numpy(batch)).numpy()
    k3_perm_r = ops.convert_to_numpy(topk(ops.convert_to_tensor(x), 0.5, ops.convert_to_tensor(batch)))
    assert np.array_equal(pyg_perm_r, k3_perm_r)

    # Min score
    x_score = np.array([0.2, 0.05, 0.23, 0.01, 0.08, 0.31, 0.15], dtype=np.float32)
    pyg_perm_m = pyg_topk(torch.from_numpy(x_score), None, torch.from_numpy(batch), min_score=0.1).numpy()
    k3_perm_m = ops.convert_to_numpy(topk(ops.convert_to_tensor(x_score), None, ops.convert_to_tensor(batch), min_score=0.1))
    assert np.array_equal(pyg_perm_m, k3_perm_m)


def test_reference_select_topk():
    torch.manual_seed(42)
    in_channels = 16
    pyg_select = PyGSelectTopK(in_channels, ratio=0.5)
    k3_select = SelectTopK(in_channels, ratio=0.5)

    k3_select.build((None, in_channels))
    with torch.no_grad():
        k3_select.weight.assign(ops.convert_to_tensor(pyg_select.weight.detach().numpy()))

    x = np.random.randn(7, in_channels).astype(np.float32)
    batch = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)

    pyg_out = pyg_select(torch.from_numpy(x), torch.from_numpy(batch))
    k3_out = k3_select(ops.convert_to_tensor(x), ops.convert_to_tensor(batch))

    assert np.array_equal(pyg_out.node_index.numpy(), ops.convert_to_numpy(k3_out.node_index))
    assert np.allclose(pyg_out.weight.detach().numpy(), ops.convert_to_numpy(k3_out.weight), atol=1e-5)


def test_reference_topk_pooling():
    torch.manual_seed(42)
    in_channels = 16
    pyg_pool = PyGTopKPooling(in_channels, ratio=0.5)
    k3_pool = TopKPooling(in_channels, ratio=0.5)

    k3_pool.select.build((None, in_channels))
    with torch.no_grad():
        k3_pool.select.weight.assign(ops.convert_to_tensor(pyg_pool.select.weight.detach().numpy()))

    edge_index = np.array([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
    ], dtype=np.int64)
    x = np.random.randn(4, in_channels).astype(np.float32)

    pyg_res = pyg_pool(torch.from_numpy(x), torch.from_numpy(edge_index))
    k3_res = k3_pool(ops.convert_to_tensor(x), ops.convert_to_tensor(edge_index))

    # Compare x, edge_index, perm, score
    assert np.allclose(pyg_res[0].detach().numpy(), ops.convert_to_numpy(k3_res[0]), atol=1e-5)
    assert np.array_equal(pyg_res[1].numpy(), ops.convert_to_numpy(k3_res[1]))
    assert np.array_equal(pyg_res[4].numpy(), ops.convert_to_numpy(k3_res[4]))
    assert np.allclose(pyg_res[5].detach().numpy(), ops.convert_to_numpy(k3_res[5]), atol=1e-5)


def test_reference_edge_pooling():
    torch.manual_seed(42)
    in_channels = 16
    pyg_pool = PyGEdgePooling(in_channels)
    k3_pool = EdgePooling(in_channels)

    k3_pool.build((None, in_channels))
    with torch.no_grad():
        k3_pool.lin.set_weights([
            pyg_pool.lin.weight.detach().numpy().T,
            pyg_pool.lin.bias.detach().numpy(),
        ])

    x = np.random.randn(5, in_channels).astype(np.float32)
    edge_index = np.array([
        [0, 0, 1, 2, 3],
        [1, 2, 3, 4, 4],
    ], dtype=np.int64)
    batch = np.array([0, 0, 0, 0, 0], dtype=np.int64)

    pyg_res = pyg_pool(torch.from_numpy(x), torch.from_numpy(edge_index), torch.from_numpy(batch))
    k3_res = k3_pool(ops.convert_to_tensor(x), ops.convert_to_tensor(edge_index), ops.convert_to_tensor(batch))

    # Compare pooled x, edge_index, batch
    assert np.allclose(pyg_res[0].detach().numpy(), ops.convert_to_numpy(k3_res[0]), atol=1e-5)
    assert np.array_equal(pyg_res[1].numpy(), ops.convert_to_numpy(k3_res[1]))
    assert np.array_equal(pyg_res[2].numpy(), ops.convert_to_numpy(k3_res[2]))

    # Test unpool
    pyg_unpool = pyg_pool.unpool(pyg_res[0], pyg_res[3])
    k3_unpool = k3_pool.unpool(k3_res[0], k3_res[3])
    assert np.allclose(pyg_unpool[0].detach().numpy(), ops.convert_to_numpy(k3_unpool[0]), atol=1e-5)


def test_reference_mem_pooling():
    torch.manual_seed(42)
    in_channels, out_channels, heads, num_clusters = 16, 32, 4, 8
    pyg_pool = PyGMemPooling(in_channels, out_channels, heads=heads, num_clusters=num_clusters)
    k3_pool = MemPooling(in_channels, out_channels, heads=heads, num_clusters=num_clusters)
    k3_pool.build((2, 5, in_channels))

    with torch.no_grad():
        k3_pool.k.assign(ops.convert_to_tensor(pyg_pool.k.detach().numpy()))
        # Conv2d weight: [1, heads, 1, 1] in PyG -> [heads, 1] in k3_pool
        conv_w = pyg_pool.conv.weight.detach().numpy().reshape(heads, 1)
        k3_pool.conv_weight.assign(ops.convert_to_tensor(conv_w))
        k3_pool.lin.set_weights([pyg_pool.lin.weight.detach().numpy().T])

    x = np.random.randn(2, 5, in_channels).astype(np.float32)
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=bool)

    pyg_x, pyg_s = pyg_pool(torch.from_numpy(x), mask=torch.from_numpy(mask))
    k3_x, k3_s = k3_pool(ops.convert_to_tensor(x), mask=ops.convert_to_tensor(mask))

    assert np.allclose(pyg_x.detach().numpy(), ops.convert_to_numpy(k3_x), atol=1e-5)
    assert np.allclose(pyg_s.detach().numpy(), ops.convert_to_numpy(k3_s), atol=1e-5)

    pyg_loss = PyGMemPooling.kl_loss(pyg_s)
    k3_loss = MemPooling.kl_loss(k3_s)
    assert np.allclose(pyg_loss.detach().numpy(), ops.convert_to_numpy(k3_loss), atol=1e-5)


def test_reference_knn_indices():
    np.random.seed(42)
    lhs = np.random.randn(10, 16).astype(np.float32)
    rhs = np.random.randn(50, 16).astype(np.float32)

    k3_l2 = L2KNNIndex(ops.convert_to_tensor(rhs))
    out_l2 = k3_l2.search(ops.convert_to_tensor(lhs), k=3)
    assert ops.shape(out_l2.score) == (10, 3)
    assert ops.shape(out_l2.index) == (10, 3)

    k3_mips = MIPSKNNIndex(ops.convert_to_tensor(rhs))
    out_mips = k3_mips.search(ops.convert_to_tensor(lhs), k=3)
    assert ops.shape(out_mips.score) == (10, 3)
    assert ops.shape(out_mips.index) == (10, 3)


def test_reference_pool_x():
    x = np.random.randn(5, 16).astype(np.float32)
    cluster = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    batch = np.array([0, 0, 0, 0, 0], dtype=np.int64)

    pyg_max, pyg_b = pyg_max_pool_x(torch.from_numpy(cluster), torch.from_numpy(x), torch.from_numpy(batch))
    k3_max, k3_b = max_pool_x(ops.convert_to_tensor(cluster), ops.convert_to_tensor(x), ops.convert_to_tensor(batch))
    assert np.allclose(pyg_max.numpy(), ops.convert_to_numpy(k3_max), atol=1e-5)
    assert np.array_equal(pyg_b.numpy(), ops.convert_to_numpy(k3_b))

    pyg_avg, pyg_b2 = pyg_avg_pool_x(torch.from_numpy(cluster), torch.from_numpy(x), torch.from_numpy(batch))
    k3_avg, k3_b2 = avg_pool_x(ops.convert_to_tensor(cluster), ops.convert_to_tensor(x), ops.convert_to_tensor(batch))
    assert np.allclose(pyg_avg.numpy(), ops.convert_to_numpy(k3_avg), atol=1e-5)
    assert np.array_equal(pyg_b2.numpy(), ops.convert_to_numpy(k3_b2))


def test_reference_pool_helpers():
    cluster = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    edge_index = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=np.int64)
    edge_attr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    pyg_ei, pyg_ea = pyg_pool_edge(torch.from_numpy(cluster), torch.from_numpy(edge_index), torch.from_numpy(edge_attr))
    k3_ei, k3_ea = pool_edge(ops.convert_to_tensor(cluster), ops.convert_to_tensor(edge_index), ops.convert_to_tensor(edge_attr))

    assert np.array_equal(pyg_ei.numpy(), ops.convert_to_numpy(k3_ei))
    assert np.allclose(pyg_ea.numpy(), ops.convert_to_numpy(k3_ea), atol=1e-5)


def test_reference_sag_pooling():
    from torch_geometric.nn import SAGPooling as PyGSAGPooling
    from k3_node.layers.pool.sag_pool import SAGPooling

    torch.manual_seed(42)
    in_channels = 16
    pyg_sag = PyGSAGPooling(in_channels, ratio=0.5)
    k3_sag = SAGPooling(in_channels, ratio=0.5)

    k3_sag.gnn.lin_rel.build((None, in_channels))
    k3_sag.gnn.lin_root.build((None, in_channels))
    k3_sag.select.build((None, 1))

    with torch.no_grad():
        k3_sag.gnn.lin_rel.set_weights([
            pyg_sag.gnn.lin_rel.weight.detach().numpy().T,
            pyg_sag.gnn.lin_rel.bias.detach().numpy(),
        ])
        k3_sag.gnn.lin_root.set_weights([pyg_sag.gnn.lin_root.weight.detach().numpy().T])
        k3_sag.select.weight.assign(ops.convert_to_tensor(pyg_sag.select.weight.detach().numpy()))

    x = torch.randn(4, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    pyg_out = pyg_sag(x, edge_index)
    k3_out = k3_sag(ops.convert_to_tensor(x.numpy()), ops.convert_to_tensor(edge_index.numpy()))

    assert np.allclose(pyg_out[0].detach().numpy(), ops.convert_to_numpy(k3_out[0]), atol=1e-5)
    assert np.array_equal(pyg_out[4].detach().numpy(), ops.convert_to_numpy(k3_out[4]))

