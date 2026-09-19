import numpy as np
from keras import ops

from k3_node.layers.pool import (
    ASAPooling,
    ApproxL2KNNIndex,
    ApproxMIPSKNNIndex,
    ClusterPooling,
    EdgePooling,
    FilterEdges,
    KNNIndex,
    L2KNNIndex,
    LEConv,
    MIPSKNNIndex,
    MemPooling,
    PANPooling,
    SAGPooling,
    SelectTopK,
    TopKPooling,
    approx_knn,
    approx_knn_graph,
    avg_pool,
    avg_pool_neighbor_x,
    avg_pool_x,
    consecutive_cluster,
    decimation_indices,
    filter_adj,
    fps,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    graclus,
    knn,
    knn_graph,
    max_pool,
    max_pool_neighbor_x,
    max_pool_x,
    nearest,
    pool_batch,
    pool_edge,
    pool_pos,
    radius,
    radius_graph,
    topk,
    voxel_grid,
)
from k3_node.layers.pool.select.base import SelectOutput


def test_global_pooling():
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ], dtype="float32")
    batch = ops.convert_to_tensor([0, 0, 1, 1], dtype="int32")

    add = global_add_pool(x, batch)
    assert np.allclose(ops.convert_to_numpy(add), [[4.0, 6.0], [12.0, 14.0]])

    mean = global_mean_pool(x, batch)
    assert np.allclose(ops.convert_to_numpy(mean), [[2.0, 3.0], [6.0, 7.0]])

    max_p = global_max_pool(x, batch)
    assert np.allclose(ops.convert_to_numpy(max_p), [[3.0, 4.0], [7.0, 8.0]])


def test_consecutive_cluster():
    src = ops.convert_to_tensor([1, 4, 4, 7, 7, 7, 10], dtype="int32")
    out, perm = consecutive_cluster(src)
    assert np.array_equal(ops.convert_to_numpy(out), [0, 1, 1, 2, 2, 2, 3])
    assert ops.shape(perm)[0] == 4


def test_filter_adj_and_layer():
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 0],
        [1, 2, 3, 0, 2],
    ], dtype="int32")
    edge_attr = ops.convert_to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    node_index = ops.convert_to_tensor([0, 1, 2], dtype="int32")

    new_edge_index, new_edge_attr = filter_adj(edge_index, edge_attr, node_index=node_index)
    assert ops.shape(new_edge_index)[0] == 2
    assert ops.shape(new_edge_index)[1] == 3  # (0,1), (1,2), (0,2)

    # FilterEdges
    layer = FilterEdges()
    select_out = SelectOutput(
        node_index=node_index,
        num_nodes=4,
        cluster_index=ops.arange(3, dtype="int32"),
        num_clusters=3,
        weight=ops.ones((3,), dtype="float32"),
    )
    batch = ops.zeros((4,), dtype="int32")
    connect_out = layer(select_out, edge_index, edge_attr, batch)
    assert ops.shape(connect_out.edge_index)[1] == 3
    assert ops.shape(connect_out.batch)[0] == 3


def test_topk_and_select_topk():
    x = ops.convert_to_tensor([2.0, 1.0, 5.0, 3.0, 8.0, 4.0], dtype="float32")
    batch = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int32")

    # Ratio test
    perm = topk(x, ratio=0.5, batch=batch)
    perm_np = ops.convert_to_numpy(perm)
    assert 2 in perm_np[:2]  # from first graph (5.0, 2.0)
    assert 4 in perm_np[1:]  # from second graph (8.0, 4.0)

    # Min score test
    perm_min = topk(x, ratio=None, batch=batch, min_score=3.5)
    perm_min_np = ops.convert_to_numpy(perm_min)
    assert 2 in perm_min_np  # 5.0
    assert 4 in perm_min_np  # 8.0
    assert 5 in perm_min_np  # 4.0

    # SelectTopK layer
    layer = SelectTopK(in_channels=4, ratio=0.5)
    feat = ops.ones((6, 4), dtype="float32")
    out = layer(feat, batch)
    assert isinstance(out, SelectOutput)
    assert ops.shape(out.node_index)[0] == 4  # 2 per graph


def test_topk_pooling():
    in_channels = 8
    layer = TopKPooling(in_channels, ratio=0.5)

    x = ops.ones((6, in_channels), dtype="float32")
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 0, 4, 5, 3],
    ], dtype="int32")
    batch = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int32")

    out_x, out_edge_index, out_edge_attr, out_batch, perm, score = layer(
        x, edge_index, batch=batch
    )
    assert ops.shape(out_x)[0] == 4  # 2 nodes per graph
    assert ops.shape(out_x)[1] == in_channels
    assert ops.shape(perm)[0] == 4
    assert ops.shape(out_batch)[0] == 4


def test_sag_pooling():
    in_channels = 8
    layer = SAGPooling(in_channels, ratio=0.5)

    x = ops.ones((4, in_channels), dtype="float32")
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 0],
        [1, 2, 3, 0, 2],
    ], dtype="int32")

    out_x, out_edge_index, out_edge_attr, out_batch, perm, score = layer(x, edge_index)
    assert ops.shape(out_x)[0] == 2
    assert ops.shape(out_x)[1] == in_channels
    assert ops.shape(perm)[0] == 2


def test_edge_pooling():
    in_channels = 8
    layer = EdgePooling(in_channels)

    x = ops.ones((4, in_channels), dtype="float32")
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ], dtype="int32")
    batch = ops.zeros((4,), dtype="int32")

    new_x, new_edge_index, new_batch, unpool_info = layer(x, edge_index, batch)
    assert ops.shape(new_x)[0] <= 4
    assert ops.shape(new_x)[1] == in_channels

    # Unpool
    unpooled_x, _, _ = layer.unpool(new_x, unpool_info)
    assert ops.shape(unpooled_x)[0] == 4
    assert ops.shape(unpooled_x)[1] == in_channels


def test_cluster_pooling():
    in_channels = 8
    layer = ClusterPooling(in_channels)

    x = ops.ones((4, in_channels), dtype="float32")
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ], dtype="int32")
    batch = ops.zeros((4,), dtype="int32")

    new_x, new_edge_index, new_batch, unpool_info = layer(x, edge_index, batch)
    assert unpool_info.edge_index is not None
    assert unpool_info.cluster is not None
    assert unpool_info.batch is not None


def test_mem_pooling():
    in_channels, out_channels, heads, num_clusters = 8, 16, 2, 4
    layer = MemPooling(in_channels, out_channels, heads=heads, num_clusters=num_clusters)

    x = ops.ones((2, 5, in_channels), dtype="float32")
    out_x, S = layer(x)
    assert ops.shape(out_x) == (2, num_clusters, out_channels)
    assert ops.shape(S) == (2, 5, num_clusters)

    loss = MemPooling.kl_loss(S)
    assert ops.shape(loss) == ()


def test_asap_pooling():
    in_channels = 8
    leconv = LEConv(in_channels, in_channels)
    x = ops.ones((4, in_channels), dtype="float32")
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 0]], dtype="int32")
    le_out = leconv(x, edge_index)
    assert ops.shape(le_out) == (4, in_channels)

    asap = ASAPooling(in_channels, ratio=0.5)
    out_x, out_edge_index, _, _, perm = asap(x, edge_index)
    assert ops.shape(out_x)[0] == 2
    assert ops.shape(out_x)[1] == in_channels


def test_pan_pooling():
    in_channels = 8
    pan = PANPooling(in_channels, ratio=0.5)
    x = ops.ones((4, in_channels), dtype="float32")
    M = ops.eye(4, dtype="float32")

    out_x, out_edge_index, _, _, perm, score = pan(x, M)
    assert ops.shape(out_x)[0] == 2
    assert ops.shape(out_x)[1] == in_channels
    assert ops.shape(out_edge_index)[0] == 2


def test_max_and_avg_pool():
    cluster = ops.convert_to_tensor([0, 0, 1, 1], dtype="int32")
    x = ops.convert_to_tensor([
        [1.0, 5.0],
        [3.0, 2.0],
        [4.0, 8.0],
        [6.0, 7.0],
    ], dtype="float32")
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ], dtype="int32")
    batch = ops.zeros((4,), dtype="int32")

    # max_pool_x & avg_pool_x
    mx, mb = max_pool_x(cluster, x, batch)
    assert np.allclose(ops.convert_to_numpy(mx), [[3.0, 5.0], [6.0, 8.0]])
    ax, ab = avg_pool_x(cluster, x, batch)
    assert np.allclose(ops.convert_to_numpy(ax), [[2.0, 3.5], [5.0, 7.5]])

    # neighbor_x with raw tensors
    mn_x = max_pool_neighbor_x(x, edge_index=edge_index)
    assert ops.shape(mn_x) == (4, 2)
    an_x = avg_pool_neighbor_x(x, edge_index=edge_index)
    assert ops.shape(an_x) == (4, 2)

    # max_pool & avg_pool (on graph)
    m_x, m_edge, m_batch = max_pool(cluster, x, edge_index, batch=batch)
    assert ops.shape(m_x)[0] == 2
    a_x, a_edge, a_batch = avg_pool(cluster, x, edge_index, batch=batch)
    assert ops.shape(a_x)[0] == 2


def test_pool_helpers():
    cluster = ops.convert_to_tensor([0, 0, 1, 1], dtype="int32")
    perm = ops.convert_to_tensor([0, 2], dtype="int32")
    batch = ops.convert_to_tensor([0, 0, 0, 0], dtype="int32")
    pos = ops.convert_to_tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype="float32")
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 3]], dtype="int32")
    edge_attr = ops.convert_to_tensor([[1.0], [2.0], [3.0]], dtype="float32")

    p_edge, p_attr = pool_edge(cluster, edge_index, edge_attr)
    assert ops.shape(p_edge)[0] == 2
    assert p_attr is not None

    p_b = pool_batch(perm, batch)
    assert ops.shape(p_b)[0] == 2

    p_p = pool_pos(cluster, pos)
    assert ops.shape(p_p)[0] == 2


def test_voxel_grid():
    pos = ops.convert_to_tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.2, 1.2],
        [2.1, 2.1],
    ], dtype="float32")
    size = ops.convert_to_tensor([1.0, 1.0], dtype="float32")

    c = voxel_grid(pos, size)
    c_np = ops.convert_to_numpy(c)
    assert c_np[0] == c_np[1]
    assert c_np[0] != c_np[2]
    assert c_np[2] != c_np[3]


def test_graclus():
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ], dtype="int32")
    c = graclus(edge_index, num_nodes=4)
    c_np = ops.convert_to_numpy(c)
    assert len(np.unique(c_np)) == 2


def test_decimation():
    ptr = ops.convert_to_tensor([0, 4, 10], dtype="int32")
    dec_idx, dec_ptr = decimation_indices(ptr, decimation_factor=2)
    assert ops.shape(dec_idx)[0] == 5  # 2 from first, 3 from second
    assert ops.shape(dec_ptr)[0] == 3


def test_knn_and_indices():
    x = ops.convert_to_tensor([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ], dtype="float32")
    y = ops.convert_to_tensor([
        [0.0, 0.0],
        [1.0, 1.0],
    ], dtype="float32")

    # KNNIndex / L2KNNIndex
    idx_layer = L2KNNIndex(x)
    out = idx_layer.search(y, k=2)
    assert ops.shape(out.score) == (2, 2)
    assert ops.shape(out.index) == (2, 2)

    # MIPSKNNIndex
    mips_layer = MIPSKNNIndex(x)
    mips_out = mips_layer.search(y, k=2)
    assert ops.shape(mips_out.score) == (2, 2)
    assert ops.shape(mips_out.index) == (2, 2)

    # knn function
    res = knn(x, y, k=2)
    assert ops.shape(res) == (2, 4)

    # knn_graph
    g = knn_graph(x, k=2)
    assert ops.shape(g)[0] == 2

    # approx knn
    approx_res = approx_knn(x, y, k=2)
    assert ops.shape(approx_res) == (2, 4)
    approx_g = approx_knn_graph(x, k=2)
    assert ops.shape(approx_g)[0] == 2


def test_point_cloud():
    pos = ops.convert_to_tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
    ], dtype="float32")

    # fps
    fps_idx = fps(pos, ratio=0.5)
    assert ops.shape(fps_idx)[0] == 2

    # radius
    rad_edge = radius(pos, pos, r=1.0)
    assert ops.shape(rad_edge)[0] == 2
    assert ops.shape(rad_edge)[1] >= 4  # at least self loops + close pairs

    # radius_graph
    rad_g = radius_graph(pos, r=1.0)
    assert ops.shape(rad_g)[0] == 2

    # nearest
    near = nearest(pos, pos)
    assert ops.shape(near)[0] == 4
