from keras import ops, random
from k3_node.models import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    MeanAggregator,
    TimeEncoder,
    LastNeighborLoader,
)


def test_time_encoder():
    enc = TimeEncoder(out_channels=16)
    t = ops.convert_to_tensor([1.0, 2.0, 3.0], dtype="float32")
    out = enc(t)
    assert out.shape == (3, 16)


def test_aggregators():
    msg = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    index = ops.convert_to_tensor([0, 0], dtype="int64")
    t = ops.convert_to_tensor([1.0, 2.0], dtype="float32")

    last_aggr = LastAggregator()
    out_last = last_aggr(msg, index, t, dim_size=1)
    assert out_last.shape == (1, 2)
    assert float(out_last[0, 0]) == 3.0

    mean_aggr = MeanAggregator()
    out_mean = mean_aggr(msg, index, t, dim_size=1)
    assert out_mean.shape == (1, 2)
    assert float(out_mean[0, 0]) == 2.0


def test_last_neighbor_loader():
    loader = LastNeighborLoader(num_nodes=4, size=2)
    src = ops.convert_to_tensor([0, 1], dtype="int64")
    dst = ops.convert_to_tensor([1, 2], dtype="int64")
    loader.insert(src, dst)

    n_id = ops.convert_to_tensor([0, 1], dtype="int64")
    unique_nodes, edge_index, e_ids = loader(n_id)
    assert len(unique_nodes.shape) == 1
    assert edge_index.shape[0] == 2


def test_tgn_memory():
    memory_dim = 16
    raw_msg_dim = 8
    time_dim = 16
    msg_module = IdentityMessage(raw_msg_dim, memory_dim, time_dim)
    aggr_module = LastAggregator()

    memory = TGNMemory(
        num_nodes=5,
        raw_msg_dim=raw_msg_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        message_module=msg_module,
        aggregator_module=aggr_module,
    )

    n_id = ops.convert_to_tensor([0, 1, 2], dtype="int64")
    mem, last_update = memory(n_id)
    assert mem.shape == (3, memory_dim)
    assert last_update.shape == (3,)

    # Update state
    src = ops.convert_to_tensor([0, 1], dtype="int64")
    dst = ops.convert_to_tensor([1, 2], dtype="int64")
    t = ops.convert_to_tensor([1.0, 2.0], dtype="float32")
    raw_msg = random.normal((2, raw_msg_dim))

    memory.update_state(src, dst, t, raw_msg)
    mem_updated, last_updated = memory(n_id)
    assert mem_updated.shape == (3, memory_dim)

