from keras import ops, random
from k3_node.models import RENet


def test_renet():
    model = RENet(num_nodes=5, num_rels=4, hidden_channels=16, seq_len=3)

    sub = ops.convert_to_tensor([0, 1], dtype="int64")
    rel = ops.convert_to_tensor([0, 1], dtype="int64")
    obj = ops.convert_to_tensor([2, 3], dtype="int64")

    h_sub = ops.convert_to_tensor([0, 1, 2], dtype="int64")
    h_sub_t = ops.convert_to_tensor([0, 1, 0], dtype="int64")
    h_sub_batch = ops.convert_to_tensor([0, 0, 1], dtype="int64")

    h_obj = ops.convert_to_tensor([1, 2, 3], dtype="int64")
    h_obj_t = ops.convert_to_tensor([1, 2, 0], dtype="int64")
    h_obj_batch = ops.convert_to_tensor([0, 0, 1], dtype="int64")

    log_prob_obj, log_prob_sub = model(
        sub, rel, obj,
        h_sub, h_sub_t, h_sub_batch,
        h_obj, h_obj_t, h_obj_batch,
    )
    assert log_prob_obj.shape == (2, 5)
    assert log_prob_sub.shape == (2, 5)

    # Test metrics
    y = ops.convert_to_tensor([2, 3], dtype="int64")
    metrics = model.test(log_prob_obj, y)
    assert metrics.shape == (4,)

