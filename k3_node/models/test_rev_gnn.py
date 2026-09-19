from keras import ops, random
from k3_node.layers.conv import GCNConv
from k3_node.models import GroupAddRev


def test_group_add_rev():
    conv1 = GCNConv(16, 16)
    conv2 = GCNConv(16, 16)
    model = GroupAddRev([conv1, conv2])

    x = random.normal((6, 32))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 0, 4, 5],
    ], dtype="int64")

    out = model(x, edge_index=edge_index)
    assert out.shape == (6, 32)

    inv = model.inverse(out, edge_index=edge_index)
    assert inv.shape == (6, 32)
    assert ops.allclose(x, inv, atol=1e-5)

