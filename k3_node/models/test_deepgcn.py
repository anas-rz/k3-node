import pytest
from keras import ops

from k3_node.layers.conv import GENConv
from k3_node.layers.norm import LayerNorm
from k3_node.models import DeepGCNLayer


@pytest.mark.parametrize("block_tuple", [("res+", 1), ("res", 1), ("dense", 2), ("plain", 1)])
def test_deepgcn(block_tuple):
    block, expansion = block_tuple
    x = ops.convert_to_tensor([[1.0] * 8] * 3, dtype="float32")
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype="int64")
    conv = GENConv(8, 8)
    norm = LayerNorm(8)
    act = ops.relu
    layer = DeepGCNLayer(conv, norm, act, block=block)
    assert str(layer) == f"DeepGCNLayer(block={block})"

    out = layer(x, edge_index)
    assert ops.shape(out) == (3, 8 * expansion)
