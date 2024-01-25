import pytest
from keras import ops, random

from k3_node.layers import PerformerAttention
from k3_node.utils import edge_index_to_adjacency_matrix


def test_performer_attention():
    x = random.normal((1, 4, 16))
    mask = ops.ones([1, 4])
    attn = PerformerAttention(channels=16, heads=4)
    out = attn(x, mask)
    assert out.shape == (1, 4, 16)
