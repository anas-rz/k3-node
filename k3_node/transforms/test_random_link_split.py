import numpy as np
from keras import ops

from k3_node.data import Data
from k3_node.transforms import RandomLinkSplit


def _graph():
    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 60, size=(2, 200))
    return Data(x=rng.random((60, 4)).astype("float32"), edge_index=edge_index, num_nodes=60)


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


def test_edge_labels_include_negatives():
    train, val, test = RandomLinkSplit(num_val=0.1, num_test=0.2)(_graph())
    for split, n_pos in ((train, 140), (val, 20), (test, 40)):
        label = _np(split.edge_label)
        assert label.dtype == np.float32
        assert label.sum() == n_pos and (label == 0).sum() == n_pos  # one negative per positive
        assert _np(split.edge_label_index).shape == (2, 2 * n_pos)
    # message passing edges: training edges for train/val, plus validation edges for test
    assert _np(train.edge_index).shape[1] == 140 and _np(test.edge_index).shape[1] == 160


def test_no_train_negatives_and_ratio():
    train, val, _ = RandomLinkSplit(num_val=0.1, num_test=0.2, add_negative_train_samples=False,
                                    neg_sampling_ratio=2.0)(_graph())
    assert (_np(train.edge_label) == 0).sum() == 0
    assert (_np(val.edge_label) == 0).sum() == 40


def test_split_labels_and_undirected():
    edge_index = _np(_graph().edge_index)
    edge_index = np.unique(np.sort(edge_index, axis=0), axis=1)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    data = Data(edge_index=np.concatenate([edge_index, edge_index[::-1]], axis=1), num_nodes=60)
    train, val, test = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True)(data)
    n = edge_index.shape[1]
    assert _np(val.pos_edge_label_index).shape[1] == int(0.1 * n)
    assert _np(val.neg_edge_label_index).shape[1] == int(0.1 * n)
    assert _np(train.edge_index).shape[1] == 2 * _np(train.pos_edge_label_index).shape[1]
