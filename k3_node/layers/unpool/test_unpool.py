from keras import ops

from k3_node.layers.unpool import knn_interpolate


def test_knn_interpolate():
    x = ops.convert_to_tensor(
        [[1.0], [10.0], [100.0], [-1.0], [-10.0], [-100.0]], dtype="float32"
    )
    pos_x = ops.convert_to_tensor([
        [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
        [-2.0, 0.0], [0.0, 0.0], [2.0, 0.0],
    ], dtype="float32")
    pos_y = ops.convert_to_tensor([
        [-1.0, -1.0], [1.0, 1.0], [-2.0, -2.0], [2.0, 2.0],
    ], dtype="float32")
    batch_x = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")
    batch_y = ops.convert_to_tensor([0, 0, 1, 1], dtype="int64")

    y = knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=2)
    assert ops.shape(y) == (4, 1)
    assert ops.convert_to_numpy(y).tolist() == [[4.0], [70.0], [-4.0], [-70.0]]


def test_knn_interpolate_no_batch():
    x = ops.convert_to_tensor([[1.0], [10.0], [100.0]], dtype="float32")
    pos_x = ops.convert_to_tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype="float32")
    pos_y = ops.convert_to_tensor([[-1.0, -1.0], [1.0, 1.0]], dtype="float32")

    y = knn_interpolate(x, pos_x, pos_y, k=2)
    assert ops.shape(y) == (2, 1)
