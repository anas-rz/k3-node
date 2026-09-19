from keras import ops

from k3_node.layers.conv.utils import scatter
from k3_node.layers.pool.knn import knn


def knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k: int = 3, num_workers: int = 1):
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.

    For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
    interpolated features :math:`\mathbf{f}(y)` are given by

    .. math::
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
        \mathbf{p}(x_i))^2}

    and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
    to :math:`y`.

    Args:
        x: Node feature matrix :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        pos_x: Node position matrix :math:`\in \mathbb{R}^{N \times d}`.
        pos_y: Upsampled node position matrix :math:`\in \mathbb{R}^{M \times d}`.
        batch_x: Batch vector assigning each node from :math:`\mathbf{X}` to
            a specific example. (default: :obj:`None`)
        batch_y: Batch vector assigning each node from :math:`\mathbf{Y}` to
            a specific example. (default: :obj:`None`)
        k (int, optional): Number of neighbors. (default: :obj:`3`)
        num_workers (int, optional): Unused, kept for API compatibility.
    """
    assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers)
    y_idx, x_idx = assign_index[0], assign_index[1]

    diff = ops.take(pos_x, x_idx, axis=0) - ops.take(pos_y, y_idx, axis=0)
    squared_distance = ops.sum(diff * diff, axis=-1, keepdims=True)
    weights = 1.0 / ops.maximum(squared_distance, 1e-16)

    num_y = ops.shape(pos_y)[0]
    y = scatter(ops.take(x, x_idx, axis=0) * weights, y_idx, dim=0, dim_size=num_y, reduce="sum")
    y = y / scatter(weights, y_idx, dim=0, dim_size=num_y, reduce="sum")

    return y
