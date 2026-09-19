from keras import ops

EPS = 1.1920929e-07  # float32 machine epsilon, matches torch.finfo().eps


def gini(w):
    r"""The Gini coefficient from the `"Improving Molecular Graph Neural
    Network Explainability with Orthonormalization and Induced Sparsity"
    <https://arxiv.org/abs/2105.04854>`_ paper.

    Computes a regularization penalty :math:`\in [0, 1]` for each row of a
    matrix according to

    .. math::
        \mathcal{L}_\textrm{Gini}^i = \sum_j^n \sum_{j'}^n \frac{|w_{ij}
         - w_{ij'}|}{2 (n^2 - n)\bar{w_i}}

    and returns an average over all rows.

    Args:
        w: A two-dimensional tensor.
    """
    num_rows = ops.shape(w)[0]
    n = ops.shape(w)[-1]

    total = 0.0
    for i in range(num_rows):
        row = ops.take(w, i, axis=0)
        t = ops.tile(ops.reshape(row, (1, -1)), (n, 1))
        u = ops.sum(ops.abs(t - ops.transpose(t))) / (
            2 * (n**2 - n) * ops.mean(ops.abs(row)) + EPS
        )
        total = total + u

    return total / num_rows
