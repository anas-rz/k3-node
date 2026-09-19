from typing import Union

from keras import ops


def bro(x, batch, p: Union[int, float] = 2):
    r"""The Batch Representation Orthogonality penalty from the `"Improving
    Molecular Graph Neural Network Explainability with Orthonormalization
    and Induced Sparsity" <https://arxiv.org/abs/2105.04854>`_ paper.

    Computes a regularization for each graph representation in a mini-batch
    according to

    .. math::
        \mathcal{L}_{\textrm{BRO}}^\mathrm{graph} =
          || \mathbf{HH}^T - \mathbf{I}||_p

    and returns an average over all graphs in the batch.
    """
    batch_np = ops.convert_to_numpy(batch)
    unique_ids = sorted(set(batch_np.tolist()))

    total = 0.0
    for graph_id in unique_ids:
        mask = ops.equal(batch, graph_id)
        where_mask = ops.where(mask)
        idx = where_mask[0] if isinstance(where_mask, (list, tuple)) else where_mask
        idx = ops.reshape(idx, (-1,))

        x_g = ops.take(x, idx, axis=0)
        n = ops.shape(x_g)[0]
        eye = ops.eye(n, dtype=x_g.dtype)
        diff = ops.matmul(x_g, ops.transpose(x_g)) - eye

        total = total + ops.power(ops.sum(ops.power(ops.abs(diff), p)), 1.0 / p)

    return total / len(unique_ids)
