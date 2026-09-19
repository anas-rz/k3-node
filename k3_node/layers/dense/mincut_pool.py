from typing import Optional, Tuple
from keras import ops


def dense_mincut_pool(
    x,
    adj,
    s,
    mask: Optional[any] = None,
    temp: float = 1.0,
) -> Tuple[any, any, any, any]:
    r"""The MinCut pooling operator from the `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    paper.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    Args:
        x: Node feature tensor [B, N, F] or [N, F].
        adj: Adjacency tensor [B, N, N] or [N, N].
        s: Assignment tensor [B, N, C] or [N, C].
        mask: Mask tensor [B, N] indicating valid nodes. (default: None)
        temp: Temperature parameter for softmax function. (default: 1.0)
    """
    if len(ops.shape(x)) == 2:
        x = ops.expand_dims(x, axis=0)
    if len(ops.shape(adj)) == 2:
        adj = ops.expand_dims(adj, axis=0)
    if len(ops.shape(s)) == 2:
        s = ops.expand_dims(s, axis=0)

    batch_size = ops.shape(x)[0]
    num_nodes = ops.shape(x)[1]
    k = ops.shape(s)[-1]

    if temp != 1.0:
        s = s / temp
    s = ops.softmax(s, axis=-1)

    if mask is not None:
        mask_m = ops.cast(ops.reshape(mask, (batch_size, num_nodes, 1)), x.dtype)
        x = x * mask_m
        s = s * mask_m

    s_t = ops.transpose(s, (0, 2, 1))

    out = ops.matmul(s_t, x)
    out_adj = ops.matmul(ops.matmul(s_t, adj), s)

    # MinCut regularization
    mincut_num = ops.sum(ops.diagonal(out_adj, axis1=1, axis2=2), axis=-1)
    d_flat = ops.sum(adj, axis=-1)
    d = ops.expand_dims(d_flat, axis=-1) * ops.expand_dims(ops.eye(num_nodes, dtype=d_flat.dtype), axis=0)
    s_d_s = ops.matmul(ops.matmul(s_t, d), s)
    mincut_den = ops.sum(ops.diagonal(s_d_s, axis1=1, axis2=2), axis=-1)
    mincut_loss = ops.mean(-(mincut_num / mincut_den))

    # Orthogonality regularization
    ss = ops.matmul(s_t, s)
    norm_ss = ops.sqrt(ops.sum(ops.power(ss, 2), axis=(-1, -2), keepdims=True))
    i_s = ops.eye(k, dtype=ss.dtype)
    norm_is = ops.sqrt(ops.cast(k, ss.dtype))
    diff = (ss / norm_ss) - (ops.expand_dims(i_s, axis=0) / norm_is)
    ortho_loss = ops.mean(ops.sqrt(ops.sum(ops.power(diff, 2), axis=(-1, -2))))

    # Fix and normalize coarsened adjacency matrix
    eye_k = ops.expand_dims(ops.eye(k, dtype=out_adj.dtype), axis=0)
    out_adj = out_adj * (1.0 - eye_k)
    d_out = ops.sum(out_adj, axis=-1, keepdims=True)
    d_sqrt = ops.sqrt(d_out) + 1e-15
    out_adj = (out_adj / d_sqrt) / ops.transpose(d_sqrt, (0, 2, 1))

    return out, out_adj, mincut_loss, ortho_loss

