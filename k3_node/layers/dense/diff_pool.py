from typing import Optional, Tuple
from keras import ops


def dense_diff_pool(
    x,
    adj,
    s,
    mask: Optional[any] = None,
    normalize: bool = True,
) -> Tuple[any, any, any, any]:
    r"""The differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper.

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
        normalize: If set to False, link prediction loss is not divided by total elements.
    """
    if len(ops.shape(x)) == 2:
        x = ops.expand_dims(x, axis=0)
    if len(ops.shape(adj)) == 2:
        adj = ops.expand_dims(adj, axis=0)
    if len(ops.shape(s)) == 2:
        s = ops.expand_dims(s, axis=0)

    batch_size = ops.shape(x)[0]
    num_nodes = ops.shape(x)[1]

    s = ops.softmax(s, axis=-1)

    if mask is not None:
        mask_m = ops.cast(ops.reshape(mask, (batch_size, num_nodes, 1)), x.dtype)
        x = x * mask_m
        s = s * mask_m

    s_t = ops.transpose(s, (0, 2, 1))

    out = ops.matmul(s_t, x)
    out_adj = ops.matmul(ops.matmul(s_t, adj), s)

    link = adj - ops.matmul(s, s_t)
    link_loss = ops.sqrt(ops.sum(ops.power(link, 2)))
    if normalize:
        numel = ops.cast(ops.prod(ops.shape(adj)), link_loss.dtype)
        link_loss = link_loss / numel

    ent_loss = ops.mean(ops.sum(-s * ops.log(s + 1e-15), axis=-1))

    return out, out_adj, link_loss, ent_loss

