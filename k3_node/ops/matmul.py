# ported from spektral

from keras import ops as k_ops


def dot(a, b):
    a_ndim = len(k_ops.shape(a))
    b_ndim = len(k_ops.shape(b))
    assert a_ndim == b_ndim, "Expected equal ranks, got {} and {}" "".format(
        a_ndim, b_ndim
    )
    return k_ops.matmul(a, b)


def mixed_mode_dot(a, b):
    a_shp = k_ops.shape(a)
    b_shp = k_ops.shape(b)

    b_t = k_ops.transpose(b, (1, 2, 0))
    b_t = k_ops.reshape(b_t, k_ops.stack((b_shp[1], -1)))
    output = dot(a, b_t)
    output = k_ops.reshape(output, k_ops.stack((a_shp[0], b_shp[2], -1)))
    output = k_ops.transpose(output, (2, 0, 1))

    return output


def modal_dot(a, b, transpose_a=False, transpose_b=False):
    a_ndim = len(k_ops.shape(a))
    b_ndim = len(k_ops.shape(b))
    assert a_ndim in (2, 3), "Expected a of rank 2 or 3, got {}".format(a_ndim)
    assert b_ndim in (2, 3), "Expected b of rank 2 or 3, got {}".format(b_ndim)

    if transpose_a:
        perm = (1, 0) if a_ndim == 2 else (0, 2, 1)
        a = k_ops.transpose(a, perm)
    if transpose_b:
        perm = (1, 0) if b_ndim == 2 else (0, 2, 1)
        b = k_ops.transpose(b, perm)
    if a_ndim == b_ndim:
        # ...ij,...jk->...ik
        return dot(a, b)
    elif a_ndim == 2:
        # ij,bjk->bik
        return mixed_mode_dot(a, b)
    else:  # a_ndim == 3
        # bij,jk->bik
        # Immediately fallback to standard dense matmul, no need to reshape
        return k_ops.matmul(a, b)
