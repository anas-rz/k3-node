from keras import ops as k_ops, backend


def degrees(A):
    return k_ops.sum(A, axis=-1)


def normalize_A(A):
    D = degrees(A)
    D = k_ops.sqrt(D)[:, None] + backend.epsilon()
    perm = (0, 2, 1) if len(k_ops.shape(A)) == 3 else (1, 0)
    output = (A / D) / k_ops.transpose(D, perm=perm)
    return output
