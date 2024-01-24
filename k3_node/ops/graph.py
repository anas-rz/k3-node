# ported from spektral
from keras import ops as k_ops, backend

from k3_node.utils import *


def degrees(A):
    return k_ops.sum(A, axis=-1)


def normalize_A(A):
    D = degrees(A)
    D = k_ops.sqrt(D)[:, None] + backend.epsilon()
    perm = (0, 2, 1) if len(k_ops.shape(A)) == 3 else (1, 0)
    output = (A / D) / k_ops.transpose(D, perm=perm)
    return output


def get_source_target(a):
    if backend.backend() == "tensorflow": 
        if isinstance(a, tf.sparse.SparseTensor):
            return a.indices[:, 0], a.indices[:, 1]
        else:
            k_ops.where(a != 0)
    else:
        return k_ops.where(a != 0)