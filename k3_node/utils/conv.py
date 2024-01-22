import numpy as np
import warnings, copy


def degree_matrix(A):
    degrees = np.array(A.sum(1)).flatten()
    D = np.diag(degrees)
    return D


def degree_power(A, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


def laplacian(A):
    return degree_matrix(A) - A


def normalized_laplacian(A, symmetric=True):
    I = np.eye(A.shape[-1], dtype=A.dtype)
    normalized_adj = normalized_adjacency(A, symmetric=symmetric)
    return I - normalized_adj


def gcn_filter(A, symmetric=True):
    out = copy.deepcopy(A)
    if isinstance(A, list) or (isinstance(A, np.ndarray) and A.ndim == 3):
        for i in range(len(A)):
            out[i] = A[i]
            out[i][np.diag_indices_from(out[i])] += 1
            out[i] = normalized_adjacency(out[i], symmetric=symmetric)
    else:
        if hasattr(out, "tocsr"):
            out = out.tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[np.diag_indices_from(out)] += 1
        out = normalized_adjacency(out, symmetric=symmetric)

    return out
