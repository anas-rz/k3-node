"""LPFormer and its personalized PageRank against PyG."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
import scipy.sparse as sp  # noqa: E402
from keras import ops  # noqa: E402
from torch_geometric.nn.models import LPFormer as PyGLPFormer  # noqa: E402
from torch_geometric.utils.ppr import _get_ppr  # noqa: E402

from k3_node.models import LPFormer  # noqa: E402
from k3_node.models.lpformer import get_ppr  # noqa: E402


def _graph(n=40):
    rng = np.random.default_rng(0)
    ei = rng.integers(0, n, (2, 120))
    ei = np.unique(np.concatenate([ei, ei[::-1]], 1), axis=1)
    return rng, np.ascontiguousarray(ei[:, ei[0] != ei[1]])


def test_ppr_matches_pyg_algorithm():
    pytest.importorskip("numba")
    _, ei = _graph()
    n = 40
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(ei[0], minlength=n))])
    js, vals = _get_ppr(rowptr, ei[1][np.lexsort((ei[1], ei[0]))], 0.15, 5e-5)  # PyG's (pure Python) algorithm
    want = np.zeros((n, n))
    for i, (j, v) in enumerate(zip(js, vals)):
        want[i, j] = v
    np.testing.assert_allclose(get_ppr(ei, n).toarray(), want, atol=1e-6)


def test_lpformer_matches_pyg():
    rng, ei = _graph()
    n = 40
    x = rng.random((n, 8)).astype("float32")
    batch = rng.integers(0, n, (2, 25))
    model = LPFormer(8, 16)
    ppr = model.calc_sparse_ppr(ei, n)
    model(batch, x, ei, ppr_matrix=ppr)
    ref = PyGLPFormer(8, 16).eval()
    st = {k: v.detach().numpy() for k, v in ref.state_dict().items()}

    def dense(layer, name):
        layer.kernel.assign(st[name + ".weight"].T)
        layer.bias.assign(st[name + ".bias"])

    def ln(layer, name):
        layer.gamma.assign(st[name + ".weight"])
        layer.beta.assign(st[name + ".bias"])

    for i, conv in enumerate(model.gnn.convs):
        conv.bias.assign(st[f"gnn.convs.{i}.bias"])
        conv.lin.kernel.assign(st[f"gnn.convs.{i}.lin.weight"].T)
    for i, norm in enumerate(model.gnn.norms):
        if norm is not None:
            norm.weight.assign(st[f"gnn.norms.{i}.weight"])
            norm.bias.assign(st[f"gnn.norms.{i}.bias"])
    ln(model.gnn_norm, "gnn_norm")
    att = model.att_layers[0]
    att.att.assign(st["att_layers.0.att"])
    att.bias.assign(st["att_layers.0.bias"])
    dense(att.lin_l, "att_layers.0.lin_l")
    dense(att.lin_r, "att_layers.0.lin_r")
    ln(att.post_att_norm, "att_layers.0.post_att_norm")
    for name in ["elementwise_lin", "ppr_encoder_cn", "ppr_encoder_onehop", "ppr_encoder_non1hop", "pairwise_lin",
                 "score_func"]:
        mlp = getattr(model, name)
        for i, lin in enumerate(mlp.linears):
            dense(lin, f"{name}.linears.{i}")
        if mlp.norm is not None:
            ln(mlp.norm, name + ".norm")

    ppr_t = torch.sparse_coo_tensor(np.stack(ppr.nonzero()), torch.tensor(ppr.data), (n, n))
    with torch.no_grad():
        want = ref(torch.tensor(batch), torch.tensor(x), torch.tensor(ei), ppr_t).numpy()
    got = ops.convert_to_numpy(model(batch, x, ei, ppr_matrix=ppr))
    np.testing.assert_allclose(got, want, atol=1e-4)
