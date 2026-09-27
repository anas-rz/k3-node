"""Cross-backend and compiled-mode consistency tests for convolution layers.

Each layer is run with identical seeded inputs and weights, and its eager output is
compared against golden outputs recorded with the torch backend (the backend that
``tests_reference`` validates against PyG). On TensorFlow and JAX the output under
``jit_compile=True`` must also match the eager output.

The input graph deliberately contains self-loops: layers that remove and re-add
self-loops must handle them without dynamic shapes when compiled.

To regenerate the golden file after an intentional numerical change::

    KERAS_BACKEND=torch python k3_node/layers/conv/test_backend_consistency.py
"""
import os
import os.path as osp

import numpy as np
import pytest
import keras
from keras import layers

import k3_node.layers as L

GOLDEN_PATH = osp.join(osp.dirname(__file__), "testdata", "backend_consistency_golden.npz")

N, E, C, O = 12, 30, 8, 5


def _graph():
    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, N, size=(2, E))
    loops = np.array([[0, 3, 7], [0, 3, 7]])  # guarantee pre-existing self-loops
    return {
        "x": rng.standard_normal((N, C)).astype("float32"),
        "pos": rng.standard_normal((N, 3)).astype("float32"),
        "normal": rng.standard_normal((N, 3)).astype("float32"),
        "edge_index": np.concatenate([edge_index, loops], axis=1).astype("int32"),
    }


def _mlp(units):
    return keras.Sequential([layers.Dense(units, activation="relu"), layers.Dense(units)])


def _xe(layer, g):
    return layer(g["x"], g["edge_index"])


# name -> (layer factory, call function)
CASES = {
    "GCNConv": (lambda: L.GCNConv(C, O), _xe),
    "GATConv": (lambda: L.GATConv(C, O, heads=2), _xe),
    "GATv2Conv": (lambda: L.GATv2Conv(C, O, heads=2), _xe),
    "SuperGATConv": (lambda: L.SuperGATConv(C, O, heads=2), _xe),
    "ClusterGCNConv": (lambda: L.ClusterGCNConv(C, O), _xe),
    "AGNNConv": (lambda: L.AGNNConv(), _xe),
    "FeaStConv": (lambda: L.FeaStConv(C, O, heads=2), _xe),
    "PointNetConv": (lambda: L.PointNetConv(local_nn=_mlp(O)), lambda l, g: l(g["x"], g["pos"], g["edge_index"])),
    "PointTransformerConv": (
        lambda: L.PointTransformerConv(C, O),
        lambda l, g: l(g["x"], g["pos"], g["edge_index"]),
    ),
    "PPFConv": (
        lambda: L.PPFConv(local_nn=_mlp(O)),
        lambda l, g: l(g["x"], g["pos"], g["normal"], g["edge_index"]),
    ),
    "SAGEConv": (lambda: L.SAGEConv(C, O), _xe),
    "GraphConv": (lambda: L.GraphConv(C, O), _xe),
    "TransformerConv": (lambda: L.TransformerConv(C, O, heads=2), _xe),
    "ChebConv": (lambda: L.ChebConv(C, O, K=3), _xe),
    "TAGConv": (lambda: L.TAGConv(C, O, K=2), _xe),
    "SGConv": (lambda: L.SGConv(C, O, K=2), _xe),
    "LEConv": (lambda: L.LEConv(C, O), _xe),
    "ResGatedGraphConv": (lambda: L.ResGatedGraphConv(C, O), _xe),
    "GENConv": (lambda: L.GENConv(C, O), _xe),
    "GINConv": (lambda: L.GINConv(_mlp(O)), _xe),
    "EdgeConv": (lambda: L.EdgeConv(keras.Sequential([layers.Dense(O)])), _xe),
    "MFConv": (lambda: L.MFConv(C, O), _xe),
    "FiLMConv": (lambda: L.FiLMConv(C, O), _xe),
    "GeneralConv": (lambda: L.GeneralConv(C, O), _xe),
    "PNAConv": (
        lambda: L.PNAConv(
            C,
            O,
            aggregators=["mean", "max", "min", "std"],
            scalers=["identity", "amplification"],
            deg=np.array([0, 2, 4, 3, 2, 1], "int32"),
        ),
        _xe,
    ),
}


class ConsistencyWrapper(keras.Model):
    def __init__(self, layer, call_fn):
        super().__init__()
        self.layer = layer
        self.call_fn = call_fn

    def call(self, inputs):
        out = self.call_fn(self.layer, inputs)
        return out[0] if isinstance(out, (tuple, list)) else out


def _build(name):
    factory, call_fn = CASES[name]
    g = _graph()
    model = ConsistencyWrapper(factory(), call_fn)
    model(g)
    rng = np.random.default_rng(123)
    weights = []
    for v in model.weights:
        w = rng.standard_normal(v.shape) * 0.3
        if "moving_variance" in v.path:
            w = np.abs(w) + 0.5
        weights.append(w.astype(v.dtype))
    model.set_weights(weights)
    shapes = np.array([str(tuple(v.shape)) for v in model.weights])
    return model, g, shapes


def _eager(model, g):
    return keras.ops.convert_to_numpy(model(g))


@pytest.fixture(scope="module")
def golden():
    if not osp.exists(GOLDEN_PATH):
        pytest.fail(f"Golden file missing; regenerate it (see module docstring): {GOLDEN_PATH}")
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize("name", sorted(CASES))
def test_eager_matches_torch_golden(name, golden):
    model, g, shapes = _build(name)
    assert list(shapes) == list(golden[f"{name}/shapes"]), (
        f"{name}: weight layout differs from the torch backend, so weights cannot be shared"
    )
    np.testing.assert_allclose(_eager(model, g), golden[f"{name}/out"], rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    keras.backend.backend() not in ("tensorflow", "jax"),
    reason="jit_compile=True means XLA only on the TensorFlow and JAX backends",
)
@pytest.mark.parametrize("name", sorted(CASES))
def test_jit_matches_eager(name):
    model, g, _ = _build(name)
    expected = _eager(model, g)
    model.compile(jit_compile=True)
    np.testing.assert_allclose(np.asarray(model.predict_on_batch(g)), expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    assert keras.backend.backend() == "torch", "Golden outputs must be generated with KERAS_BACKEND=torch"
    arrays = {}
    for case in sorted(CASES):
        m, graph, layer_shapes = _build(case)
        arrays[f"{case}/out"] = _eager(m, graph)
        arrays[f"{case}/shapes"] = layer_shapes
    os.makedirs(osp.dirname(GOLDEN_PATH), exist_ok=True)
    np.savez(GOLDEN_PATH, **arrays)
    print(f"Wrote {len(CASES)} golden outputs to {GOLDEN_PATH}")
