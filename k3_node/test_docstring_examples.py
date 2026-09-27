"""Runs the ``Example:`` code in the docstrings of public layers and models.

Every example must run on the current backend, and each ``print(...)  # expected`` line must print
the value in its comment, so the documented output shapes cannot drift from the code.
"""
import ast
import inspect
import re
import textwrap

import keras
import pytest

import k3_node.layers
import k3_node.models

# Public objects that intentionally have no forward-pass example.
NO_EXAMPLE = {
    "Aggregation",  # abstract base class; see the concrete aggregations
    "KGEModel",  # abstract base class; see TransE, DistMult, ComplEx, RotatE
    "Connect",  # abstract base class of the pooling "connect" step
    "Select",  # abstract base class of the pooling "select" step
    "BasicGNN",  # abstract base class; see GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN
}

_FENCE = re.compile(r"```python\n(.*?)```", re.S)


def _public_objects():
    """Unique public layer/model classes (and layer functions), keyed by public name."""
    found, seen = {}, set()
    for package in (k3_node.layers, k3_node.models):
        names = getattr(package, "__all__", None) or [n for n in dir(package) if not n.startswith("_")]
        for name in sorted(names):
            obj = getattr(package, name, None)
            is_layer = inspect.isclass(obj) and issubclass(obj, keras.layers.Layer)
            is_layer_fn = (
                inspect.isfunction(obj) and package is k3_node.layers and obj.__module__.startswith("k3_node.layers")
            )
            if (is_layer or is_layer_fn) and id(obj) not in seen and obj.__module__.startswith("k3_node"):
                seen.add(id(obj))
                found[f"{package.__name__}.{name}"] = obj
    return found


PUBLIC = _public_objects()


def _examples(obj):
    doc = obj.__doc__ or ""
    section = doc[doc.find("Example") :] if "Example" in doc else ""
    return [textwrap.dedent(block) for block in _FENCE.findall(section)]


def _expected_prints(code):
    """Maps the line number of each top-level print() to the text of its trailing comment."""
    expected = {}
    lines = code.split("\n")
    for node in ast.parse(code).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "print":
            _, sep, comment = lines[node.lineno - 1].partition("  # ")
            if sep:
                expected[node.lineno] = comment
    return expected


WITH_EXAMPLES = [name for name, obj in PUBLIC.items() if _examples(obj)]


@pytest.mark.parametrize("name", WITH_EXAMPLES)
def test_docstring_example_runs(name):
    for code in _examples(PUBLIC[name]):
        expected = _expected_prints(code)
        printed = []

        def capture(*args, **kwargs):
            frame = inspect.currentframe().f_back
            printed.append((frame.f_lineno, " ".join(str(a) for a in args)))

        exec(compile(code, f"<{name} example>", "exec"), {"print": capture, "__name__": "__example__"})
        for lineno, text in printed:
            if lineno in expected:
                assert expected[lineno].startswith(text), (
                    f"{name}: line {lineno} printed {text!r}, but the docstring says {expected[lineno]!r}"
                )


def test_every_public_layer_and_model_has_an_example():
    missing = sorted(
        name for name, obj in PUBLIC.items()
        if not _examples(obj) and name.rpartition(".")[2] not in NO_EXAMPLE and inspect.isclass(obj)
    )
    assert not missing, f"{len(missing)} public layers/models have no docstring example: {missing}"
