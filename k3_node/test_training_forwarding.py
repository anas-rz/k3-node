"""Every k3 layer must forward `training` to its dropout / batch-norm sublayers.

On the JAX backend Keras does not propagate `training` from a model to nested layers during
`fit`, so a layer that calls its dropout or batch norm without passing `training` silently runs
it in inference mode. This test reproduces that: it disables Keras' propagation, enables dropout
in every constructor, runs all docstring examples with `training=True`, and reports any nested
call that did not receive `training` from its parent.
"""
import collections
import inspect
import re
import textwrap

import keras
import pytest
from keras.src.layers.layer import Layer

import k3_node.layers
import k3_node.models

_TRAINING_SENSITIVE = ("BatchNormalization", "BatchNorm", "InstanceNorm", "HeteroBatchNorm")


def _training_sensitive(layer):
    for sub in layer._flatten_layers(include_self=True):
        if type(sub).__name__ in _TRAINING_SENSITIVE:
            return True
        if isinstance(sub, keras.layers.Dropout) and float(sub.rate or 0) > 0:
            return True
        if isinstance(getattr(sub, "dropout", None), float) and sub.dropout > 0:  # e.g. attention dropout
            return True
    return False


def _enable_dropout_defaults(cls):
    """Sets zero-valued dropout defaults of cls.__init__ to 0.1 (so dropout paths are exercised)."""
    init = cls.__dict__.get("__init__")
    if init is None:
        return None
    saved = (init.__defaults__, dict(init.__kwdefaults__ or {}))
    params = [p for p in inspect.signature(init).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if init.__defaults__:
        defaults = list(init.__defaults__)
        for i, p in enumerate(params[len(params) - len(defaults):]):
            if "drop" in p.name and not isinstance(defaults[i], bool) and defaults[i] == 0:
                defaults[i] = 0.1
        init.__defaults__ = tuple(defaults)
    for k, v in (init.__kwdefaults__ or {}).items():
        if "drop" in k and not isinstance(v, bool) and v == 0:
            init.__kwdefaults__[k] = 0.1
    return init, saved


@pytest.fixture
def no_training_propagation(monkeypatch):
    stack, gaps = [], collections.Counter()
    original_resolve = Layer._resolve_and_populate_arg
    original_call = Layer.__call__

    def resolve(self, arg_name, call_spec, call_context, kwargs):
        if arg_name != "training":
            return original_resolve(self, arg_name, call_spec, call_context, kwargs)
        passed = arg_name in call_spec.user_arguments_dict
        value = call_spec.user_arguments_dict.get(arg_name) if passed else (True if len(stack) <= 1 else None)
        if self._call_has_context_arg.get(arg_name, False) and value is not None:
            kwargs[arg_name] = value
        parent = stack[-2] if len(stack) > 1 else None
        if (parent is not None and not passed and type(parent).__module__.startswith("k3_node")
                and self._call_has_context_arg.get("training", False) and _training_sensitive(self)):
            gaps[f"{type(parent).__module__}.{type(parent).__name__} -> {type(self).__name__}"] += 1

    def call(self, *args, **kwargs):
        stack.append(self)
        try:
            return original_call(self, *args, **kwargs)
        finally:
            stack.pop()

    monkeypatch.setattr(Layer, "_resolve_and_populate_arg", resolve)
    monkeypatch.setattr(Layer, "__call__", call)
    return gaps


@pytest.mark.skipif(keras.backend.backend() != "torch", reason="checks the code, not a backend; torch is fastest")
def test_layers_forward_training_to_dropout_and_batch_norm(no_training_propagation):
    import sys

    patched = []
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("k3_node"):
            for obj in list(vars(module).values()):
                if inspect.isclass(obj) and issubclass(obj, Layer) and obj.__module__.startswith("k3_node"):
                    result = _enable_dropout_defaults(obj)
                    if result:
                        patched.append(result)
    try:
        seen = set()
        for package in (k3_node.layers, k3_node.models):
            for name in dir(package):
                obj = getattr(package, name)
                if id(obj) in seen or not getattr(obj, "__module__", "").startswith("k3_node"):
                    continue
                seen.add(id(obj))
                doc = obj.__doc__ or ""
                for code in re.findall(r"```python\n(.*?)```", doc[doc.find("Example"):] if "Example" in doc else "", re.S):
                    exec(compile(textwrap.dedent(code), name, "exec"), {"print": lambda *a, **k: None})
    finally:
        for init, (defaults, kwdefaults) in patched:
            init.__defaults__ = defaults
            if init.__kwdefaults__ is not None:
                init.__kwdefaults__.clear()
                init.__kwdefaults__.update(kwdefaults)
    assert not no_training_propagation, (
        "These layers call a dropout / batch-norm sublayer without forwarding `training` "
        f"(it would never train on JAX): {sorted(no_training_propagation)}"
    )
