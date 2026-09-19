import re
import inspect
from typing import List, Optional, Union

import keras
from keras import ops

import k3_node.layers.norm as norm_module


def _normalize_string(s: str) -> str:
    return re.sub(r"[_\-\s]", "", s).lower()


def _normalization_resolver(query, *args, **kwargs):
    if query is None:
        return None
    if not isinstance(query, str):
        return query

    norms = {
        _normalize_string(name): cls
        for name, cls in vars(norm_module).items()
        if isinstance(cls, type)
    }
    key = _normalize_string(query)
    if key in norms:
        return norms[key](*args, **kwargs)
    if key + "norm" in norms:
        return norms[key + "norm"](*args, **kwargs)
    if key.endswith("norm") and key[:-4] in norms:
        return norms[key[:-4]](*args, **kwargs)
    raise ValueError(f"Could not resolve normalization layer '{query}'")


class MLP(keras.layers.Layer):
    r"""A Multi-Layer Perceptron (MLP) model.

    There exists two ways to instantiate an `MLP`:

    1. By specifying explicit channel sizes, e.g., `MLP([16, 32, 64, 128])`
       creates a three-layer MLP with **differently** sized hidden layers.

    2. By specifying fixed hidden channel sizes over a number of layers,
       e.g., `MLP(in_channels=16, hidden_channels=32, out_channels=128,
       num_layers=3)` creates a three-layer MLP with **equally** sized
       hidden layers.

    Args:
        channel_list (List[int] or int, optional): List of input,
            intermediate and output channels such that
            `len(channel_list) - 1` denotes the number of layers of the
            MLP. (default: `None`)
        in_channels (int, optional): Size of each input sample. Will
            override `channel_list`. (default: `None`)
        hidden_channels (int, optional): Size of each hidden sample. Will
            override `channel_list`. (default: `None`)
        out_channels (int, optional): Size of each output sample. Will
            override `channel_list`. (default: `None`)
        num_layers (int, optional): The number of layers. Will override
            `channel_list`. (default: `None`)
        dropout (float or List[float], optional): Dropout probability of
            each hidden embedding. (default: `0.`)
        act (str or Callable, optional): The non-linear activation function
            to use. (default: `"relu"`)
        act_first (bool, optional): If set to `True`, activation is applied
            before normalization. (default: `False`)
        norm (str or Callable, optional): The normalization function to
            use. (default: `"batch_norm"`)
        norm_kwargs (dict, optional): Arguments passed to the respective
            normalization function. (default: `None`)
        plain_last (bool, optional): If set to `False`, will apply
            non-linearity, normalization and dropout to the last layer as
            well. (default: `True`)
        bias (bool or List[bool], optional): If set to `False`, the module
            will not learn additive biases. (default: `True`)
    """
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Union[float, List[float]] = 0.0,
        act="relu",
        act_first: bool = False,
        norm="batch_norm",
        norm_kwargs: Optional[dict] = None,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(channel_list, int):
            in_channels = channel_list
            channel_list = None

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(
                    f"Argument `hidden_channels` must be given for `num_layers={num_layers}`"
                )
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = list(channel_list)

        self.act = keras.activations.get(act) if act is not None else None
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.0
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)}) does not "
                f"match the number of layers specified ({len(channel_list) - 1})"
            )
        self.dropout_rate = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list) - 1})"
            )

        self.lins = []
        for in_c, out_c, _bias in zip(channel_list[:-1], channel_list[1:], bias):
            lin = keras.layers.Dense(out_c, use_bias=_bias)
            lin.build((None, in_c))
            self.lins.append(lin)

        self.norms = []
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hc in iterator:
            norm_layer = _normalization_resolver(norm, hc, **(norm_kwargs or {}))
            self.norms.append(norm_layer)

        self.dropouts = [keras.layers.Dropout(p) if p > 0.0 else None for p in self.dropout_rate]

        self.supports_norm_batch = False
        if len(self.norms) > 0 and self.norms[0] is not None:
            norm_params = inspect.signature(self.norms[0].call).parameters
            self.supports_norm_batch = "batch" in norm_params

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            if hasattr(lin, "kernel_initializer") and lin.kernel is not None:
                lin.kernel.assign(lin.kernel_initializer(ops.shape(lin.kernel)))
                if lin.bias is not None:
                    lin.bias.assign(lin.bias_initializer(ops.shape(lin.bias)))
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def call(self, x, batch=None, batch_size=None, return_emb=None, training=None):
        emb = None

        # If `plain_last=True`, `len(norms) == len(lins) - 1`, thus skipping
        # execution of the last layer inside the loop.
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if norm is not None:
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x, training=training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            if self.dropouts[-1] is not None:
                x = self.dropouts[-1](x, training=training)

        return (x, emb) if isinstance(return_emb, bool) else x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"
