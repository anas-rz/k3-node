from typing import List, Optional, Union
from keras import layers, ops


class LayerNorm(layers.Layer):
    r"""Applies layer normalization over each individual example in a batch
    of features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode (str, optional): The normalization mode to use for layer
            normalization (:obj:`"graph"` or :obj:`"node"`). If :obj:`"graph"`
            is used, each graph will be considered as an element to be
            normalized. If `"node"` is used, each node will be considered as
            an element to be normalized. (default: :obj:`"graph"`)
    """
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = 'graph',
        **kwargs
    ):
        super().__init__(**kwargs)
        if mode not in ('graph', 'node'):
            raise ValueError(f"Unknown normalization mode: {mode}")

        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine
        self.mode = mode

        if affine:
            self.weight = self.add_weight(
                shape=(in_channels,),
                initializer="ones",
                trainable=True,
                name="weight",
            )
            self.bias = self.add_weight(
                shape=(in_channels,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.weight = None
            self.bias = None

    def reset_parameters(self):
        if self.affine:
            self.weight.assign(ops.ones(self.weight.shape, dtype=self.weight.dtype))
            self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))

    def call(self, x, batch=None, batch_size=None):
        if batch is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, batch = x
            elif len(x) == 3:
                x, batch, batch_size = x

        if self.mode == 'graph':
            if batch is None:
                mean = ops.mean(x)
                var = ops.mean(ops.power(x - mean, 2))
                out = (x - mean) / ops.sqrt(var + self.eps)
            else:
                if batch_size is not None and not isinstance(batch_size, int):
                    try:
                        batch_size = int(batch_size)
                    except Exception:
                        pass
                elif batch_size is None:
                    batch_size = ops.cast(ops.max(batch), "int32") + 1

                batch = ops.cast(batch, "int32")
                in_channels = ops.cast(ops.shape(x)[-1], dtype=x.dtype)
                ones = ops.ones((ops.shape(x)[0], 1), dtype=x.dtype)
                node_counts = ops.maximum(ops.segment_sum(ones, batch, num_segments=batch_size), 1.0)
                total_count = node_counts * in_channels

                sum_x = ops.sum(ops.segment_sum(x, batch, num_segments=batch_size), axis=-1, keepdims=True)
                mean = sum_x / total_count
                x_centered = x - ops.take(mean, batch, axis=0)

                sum_sq = ops.sum(ops.segment_sum(ops.power(x_centered, 2), batch, num_segments=batch_size), axis=-1, keepdims=True)
                var = sum_sq / total_count
                std_x = ops.take(ops.sqrt(var + self.eps), batch, axis=0)
                out = x_centered / std_x

            if self.affine:
                out = out * self.weight + self.bias
            return out

        elif self.mode == 'node':
            mean = ops.mean(x, axis=-1, keepdims=True)
            var = ops.var(x, axis=-1, keepdims=True)
            out = (x - mean) / ops.sqrt(var + self.eps)
            if self.affine:
                out = out * self.weight + self.bias
            return out

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape


class HeteroLayerNorm(layers.Layer):
    r"""Applies layer normalization over each individual example in a batch
    of heterogeneous features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.
    Compared to :class:`LayerNorm`, :class:`HeteroLayerNorm` applies
    normalization individually for each node or edge type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode (str, optional): The normalization mode to use for layer
            normalization (:obj:`"node"`). (default: :obj:`"node"`)
    """
    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = 'node',
        **kwargs
    ):
        super().__init__(**kwargs)
        if mode != 'node':
            raise ValueError(f"HeteroLayerNorm only supports mode='node' (got '{mode}')")

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.affine = affine
        self.mode = mode

        if affine:
            self.weight = self.add_weight(
                shape=(num_types, in_channels),
                initializer="ones",
                trainable=True,
                name="weight",
            )
            self.bias = self.add_weight(
                shape=(num_types, in_channels),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.weight = None
            self.bias = None

    def reset_parameters(self):
        if self.affine:
            self.weight.assign(ops.ones(self.weight.shape, dtype=self.weight.dtype))
            self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))

    def call(
        self,
        x,
        type_vec=None,
        type_ptr: Optional[Union[list, tuple]] = None,
    ):
        if type_vec is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, type_vec = x
            elif len(x) == 3:
                x, type_vec, type_ptr = x

        if type_vec is None and type_ptr is None:
            raise ValueError("Either 'type_vec' or 'type_ptr' must be given")

        mean = ops.mean(x, axis=-1, keepdims=True)
        var = ops.var(x, axis=-1, keepdims=True)
        out = (x - mean) / ops.sqrt(var + self.eps)

        if self.affine:
            if type_ptr is not None:
                parts = []
                for i in range(len(type_ptr) - 1):
                    s, e = type_ptr[i], type_ptr[i + 1]
                    part = out[s:e] * self.weight[i] + self.bias[i]
                    parts.append(part)
                out = ops.concatenate(parts, axis=0)
            else:
                type_vec = ops.cast(type_vec, "int32")
                w = ops.take(self.weight, type_vec, axis=0)
                b = ops.take(self.bias, type_vec, axis=0)
                out = out * w + b

        return out

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape
