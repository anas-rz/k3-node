import math
from typing import Any, Dict, List, Optional, Union
from keras import initializers, layers, ops


def _get_weight_initializer(initializer: Optional[str], in_channels: int):
    if initializer in ('glorot', 'xavier_uniform'):
        return initializers.GlorotUniform()
    elif initializer in ('kaiming_uniform', 'he_uniform'):
        return initializers.HeUniform()
    elif initializer == 'uniform' or initializer is None:
        if in_channels > 0:
            bound = 1.0 / math.sqrt(in_channels)
            return initializers.RandomUniform(minval=-bound, maxval=bound)
        return initializers.GlorotUniform()
    return initializers.get(initializer)


def _get_bias_initializer(initializer: Optional[str], in_channels: int):
    if initializer == 'zeros':
        return initializers.Zeros()
    elif initializer is None:
        if in_channels > 0:
            bound = 1.0 / math.sqrt(in_channels)
            return initializers.RandomUniform(minval=-bound, maxval=bound)
        return initializers.Zeros()
    return initializers.get(initializer)


class Linear(layers.Layer):
    r"""Applies a linear transformation to the incoming data:

    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W} + \mathbf{b}

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`). (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`). (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_initializer: Optional[str] = None,
        bias_initializer: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weight = None
        self.bias = None

        if self.in_channels > 0:
            self._build_weights(self.in_channels)

    def _build_weights(self, in_features: int):
        self.in_channels = in_features
        w_init = _get_weight_initializer(self.weight_initializer, in_features)
        b_init = _get_bias_initializer(self.bias_initializer, in_features)

        self.weight = self.add_weight(
            shape=(in_features, self.out_channels),
            initializer=w_init,
            trainable=True,
            name="weight",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer=b_init,
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape):
        if not hasattr(self, 'weight') or self.weight is None:
            self._build_weights(input_shape[-1])
        super().build(input_shape)

    def reset_parameters(self):
        if hasattr(self, 'weight') and self.weight is not None:
            w_init = _get_weight_initializer(self.weight_initializer, self.in_channels)
            self.weight.assign(w_init(self.weight.shape, dtype=self.weight.dtype))
        if hasattr(self, 'bias') and self.bias is not None:
            b_init = _get_bias_initializer(self.bias_initializer, self.in_channels)
            self.bias.assign(b_init(self.bias.shape, dtype=self.bias.dtype))

    def call(self, x):
        if not hasattr(self, 'weight') or self.weight is None:
            self._build_weights(ops.shape(x)[-1])

        out = ops.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.use_bias})')


class HeteroLinear(layers.Layer):
    r"""Applies separate linear transformations to the incoming data according
    to types.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \mathbf{\Theta}_{\kappa_i} +
        \mathbf{b}_{\kappa_i}

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_types (int): The number of types.
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`type_vec` is sorted. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`). (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`). (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_types: int,
        is_sorted: bool = False,
        bias: bool = True,
        weight_initializer: Optional[str] = None,
        bias_initializer: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_types = num_types
        self.is_sorted = is_sorted
        self.use_bias = bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weight = None
        self.bias = None

        if self.in_channels > 0:
            self._build_weights(self.in_channels)

    def _build_weights(self, in_features: int):
        self.in_channels = in_features
        w_init = _get_weight_initializer(self.weight_initializer, in_features)
        b_init = _get_bias_initializer(self.bias_initializer, in_features)

        self.weight = self.add_weight(
            shape=(self.num_types, in_features, self.out_channels),
            initializer=w_init,
            trainable=True,
            name="weight",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.num_types, self.out_channels),
                initializer=b_init,
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape):
        if self.in_channels <= 0:
            self._build_weights(input_shape[-1])
        super().build(input_shape)

    def reset_parameters(self):
        if self.in_channels > 0 and self.weight is not None:
            w_init = _get_weight_initializer(self.weight_initializer, self.in_channels)
            self.weight.assign(w_init(self.weight.shape, dtype=self.weight.dtype))
            if self.bias is not None:
                b_init = _get_bias_initializer(self.bias_initializer, self.in_channels)
                self.bias.assign(b_init(self.bias.shape, dtype=self.bias.dtype))

    def call(self, x, type_vec):
        if self.in_channels <= 0 and not self.built:
            self.build(ops.shape(x))

        num_nodes = ops.shape(x)[0]
        type_vec = ops.cast(type_vec, dtype="int32")

        w_selected = ops.take(self.weight, type_vec, axis=0)
        x_exp = ops.expand_dims(x, axis=1)
        out = ops.squeeze(ops.matmul(x_exp, w_selected), axis=1)

        if self.bias is not None:
            b_selected = ops.take(self.bias, type_vec, axis=0)
            out = out + b_selected

        return out

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape
        return (*x_shape[:-1], self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_types={self.num_types}, '
                f'bias={self.use_bias})')


class HeteroDictLinear(layers.Layer):
    r"""Applies separate linear transformations to the incoming data
    dictionary.

    For key :math:`\kappa`, it computes

    .. math::
        \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
        \mathbf{W}_{\kappa} + \mathbf{b}_{\kappa}.

    Args:
        in_channels (int or Dict[Any, int]): Size of each input sample.
        out_channels (int): Size of each output sample.
        types (List[Any], optional): The keys of the input dictionary.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[Any, int]],
        out_channels: int,
        types: Optional[List[Any]] = None,
        **kwargs
    ):
        layer_kwargs = {k: v for k, v in kwargs.items() if k in ["name", "trainable", "dtype", "autocast"]}
        super().__init__(**layer_kwargs)

        if isinstance(in_channels, dict):
            self.types = list(in_channels.keys())
            if types is not None and set(self.types) != set(types):
                raise ValueError("The provided 'types' do not match with the "
                                 "keys in the 'in_channels' dictionary")
        else:
            if types is None:
                raise ValueError("Please provide a list of 'types' if passing "
                                 "'in_channels' as an integer")
            self.types = types
            in_channels = {node_type: in_channels for node_type in types}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs

        lin_kwargs = {k: v for k, v in kwargs.items() if k not in ["name"]}
        self.lins = {
            str(key): Linear(channels, self.out_channels, name=f"lin_{key}", **lin_kwargs)
            for key, channels in self.in_channels.items()
        }

    def reset_parameters(self):
        for lin in self.lins.values():
            lin.reset_parameters()

    def call(self, x_dict: Dict[str, Any]) -> Dict[str, Any]:
        out_dict = {}
        for key, x in x_dict.items():
            str_key = str(key)
            if str_key in self.lins:
                out_dict[key] = self.lins[str_key](x)
        return out_dict

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.kwargs.get("bias", True)})')
