from typing import Dict, List, Optional

import keras
from keras import ops


class JumpingKnowledge(keras.layers.Layer):
    r"""The Jumping Knowledge layer aggregation module from the
    `"Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper.

    Jumping knowledge is performed based on either **concatenation**
    (:obj:`"cat"`)

    .. math::

        \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)},

    **max pooling** (:obj:`"max"`)

    .. math::

        \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right),

    or **weighted summation**

    .. math::

        \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

    with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
    LSTM (:obj:`"lstm"`).

    Args:
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        channels (int, optional): The number of channels per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)
    """

    def __init__(
        self,
        mode: str,
        channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm'], \
            f"mode must be 'cat', 'max', or 'lstm', got '{mode}'"

        self.channels = channels
        self.num_layers = num_layers

        if self.mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            lstm_units = (num_layers * channels) // 2
            self.lstm = keras.layers.Bidirectional(
                keras.layers.LSTM(lstm_units, return_sequences=True),
                merge_mode='concat',
            )
            self.att = keras.layers.Dense(1, use_bias=False)
        else:
            self.lstm = None
            self.att = None

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        # Keras layers reinitialize on next call; explicit rebuild if needed
        if self.lstm is not None and self.lstm.built:
            for layer in self.lstm.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    layer.kernel.assign(
                        keras.initializers.GlorotUniform()(layer.kernel.shape)
                    )
                if hasattr(layer, 'recurrent_kernel') and layer.recurrent_kernel is not None:
                    layer.recurrent_kernel.assign(
                        keras.initializers.Orthogonal()(layer.recurrent_kernel.shape)
                    )
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.assign(ops.zeros(layer.bias.shape))
        if self.att is not None and self.att.built:
            if self.att.kernel is not None:
                self.att.kernel.assign(
                    keras.initializers.GlorotUniform()(self.att.kernel.shape)
                )

    def call(self, xs: List) -> object:
        r"""Forward pass.

        Args:
            xs (List[Tensor]): List containing the layer-wise representations.
        """
        if self.mode == 'cat':
            return ops.concatenate(xs, axis=-1)
        elif self.mode == 'max':
            return ops.max(ops.stack(xs, axis=-1), axis=-1)
        else:  # lstm
            assert self.lstm is not None and self.att is not None
            x = ops.stack(xs, axis=1)  # [num_nodes, num_layers, num_channels]
            alpha = self.lstm(x)        # [num_nodes, num_layers, 2*lstm_units]
            alpha = self.att(alpha)     # [num_nodes, num_layers, 1]
            alpha = ops.squeeze(alpha, axis=-1)  # [num_nodes, num_layers]
            alpha = ops.softmax(alpha, axis=-1)
            return ops.sum(x * ops.expand_dims(alpha, axis=-1), axis=1)

    def __repr__(self) -> str:
        if self.mode == 'lstm':
            return (f'{self.__class__.__name__}({self.mode}, '
                    f'channels={self.channels}, layers={self.num_layers})')
        return f'{self.__class__.__name__}({self.mode})'


class HeteroJumpingKnowledge(keras.layers.Layer):
    r"""A heterogeneous version of the :class:`JumpingKnowledge` module.

    Args:
        types (List[str]): The keys of the input dictionary.
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        channels (int, optional): The number of channels per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)
    """

    def __init__(
        self,
        types: List[str],
        mode: str,
        channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mode = mode.lower()
        self.types = list(types)

        self.jk_dict = {
            key: JumpingKnowledge(mode, channels, num_layers)
            for key in types
        }

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        for jk in self.jk_dict.values():
            jk.reset_parameters()

    def call(self, xs_dict: Dict[str, List]) -> Dict[str, object]:
        r"""Forward pass.

        Args:
            xs_dict (Dict[str, List[Tensor]]): A dictionary holding a
                list of layer-wise representation for each type.
        """
        return {key: self.jk_dict[key](xs_dict[key]) for key in self.types}

    def __repr__(self) -> str:
        if self.mode == 'lstm':
            jk = next(iter(self.jk_dict.values()))
            return (f'{self.__class__.__name__}('
                    f'num_types={len(self.jk_dict)}, '
                    f'mode={self.mode}, channels={jk.channels}, '
                    f'layers={jk.num_layers})')
        return (f'{self.__class__.__name__}(num_types={len(self.jk_dict)}, '
                f'mode={self.mode})')

