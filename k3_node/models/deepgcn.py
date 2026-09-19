import keras
from keras import ops

from k3_node.models.utils import reset


class DeepGCNLayer(keras.layers.Layer):
    r"""The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
    The implemented skip connections includes the pre-activation residual
    connection (`"res+"`), the residual connection (`"res"`), the dense
    connection (`"dense"`) and no connections (`"plain"`).

    * **Res+** (`"res+"`):

    .. math::
        \text{Normalization}\to\text{Activation}\to\text{Dropout}\to
        \text{GraphConv}\to\text{Res}

    * **Res** (`"res"`) / **Dense** (`"dense"`) / **Plain** (`"plain"`):

    .. math::
        \text{GraphConv}\to\text{Normalization}\to\text{Activation}\to
        \text{Res/Dense/Plain}\to\text{Dropout}

    Args:
        conv (optional): the GCN operator. (default: `None`)
        norm (optional): the normalization layer. (default: `None`)
        act (optional): the activation layer. (default: `None`)
        block (str, optional): The skip connection operation to use
            (`"res+"`, `"res"`, `"dense"` or `"plain"`). (default: `"res+"`)
        dropout (float, optional): Whether to apply dropout. (default: `0.`)
        ckpt_grad (bool, optional): Kept for API compatibility; a no-op,
            since there is no gradient-checkpointing API shared uniformly
            across Keras backends. (default: `False`)
    """
    def __init__(
        self,
        conv=None,
        norm=None,
        act=None,
        block: str = "res+",
        dropout: float = 0.0,
        ckpt_grad: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        assert self.block in ["res+", "res", "dense", "plain"]
        self.dropout_rate = dropout
        self.ckpt_grad = ckpt_grad
        self.dropout = keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.conv)
        reset(self.norm)

    def call(self, x, edge_index, training=None, **kwargs):
        if self.block == "res+":
            h = x
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            if self.dropout is not None:
                h = self.dropout(h, training=training)
            h = self.conv(h, edge_index, **kwargs)

            return x + h

        else:
            h = self.conv(x, edge_index, **kwargs)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)

            if self.block == "res":
                h = x + h
            elif self.block == "dense":
                h = ops.concatenate([x, h], axis=-1)
            elif self.block == "plain":
                pass

            if self.dropout is not None:
                h = self.dropout(h, training=training)
            return h

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(block={self.block})"
