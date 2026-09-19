import copy
from typing import Optional, Union, Tuple, Callable
import keras
from keras import ops, activations
from keras.layers import Dense, Layer

from k3_node.layers.conv.message_passing import MessagePassing


class FiLMConv(MessagePassing):
    r"""The FiLM graph convolutional operator from the
    `"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
    <https://arxiv.org/abs/1906.12192>`_ paper.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int = 1,
        nn: Optional[Callable] = None,
        act: Optional[Union[str, Callable]] = "relu",
        aggr: str = "mean",
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = max(num_relations, 1)
        self.act = activations.get(act) if act is not None else None

        if isinstance(in_channels, int):
            self.in_channels_l = in_channels
            self.in_channels_r = in_channels
        else:
            self.in_channels_l, self.in_channels_r = in_channels

        self.lins = [
            Dense(out_channels, use_bias=False) for _ in range(self.num_relations)
        ]
        self.films = []
        for _ in range(self.num_relations):
            if nn is None:
                self.films.append(Dense(2 * out_channels, use_bias=True))
            else:
                self.films.append(copy.deepcopy(nn))

        self.lin_skip = Dense(out_channels, use_bias=False)
        if nn is None:
            self.film_skip = Dense(2 * out_channels, use_bias=False)
        else:
            self.film_skip = copy.deepcopy(nn)

    def build(self, input_shape=None):
        for lin in self.lins:
            lin.build((None, self.in_channels_l))
        for film in self.films:
            film.build((None, self.in_channels_r))
        self.lin_skip.build((None, self.in_channels_r))
        self.film_skip.build((None, self.in_channels_r))
        self.built = True

    def call(self, inputs, edge_index=None, edge_type=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, edge_index, edge_type = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected inputs with edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_l, x_r = x
        else:
            x_l = x_r = x

        edge_index = ops.cast(edge_index, "int32")
        if edge_type is not None:
            edge_type = ops.cast(edge_type, "int32")

        # Skip connection
        film_s = self.film_skip(x_r)
        beta_s, gamma_s = ops.split(film_s, 2, axis=-1)
        out = gamma_s * self.lin_skip(x_r) + beta_s
        if self.act is not None:
            out = self.act(out)

        # Message passing per relation
        num_nodes = ops.shape(x_r)[0]
        size = (ops.shape(x_l)[0], num_nodes)

        for i in range(self.num_relations):
            if edge_type is not None:
                mask = ops.equal(edge_type, i)
                idx = ops.reshape(ops.where(mask), (-1,))
                idx = ops.cast(idx, "int32")
                if ops.shape(idx)[0] == 0:
                    continue
                edge_index_i = ops.take(edge_index, idx, axis=1)
            else:
                edge_index_i = edge_index

            film_val = self.films[i](x_r)
            beta, gamma = ops.split(film_val, 2, axis=-1)
            h = self.propagate(
                edge_index_i,
                x=self.lins[i](x_l),
                beta=beta,
                gamma=gamma,
                size=size,
            )
            out = out + h

        return out

    def message(self, x_j, beta_i, gamma_i):
        out = gamma_i * x_j + beta_i
        if self.act is not None:
            out = self.act(out)
        return out

