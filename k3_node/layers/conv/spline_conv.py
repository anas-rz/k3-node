from typing import List, Optional, Tuple, Union
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing


class SplineConv(MessagePassing):
    r"""The spline-based convolutional operator from the `"SplineCNN: Fast
    Geometric Deep Learning with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper.

    Args:
        in_channels (int or tuple): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or List[int]): Size of the convolving kernel.
        is_open_spline (bool or List[bool], optional): If set to :obj:`False`,
            uses closed B-spline basis. (default: :obj:`True`)
        degree (int, optional): B-spline basis degree. (default: :obj:`1`)
        aggr (str, optional): The aggregation scheme to use (:obj:`"mean"`,
            :obj:`"add"`, :obj:`"max"`). (default: :obj:`"mean"`)
        root_weight (bool, optional): Whether to add transformed root node
            features. (default: :obj:`True`)
        bias (bool, optional): Whether to learn an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        dim: int,
        kernel_size: Union[int, List[int]],
        is_open_spline: Union[bool, List[bool]] = True,
        degree: int = 1,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        assert len(kernel_size) == dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.degree = degree
        self.root_weight = root_weight
        self.use_bias = bias

        # Total number of spline basis elements
        k_prod = 1
        for k in kernel_size:
            k_prod *= k
        self.K = k_prod

        # Compute strides for multidimensional indexing
        strides = []
        stride = 1
        for k in reversed(kernel_size):
            strides.insert(0, stride)
            stride *= k
        self.strides = strides

    def build(self, input_shape=None):
        in_dim = self.in_channels[0]
        if in_dim == -1 and input_shape is not None:
            if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
                in_dim = input_shape[0][-1]
            elif isinstance(input_shape, (list, tuple)):
                in_dim = input_shape[-1]
            self.in_channels = (in_dim, self.in_channels[1] if self.in_channels[1] != -1 else in_dim)

        self.weight = self.add_weight(
            shape=(self.K, self.in_channels[0], self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
            name="weight",
        )

        if self.root_weight:
            self.root_lin = keras.layers.Dense(self.out_channels, use_bias=False)
            self.root_lin.build((None, self.in_channels[1]))

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

        super().build(input_shape)

    def _spline_basis_1d(self, e_d, k_d):
        # e_d: (E,) in [0, 1]
        u = ops.clip(e_d * (k_d - 1), 0.0, float(k_d - 1))
        i0 = ops.cast(ops.floor(u), "int32")
        i0 = ops.clip(i0, 0, k_d - 1)
        i1 = ops.clip(i0 + 1, 0, k_d - 1)
        w1 = u - ops.cast(i0, u.dtype)
        w0 = 1.0 - w1
        return [(i0, w0), (i1, w1)]

    def _compute_kernel(self, edge_attr):
        # edge_attr: (E, D)
        E = ops.shape(edge_attr)[0]
        # Basis product over dimensions
        dim_bases = [
            self._spline_basis_1d(edge_attr[:, d], self.kernel_size[d])
            for d in range(self.dim)
        ]

        # Cartesian product of basis across dimensions
        basis_combinations = [([], 1.0)]
        for d in range(self.dim):
            new_combinations = []
            stride = self.strides[d]
            for curr_indices, curr_weight in basis_combinations:
                for idx, w in dim_bases[d]:
                    new_idx = curr_indices + [idx * stride]
                    new_w = curr_weight * w
                    new_combinations.append((new_idx, new_w))
            basis_combinations = new_combinations

        # Construct edge-wise weight matrix W_eff: (E, C_in, C_out)
        W_eff = ops.zeros((E, self.in_channels[0], self.out_channels), dtype=self.weight.dtype)
        for idx_parts, basis_w in basis_combinations:
            total_idx = idx_parts[0]
            for part in idx_parts[1:]:
                total_idx = total_idx + part
            # total_idx: (E,)
            basis_weights = ops.take(self.weight, total_idx, axis=0)  # (E, C_in, C_out)
            basis_w_expanded = ops.expand_dims(ops.expand_dims(basis_w, axis=-1), axis=-1)
            W_eff = W_eff + basis_w_expanded * basis_weights

        return W_eff

    def call(self, x, edge_index, edge_attr=None, **kwargs):
        if isinstance(x, (list, tuple)):
            x_src, x_dst = x[0], x[1]
        else:
            x_src = x_dst = x

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)

        if self.root_weight and x_dst is not None:
            out = out + self.root_lin(x_dst)

        if self.use_bias:
            out = out + self.bias

        return out

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return x_j
        W_eff = self._compute_kernel(edge_attr)  # (E, C_in, C_out)
        # x_j: (E, C_in)
        # m = sum_c x_j * W_eff
        return ops.einsum("ec,ecd->ed", x_j, W_eff)

