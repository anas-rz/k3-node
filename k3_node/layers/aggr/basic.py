from typing import Optional
from keras import initializers, ops

from .base import Aggregation


class SumAggregation(Aggregation):
    r"""An aggregation operator that sums up features across a set of elements."""

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        return self.reduce(x, index, ptr, dim_size, dim, reduce="sum")


class MeanAggregation(Aggregation):
    r"""An aggregation operator that averages features across a set of elements."""

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        return self.reduce(x, index, ptr, dim_size, dim, reduce="mean")


class MaxAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise maximum across a set of elements."""

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        return self.reduce(x, index, ptr, dim_size, dim, reduce="max")


class MinAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise minimum across a set of elements."""

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        return self.reduce(x, index, ptr, dim_size, dim, reduce="min")


class MulAggregation(Aggregation):
    r"""An aggregation operator that multiplies features across a set of elements."""

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        self.assert_index_present(index)
        return self.reduce(x, index, ptr, dim_size, dim, reduce="mul")


class VarAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise variance across a set of elements."""

    def __init__(self, semi_grad: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.semi_grad = semi_grad

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        x_sq = ops.power(x, 2)
        if self.semi_grad:
            x_sq = ops.stop_gradient(x_sq)
        mean2 = self.reduce(x_sq, index, ptr, dim_size, dim, reduce="mean")
        return mean2 - ops.power(mean, 2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(semi_grad={self.semi_grad})"


class StdAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise standard deviation across a set of elements."""

    def __init__(self, semi_grad: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.semi_grad = semi_grad
        self.var_aggr = VarAggregation(semi_grad=semi_grad)

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        var = self.var_aggr(x, index, ptr, dim_size, dim)
        out = ops.sqrt(ops.maximum(var, 1e-5))
        out = ops.where(out <= (1e-5**0.5), 0.0, out)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(semi_grad={self.semi_grad})"


class SoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term."""

    def __init__(
        self,
        t: float = 1.0,
        learn: bool = False,
        semi_grad: bool = False,
        channels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable"
            )

        if not learn and channels != 1:
            raise ValueError(
                f"Cannot set 'channels' greater than '1' in case "
                f"'{self.__class__.__name__}' is not trainable"
            )

        self._init_t = t
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels

        if learn:
            self.t = self.add_weight(
                shape=(channels,),
                initializer=initializers.Constant(t),
                trainable=True,
                name="t",
            )
        else:
            self.t = t

    def reset_parameters(self):
        if self.learn:
            self.t.assign(ops.full((self.channels,), self._init_t, dtype=self.t.dtype))

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        t = self.t
        if self.channels != 1:
            self.assert_two_dimensional_input(x, dim)
            t = ops.reshape(t, (1, self.channels))

        alpha = x
        if self.learn or t != 1.0:
            alpha = x * t

        if not self.learn and self.semi_grad:
            alpha = ops.stop_gradient(alpha)

        # Graph-wise softmax over segments
        index = ops.cast(index, dtype="int32")
        dim_size = dim_size or (int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0)

        max_val = ops.segment_max(alpha, index, num_segments=dim_size)
        max_exp = ops.take(max_val, index, axis=0)
        exp_alpha = ops.exp(alpha - max_exp)
        sum_exp = ops.segment_sum(exp_alpha, index, num_segments=dim_size)
        sum_exp_taken = ops.take(sum_exp, index, axis=0)
        alpha_sm = exp_alpha / ops.maximum(sum_exp_taken, 1e-12)

        return self.reduce(x * alpha_sm, index, ptr, dim_size, dim, reduce="sum")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(learn={self.learn})"


class PowerMeanAggregation(Aggregation):
    r"""The powermean aggregation operator based on a power term."""

    def __init__(
        self,
        p: float = 1.0,
        learn: bool = False,
        channels: int = 1,
        clamp_min: Optional[float] = 1e-4,
        clamp_max: Optional[float] = 100.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not learn and channels != 1:
            raise ValueError(
                f"Cannot set 'channels' greater than '1' in case "
                f"'{self.__class__.__name__}' is not trainable"
            )

        self._init_p = p
        self.learn = learn
        self.channels = channels
        self.min_value = clamp_min if clamp_min is not None else 1e-4
        self.max_value = clamp_max if clamp_max is not None else 100.0

        if learn:
            self.p = self.add_weight(
                shape=(channels,),
                initializer=initializers.Constant(p),
                trainable=True,
                name="p",
            )
        else:
            self.p = p

    def reset_parameters(self):
        if self.learn:
            self.p.assign(ops.full((self.channels,), self._init_p, dtype=self.p.dtype))

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        p = self.p
        if self.channels != 1:
            self.assert_two_dimensional_input(x, dim)
            p = ops.reshape(p, (-1, self.channels))

        if self.learn or p != 1.0:
            x = ops.power(ops.clip(x, self.min_value, self.max_value), p)

        out = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")

        if self.learn or p != 1.0:
            out = ops.power(ops.clip(out, self.min_value, self.max_value), 1.0 / p)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(learn={self.learn})"
