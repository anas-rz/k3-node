# ported from PyTorch Geometric

from keras import layers, ops


class Aggregation(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        x,
        index=None,
        ptr=None,
        dim_size=None,
        dim=-2,
        max_num_elements=None,
    ):
        pass

    def assert_index_present(self, index):
        if index is None:
            raise NotImplementedError("Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index):
        if index is not None and not ops.all(index[:-1] <= index[1:]):
            raise ValueError(
                "Can not perform aggregation since the 'index' tensor is not sorted. "
                "Specifically, if you use this aggregation as part of 'MessagePassing', "
                "ensure that 'edge_index' is sorted by destination nodes."
            )

    def assert_two_dimensional_input(self, x, axis):
        if len(ops.shape(x)) != 2:
            raise ValueError(
                f"Aggregation requires two-dimensional inputs (got '{x.ndim}')"
            )

        if axis not in [-2, 0]:
            raise ValueError(
                f"Aggregation needs to perform aggregation in first dimension (got '{dim}')"
            )

    def reduce(self, x, index=None, axis=-2, reduce_fn=ops.segment_sum):
        self.assert_two_dimensional_input(x, axis)
        if index is None:
            raise NotImplementedError("Aggregation requires 'index' to be specified")
        return reduce_fn(x, index)
