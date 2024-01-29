from keras import layers, ops


class MeanSubtractionNorm(layers.Layer):
    """
    `k3_node.layers.MeanSubtractionNorm`
    Implementation of Mean Subtraction Norm layer

    Args:
        **kwargs: Additional arguments to pass to the `Layer` superclass.

    Call Arguments:
        `x` Input tensor.
    
    Call Returns:
        Output tensor of the same shape as `x`.
    ```python
    import numpy as np
    from k3_node.layers import MeanSubtractionNorm
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    layer = MeanSubtractionNorm()
    layer(x).numpy()
    ```
    """
    def __init__(self, **kwargs):
        super(MeanSubtractionNorm, self).__init__(**kwargs)

    def call(self, x):
        return x - ops.mean(x, axis=0, keepdims=True)
