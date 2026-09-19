
import keras
from keras import ops

from k3_node.models.label_prop import LabelPropagation


class CorrectAndSmooth(keras.layers.Layer):
    r"""The correct and smooth (C&S) post-processing model from the
    `"Combining Label Propagation And Simple Models Out-performs Graph Neural
    Networks" <https://arxiv.org/abs/2010.13993>`_ paper.

    Args:
        num_correction_layers (int): The number of propagations :math:`L_1`.
        correction_alpha (float): The :math:`\alpha_1` coefficient.
        num_smoothing_layers (int): The number of propagations :math:`L_2`.
        smoothing_alpha (float): The :math:`\alpha_2` coefficient.
        autoscale (bool, optional): If set to :obj:`True`, will automatically
            determine the scaling factor :math:`\gamma`. (default: :obj:`True`)
        scale (float, optional): The scaling factor :math:`\gamma`, in case
            :obj:`autoscale = False`. (default: :obj:`1.0`)
    """
    def __init__(
        self,
        num_correction_layers: int,
        correction_alpha: float,
        num_smoothing_layers: int,
        smoothing_alpha: float,
        autoscale: bool = True,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers, correction_alpha)
        self.prop2 = LabelPropagation(num_smoothing_layers, smoothing_alpha)

    def build(self, input_shape=None):
        self.built = True

    def call(self, y_soft, *args, **kwargs):
        y_soft = self.correct(y_soft, *args, **kwargs)
        return self.smooth(y_soft, *args, **kwargs)

    def correct(self, y_soft, y_true, mask, edge_index, edge_weight=None):
        num_classes = ops.shape(y_soft)[-1]
        y_true_shape = ops.shape(y_true)
        if len(y_true_shape) == 1:
            y_true = ops.one_hot(y_true, num_classes)
        y_true = ops.cast(y_true, y_soft.dtype)

        mask_shape = ops.shape(mask)
        is_bool_mask = (len(mask_shape) == 1 and 'bool' in str(mask.dtype))
        if is_bool_mask:
            indices = ops.where(mask)[0]
        else:
            indices = mask

        numel = float(ops.shape(indices)[0])
        if ops.shape(y_true)[0] == ops.shape(y_soft)[0]:
            y_true_sub = ops.take(y_true, indices, axis=0)
        else:
            y_true_sub = y_true

        error_zeros = ops.zeros_like(y_soft)
        error = ops.scatter_update(
            error_zeros,
            ops.expand_dims(indices, -1),
            y_true_sub - ops.take(y_soft, indices, axis=0)
        )

        if self.autoscale:
            smoothed_error = self.prop1(
                error, edge_index, edge_weight=edge_weight,
                post_step=lambda x: ops.clip(x, -1.0, 1.0)
            )

            error_masked = ops.take(error, indices, axis=0)
            sigma = ops.sum(ops.abs(error_masked)) / max(numel, 1.0)
            sum_smoothed = ops.sum(ops.abs(smoothed_error), axis=1, keepdims=True)
            scale = sigma / ops.maximum(sum_smoothed, 1e-12)
            scale = ops.where(scale > 1000.0, ops.ones_like(scale), scale)
            return y_soft + scale * smoothed_error
        else:
            def fix_input(x):
                return ops.scatter_update(x, ops.expand_dims(indices, -1), ops.take(error, indices, axis=0))

            smoothed_error = self.prop1(
                error, edge_index, edge_weight=edge_weight,
                post_step=fix_input,
            )
            return y_soft + self.scale * smoothed_error

    def smooth(self, y_soft, y_true, mask, edge_index, edge_weight=None):
        num_classes = ops.shape(y_soft)[-1]
        y_true_shape = ops.shape(y_true)
        if len(y_true_shape) == 1:
            y_true = ops.one_hot(y_true, num_classes)
        y_true = ops.cast(y_true, y_soft.dtype)

        mask_shape = ops.shape(mask)
        is_bool_mask = (len(mask_shape) == 1 and 'bool' in str(mask.dtype))
        if is_bool_mask:
            indices = ops.where(mask)[0]
        else:
            indices = mask

        if ops.shape(y_true)[0] == ops.shape(y_soft)[0]:
            y_true_sub = ops.take(y_true, indices, axis=0)
        else:
            y_true_sub = y_true

        y_soft = ops.scatter_update(y_soft, ops.expand_dims(indices, -1), y_true_sub)
        return self.prop2(y_soft, edge_index, edge_weight=edge_weight)

    def __repr__(self):
        L1, alpha1 = self.prop1.num_layers, self.prop1.alpha
        L2, alpha2 = self.prop2.num_layers, self.prop2.alpha
        return (f'{self.__class__.__name__}(\n'
                f'  correct: num_layers={L1}, alpha={alpha1}\n'
                f'  smooth:  num_layers={L2}, alpha={alpha2}\n'
                f'  autoscale={self.autoscale}, scale={self.scale}\n'
                ')')
