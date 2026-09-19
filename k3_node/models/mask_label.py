from typing import Optional

import keras
from keras import ops
import numpy as np


class MaskLabel(keras.layers.Layer):
    r"""The label embedding and masking layer from the `"Masked Label
    Prediction: Unified Message Passing Model for Semi-Supervised
    Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

    Here, node labels :obj:`y` are merged to the initial node features :obj:`x`
    for a subset of their nodes according to :obj:`mask`.

    Args:
        num_classes (int): The number of classes.
        out_channels (int): Size of each output sample.
        method (str, optional): If set to :obj:`"add"`, label embeddings are
            added to the input. If set to :obj:`"concat"`, label embeddings are
            concatenated. In case :obj:`method="add"`, then :obj:`out_channels`
            needs to be identical to the input dimensionality of node features.
            (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        method: str = "add",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.out_channels = out_channels
        self.method = method

        if method not in ["add", "concat"]:
            raise ValueError(
                f"'method' must be either 'add' or 'concat' (got '{method}')"
            )

        self.emb = keras.layers.Embedding(num_classes, out_channels)

    def build(self, input_shape=None):
        self.emb.build((None,))
        self.built = True

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        if self.emb.built:
            self.emb.embeddings.assign(
                keras.initializers.Uniform()(self.emb.embeddings.shape)
            )

    def call(self, x, y, mask):
        """Forward pass.

        Args:
            x (Tensor): The input node features of shape ``[N, in_channels]``.
            y (Tensor): The node labels of shape ``[N]`` (integer class indices).
            mask (Tensor): Boolean tensor of shape ``[N]`` indicating which
                nodes have ground-truth labels to embed.
        """
        # Embed all labels; then zero-out non-masked entries
        all_emb = self.emb(y)  # [N, out_channels]

        # Build a float mask: 1.0 where mask is True, 0.0 elsewhere
        float_mask = ops.cast(mask, dtype=all_emb.dtype)  # [N]
        float_mask = ops.expand_dims(float_mask, axis=-1)  # [N, 1]
        masked_emb = all_emb * float_mask  # [N, out_channels]

        if self.method == "concat":
            return ops.concatenate([x, masked_emb], axis=-1)
        else:
            return x + masked_emb

    @staticmethod
    def ratio_mask(mask, ratio: float):
        r"""Modifies :obj:`mask` by setting :obj:`ratio` of :obj:`True`
        entries to :obj:`False`. Does not operate in-place.

        Args:
            mask (Tensor): The boolean mask to re-mask.
            ratio (float): The ratio of True entries to keep.
        """
        mask_np = ops.convert_to_numpy(mask).astype(bool)
        n = int(mask_np.sum())
        out_np = mask_np.copy()
        if n > 0:
            keep = np.random.rand(n) < ratio
            true_indices = np.where(mask_np)[0]
            out_np[true_indices] = keep
        return ops.convert_to_tensor(out_np, dtype="bool")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

