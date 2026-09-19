import copy

import keras
from keras import ops

from k3_node.models.utils import reset, uniform_

EPS = 1e-15


class DeepGraphInfomax(keras.layers.Layer):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder: The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """
    def __init__(self, hidden_channels, encoder, summary, corruption, **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = self.add_weight(
            shape=(hidden_channels, hidden_channels),
            initializer="zeros",
            name="weight",
        )
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.summary)
        uniform_(self.hidden_channels, self.weight)

    def call(self, *args, **kwargs):
        r"""Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor,)
        cor_args = cor[: len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value

        neg_z = self.encoder(*cor_args, **cor_kwargs)

        summary = self.summary(pos_z, *args, **kwargs)

        return pos_z, neg_z, summary

    def discriminate(self, z, summary, sigmoid: bool = True):
        r"""Given the patch-summary pair `z` and `summary`, computes the
        probability scores assigned to this patch-summary pair."""
        summary = ops.transpose(summary) if len(ops.shape(summary)) > 1 else summary
        value = ops.matmul(z, ops.matmul(self.weight, summary))
        return ops.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -ops.mean(ops.log(self.discriminate(pos_z, summary, sigmoid=True) + EPS))
        neg_loss = -ops.mean(ops.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS))

        return pos_loss + neg_loss

    def test(self, train_z, train_y, test_z, test_y, solver="lbfgs", *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression
        downstream task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(*args, solver=solver, **kwargs).fit(
            ops.convert_to_numpy(train_z), ops.convert_to_numpy(train_y)
        )
        return clf.score(ops.convert_to_numpy(test_z), ops.convert_to_numpy(test_y))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.hidden_channels})"
