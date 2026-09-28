import keras
from keras import ops

from k3_node.models.utils import negative_sampling, reset

EPS = 1e-15
MAX_LOGSTD = 10


def _randn_like(x):
    return keras.random.normal(ops.shape(x), dtype=x.dtype)


class InnerProductDecoder:
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """
    def __call__(self, z, edge_index, sigmoid: bool = True):
        r"""Decodes the latent variables `z` into edge probabilities for
        the given node-pairs `edge_index`."""
        row, col = edge_index[0], edge_index[1]
        value = ops.sum(ops.take(z, row, axis=0) * ops.take(z, col, axis=0), axis=1)
        return ops.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid: bool = True):
        r"""Decodes the latent variables `z` into a probabilistic dense
        adjacency matrix."""
        adj = ops.matmul(z, ops.transpose(z))
        return ops.sigmoid(adj) if sigmoid else adj


class GAE:
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder: The encoder module.
        decoder (optional): The decoder module. If set to `None`, will
            default to `InnerProductDecoder`. (default: `None`)
    """
    def __init__(self, encoder, decoder=None):
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def __call__(self, *args, **kwargs):
        r"""Alias for `encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def to(self, *args, **kwargs):
        return self

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables `z`, computes the binary cross entropy
        loss for positive edges `pos_edge_index` and negative sampled
        edges."""
        pos_loss = -ops.mean(ops.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS))

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, ops.shape(z)[0])
        neg_loss = -ops.mean(ops.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS))

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables `z`, positive edges `pos_edge_index` and
        negative edges `neg_edge_index`, computes area under the ROC curve
        (AUC) and average precision (AP) scores."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = ops.ones((ops.shape(pos_edge_index)[1],))
        neg_y = ops.zeros((ops.shape(neg_edge_index)[1],))
        y = ops.concatenate([pos_y, neg_y], axis=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = ops.concatenate([pos_pred, neg_pred], axis=0)

        y, pred = ops.convert_to_numpy(y), ops.convert_to_numpy(pred)

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    # ---- Keras-style training -------------------------------------------------------------------
    def compile(self, optimizer, discriminator_optimizer=None, discriminator_steps: int = 5):
        r"""Sets the optimizers used by :meth:`fit`.

        Args:
            optimizer (keras.optimizers.Optimizer): Trains the encoder (and decoder).
            discriminator_optimizer (keras.optimizers.Optimizer, optional): Trains the
                discriminator of adversarial models (:class:`ARGA`, :class:`ARGVA`).
            discriminator_steps (int): Discriminator updates per encoder update. (default: ``5``)
        """
        self.optimizer = optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_steps = discriminator_steps

    def _encoder_loss(self, data):
        z = self.encode(data.x, data.edge_index, training=True)
        loss = self.recon_loss(z, data.pos_edge_label_index)
        if isinstance(self, ARGA):
            loss = loss + self.reg_loss(z)
        if hasattr(self, "kl_loss"):
            loss = loss + (1 / data.num_nodes) * self.kl_loss()
        return loss

    def train_step(self, data):
        r"""Runs one training step on ``data`` and returns the loss."""
        from k3_node.training import gradient_step

        self.train()
        if isinstance(self, ARGA):
            z = ops.stop_gradient(self.encode(data.x, data.edge_index, training=True))
            for _ in range(self.discriminator_steps):
                gradient_step(lambda: self.discriminator_loss(z), self.discriminator.trainable_variables,
                              self.discriminator_optimizer)
        return gradient_step(lambda: self._encoder_loss(data), self._trainable_variables(), self.optimizer)

    def _trainable_variables(self):
        variables = list(self.encoder.trainable_variables)
        return variables + list(getattr(self.decoder, "trainable_variables", []))

    def fit(self, data, epochs: int = 1, validation_data=None, verbose: int = 1):
        r"""Trains the model on one graph for ``epochs`` full-graph steps.

        Args:
            data (Data): The training graph with node features ``x``, the message passing edges
                ``edge_index`` and the edges to reconstruct ``pos_edge_label_index``, as created
                by :class:`~k3_node.transforms.RandomLinkSplit` with ``split_labels=True``.
            epochs (int): The number of training steps. (default: ``1``)
            validation_data (Data, optional): A graph with ``pos_edge_label_index`` and
                ``neg_edge_label_index`` on which AUC and average precision are reported.
            verbose (int): ``0`` is silent, otherwise one line is printed per epoch.

        Returns:
            dict: The loss (and validation metrics) of every epoch.
        """
        if getattr(self, "optimizer", None) is None:
            raise ValueError("Call `compile(optimizer=...)` before `fit`.")
        if isinstance(self, ARGA) and self.discriminator_optimizer is None:
            raise ValueError("Adversarial models need `compile(..., discriminator_optimizer=...)`.")
        # Create the variables before the first gradient step
        z = self.encode(data.x, data.edge_index)
        if isinstance(self, ARGA):
            self.discriminator(z)

        history = {"loss": []}
        for epoch in range(1, epochs + 1):
            logs = {"loss": self.train_step(data)}
            if validation_data is not None:
                logs.update({f"val_{k}": v for k, v in self.evaluate(validation_data).items()})
            for key, value in logs.items():
                history.setdefault(key, []).append(value)
            if verbose:
                print(f"Epoch {epoch:03d}: " + ", ".join(f"{k}: {v:.4f}" for k, v in logs.items()))
        return history

    def evaluate(self, data):
        r"""Returns the link prediction AUC and average precision on ``data`` (which needs
        ``pos_edge_label_index`` and ``neg_edge_label_index``)."""
        z = self.embed(data)
        auc, ap = self.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        return {"auc": float(auc), "ap": float(ap)}

    def embed(self, data):
        r"""Returns the node embeddings of ``data`` (without sampling noise)."""
        from k3_node.training import no_grad

        self.eval()
        with no_grad():
            return self.encode(data.x, data.edge_index, training=False)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder: The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (optional): The decoder module. If set to `None`, will
            default to `InnerProductDecoder`. (default: `None`)
    """
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)
        self.training = True

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + _randn_like(logstd) * ops.exp(logstd)
        return mu

    def encode(self, *args, **kwargs):
        self._mu, self._logstd = self.encoder(*args, **kwargs)
        self._logstd = ops.minimum(self._logstd, MAX_LOGSTD)
        z = self.reparametrize(self._mu, self._logstd)
        return z

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments `mu` and
        `logstd`, or based on latent variables from last encoding."""
        mu = self._mu if mu is None else mu
        logstd = self._logstd if logstd is None else ops.minimum(logstd, MAX_LOGSTD)
        return -0.5 * ops.mean(
            ops.sum(1 + 2 * logstd - ops.square(mu) - ops.square(ops.exp(logstd)), axis=1)
        )

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.

    Args:
        encoder: The encoder module.
        discriminator: The discriminator module.
        decoder (optional): The decoder module. If set to `None`, will
            default to `InnerProductDecoder`. (default: `None`)
    """
    def __init__(self, encoder, discriminator, decoder=None):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        reset(self.discriminator)

    def reset_parameters(self):
        super().reset_parameters()
        reset(getattr(self, "discriminator", None))

    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder."""
        real = ops.sigmoid(self.discriminator(z))
        return -ops.mean(ops.log(real + EPS))

    def discriminator_loss(self, z):
        r"""Computes the loss of the discriminator."""
        real = ops.sigmoid(self.discriminator(_randn_like(z)))
        fake = ops.sigmoid(self.discriminator(ops.stop_gradient(z)))
        real_loss = -ops.mean(ops.log(real + EPS))
        fake_loss = -ops.mean(ops.log(1 - fake + EPS))
        return real_loss + fake_loss


class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model
    from the `"Adversarially Regularized Graph Autoencoder for Graph
    Embedding" <https://arxiv.org/abs/1802.04407>`_ paper.

    Args:
        encoder: The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        discriminator: The discriminator module.
        decoder (optional): The decoder module. If set to `None`, will
            default to `InnerProductDecoder`. (default: `None`)
    """
    def __init__(self, encoder, discriminator, decoder=None):
        super().__init__(encoder, discriminator, decoder)
        self.vgae = VGAE(encoder, decoder)

    @property
    def _mu(self):
        return self.vgae._mu

    @property
    def _logstd(self):
        return self.vgae._logstd

    def reparametrize(self, mu, logstd):
        return self.vgae.reparametrize(mu, logstd)

    def encode(self, *args, **kwargs):
        return self.vgae.encode(*args, **kwargs)

    def kl_loss(self, mu=None, logstd=None):
        return self.vgae.kl_loss(mu, logstd)

    def eval(self):
        self.training = False
        self.vgae.eval()
        return self

    def train(self):
        self.training = True
        self.vgae.train()
        return self
