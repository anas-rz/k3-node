import numpy as np
from keras import ops

from k3_node.models import ARGA, ARGVA, GAE, VGAE


def test_gae():
    model = GAE(encoder=lambda x: x)
    model.reset_parameters()

    x = ops.convert_to_tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]], dtype="float32")
    z = model.encode(x)
    assert np.allclose(ops.convert_to_numpy(z), ops.convert_to_numpy(x))

    adj = model.decoder.forward_all(z)
    expected = ops.sigmoid(ops.convert_to_tensor([
        [2.0, -1.0, 1.0],
        [-1.0, 5.0, 4.0],
        [1.0, 4.0, 5.0],
    ], dtype="float32"))
    assert np.allclose(ops.convert_to_numpy(adj), ops.convert_to_numpy(expected), atol=1e-5)

    edge_index = ops.convert_to_tensor([[0, 1], [1, 2]], dtype="int64")
    value = model.decode(z, edge_index)
    expected_value = ops.sigmoid(ops.convert_to_tensor([-1.0, 4.0], dtype="float32"))
    assert np.allclose(ops.convert_to_numpy(value), ops.convert_to_numpy(expected_value), atol=1e-5)

    pos_edge_index = ops.convert_to_tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype="int64"
    )
    z2 = ops.convert_to_tensor(np.random.randn(11, 16).astype("float32"))
    loss = model.recon_loss(z2, pos_edge_index)
    assert float(ops.convert_to_numpy(loss)) > 0

    auc, ap = model.test(z2, pos_edge_index, pos_edge_index)
    assert 0 <= auc <= 1 and 0 <= ap <= 1


def test_vgae():
    model = VGAE(encoder=lambda x: (x, x))

    x = ops.convert_to_tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]], dtype="float32")
    model.encode(x)
    assert float(ops.convert_to_numpy(model.kl_loss())) > 0

    model.eval()
    model.encode(x)


def test_arga():
    model = ARGA(encoder=lambda x: x, discriminator=lambda x: ops.convert_to_tensor([0.5]))
    model.reset_parameters()

    x = ops.convert_to_tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]], dtype="float32")
    z = model.encode(x)

    assert float(ops.convert_to_numpy(model.reg_loss(z))) > 0
    assert float(ops.convert_to_numpy(model.discriminator_loss(z))) > 0


def test_argva():
    model = ARGVA(encoder=lambda x: (x, x), discriminator=lambda x: ops.convert_to_tensor([0.5]))

    x = ops.convert_to_tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]], dtype="float32")
    model.encode(x)
    model.reparametrize(model._mu, model._logstd)
    assert float(ops.convert_to_numpy(model.kl_loss())) > 0


def test_init():
    import keras

    encoder = keras.layers.Dense(32)
    encoder.build((None, 16))
    decoder = keras.layers.Dense(16)
    decoder.build((None, 32))
    discriminator = keras.layers.Dense(1)
    discriminator.build((None, 32))

    GAE(encoder, decoder)
    VGAE(encoder, decoder)
    ARGA(encoder, discriminator, decoder)
    ARGVA(encoder, discriminator, decoder)


def test_fit_evaluate_all_variants():
    import keras
    import numpy as np
    from k3_node.data import Data
    from k3_node.layers import GCNConv

    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 20, size=(2, 60))
    data = Data(x=rng.random((20, 8)).astype("float32"), edge_index=edge_index,
                pos_edge_label_index=edge_index, neg_edge_label_index=rng.integers(0, 20, size=(2, 60)))

    class Encoder(keras.Model):
        def __init__(self, variational):
            super().__init__()
            self.variational = variational
            self.conv_mu, self.conv_logstd = GCNConv(8, 4), GCNConv(8, 4)

        def call(self, x, edge_index):
            mu = self.conv_mu(x, edge_index)
            return (mu, self.conv_logstd(x, edge_index)) if self.variational else mu

    def discriminator():
        return keras.Sequential([keras.layers.Dense(8, activation="relu"), keras.layers.Dense(1)])

    for model in [GAE(Encoder(False)), VGAE(Encoder(True)), ARGA(Encoder(False), discriminator()),
                  ARGVA(Encoder(True), discriminator())]:
        model.compile(keras.optimizers.Adam(0.01), discriminator_optimizer=keras.optimizers.Adam(0.01))
        model.embed(data)  # builds the encoder
        before = [np.array(v) for v in model.encoder.trainable_variables]
        history = model.fit(data, epochs=2, validation_data=data, verbose=0)
        assert set(history) == {"loss", "val_auc", "val_ap"} and len(history["loss"]) == 2
        assert any(not np.allclose(b, np.array(v)) for b, v in zip(before, model.encoder.trainable_variables))
        assert 0.0 <= model.evaluate(data)["auc"] <= 1.0
        assert tuple(model.embed(data).shape) == (20, 4)
