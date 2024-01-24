from keras.utils import normalize
from keras import layers


class MessageNorm(layers.Layer):
    def __init__(self, learn_scale=False, **kwargs):
        super().__init__(**kwargs)
        self.learn_scale = learn_scale
        self.scale = self.add_weight((1,), initializer="ones", trainable=learn_scale)

    def call(self, inputs, p=2):
        x, msg = inputs
        msg = normalize(msg, axis=-1, order=p)
        x_norm = normalize(x, axis=-1, order=p)
        return msg * x_norm * self.scale
