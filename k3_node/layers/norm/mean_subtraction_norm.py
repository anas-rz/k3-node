from keras import layers, ops
class MeanSubtractionNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(MeanSubtractionNorm, self).__init__(**kwargs)

    def call(self, x):
        return x - ops.mean(x, axis=0, keepdims=True)
