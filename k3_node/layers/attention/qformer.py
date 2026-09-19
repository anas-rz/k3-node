from keras import layers, ops


class QFormerEncoderLayer(layers.Layer):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        dropout=0.0,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dim // num_heads,
            dropout=dropout,
            use_bias=True,
        )

        self.linear1 = layers.Dense(hidden_dim)
        self.linear2 = layers.Dense(input_dim)

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.dropout = layers.Dropout(dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

        self.activation = activation

    def call(self, x, training=None):
        # PyTorch TransformerEncoderLayer, norm_first=False

        attn = self.self_attn(
            x,
            x,
            training=training,
        )

        x = self.norm1(
            x + self.dropout1(attn, training=training)
        )

        ff = self.linear1(x)

        if self.activation == "relu":
            ff = ops.relu(ff)
        elif self.activation == "gelu":
            ff = ops.gelu(ff)
        else:
            raise ValueError(
                f"Unsupported activation: {self.activation}"
            )

        ff = self.dropout(ff, training=training)
        ff = self.linear2(ff)

        x = self.norm2(
            x + self.dropout2(ff, training=training)
        )

        return x


class QFormer(layers.Layer):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_heads,
        num_layers,
        dropout=0.0,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-5
        )

        self.encoder = [
            QFormerEncoderLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ]

        self.project = layers.Dense(output_dim)

    def call(self, x, training=None):
        x = self.layer_norm(x)

        for layer in self.encoder:
            x = layer(x, training=training)

        return self.project(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        })
        return config