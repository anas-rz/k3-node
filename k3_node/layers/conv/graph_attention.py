# ported from stellargraph
from keras import ops
from keras import activations, constraints, initializers, regularizers
from keras.layers import Layer, LeakyReLU, Dropout


class GraphAttention(Layer):
    """
    `k3_node.layers.GraphAttention`
    Implementation of Graph Attention (GAT) layer

    Args:
        units: Positive integer, dimensionality of the output space.
        attn_heads: Positive integer, number of attention heads.
        attn_heads_reduction: {'concat', 'average'} Method for reducing attention heads.
        in_dropout_rate: Dropout rate applied to the input (node features).
        attn_dropout_rate: Dropout rate applied to attention coefficients.
        activation: Activation function to use.
        use_bias: Whether to add a bias to the linear transformation.
        final_layer: Deprecated, use tf.gather or GatherIndices instead.
        saliency_map_support: Whether to support saliency map calculations.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer for the `kernel` weights matrix.
        kernel_constraint: Constraint for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        bias_regularizer: Regularizer for the bias vector.
        bias_constraint: Constraint for the bias vector.
        attn_kernel_initializer: Initializer for the attention kernel weights matrix.
        attn_kernel_regularizer: Regularizer for the attention kernel weights matrix.
        attn_kernel_constraint: Constraint for the attention kernel weights matrix.
        **kwargs: Additional arguments to pass to the `Layer` superclass.
    """
    def __init__(
        self,
        units,
        attn_heads=1,
        attn_heads_reduction="concat",  # {'concat', 'average'}
        in_dropout_rate=0.0,
        attn_dropout_rate=0.0,
        activation="relu",
        use_bias=True,
        final_layer=None,
        saliency_map_support=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        attn_kernel_initializer="glorot_uniform",
        attn_kernel_regularizer=None,
        attn_kernel_constraint=None,
        **kwargs,
    ):
        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError(
                "{}: Possible heads reduction methods: concat, average; received {}".format(
                    type(self).__name__, attn_heads_reduction
                )
            )

        self.units = units  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.in_dropout_rate = in_dropout_rate  # dropout rate for node features
        self.attn_dropout_rate = attn_dropout_rate  # dropout rate for attention coefs
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        if final_layer is not None:
            raise ValueError(
                "'final_layer' is not longer supported, use 'tf.gather' or 'GatherIndices' separately"
            )

        self.saliency_map_support = saliency_map_support
        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F')
            self.output_dim = self.units * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.units

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        super().__init__(**kwargs)

    def build(self, input_shapes):
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        # Variables to support integrated gradients
        self.delta = self.add_weight(
            name="ig_delta", shape=(), trainable=False, initializer=initializers.ones()
        )
        self.non_exist_edge = self.add_weight(
            name="ig_non_exist_edge",
            shape=(),
            trainable=False,
            initializer=initializers.zeros(),
        )

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel_{}".format(head),
            )
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    shape=(self.units,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias_{}".format(head),
                )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                shape=(self.units, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_self_{}".format(head),
            )
            attn_kernel_neighs = self.add_weight(
                shape=(self.units, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_neigh_{}".format(head),
            )
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (1 x N x F)
        A = inputs[1]  # Adjacency matrix (1 X N x N)
        N = ops.shape(A)[-1]

        assert len(ops.shape(A)) == 2, f"Adjacency matrix A should be 2-D"

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[
                head
            ]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network

            features = ops.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = ops.dot(
                features, attention_kernel[0]
            )  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = ops.dot(
                features, attention_kernel[1]
            )  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + ops.transpose(
                attn_for_neighs
            )  # (N x N) via broadcasting

            dense = LeakyReLU(0.2)(dense)

            if not self.saliency_map_support:
                mask = -10e9 * (1.0 - A)
                dense += mask
                dense = ops.softmax(dense)  # (N x N), Eq. 3 of the paper

            else:
                # dense = dense - tf.reduce_max(dense)
                # GAT with support for saliency calculations
                W = (self.delta * A) * ops.exp(
                    dense - ops.max(dense, axis=1, keepdims=True)
                ) * (1 - self.non_exist_edge) + self.non_exist_edge * (
                    A + self.delta * (ops.ones((N, N)) - A) + ops.eye(N)
                ) * ops.exp(
                    dense - ops.max(dense, axis=1, keepdims=True)
                )
                dense = W / ops.sum(W, axis=1, keepdims=True)

            # Apply dropout to features and attention coefficients
            dropout_feat = Dropout(self.in_dropout_rate)(features)  # (N x F')
            dropout_attn = Dropout(self.attn_dropout_rate)(dense)  # (N x N)

            # Linear combination with neighbors' features [YT: see Eq. 4]
            node_features = ops.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = ops.add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = ops.concatenate(outputs, axis=1)  # (N x KF')
        else:
            output = ops.mean(ops.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)

        return output
