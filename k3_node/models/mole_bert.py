import os
import os.path as osp
import shutil
from typing import Optional, Tuple, Union

import keras
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import add_self_loops
from k3_node.layers.pool import global_add_pool, global_max_pool, global_mean_pool
from k3_node.data.download import download_url


MOLE_BERT_URL = (
    "https://github.com/junxia97/Mole-BERT/raw/refs/heads/main/model_gin/Mole-BERT.pth"
)


class MoleBERTGINConv(MessagePassing):
    """Extension of GIN aggregation to incorporate categorical edge information with self-loops.

    Matches the GINConv variant from Hu et al. used in Mole-BERT.
    """

    def __init__(
        self,
        emb_dim: int = 300,
        out_dim: Optional[int] = None,
        num_bond_type: int = 6,
        num_bond_direction: int = 3,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.emb_dim = emb_dim
        self.out_dim = emb_dim if out_dim is None else out_dim
        self.num_bond_type = num_bond_type
        self.num_bond_direction = num_bond_direction

        self.mlp = keras.Sequential(
            [
                layers.Dense(2 * emb_dim, activation="relu", name="mlp_0"),
                layers.Dense(self.out_dim, name="mlp_2"),
            ],
            name="mlp",
        )
        self.edge_embedding1 = layers.Embedding(
            num_bond_type, emb_dim, name="edge_embedding1"
        )
        self.edge_embedding2 = layers.Embedding(
            num_bond_direction, emb_dim, name="edge_embedding2"
        )

    def build(self, input_shape=None):
        self.mlp.build((None, self.emb_dim))
        self.edge_embedding1.build((None,))
        self.edge_embedding2.build((None,))
        super().build(input_shape)

    def call(self, x, edge_index, edge_attr):
        num_nodes = ops.shape(x)[0]

        # Add self-loops to edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Add features corresponding to self-loop edges: [4, 0]
        self_loop_attr = ops.stack(
            [
                ops.full((num_nodes,), 4, dtype=edge_attr.dtype),
                ops.zeros((num_nodes,), dtype=edge_attr.dtype),
            ],
            axis=1,
        )
        edge_attr = ops.concatenate([edge_attr, self_loop_attr], axis=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embeddings, size=(num_nodes, num_nodes)
        )

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class MoleBERTGNN(layers.Layer):
    """5-layer GIN encoder backbone of Mole-BERT with Jumping Knowledge."""

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        num_atom_type: int = 120,
        num_chirality_tag: int = 3,
        JK: str = "last",
        drop_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_atom_type = num_atom_type
        self.num_chirality_tag = num_chirality_tag
        self.JK = JK
        self.drop_ratio = drop_ratio

        self.x_embedding1 = layers.Embedding(
            num_atom_type, emb_dim, name="x_embedding1"
        )
        self.x_embedding2 = layers.Embedding(
            num_chirality_tag, emb_dim, name="x_embedding2"
        )

        self.gnns = [
            MoleBERTGINConv(emb_dim=emb_dim, name=f"gnns_{i}")
            for i in range(num_layer)
        ]
        self.batch_norms = [
            layers.BatchNormalization(
                axis=-1, epsilon=1e-5, momentum=0.9, name=f"batch_norms_{i}"
            )
            for i in range(num_layer)
        ]
        self.dropout_layer = layers.Dropout(drop_ratio)

    def build(self, input_shape=None):
        self.x_embedding1.build((None,))
        self.x_embedding2.build((None,))
        for i in range(self.num_layer):
            self.gnns[i].build(None)
            self.batch_norms[i].build((None, self.emb_dim))
        super().build(input_shape)

    def call(self, x, edge_index, edge_attr, training: bool = False):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [h]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h, training=training)
            if layer == self.num_layer - 1:
                # Remove ReLU for the last layer
                h = self.dropout_layer(h, training=training)
            else:
                h = self.dropout_layer(ops.relu(h), training=training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = ops.concatenate(h_list, axis=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            node_representation = ops.max(ops.stack(h_list, axis=0), axis=0)
        elif self.JK == "sum":
            node_representation = ops.sum(ops.stack(h_list, axis=0), axis=0)
        else:
            raise ValueError(f"Unknown JK mode: {self.JK}")

        return node_representation


class MoleBERT(keras.Model):
    """Complete Mole-BERT Model with graph-level pooling and property prediction head."""

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        num_tasks: Optional[int] = None,
        JK: str = "last",
        drop_ratio: float = 0.0,
        graph_pooling: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling

        self.gnn = MoleBERTGNN(
            num_layer=num_layer,
            emb_dim=emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            name="gnn",
        )

        if graph_pooling in ("sum", "add"):
            self.pool_fn = global_add_pool
        elif graph_pooling == "mean":
            self.pool_fn = global_mean_pool
        elif graph_pooling == "max":
            self.pool_fn = global_max_pool
        else:
            raise ValueError(f"Invalid graph pooling type: '{graph_pooling}'")

        if num_tasks is not None:
            mult = (num_layer + 1) if JK == "concat" else 1
            self.graph_pred_linear = layers.Dense(
                num_tasks, name="graph_pred_linear"
            )
        else:
            self.graph_pred_linear = None

    def build(self, input_shape=None):
        self.gnn.build(None)
        if self.graph_pred_linear is not None:
            mult = (self.num_layer + 1) if self.JK == "concat" else 1
            self.graph_pred_linear.build((None, mult * self.emb_dim))
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        """Call MoleBERT model.

        inputs can be a tuple: `(x, edge_index, edge_attr)` or `(x, edge_index, edge_attr, batch)`.
        """
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3:
                x, edge_index, edge_attr = inputs
                batch = None
            elif len(inputs) == 4:
                x, edge_index, edge_attr, batch = inputs
            else:
                raise ValueError("Expected 3 or 4 input tensors.")
        elif isinstance(inputs, dict):
            x = inputs["x"]
            edge_index = inputs["edge_index"]
            edge_attr = inputs["edge_attr"]
            batch = inputs.get("batch", None)
        else:
            raise ValueError("inputs must be a tuple, list, or dict.")

        node_rep = self.gnn(x, edge_index, edge_attr, training=training)

        if batch is not None:
            graph_rep = self.pool_fn(node_rep, batch)
            if self.graph_pred_linear is not None:
                logits = self.graph_pred_linear(graph_rep)
                return logits, node_rep
            return graph_rep, node_rep

        if self.graph_pred_linear is not None:
            # If no batch given, default all nodes to batch 0
            num_nodes = ops.shape(node_rep)[0]
            batch_zeros = ops.zeros((num_nodes,), dtype="int32")
            graph_rep = self.pool_fn(node_rep, batch_zeros)
            logits = self.graph_pred_linear(graph_rep)
            return logits, node_rep

        return node_rep


def load_mole_bert_weights(model: Union[MoleBERT, MoleBERTGNN], checkpoint_path: str):
    """Loads official PyTorch Mole-BERT.pth checkpoint state dict into Keras 3 MoleBERT model."""
    import torch

    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    def get_np(key):
        return state_dict[key].detach().cpu().float().numpy()

    if not model.built:
        model.build(None)

    gnn = model.gnn if hasattr(model, "gnn") else model

    # Embeddings
    gnn.x_embedding1.set_weights([get_np("x_embedding1.weight")])
    gnn.x_embedding2.set_weights([get_np("x_embedding2.weight")])

    for l in range(gnn.num_layer):
        conv = gnn.gnns[l]
        bn = gnn.batch_norms[l]

        # GINConv edge embeddings
        conv.edge_embedding1.set_weights([get_np(f"gnns.{l}.edge_embedding1.weight")])
        conv.edge_embedding2.set_weights([get_np(f"gnns.{l}.edge_embedding2.weight")])

        # GINConv MLP layers
        w0 = get_np(f"gnns.{l}.mlp.0.weight").T
        b0 = get_np(f"gnns.{l}.mlp.0.bias")
        conv.mlp.layers[0].set_weights([w0, b0])

        w2 = get_np(f"gnns.{l}.mlp.2.weight").T
        b2 = get_np(f"gnns.{l}.mlp.2.bias")
        conv.mlp.layers[1].set_weights([w2, b2])

        # Batch Normalization
        gamma = get_np(f"batch_norms.{l}.weight")
        beta = get_np(f"batch_norms.{l}.bias")
        mean = get_np(f"batch_norms.{l}.running_mean")
        var = get_np(f"batch_norms.{l}.running_var")
        bn.set_weights([gamma, beta, mean, var])


def download_mole_bert_checkpoint(cache_dir: Optional[str] = None) -> str:
    """Downloads official Mole-BERT.pth checkpoint from GitHub."""
    if cache_dir is None:
        cache_dir = osp.expanduser("~/.cache/k3_node/mole_bert")

    os.makedirs(cache_dir, exist_ok=True)
    target_path = osp.join(cache_dir, "Mole-BERT.pth")

    if osp.exists(target_path) and osp.getsize(target_path) > 1000:
        return target_path

    # Check local path in Mole-BERT/model_gin/Mole-BERT.pth
    local_path = osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
        "Mole-BERT",
        "model_gin",
        "Mole-BERT.pth",
    )
    if osp.exists(local_path) and osp.getsize(local_path) > 1000:
        shutil.copyfile(local_path, target_path)
        return target_path

    return download_url(MOLE_BERT_URL, cache_dir, filename="Mole-BERT.pth")

