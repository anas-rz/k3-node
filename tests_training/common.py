import os
import numpy as np
import pytest
import keras
from keras import layers, ops
import k3_node.layers as k3_layers

CONV_LAYERS = [
    "AGNNConv",
    "AntiSymmetricConv",
    "APPNP",
    "APPNPConv",
    "ARMAConv",
    "CGConv",
    "ChebConv",
    "ClusterGCNConv",
    "CrystalConv",
    "CuGraphGATConv",
    "CuGraphSAGEConv",
    "CuGraphRGCNConv",
    "DiffusionConv",
    "DirGNNConv",
    "DNAConv",
    "DynamicEdgeConv",
    "EdgeConv",
    "EGConv",
    "FAConv",
    "FastRGCNConv",
    "FeaStConv",
    "FiLMConv",
    "FusedGATConv",
    "GATConv",
    "GatedGraphConv",
    "GATv2Conv",
    "GCN2Conv",
    "GCNConv",
    "GENConv",
    "GeneralConv",
    "GINConv",
    "GINEConv",
    "GMMConv",
    "GPSConv",
    "GraphAttention",
    "GraphConvolution",
    "GraphConv",
    "GravNetConv",
    "HANConv",
    "HEATConv",
    "HeteroConv",
    "HGTConv",
    "HypergraphConv",
    "LEConv",
    "LGConv",
    "MeshCNNConv",
    "MessagePassing",
    "MFConv",
    "MixHopConv",
    "ECConv",
    "NNConv",
    "PANConv",
    "PDNConv",
    "PNAConv",
    "PointConv",
    "PointNetConv",
    "PointGNNConv",
    "PointTransformerConv",
    "PPFConv",
    "PPNPPropagation",
    "ResGatedGraphConv",
    "RGATConv",
    "RGCNConv",
    "SAGEConv",
    "SGConv",
    "SignedConv",
    "SimpleConv",
    "SplineConv",
    "SSGConv",
    "SuperGATConv",
    "TAGConv",
    "TransformerConv",
    "WLConv",
    "WLConvContinuous",
    "XConv",
]

NORM_LAYERS = [
    "BatchNorm",
    "DiffGroupNorm",
    "GraphNorm",
    "GraphSizeNorm",
    "HeteroBatchNorm",
    "HeteroLayerNorm",
    "InstanceNorm",
    "LayerNorm",
    "MeanSubtractionNorm",
    "MessageNorm",
    "PairNorm",
]

POOL_LAYERS = [
    "ASAPooling",
    "ClusterPooling",
    "Connect",
    "EdgePooling",
    "FilterEdges",
    "MemPooling",
    "PANPooling",
    "SAGPooling",
    "Select",
    "SelectTopK",
    "TopKPooling",
]

AGGR_LAYERS = [
    "Aggregation",
    "AttentionalAggregation",
    "DeepSetsAggregation",
    "DegreeScalerAggregation",
    "EquilibriumAggregation",
    "FusedAggregation",
    "GraphMultisetTransformer",
    "GRUAggregation",
    "LCMAggregation",
    "LSTMAggregation",
    "MaxAggregation",
    "MeanAggregation",
    "MedianAggregation",
    "MinAggregation",
    "MLPAggregation",
    "MulAggregation",
    "MultiAggregation",
    "PatchTransformerAggregation",
    "PowerMeanAggregation",
    "QuantileAggregation",
    "Set2Set",
    "SetTransformerAggregation",
    "SoftmaxAggregation",
    "SortAggregation",
    "StdAggregation",
    "SumAggregation",
    "VarAggregation",
    "VariancePreservingAggregation",
]

DENSE_LAYERS = [
    "DenseGATConv",
    "DenseGCNConv",
    "DenseGINConv",
    "DenseGraphConv",
    "DenseSAGEConv",
    "DMoNPooling",
    "HeteroDictLinear",
    "HeteroLinear",
    "Linear",
]

ATTENTION_LAYERS = [
    "PerformerAttention",
    "PerformerProjection",
    "PolynormerAttention",
    "QFormer",
    "QFormerEncoderLayer",
    "SGFormerAttention",
]

KGE_LAYERS = [
    "ComplEx",
    "DistMult",
    "KGEModel",
    "RotatE",
    "TransE",
]


def get_layer_test(layer_name):
    """Returns (model_factory, inputs_factory, target_factory, has_weights, skip_reason)."""
    in_c = 8
    out_c = 4
    N = 10
    E = 20

    # Base / Abstract / Vendor layers
    if layer_name in ["MessagePassing", "Aggregation", "Connect", "Select", "KGEModel"]:
        return None, None, None, False, "Base/Abstract class"
    if layer_name.startswith("CuGraph"):
        return None, None, None, False, "cuGraph vendor layer (requires GPU cuGraph bindings)"
    if layer_name == "WLConv":
        return None, None, None, False, "Discrete non-differentiable Weisfeiler-Lehman graph coloring operator"

    # =========================================================================
    # Group 1: CONVOLUTION LAYERS
    # =========================================================================
    std_convs = {
        "AGNNConv": lambda: k3_layers.AGNNConv(requires_grad=True),
        "AntiSymmetricConv": lambda: k3_layers.AntiSymmetricConv(in_c),
        "APPNP": lambda: k3_layers.APPNP(K=2, alpha=0.1),
        "APPNPConv": lambda: k3_layers.APPNPConv(out_c, alpha=0.1, propagations=2),
        "ARMAConv": lambda: k3_layers.ARMAConv(in_c, out_c, num_stacks=1, num_layers=1),
        "ChebConv": lambda: k3_layers.ChebConv(in_c, out_c, K=2),
        "ClusterGCNConv": lambda: k3_layers.ClusterGCNConv(in_c, out_c),
        "DNAConv": lambda: k3_layers.DNAConv(in_c, heads=1, groups=1),
        "FeaStConv": lambda: k3_layers.FeaStConv(in_c, out_c, num_heads=2),
        "FiLMConv": lambda: k3_layers.FiLMConv(in_c, out_c),
        "FusedGATConv": lambda: k3_layers.FusedGATConv(in_c, out_c),
        "GATConv": lambda: k3_layers.GATConv(in_c, out_c),
        "GATv2Conv": lambda: k3_layers.GATv2Conv(in_c, out_c),
        "GatedGraphConv": lambda: k3_layers.GatedGraphConv(out_c, num_layers=2),
        "GCNConv": lambda: k3_layers.GCNConv(in_c, out_c),
        "GraphConvolution": lambda: k3_layers.GraphConvolution(in_c, out_c),
        "GraphConv": lambda: k3_layers.GraphConv(in_c, out_c),
        "LEConv": lambda: k3_layers.LEConv(in_c, out_c),
        "LGConv": lambda: k3_layers.LGConv(),
        "MFConv": lambda: k3_layers.MFConv(in_c, out_c),
        "MixHopConv": lambda: k3_layers.MixHopConv(in_c, out_c, powers=[0, 1]),
        "PANConv": lambda: k3_layers.PANConv(in_c, out_c, filter_size=2),
        "ResGatedGraphConv": lambda: k3_layers.ResGatedGraphConv(in_c, out_c),
        "SAGEConv": lambda: k3_layers.SAGEConv(in_c, out_c),
        "SGConv": lambda: k3_layers.SGConv(in_c, out_c, K=2),
        "SimpleConv": lambda: k3_layers.SimpleConv(),
        "SSGConv": lambda: k3_layers.SSGConv(in_c, out_c, alpha=0.1, K=2),
        "SuperGATConv": lambda: k3_layers.SuperGATConv(in_c, out_c),
        "TAGConv": lambda: k3_layers.TAGConv(in_c, out_c, K=2),
        "TransformerConv": lambda: k3_layers.TransformerConv(in_c, out_c),
        "WLConvContinuous": lambda: k3_layers.WLConvContinuous(),
        "PPNPPropagation": lambda: k3_layers.PPNPPropagation(units=out_c),
        "DiffusionConv": lambda: k3_layers.DiffusionConv(in_c, out_c, K=2),
        "EGConv": lambda: k3_layers.EGConv(in_c, out_c, num_heads=1),
    }

    if layer_name in std_convs:
        def model_factory():
            layer = std_convs[layer_name]()
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.pre = layers.Dense(in_c)
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    x, edge_index = inputs["x"], inputs["edge_index"]
                    h = self.pre(x)
                    h = self.layer(h, edge_index)
                    if isinstance(h, (tuple, list)):
                        h = h[0]
                    return self.post(h)
            return Model()

        def inputs_factory():
            x = np.random.randn(N, in_c).astype(np.float32)
            edge_index = np.random.randint(0, N, size=(2, E)).astype(np.int32)
            return {"x": x, "edge_index": edge_index}

        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)

        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "GINConv":
        def model_factory():
            mlp = keras.Sequential([layers.Dense(out_c, activation="relu"), layers.Dense(out_c)])
            layer = k3_layers.GINConv(mlp)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["edge_index"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "GINEConv":
        edge_dim = 3
        def model_factory():
            mlp = keras.Sequential([layers.Dense(out_c, activation="relu"), layers.Dense(out_c)])
            layer = k3_layers.GINEConv(mlp, edge_dim=edge_dim)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["edge_index"], inputs["edge_attr"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
                "edge_attr": np.random.randn(E, edge_dim).astype(np.float32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    edge_convs = {
        "CGConv": lambda: k3_layers.CGConv(in_c, dim=3),
        "CrystalConv": lambda: k3_layers.CrystalConv(in_c, edge_dim=3),
        "ECConv": lambda: k3_layers.ECConv(in_c, out_c, nn=layers.Dense(in_c * out_c)),
        "NNConv": lambda: k3_layers.NNConv(in_c, out_c, nn=layers.Dense(in_c * out_c)),
        "GENConv": lambda: k3_layers.GENConv(in_c, out_c, edge_dim=3),
        "GeneralConv": lambda: k3_layers.GeneralConv(in_c, out_c, in_edge_channels=3),
        "GMMConv": lambda: k3_layers.GMMConv(in_c, out_c, dim=3, kernel_size=2),
        "PDNConv": lambda: k3_layers.PDNConv(in_c, out_c, edge_dim=3, hidden_channels=16),
        "SplineConv": lambda: k3_layers.SplineConv(in_c, out_c, dim=1, kernel_size=2),
    }

    if layer_name in edge_convs:
        def model_factory():
            layer = edge_convs[layer_name]()
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.layer(inputs["x"], inputs["edge_index"], inputs["edge_attr"])
                    return self.post(h)
            return Model()

        def inputs_factory():
            edge_attr_dim = 1 if layer_name == "SplineConv" else 3
            if layer_name == "SplineConv":
                attr = np.random.uniform(0.01, 0.99, size=(E, edge_attr_dim)).astype(np.float32)
            else:
                attr = np.random.randn(E, edge_attr_dim).astype(np.float32)
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
                "edge_attr": attr,
            }

        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)

        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["RGCNConv", "FastRGCNConv", "RGATConv"]:
        num_rels = 3
        def model_factory():
            if layer_name == "RGCNConv":
                layer = k3_layers.RGCNConv(in_c, out_c, num_relations=num_rels)
            elif layer_name == "FastRGCNConv":
                layer = k3_layers.FastRGCNConv(in_c, out_c, num_relations=num_rels)
            else:
                layer = k3_layers.RGATConv(in_c, out_c, num_relations=num_rels)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["edge_index"], inputs["edge_type"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
                "edge_type": np.random.randint(0, num_rels, size=(E,)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "SignedConv":
        def model_factory():
            layer = k3_layers.SignedConv(in_c, out_c, first_aggr=True)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["pos_edge_index"], inputs["neg_edge_index"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "pos_edge_index": np.random.randint(0, N, size=(2, E // 2)).astype(np.int32),
                "neg_edge_index": np.random.randint(0, N, size=(2, E // 2)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, 2 * out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["EdgeConv", "DynamicEdgeConv"]:
        def model_factory():
            nn = keras.Sequential([layers.Dense(out_c, activation="relu"), layers.Dense(out_c)])
            layer = k3_layers.EdgeConv(nn) if layer_name == "EdgeConv" else k3_layers.DynamicEdgeConv(nn, k=3)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    if layer_name == "EdgeConv":
                        return self.layer(inputs["x"], inputs["edge_index"])
                    return self.layer(inputs["x"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["PointConv", "PointNetConv", "PointGNNConv", "PointTransformerConv"]:
        def model_factory():
            if layer_name in ["PointConv", "PointNetConv"]:
                local_nn = keras.Sequential([layers.Dense(out_c, activation="relu"), layers.Dense(out_c)])
                layer = k3_layers.PointConv(local_nn=local_nn) if layer_name == "PointConv" else k3_layers.PointNetConv(local_nn=local_nn)
            elif layer_name == "PointGNNConv":
                mlp_h = keras.layers.Dense(3)
                mlp_f = keras.layers.Dense(out_c)
                mlp_g = keras.layers.Dense(in_c)
                layer = k3_layers.PointGNNConv(mlp_h=mlp_h, mlp_f=mlp_f, mlp_g=mlp_g)
            else:
                layer = k3_layers.PointTransformerConv(in_c, out_c)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.layer(inputs["x"], inputs["pos"], inputs["edge_index"])
                    return self.post(h)
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "pos": np.random.randn(N, 3).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "PPFConv":
        def model_factory():
            local_nn = keras.Sequential([layers.Dense(out_c, activation="relu"), layers.Dense(out_c)])
            layer = k3_layers.PPFConv(local_nn=local_nn)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["pos"], inputs["normal"], inputs["edge_index"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "pos": np.random.randn(N, 3).astype(np.float32),
                "normal": np.random.randn(N, 3).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "GravNetConv":
        def model_factory():
            layer = k3_layers.GravNetConv(in_c, out_c, space_dimensions=3, propagate_dimensions=4, k=3)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"])
            return Model()
        def inputs_factory():
            return {"x": np.random.randn(N, in_c).astype(np.float32)}
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "XConv":
        def model_factory():
            layer = k3_layers.XConv(in_c, out_c, dim=3, kernel_size=2)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["pos"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "pos": np.random.randn(N, 3).astype(np.float32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "GCN2Conv":
        def model_factory():
            layer = k3_layers.GCN2Conv(in_c, alpha=0.1, theta=0.5)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.layer(inputs["x"], inputs["x_0"], inputs["edge_index"])
                    return self.post(h)
            return Model()
        def inputs_factory():
            x = np.random.randn(N, in_c).astype(np.float32)
            return {
                "x": x,
                "x_0": x,
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "FAConv":
        def model_factory():
            layer = k3_layers.FAConv(in_c, eps=0.1)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.layer(inputs["x"], inputs["x_0"], inputs["edge_index"])
                    return self.post(h)
            return Model()
        def inputs_factory():
            x = np.random.randn(N, in_c).astype(np.float32)
            return {
                "x": x,
                "x_0": x,
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "DirGNNConv":
        def model_factory():
            base_conv = k3_layers.GCNConv(in_c, out_c)
            layer = k3_layers.DirGNNConv(base_conv)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["edge_index"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "HypergraphConv":
        def model_factory():
            layer = k3_layers.HypergraphConv(in_c, out_c)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["hyperedge_index"], num_edges=2)
            return Model()
        def inputs_factory():
            h_idx = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]], dtype=np.int32)
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "hyperedge_index": h_idx,
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "PNAConv":
        def model_factory():
            deg = np.array([2, 3, 1, 4, 2, 1, 3, 2, 1, 1], dtype=np.int32)
            layer = k3_layers.PNAConv(in_c, out_c, aggregators=["sum", "mean", "max"], scalers=["identity"], deg=deg)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["edge_index"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "GPSConv":
        def model_factory():
            conv = k3_layers.GCNConv(in_c, in_c)
            layer = k3_layers.GPSConv(in_c, conv=conv, heads=2)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.layer(inputs["x"], inputs["edge_index"])
                    return self.post(h)
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "GraphAttention":
        def model_factory():
            layer = k3_layers.GraphAttention(in_c, out_c)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["edge_index"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "MeshCNNConv":
        def model_factory():
            layer = k3_layers.MeshCNNConv(in_c, out_c)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["edge_index"])
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["HEATConv", "HeteroConv", "HGTConv", "HANConv"]:
        return None, None, None, False, "Heterogeneous GNN layer (tested in hetero suite)"

    # =========================================================================
    # Group 2: NORMALIZATION LAYERS
    # =========================================================================
    if layer_name in ["BatchNorm", "DiffGroupNorm"]:
        def model_factory():
            layer = k3_layers.BatchNorm(in_c) if layer_name == "BatchNorm" else k3_layers.DiffGroupNorm(in_c, groups=2)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.pre = layers.Dense(in_c)
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    return self.post(self.layer(self.pre(inputs["x"])))
            return Model()
        def inputs_factory():
            return {"x": np.random.randn(N, in_c).astype(np.float32)}
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["GraphNorm", "GraphSizeNorm", "InstanceNorm", "LayerNorm", "MeanSubtractionNorm", "PairNorm"]:
        def model_factory():
            if layer_name == "GraphNorm":
                layer = k3_layers.GraphNorm(in_c)
            elif layer_name == "GraphSizeNorm":
                layer = k3_layers.GraphSizeNorm()
            elif layer_name == "InstanceNorm":
                layer = k3_layers.InstanceNorm(in_c)
            elif layer_name == "LayerNorm":
                layer = k3_layers.LayerNorm(in_c)
            elif layer_name == "MeanSubtractionNorm":
                layer = k3_layers.MeanSubtractionNorm()
            else:
                layer = k3_layers.PairNorm()
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.pre = layers.Dense(in_c)
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.pre(inputs["x"])
                    try:
                        h = self.layer(h, inputs["batch"], batch_size=2)
                    except TypeError:
                        h = self.layer(h, inputs["batch"], dim_size=2)
                    return self.post(h)
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "batch": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "MessageNorm":
        def model_factory():
            layer = k3_layers.MessageNorm(learn_scale=True)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.layer(inputs["x"], inputs["msg"])
                    return self.post(h)
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "msg": np.random.randn(N, in_c).astype(np.float32),
            }
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["HeteroBatchNorm", "HeteroLayerNorm"]:
        return None, None, None, False, "Heterogeneous Norm layer (tested in hetero suite)"

    # =========================================================================
    # Group 3: POOLING LAYERS
    # =========================================================================
    if layer_name in ["TopKPooling", "SAGPooling", "ASAPooling"]:
        def model_factory():
            if layer_name == "TopKPooling":
                layer = k3_layers.TopKPooling(in_c, ratio=0.5)
            elif layer_name == "SAGPooling":
                layer = k3_layers.SAGPooling(in_c, ratio=0.5)
            else:
                layer = k3_layers.ASAPooling(in_c, ratio=0.5)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.pre = layers.Dense(in_c)
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.pre(inputs["x"])
                    res = self.layer(h, inputs["edge_index"], batch=inputs["batch"])
                    x_p, edge_index_p, edge_attr_p, batch_p = res[0], res[1], res[2], res[3]
                    pooled = k3_layers.global_add_pool(x_p, batch_p)
                    return self.post(pooled)
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
                "batch": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            }
        def target_factory():
            return np.random.randn(2, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["EdgePooling", "PANPooling", "MemPooling", "ClusterPooling", "DMoNPooling"]:
        def model_factory():
            if layer_name == "EdgePooling":
                layer = k3_layers.EdgePooling(in_c)
            elif layer_name == "PANPooling":
                layer = k3_layers.PANPooling(in_c, ratio=0.5)
            elif layer_name == "MemPooling":
                layer = k3_layers.MemPooling(in_c, out_c, heads=2, num_clusters=2)
            elif layer_name == "ClusterPooling":
                layer = k3_layers.ClusterPooling(in_c)
            else:
                layer = k3_layers.DMoNPooling(in_c, k=2)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.pre = layers.Dense(in_c)
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.pre(inputs["x"])
                    if layer_name == "EdgePooling":
                        x_p, edge_index_p, batch_p, _ = self.layer(h, inputs["edge_index"], inputs["batch"])
                        pooled = k3_layers.global_add_pool(x_p, batch_p)
                        return self.post(pooled)
                    elif layer_name == "MemPooling":
                        out = self.layer(h, inputs["batch"])
                        return self.post(out[0] if isinstance(out, (list, tuple)) else out)
                    elif layer_name == "DMoNPooling":
                        x_dense = ops.expand_dims(h, 0)
                        adj_dense = ops.expand_dims(inputs["adj"], 0)
                        res = self.layer(x_dense, adj_dense)
                        x_p = res[1]
                        return self.post(ops.mean(x_p, axis=1))
                    elif layer_name == "ClusterPooling":
                        x_p, edge_index_p, batch_p, _ = self.layer(h, inputs["edge_index"], inputs["batch"])
                        pooled = k3_layers.global_add_pool(x_p, batch_p)
                        return self.post(pooled)
                    else:
                        res = self.layer(h, inputs["edge_index"], batch=inputs["batch"])
                        x_p, batch_p = res[0], res[3]
                        pooled = k3_layers.global_add_pool(x_p, batch_p)
                        return self.post(pooled)
            return Model()
        def inputs_factory():
            adj = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                adj[i, (i + 1) % N] = 1.0
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
                "batch": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
                "adj": adj,
                "cluster": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            }
        def target_factory():
            if layer_name == "DMoNPooling":
                return np.random.randn(1, out_c).astype(np.float32)
            return np.random.randn(2, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "SelectTopK":
        def model_factory():
            layer = k3_layers.SelectTopK(in_c, ratio=0.5)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    sel = self.layer(inputs["x"], inputs["batch"])
                    pooled = ops.take(inputs["x"], sel.node_index, axis=0) * ops.expand_dims(sel.weight, -1)
                    return self.post(ops.mean(pooled, axis=0, keepdims=True))
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "batch": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            }
        def target_factory():
            return np.random.randn(1, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name == "FilterEdges":
        def model_factory():
            layer = k3_layers.FilterEdges()
            sel = k3_layers.SelectTopK(in_c, ratio=0.5)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.sel = sel
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    s = self.sel(inputs["x"], inputs["batch"])
                    c = self.layer.call(s, inputs["edge_index"], batch=inputs["batch"])
                    pooled = ops.take(inputs["x"], s.node_index, axis=0)
                    return self.post(ops.mean(pooled, axis=0, keepdims=True))
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
                "batch": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            }
        def target_factory():
            return np.random.randn(1, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    # =========================================================================
    # Group 4: AGGREGATION LAYERS
    # =========================================================================
    aggr_classes = {
        "SumAggregation": lambda: k3_layers.SumAggregation(),
        "MeanAggregation": lambda: k3_layers.MeanAggregation(),
        "MaxAggregation": lambda: k3_layers.MaxAggregation(),
        "MinAggregation": lambda: k3_layers.MinAggregation(),
        "MulAggregation": lambda: k3_layers.MulAggregation(),
        "StdAggregation": lambda: k3_layers.StdAggregation(),
        "VarAggregation": lambda: k3_layers.VarAggregation(),
        "SoftmaxAggregation": lambda: k3_layers.SoftmaxAggregation(learn=True),
        "PowerMeanAggregation": lambda: k3_layers.PowerMeanAggregation(learn=True),
        "MedianAggregation": lambda: k3_layers.MedianAggregation(),
        "QuantileAggregation": lambda: k3_layers.QuantileAggregation(q=0.5),
        "AttentionalAggregation": lambda: k3_layers.AttentionalAggregation(gate_nn=layers.Dense(1)),
        "Set2Set": lambda: k3_layers.Set2Set(in_c, processing_steps=2),
        "GraphMultisetTransformer": lambda: k3_layers.GraphMultisetTransformer(in_c, k=2),
        "SetTransformerAggregation": lambda: k3_layers.SetTransformerAggregation(in_c, num_seed_points=2),
        "DeepSetsAggregation": lambda: k3_layers.DeepSetsAggregation(local_nn=layers.Dense(in_c), global_nn=layers.Dense(in_c)),
        "MLPAggregation": lambda: k3_layers.MLPAggregation(in_c, out_c, max_num_elements=N),
        "GRUAggregation": lambda: k3_layers.GRUAggregation(in_c, out_c),
        "LSTMAggregation": lambda: k3_layers.LSTMAggregation(in_c, out_c),
        "MultiAggregation": lambda: k3_layers.MultiAggregation(aggrs=["sum", "mean", "max"]),
        "DegreeScalerAggregation": lambda: k3_layers.DegreeScalerAggregation(
            k3_layers.SumAggregation(),
            scaler="identity",
            deg=np.ones((N,), dtype=np.float32),
        ),
        "VariancePreservingAggregation": lambda: k3_layers.VariancePreservingAggregation(),
        "LCMAggregation": lambda: k3_layers.LCMAggregation(in_c, out_c),
        "PatchTransformerAggregation": lambda: k3_layers.PatchTransformerAggregation(in_c, out_c, patch_size=2, hidden_channels=in_c),
        "EquilibriumAggregation": lambda: k3_layers.EquilibriumAggregation(in_c, out_c, num_layers=[in_c]),
        "FusedAggregation": lambda: k3_layers.FusedAggregation(aggrs=["sum", "mean"]),
        "SortAggregation": lambda: k3_layers.SortAggregation(k=2),
    }

    if layer_name in aggr_classes:
        num_segments = 2
        def model_factory():
            aggr = aggr_classes[layer_name]()
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.pre = layers.Dense(in_c)
                    self.aggr = aggr
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    h = self.pre(inputs["x"])
                    res = self.aggr(h, index=inputs["index"], dim_size=num_segments)
                    if isinstance(res, (list, tuple)):
                        res = ops.concatenate(res, axis=-1)
                    return self.post(res)
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(N, in_c).astype(np.float32),
                "index": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            }
        def target_factory():
            return np.random.randn(num_segments, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    # =========================================================================
    # Group 5: DENSE / LINEAR LAYERS
    # =========================================================================
    if layer_name == "Linear":
        def model_factory():
            layer = k3_layers.Linear(in_c, out_c)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"])
            return Model()
        def inputs_factory():
            return {"x": np.random.randn(N, in_c).astype(np.float32)}
        def target_factory():
            return np.random.randn(N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    if layer_name in ["HeteroLinear", "HeteroDictLinear"]:
        return None, None, None, False, "Heterogeneous Linear layer (tested in hetero suite)"

    dense_convs = {
        "DenseGCNConv": lambda: k3_layers.DenseGCNConv(in_c, out_c),
        "DenseSAGEConv": lambda: k3_layers.DenseSAGEConv(in_c, out_c),
        "DenseGINConv": lambda: k3_layers.DenseGINConv(keras.Sequential([layers.Dense(out_c)])),
        "DenseGraphConv": lambda: k3_layers.DenseGraphConv(in_c, out_c),
        "DenseGATConv": lambda: k3_layers.DenseGATConv(in_c, out_c),
    }

    if layer_name in dense_convs:
        def model_factory():
            layer = dense_convs[layer_name]()
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["x"], inputs["adj"])
            return Model()
        def inputs_factory():
            adj = np.eye(N, dtype=np.float32)
            return {
                "x": np.random.randn(1, N, in_c).astype(np.float32),
                "adj": np.expand_dims(adj, 0).astype(np.float32),
            }
        def target_factory():
            return np.random.randn(1, N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    # =========================================================================
    # Group 6: ATTENTION LAYERS
    # =========================================================================
    if layer_name in ["PerformerAttention", "PerformerProjection", "PolynormerAttention", "QFormer", "QFormerEncoderLayer", "SGFormerAttention"]:
        def model_factory():
            if layer_name == "PerformerAttention":
                layer = k3_layers.PerformerAttention(channels=out_c, heads=2)
            elif layer_name == "PerformerProjection":
                layer = k3_layers.PerformerProjection(num_cols=in_c)
            elif layer_name == "PolynormerAttention":
                layer = k3_layers.PolynormerAttention(channels=in_c, heads=2)
            elif layer_name == "QFormer":
                layer = k3_layers.QFormer(input_dim=in_c, hidden_dim=in_c, output_dim=out_c, num_heads=2, num_layers=1)
            elif layer_name == "QFormerEncoderLayer":
                layer = k3_layers.QFormerEncoderLayer(input_dim=in_c, hidden_dim=in_c, num_heads=2)
            else:
                layer = k3_layers.SGFormerAttention(channels=in_c, heads=2)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                    self.post = layers.Dense(out_c)
                def call(self, inputs):
                    x = inputs["x"]
                    if layer_name in ["PerformerAttention", "PolynormerAttention"]:
                        mask = ops.ones((1, N))
                        out = self.layer(x, mask)
                    elif layer_name in ["QFormer", "QFormerEncoderLayer", "SGFormerAttention"]:
                        out = self.layer(x)
                    elif layer_name == "PerformerProjection":
                        x_h = ops.expand_dims(x, 1)
                        out = self.layer(x_h, x_h, x_h)
                        out = ops.reshape(out, (1, N, -1))
                    else:
                        out = self.layer(x)
                    return self.post(out)
            return Model()
        def inputs_factory():
            return {
                "x": np.random.randn(1, N, in_c).astype(np.float32),
                "edge_index": np.random.randint(0, N, size=(2, E)).astype(np.int32),
            }
        def target_factory():
            return np.random.randn(1, N, out_c).astype(np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    # =========================================================================
    # Group 7: KNOWLEDGE GRAPH EMBEDDING LAYERS
    # =========================================================================
    if layer_name in ["TransE", "ComplEx", "DistMult", "RotatE"]:
        num_nodes = 20
        num_relations = 5
        hidden_channels = 8
        def model_factory():
            if layer_name == "TransE":
                layer = k3_layers.TransE(num_nodes, num_relations, hidden_channels)
            elif layer_name == "ComplEx":
                layer = k3_layers.ComplEx(num_nodes, num_relations, hidden_channels)
            elif layer_name == "DistMult":
                layer = k3_layers.DistMult(num_nodes, num_relations, hidden_channels)
            else:
                layer = k3_layers.RotatE(num_nodes, num_relations, hidden_channels)
            class Model(keras.Model):
                def __init__(self):
                    super().__init__()
                    self.layer = layer
                def call(self, inputs):
                    return self.layer(inputs["head"], inputs["rel"], inputs["tail"])
            return Model()
        def inputs_factory():
            B = 10
            return {
                "head": np.random.randint(0, num_nodes, size=(B,)).astype(np.int32),
                "rel": np.random.randint(0, num_relations, size=(B,)).astype(np.int32),
                "tail": np.random.randint(0, num_nodes, size=(B,)).astype(np.int32),
            }
        def target_factory():
            return np.ones((10,), dtype=np.float32)
        return model_factory, inputs_factory, target_factory, True, None

    return None, None, None, False, f"Test builder for {layer_name} not yet added"


def train_and_verify_layer(layer_name, epochs=10, lr=0.02):
    """Instantiates a model containing layer_name, trains it, and asserts loss decreases."""
    model_factory, inputs_factory, target_factory, has_weights, skip_reason = get_layer_test(layer_name)
    if skip_reason:
        pytest.skip(skip_reason)

    model = model_factory()
    inputs = inputs_factory()
    target = target_factory()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    l0 = float(model.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(model.train_on_batch(inputs, target))

    assert l_last < l0, (
        f"Layer {layer_name} did not reduce loss during training: "
        f"initial_loss={l0:.6f}, final_loss={l_last:.6f}"
    )
