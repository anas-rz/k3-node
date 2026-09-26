"""High-level Task APIs for K3-Node."""

from k3_node.tasks.base import BaseTask
from k3_node.tasks.backbone_resolver import resolve_backbone
from k3_node.tasks.node_classification import NodeClassifier
from k3_node.tasks.node_regression import NodeRegressor
from k3_node.tasks.graph_classification import GraphClassifier
from k3_node.tasks.graph_regression import GraphRegressor
from k3_node.tasks.link_prediction import LinkPredictor

__all__ = [
    "BaseTask",
    "resolve_backbone",
    "NodeClassifier",
    "NodeRegressor",
    "GraphClassifier",
    "GraphRegressor",
    "LinkPredictor",
]
