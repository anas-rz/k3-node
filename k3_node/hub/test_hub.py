"""Tests for Hugging Face Hub integration in K3-Node."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from keras import ops

import k3_node
from k3_node.data import Data
from k3_node.tasks import (
    NodeClassifier,
    GraphClassifier,
    GraphRegressor,
    LinkPredictor,
)
from k3_node.hub import (
    K3NodeHubMixin,
    from_pretrained,
    push_to_hub,
    save_pretrained,
    generate_model_card,
    generate_dataset_card,
    save_graph_dataset,
    load_graph_dataset,
    push_dataset_to_hub,
    load_dataset_from_hub,
)


def _create_synthetic_node_data(num_nodes=12, in_channels=8, num_classes=2):
    x = ops.convert_to_tensor(np.random.randn(num_nodes, in_channels), dtype="float32")
    src = np.arange(num_nodes, dtype="int64")
    dst = (src + 1) % num_nodes
    edge_index = ops.convert_to_tensor(np.stack([src, dst], axis=0), dtype="int64")
    y = ops.convert_to_tensor(np.random.randint(0, num_classes, size=(num_nodes,)), dtype="int64")
    return Data(x=x, edge_index=edge_index, y=y)


def _create_synthetic_graph_dataset(num_graphs=6, nodes_per_graph=4, in_channels=6):
    graphs = []
    for i in range(num_graphs):
        x = ops.convert_to_tensor(np.random.randn(nodes_per_graph, in_channels), dtype="float32")
        edge_index = ops.convert_to_tensor(np.array([[0, 1, 2], [1, 2, 3]], dtype="int64"), dtype="int64")
        y = ops.convert_to_tensor(np.array([i % 2], dtype="int64"), dtype="int64")
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


def test_hub_module_exports():
    assert hasattr(k3_node, "hub")
    assert hasattr(k3_node, "from_pretrained")
    assert hasattr(k3_node, "push_to_hub")
    assert hasattr(k3_node, "save_pretrained")
    assert hasattr(k3_node, "load_dataset_from_hub")
    assert hasattr(k3_node, "push_dataset_to_hub")


def test_generate_model_card():
    config = {
        "task_type": "NodeClassifier",
        "backbone": "gcn",
        "in_channels": 16,
        "hidden_channels": 32,
        "out_channels": 3,
        "num_layers": 2,
        "dropout": 0.2,
    }
    card = generate_model_card(
        task_type="NodeClassifier",
        backbone="gcn",
        config=config,
        metrics={"accuracy": 0.845, "loss": 0.32},
        dataset_name="Cora",
        repo_id="anas-rz/cora-gcn",
    )
    assert "pipeline_tag: graph-ml" in card
    assert "graph-machine-learning" in card
    assert "# anas-rz/cora-gcn" in card
    assert "Cora" in card
    assert "0.8450" in card
    assert "NodeClassifier.from_pretrained" in card


def test_generate_dataset_card():
    data = _create_synthetic_node_data()
    card = generate_dataset_card(data, repo_id="anas-rz/synthetic-graph", description="Synthetic graph test dataset")
    assert "pipeline_tag: graph-ml" in card or "graph-dataset" in card
    assert "Total Graphs" in card
    assert "anas-rz/synthetic-graph" in card


def test_save_and_from_pretrained_node_classifier():
    data = _create_synthetic_node_data(num_nodes=10, in_channels=8, num_classes=2)
    clf = NodeClassifier(backbone="gcn", in_channels=8, out_channels=2, hidden_channels=16, num_layers=2)
    clf.fit(data, epochs=2, verbose=0)
    orig_preds = clf.predict(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save_pretrained(tmpdir, metrics={"accuracy": 1.0})
        p = Path(tmpdir)
        assert (p / "config.json").exists()
        assert (p / "model.weights.h5").exists()
        assert (p / "README.md").exists()

        # Load with specific class
        loaded_clf = NodeClassifier.from_pretrained(tmpdir)
        assert isinstance(loaded_clf, NodeClassifier)
        new_preds = loaded_clf.predict(data)
        np.testing.assert_array_equal(ops.convert_to_numpy(orig_preds), ops.convert_to_numpy(new_preds))

        # Load with generic from_pretrained
        generic_loaded = from_pretrained(tmpdir)
        assert isinstance(generic_loaded, NodeClassifier)
        gen_preds = generic_loaded.predict(data)
        np.testing.assert_array_equal(ops.convert_to_numpy(orig_preds), ops.convert_to_numpy(gen_preds))


def test_save_and_from_pretrained_graph_classifier():
    dataset = _create_synthetic_graph_dataset(num_graphs=4, nodes_per_graph=4, in_channels=6)
    clf = GraphClassifier(backbone="gin", in_channels=6, num_classes=2, hidden_channels=16, num_layers=2)
    clf.fit(dataset, epochs=2, batch_size=2, verbose=0)
    orig_preds = clf.predict(dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save_pretrained(tmpdir)
        loaded = GraphClassifier.from_pretrained(tmpdir)
        assert isinstance(loaded, GraphClassifier)
        new_preds = loaded.predict(dataset, batch_size=2)
        np.testing.assert_array_equal(ops.convert_to_numpy(orig_preds), ops.convert_to_numpy(new_preds))


def test_save_and_from_pretrained_graph_regressor():
    dataset = _create_synthetic_graph_dataset(num_graphs=4, nodes_per_graph=4, in_channels=6)
    reg = GraphRegressor(backbone="gin", in_channels=6, out_channels=1, hidden_channels=16, num_layers=2)
    reg.fit(dataset, epochs=2, batch_size=2, verbose=0)
    orig_preds = reg.predict(dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        reg.save_pretrained(tmpdir)
        loaded = GraphRegressor.from_pretrained(tmpdir)
        assert isinstance(loaded, GraphRegressor)
        new_preds = loaded.predict(dataset, batch_size=2)
        np.testing.assert_allclose(ops.convert_to_numpy(orig_preds), ops.convert_to_numpy(new_preds), rtol=1e-5)


def test_save_and_from_pretrained_link_predictor():
    data = _create_synthetic_node_data(num_nodes=8, in_channels=6)
    lp = LinkPredictor(backbone="gcn", in_channels=6, hidden_channels=16, out_channels=16, num_layers=2)
    lp.fit(data, epochs=2, verbose=0)
    orig_probs = lp.predict_proba(data, edge_label_index=data.edge_index)

    with tempfile.TemporaryDirectory() as tmpdir:
        lp.save_pretrained(tmpdir)
        loaded = LinkPredictor.from_pretrained(tmpdir)
        assert isinstance(loaded, LinkPredictor)
        new_probs = loaded.predict_proba(data, edge_label_index=data.edge_index)
        np.testing.assert_allclose(ops.convert_to_numpy(orig_probs), ops.convert_to_numpy(new_probs), rtol=1e-4)


def test_save_and_load_single_graph_dataset():
    data = _create_synthetic_node_data(num_nodes=8, in_channels=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "single_graph.npz")
        saved_path = save_graph_dataset(data, path)
        assert os.path.exists(saved_path)

        loaded = load_graph_dataset(saved_path)
        assert isinstance(loaded, Data)
        np.testing.assert_allclose(ops.convert_to_numpy(data.x), ops.convert_to_numpy(loaded.x))
        np.testing.assert_array_equal(ops.convert_to_numpy(data.edge_index), ops.convert_to_numpy(loaded.edge_index))
        np.testing.assert_array_equal(ops.convert_to_numpy(data.y), ops.convert_to_numpy(loaded.y))


def test_save_and_load_graph_list_dataset():
    graphs = _create_synthetic_graph_dataset(num_graphs=4, nodes_per_graph=3, in_channels=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "graph_list.npz")
        saved_path = save_graph_dataset(graphs, path)
        assert os.path.exists(saved_path)

        loaded = load_graph_dataset(saved_path)
        assert isinstance(loaded, list)
        assert len(loaded) == 4
        for orig, rec in zip(graphs, loaded):
            np.testing.assert_allclose(ops.convert_to_numpy(orig.x), ops.convert_to_numpy(rec.x))
            np.testing.assert_array_equal(ops.convert_to_numpy(orig.edge_index), ops.convert_to_numpy(rec.edge_index))


def test_push_to_hub_mocked():
    data = _create_synthetic_node_data(num_nodes=6, in_channels=4)
    clf = NodeClassifier(backbone="gcn", in_channels=4, out_channels=2, hidden_channels=8)
    clf.fit(data, epochs=1, verbose=0)

    with patch("huggingface_hub.HfApi") as MockApi:
        mock_api_instance = MagicMock()
        MockApi.return_value = mock_api_instance

        url = clf.push_to_hub("test-user/test-gcn", token="dummy_token")
        assert url == "https://huggingface.co/test-user/test-gcn"
        mock_api_instance.create_repo.assert_called_once()
        mock_api_instance.upload_folder.assert_called_once()
        args, kwargs = mock_api_instance.upload_folder.call_args
        assert kwargs.get("repo_id") == "test-user/test-gcn"
        assert kwargs.get("repo_type") == "model"


def test_push_dataset_to_hub_mocked():
    data = _create_synthetic_node_data(num_nodes=6, in_channels=4)

    with patch("huggingface_hub.HfApi") as MockApi:
        mock_api_instance = MagicMock()
        MockApi.return_value = mock_api_instance

        url = push_dataset_to_hub(data, "test-user/test-dataset", token="dummy_token")
        assert url == "https://huggingface.co/datasets/test-user/test-dataset"
        mock_api_instance.create_repo.assert_called_once()
        mock_api_instance.upload_folder.assert_called_once()
        args, kwargs = mock_api_instance.upload_folder.call_args
        assert kwargs.get("repo_id") == "test-user/test-dataset"
        assert kwargs.get("repo_type") == "dataset"


def test_load_from_hub_mocked():
    data = _create_synthetic_node_data(num_nodes=8, in_channels=6, num_classes=2)
    clf = NodeClassifier(backbone="gcn", in_channels=6, out_channels=2, hidden_channels=16)
    clf.fit(data, epochs=1, verbose=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save_pretrained(tmpdir)
        config_file = os.path.join(tmpdir, "config.json")
        weights_file = os.path.join(tmpdir, "model.weights.h5")

        def mock_download(repo_id, filename, **kwargs):
            if filename == "config.json":
                return config_file
            elif filename == "model.weights.h5":
                return weights_file
            raise FileNotFoundError(filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            loaded = NodeClassifier.from_pretrained("test-user/test-gcn")
            assert isinstance(loaded, NodeClassifier)
            preds = loaded.predict(data)
            assert ops.shape(preds)[0] == 8


def test_load_dataset_from_hub_mocked():
    data = _create_synthetic_node_data(num_nodes=6, in_channels=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "graph_data.npz")
        save_graph_dataset(data, dataset_path)

        with patch("huggingface_hub.hf_hub_download", return_value=dataset_path):
            loaded = load_dataset_from_hub("test-user/test-graph-dataset")
            assert isinstance(loaded, Data)
            np.testing.assert_allclose(ops.convert_to_numpy(data.x), ops.convert_to_numpy(loaded.x))


def test_schnet_hub_save_load_predict():
    """Test k3.models.SchNet.from_pretrained and model.predict(molecule_data)."""
    from k3_node.models import SchNet

    model = SchNet(
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        num_gaussians=10,
        cutoff=5.0,
    )
    z = ops.convert_to_tensor([1, 6, 8, 1])
    pos = ops.convert_to_tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype="float32")
    molecule_data = Data(z=z, pos=pos)

    energy = model.predict(molecule_data)
    assert energy.shape == (1, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, repo_id="k3-node/schnet-qm9")

        # Load pre-trained weights with one line
        reloaded = SchNet.from_pretrained(tmpdir)
        assert isinstance(reloaded, SchNet)
        energy_reloaded = reloaded.predict(molecule_data)
        np.testing.assert_allclose(
            ops.convert_to_numpy(energy),
            ops.convert_to_numpy(energy_reloaded),
            atol=1e-5,
        )

        # Test generic hub.from_pretrained
        generic_loaded = from_pretrained(tmpdir)
        assert isinstance(generic_loaded, SchNet)
        np.testing.assert_allclose(
            ops.convert_to_numpy(energy),
            ops.convert_to_numpy(generic_loaded.predict(molecule_data)),
            atol=1e-5,
        )

    # Push community checkpoints directly to the hub (mocked)
    with patch("huggingface_hub.HfApi") as MockApi:
        mock_api_instance = MagicMock()
        MockApi.return_value = mock_api_instance
        url = model.push_to_hub("k3-node/schnet-qm9", token="dummy_token")
        assert url == "https://huggingface.co/k3-node/schnet-qm9"


def test_gcn_hub_save_load_predict():
    """Test k3.models.GCN save_pretrained, from_pretrained, and predict."""
    from k3_node.models import GCN

    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    x = ops.zeros((4, 8), dtype="float32")
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
    graph = Data(x=x, edge_index=edge_index)

    p1 = gcn.predict(graph)
    assert p1.shape == (4, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        gcn.save_pretrained(tmpdir)
        reloaded = GCN.from_pretrained(tmpdir)
        assert isinstance(reloaded, GCN)
        p2 = reloaded.predict(graph)
        np.testing.assert_allclose(
            ops.convert_to_numpy(p1),
            ops.convert_to_numpy(p2),
            atol=1e-5,
        )


def test_chgnet_hub_save_load_predict():
    """Test k3.models.CHGNet.from_pretrained, predict, and push_to_hub."""
    from k3_node.models.materials import CHGNet

    model = CHGNet(
        dim_atom_embedding=16,
        dim_bond_embedding=16,
        dim_angle_embedding=16,
        num_blocks=2,
        atom_conv_hidden_dims=(16,),
        bond_conv_hidden_dims=(16,),
    )

    crystal = {
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.2, 0.8], [1.5, 1.5, 1.0]], dtype=np.float32),
        "edge_index": np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=np.int32),
        "line_edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32),
        "node_type": np.array([6, 8, 1, 6], dtype=np.int32),
        "batch": np.array([0, 0, 0, 0], dtype=np.int32),
        "state_attr": np.array([[0.0, 0.0]], dtype=np.float32),
    }

    pred = model.predict(crystal)
    assert pred.shape == () or pred.shape == (1,)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, repo_id="anas-rz/chgnet-mp-2026")
        reloaded = CHGNet.from_pretrained(tmpdir)
        assert isinstance(reloaded, CHGNet)
        pred_reloaded = reloaded.predict(crystal)
        np.testing.assert_allclose(
            ops.convert_to_numpy(pred),
            ops.convert_to_numpy(pred_reloaded),
            atol=1e-5,
        )

    # Push community checkpoints directly to the hub (mocked)
    with patch("huggingface_hub.HfApi") as MockApi:
        mock_api_instance = MagicMock()
        MockApi.return_value = mock_api_instance
        url = model.push_to_hub("anas-rz/chgnet-mp-2026", token="dummy_token")
        assert url == "https://huggingface.co/anas-rz/chgnet-mp-2026"



def test_hub_injection_keeps_model_specific_from_pretrained():
    """Models that load original checkpoints must keep their own ``from_pretrained``."""
    from k3_node.models import GraphMAE2, Graphormer, Graphormer3D

    for cls in (GraphMAE2, Graphormer, Graphormer3D):
        assert cls.from_pretrained.__func__ is vars(cls)["from_pretrained"].__func__, cls.__name__


def test_hub_injection_keeps_keras_batched_predict():
    """Array inputs must still go through Keras' batched ``Model.predict``; graph inputs use the hub path."""
    from k3_node.models import MLP, GCN

    mlp = MLP([8, 16, 3])
    x = np.random.randn(10, 8).astype("float32")
    expected = ops.convert_to_numpy(mlp(x, training=False))
    np.testing.assert_allclose(mlp.predict(x, batch_size=4, verbose=0), expected, rtol=1e-5, atol=1e-6)

    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = Data(x=x, edge_index=np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int32"))
    assert tuple(ops.shape(gcn.predict(graph))) == (10, 3)
