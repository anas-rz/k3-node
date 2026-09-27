"""Unit and integration tests for turn-key ONNX, TFLite, and TensorRT export."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import keras
from keras import ops

import k3_node
from k3_node.data import Data
from k3_node.models import GCN
from k3_node.tasks import NodeClassifier
from k3_node.export import (
    export_onnx,
    export_tflite,
    export_tensorrt,
    generate_triton_config,
    ONNXModel,
    TFLiteModel,
)


def _create_sample_graph(num_nodes=6, in_channels=8):
    x = ops.zeros((num_nodes, in_channels), dtype="float32")
    src = np.arange(num_nodes, dtype="int64")
    dst = (src + 1) % num_nodes
    edge_index = ops.convert_to_tensor(np.stack([src, dst], axis=0), dtype="int64")
    y = ops.convert_to_tensor(np.array([i % 2 for i in range(num_nodes)], dtype="int64"), dtype="int64")
    return Data(x=x, edge_index=edge_index, y=y)


def test_export_module_exports():
    """Verify all export utilities are exported from top-level and export modules."""
    assert hasattr(k3_node, "export_onnx")
    assert hasattr(k3_node, "export_tflite")
    assert hasattr(k3_node, "export_tensorrt")
    assert hasattr(k3_node, "ONNXModel")
    assert hasattr(k3_node, "TFLiteModel")


def test_export_onnx_model_and_runtime_inference():
    """Test exporting GCN to ONNX and serving with ONNXModel."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    # Initial Keras prediction
    keras_pred = ops.convert_to_numpy(gcn.predict(graph))

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "gcn.onnx"
        out_path = export_onnx(gcn, onnx_file, dummy_inputs=graph)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Load into lightweight serving runtime
        serving_model = ONNXModel(onnx_file)
        onnx_pred = serving_model.predict(graph)
        assert onnx_pred.shape == (6, 3)

        # Numerical parity check
        np.testing.assert_allclose(keras_pred, onnx_pred, atol=1e-4)

        # Dynamic graph sizing check: serve a graph with different number of nodes & edges!
        large_graph = _create_sample_graph(num_nodes=14, in_channels=8)
        large_pred = serving_model.predict(large_graph)
        assert large_pred.shape == (14, 3)


def test_export_onnx_task_estimator():
    """Test exporting NodeClassifier task estimator to ONNX."""
    data = _create_sample_graph(num_nodes=8, in_channels=4)
    clf = NodeClassifier(backbone="gcn", in_channels=4, out_channels=2, hidden_channels=8)
    clf.fit(data, epochs=1, verbose=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "classifier.onnx"
        out_path = clf.export_onnx(onnx_file)
        assert out_path.exists()

        runtime = ONNXModel(onnx_file)
        preds = runtime.predict(data)
        assert preds.shape == (8, 2)


def test_export_tflite_fp32_and_runtime_inference():
    """Test exporting GCN to TFLite (FP32) and serving with TFLiteModel."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)
    keras_pred = ops.convert_to_numpy(gcn.predict(graph))

    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_file = Path(tmpdir) / "gcn.tflite"
        out_path = export_tflite(gcn, tflite_file, dummy_inputs=graph)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Serve with TFLiteModel runtime
        tflite_runner = TFLiteModel(tflite_file)
        tflite_pred = tflite_runner.predict(graph)
        assert tflite_pred.shape == (6, 3)
        np.testing.assert_allclose(keras_pred, tflite_pred, atol=1e-4)


def test_export_tflite_fp16_quantization():
    """Test TFLite FP16 quantization."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_file = Path(tmpdir) / "gcn_fp16.tflite"
        out_path = gcn.export_tflite(tflite_file, dummy_inputs=graph, quantization="fp16")
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_export_tflite_int8_dynamic_quantization():
    """Test TFLite dynamic range INT8 quantization."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_file = Path(tmpdir) / "gcn_int8.tflite"
        out_path = gcn.export_tflite(tflite_file, dummy_inputs=graph, quantization="int8_dynamic")
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_generate_triton_config():
    """Test generating Triton Inference Server model repository and config.pbtxt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = generate_triton_config(
            model_name="cora_gcn",
            output_dir=tmpdir,
            in_channels=1433,
            out_channels=7,
            backend="onnxruntime",
        )
        assert cfg_path.exists()
        with open(cfg_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert 'name: "cora_gcn"' in content
        assert "onnxruntime_onnx" in content
        assert "dims: [ -1, 1433 ]" in content
        assert "dims: [ -1, 7 ]" in content
        assert (Path(tmpdir) / "cora_gcn" / "1").exists()


def test_export_tensorrt_mocked():
    """Test TensorRT exporter fallback and compilation logic with mock."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        engine_file = Path(tmpdir) / "model.engine"

        real_run = subprocess.run
        with patch("shutil.which", return_value="/usr/local/bin/trtexec"), patch("subprocess.run") as mock_sub:
            # Create a mock engine file to simulate trtexec output
            def fake_run(cmd, *args, **kwargs):
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                if "trtexec" in cmd_str:
                    with open(engine_file, "wb") as f:
                        f.write(b"TRT_ENGINE_BYTES")
                    return MagicMock(returncode=0)
                return real_run(cmd, *args, **kwargs)

            mock_sub.side_effect = fake_run
            out = gcn.export_tensorrt(engine_file, dummy_inputs=graph)
            assert out.exists()
            assert mock_sub.called
            args, _ = mock_sub.call_args
            assert "--saveEngine=" in " ".join(args[0])


def test_export_tensorrt_python_api_mocked():
    """Test TensorRT Python API compilation path with optimization profile."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        engine_file = Path(tmpdir) / "model.engine"

        # Mock tensorrt module
        mock_trt = MagicMock()
        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_builder.platform_has_fast_fp16 = True

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config
        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        # Return serialized engine bytes
        mock_builder.build_serialized_network.return_value = b"TENSORRT_SERIALIZED_PLAN"

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            out = export_tensorrt(
                gcn,
                engine_file,
                dummy_inputs=graph,
                precision="fp16",
                min_shapes={"x": (1, 8), "edge_index": (2, 1)},
                opt_shapes={"x": (6, 8), "edge_index": (2, 6)},
                max_shapes={"x": (50, 8), "edge_index": (2, 100)},
            )
            assert out.exists()
            assert out.read_bytes() == b"TENSORRT_SERIALIZED_PLAN"
            assert mock_builder.create_builder_config.called
            assert mock_builder.create_optimization_profile.called
            assert mock_config.add_optimization_profile.called
            assert mock_config.set_flag.called


def test_export_tensorrt_direct_onnx_input():
    """Test export_tensorrt when supplied directly with an existing .onnx file."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "pre_exported.onnx"
        export_onnx(gcn, onnx_file, dummy_inputs=graph)
        assert onnx_file.exists()

        engine_file = Path(tmpdir) / "pre_exported.engine"

        real_run = subprocess.run
        with patch("shutil.which", return_value="/usr/local/bin/trtexec"), patch("subprocess.run") as mock_sub:
            def fake_run(cmd, *args, **kwargs):
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                if "trtexec" in cmd_str:
                    with open(engine_file, "wb") as f:
                        f.write(b"DIRECT_ONNX_TRT_BYTES")
                    return MagicMock(returncode=0)
                return real_run(cmd, *args, **kwargs)

            mock_sub.side_effect = fake_run
            out = export_tensorrt(str(onnx_file), engine_file)
            assert out.exists()
            assert out.read_bytes() == b"DIRECT_ONNX_TRT_BYTES"
            args, _ = mock_sub.call_args
            cmd_line = " ".join(args[0])
            assert f"--onnx={onnx_file}" in cmd_line
            assert f"--saveEngine={engine_file}" in cmd_line


def test_export_tensorrt_precision_and_dynamic_shapes_flags():
    """Test trtexec dynamic shape and INT8 precision flag generation."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        engine_file = Path(tmpdir) / "int8_model.engine"

        real_run = subprocess.run
        with patch("shutil.which", return_value="/usr/bin/trtexec"), patch("subprocess.run") as mock_sub:
            captured_cmd = []

            def fake_run(cmd, *args, **kwargs):
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                if "trtexec" in cmd_str:
                    captured_cmd.extend(cmd)
                    with open(engine_file, "wb") as f:
                        f.write(b"INT8_ENGINE")
                    return MagicMock(returncode=0)
                return real_run(cmd, *args, **kwargs)

            mock_sub.side_effect = fake_run
            out = gcn.export_tensorrt(
                engine_file,
                dummy_inputs=graph,
                precision="int8",
                min_shapes={"x": (1, 8), "edge_index": (2, 2)},
                opt_shapes={"x": (10, 8), "edge_index": (2, 20)},
                max_shapes={"x": (100, 8), "edge_index": (2, 200)},
            )
            assert out.exists()
            cmd_line = " ".join(captured_cmd)
            assert "--int8" in cmd_line
            assert "--minShapes=x:1x8,edge_index:2x2" in cmd_line
            assert "--optShapes=x:10x8,edge_index:2x20" in cmd_line
            assert "--maxShapes=x:100x8,edge_index:2x200" in cmd_line


def test_export_tensorrt_missing_tools_raises_import_error():
    """Test that informative ImportError is raised when neither tensorrt nor trtexec is present."""
    gcn = GCN(in_channels=8, hidden_channels=16, num_layers=2, out_channels=3)
    graph = _create_sample_graph(num_nodes=6, in_channels=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        engine_file = Path(tmpdir) / "model.engine"

        # Mock absence of trtexec and ensure tensorrt import fails
        with patch("shutil.which", return_value=None):
            with pytest.raises(ImportError, match="Neither NVIDIA `tensorrt` Python package nor `trtexec` binary"):
                export_tensorrt(gcn, engine_file, dummy_inputs=graph)

        # Confirm intermediate ONNX file was preserved
        intermediate_onnx = engine_file.with_suffix(".onnx")
        assert intermediate_onnx.exists()
        assert intermediate_onnx.stat().st_size > 0


def test_export_tensorrt_task_estimator():
    """Test task estimator .export_tensorrt() method."""
    data = _create_sample_graph(num_nodes=8, in_channels=4)
    clf = NodeClassifier(backbone="gcn", in_channels=4, out_channels=2, hidden_channels=8)
    clf.fit(data, epochs=1, verbose=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        engine_file = Path(tmpdir) / "task_model.engine"

        real_run = subprocess.run
        with patch("shutil.which", return_value="/usr/local/bin/trtexec"), patch("subprocess.run") as mock_sub:
            def fake_run(cmd, *args, **kwargs):
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                if "trtexec" in cmd_str:
                    with open(engine_file, "wb") as f:
                        f.write(b"TASK_TRT_BYTES")
                    return MagicMock(returncode=0)
                return real_run(cmd, *args, **kwargs)

            mock_sub.side_effect = fake_run
            out = clf.export_tensorrt(engine_file, dummy_inputs=data)
            assert out.exists()
            assert out.read_bytes() == b"TASK_TRT_BYTES"

