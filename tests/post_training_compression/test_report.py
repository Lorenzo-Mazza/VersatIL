"""Tests for versatil.post_training_compression.report module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.post_training_compression.report import QuantizationReport
from versatil.quantization.constants import ReportMetricKey


@pytest.fixture
def float_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    def factory(
        input_features: int = 4,
        output_features: int = 2,
        num_outputs: int = 1,
    ) -> nn.Module:
        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = nn.ModuleList(
                    [
                        nn.Linear(
                            in_features=input_features,
                            out_features=output_features,
                        )
                        for _ in range(num_outputs)
                    ]
                )

            def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
                combined = inputs[0]
                return tuple(head(combined) for head in self.heads)

        model = MultiOutputModel()
        with torch.no_grad():
            for parameter in model.parameters():
                data = rng.standard_normal(parameter.shape).astype(np.float32)
                parameter.copy_(torch.from_numpy(data))
        return model

    return factory


@pytest.fixture
def example_inputs_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    def factory(
        batch_size: int = 2,
        input_features: int = 4,
    ) -> tuple[torch.Tensor, ...]:
        data = rng.standard_normal((batch_size, input_features)).astype(np.float32)
        return (torch.from_numpy(data),)

    return factory


@pytest.fixture
def report_factory(
    float_model_factory: Callable[..., nn.Module],
    example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
) -> Callable[..., QuantizationReport]:
    def factory(
        input_features: int = 4,
        output_features: int = 2,
        num_outputs: int = 2,
        batch_size: int = 2,
        action_keys: list[str] | None = None,
        quantized_model: nn.Module | None = None,
    ) -> QuantizationReport:
        if action_keys is None:
            action_keys = [f"action_{index}" for index in range(num_outputs)]
        float_model = float_model_factory(
            input_features=input_features,
            output_features=output_features,
            num_outputs=num_outputs,
        )
        if quantized_model is None:
            quantized_model = float_model_factory(
                input_features=input_features,
                output_features=output_features,
                num_outputs=num_outputs,
            )
        example_inputs = example_inputs_factory(
            batch_size=batch_size,
            input_features=input_features,
        )
        return QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=action_keys,
        )

    return factory


@pytest.fixture
def quantized_model_mock_factory() -> Callable[..., MagicMock]:
    """Factory for mock quantized models with configurable FX graph nodes."""

    def factory(
        node_targets: list[str] | None = None,
        node_args: list[list] | None = None,
    ) -> MagicMock:
        if node_targets is None:
            node_targets = []
        nodes = []
        for index, target in enumerate(node_targets):
            node = MagicMock()
            node.target = target
            node.args = node_args[index] if node_args and index < len(node_args) else []
            nodes.append(node)
        mock_graph = MagicMock()
        mock_graph.nodes = nodes
        model = MagicMock(spec=nn.Module)
        model.graph = mock_graph
        model.parameters = MagicMock(return_value=iter([]))
        model.buffers = MagicMock(return_value=iter([]))
        return model

    return factory


@pytest.mark.unit
class TestOperatorCoverage:
    def test_returns_zero_counts_when_model_has_no_graph(
        self,
        report_factory: Callable[..., QuantizationReport],
    ):
        report = report_factory()

        coverage = report.compute_operator_coverage()

        assert coverage["conv2d"][ReportMetricKey.QUANTIZED.value] == 0
        assert coverage["conv2d"][ReportMetricKey.TOTAL.value] == 0
        assert coverage["linear"][ReportMetricKey.QUANTIZED.value] == 0
        assert coverage["linear"][ReportMetricKey.TOTAL.value] == 0

    @pytest.mark.parametrize(
        "target, expected_key",
        [
            ("torch.ops.aten.linear.default", "linear"),
            ("torch.ops.aten.conv2d.default", "conv2d"),
            ("torch.ops.aten.addmm.default", "linear"),
        ],
    )
    def test_counts_operator_nodes_in_fx_graph(
        self,
        float_model_factory: Callable[..., nn.Module],
        example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
        quantized_model_mock_factory: Callable[..., MagicMock],
        target: str,
        expected_key: str,
    ):
        float_model = float_model_factory(num_outputs=1)
        quantized_model = quantized_model_mock_factory(
            node_targets=[target],
        )

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs_factory(),
            action_keys=["action_0"],
        )

        coverage = report.compute_operator_coverage()

        assert coverage[expected_key][ReportMetricKey.TOTAL.value] == 1
        assert coverage[expected_key][ReportMetricKey.QUANTIZED.value] == 0

    def test_detects_quantized_nodes_with_dequantize_args(
        self,
        float_model_factory: Callable[..., nn.Module],
        example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
        quantized_model_mock_factory: Callable[..., MagicMock],
    ):
        mock_dequant_arg = MagicMock()
        mock_dequant_arg.target = (
            "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
        )
        quantized_model = quantized_model_mock_factory(
            node_targets=["torch.ops.aten.linear.default"],
            node_args=[[mock_dequant_arg]],
        )

        report = QuantizationReport(
            float_model=float_model_factory(num_outputs=1),
            quantized_model=quantized_model,
            example_inputs=example_inputs_factory(),
            action_keys=["action_0"],
        )

        coverage = report.compute_operator_coverage()

        assert coverage["linear"][ReportMetricKey.TOTAL.value] == 1
        assert coverage["linear"][ReportMetricKey.QUANTIZED.value] == 1


@pytest.mark.unit
class TestOutputDivergence:
    def test_computes_correct_differences_for_known_outputs(
        self,
        example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
    ):
        example_inputs = example_inputs_factory()

        float_model = MagicMock(spec=nn.Module)
        float_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        float_model.return_value = (float_output,)

        quantized_model = MagicMock(spec=nn.Module)
        quantized_output = torch.tensor([[1.1, 2.0], [2.9, 4.2]])
        quantized_model.return_value = (quantized_output,)

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=["position"],
        )

        divergence = report.compute_output_divergence()

        assert "position" in divergence
        expected_max = 0.2
        expected_mean = (0.1 + 0.0 + 0.1 + 0.2) / 4.0
        assert (
            abs(
                divergence["position"][ReportMetricKey.MAX_DIFFERENCE.value]
                - expected_max
            )
            < 1e-5
        )
        assert (
            abs(
                divergence["position"][ReportMetricKey.MEAN_DIFFERENCE.value]
                - expected_mean
            )
            < 1e-5
        )

    def test_returns_zero_divergence_for_identical_models(
        self,
        example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
    ):
        example_inputs = example_inputs_factory()

        output_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        float_model = MagicMock(spec=nn.Module)
        float_model.return_value = (output_tensor,)

        quantized_model = MagicMock(spec=nn.Module)
        quantized_model.return_value = (output_tensor,)

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=["position"],
        )

        divergence = report.compute_output_divergence()

        assert divergence["position"][ReportMetricKey.MAX_DIFFERENCE.value] == 0.0
        assert divergence["position"][ReportMetricKey.MEAN_DIFFERENCE.value] == 0.0

    def test_computes_divergence_per_action_key(
        self,
        example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
    ):
        example_inputs = example_inputs_factory()

        float_model = MagicMock(spec=nn.Module)
        float_model.return_value = (
            torch.tensor([[1.0]]),
            torch.tensor([[2.0]]),
        )

        quantized_model = MagicMock(spec=nn.Module)
        quantized_model.return_value = (
            torch.tensor([[1.5]]),
            torch.tensor([[2.0]]),
        )

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=["position", "gripper"],
        )

        divergence = report.compute_output_divergence()

        assert divergence["position"][
            ReportMetricKey.MAX_DIFFERENCE.value
        ] == pytest.approx(0.5)
        assert divergence["gripper"][ReportMetricKey.MAX_DIFFERENCE.value] == 0.0


@pytest.mark.unit
class TestSizeReduction:
    def test_computes_correct_bytes_for_known_parameters(self):
        float_model = nn.Linear(in_features=10, out_features=5, bias=False)
        quantized_model = nn.Linear(in_features=10, out_features=5, bias=False)
        example_inputs = (torch.zeros(1, 10),)

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=["position"],
        )

        size = report.compute_size_reduction()

        # 10 * 5 = 50 parameters, float32 = 4 bytes each = 200 bytes
        expected_float_bytes = 50 * 4
        assert size[ReportMetricKey.FLOAT_BYTES.value] == expected_float_bytes
        assert size[ReportMetricKey.QUANTIZED_BYTES.value] == expected_float_bytes
        assert size[ReportMetricKey.COMPRESSION_RATIO.value] == pytest.approx(1.0)

    def test_includes_buffers_in_quantized_bytes(self):
        float_model = nn.Linear(in_features=4, out_features=2, bias=False)

        quantized_model = nn.Linear(in_features=4, out_features=2, bias=False)
        quantized_model.register_buffer("scale", torch.tensor([1.0]))
        quantized_model.register_buffer(
            "zero_point", torch.tensor([0], dtype=torch.int32)
        )

        example_inputs = (torch.zeros(1, 4),)

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=["position"],
        )

        size = report.compute_size_reduction()

        float_param_bytes = 4 * 2 * 4  # 8 params * 4 bytes
        quantized_param_bytes = 4 * 2 * 4  # 8 params * 4 bytes
        buffer_bytes = 1 * 4 + 1 * 4  # scale (float32) + zero_point (int32)
        assert size[ReportMetricKey.FLOAT_BYTES.value] == float_param_bytes
        assert (
            size[ReportMetricKey.QUANTIZED_BYTES.value]
            == quantized_param_bytes + buffer_bytes
        )

    def test_compression_ratio_handles_zero_quantized_bytes(self):
        float_model = nn.Linear(in_features=4, out_features=2, bias=False)

        quantized_model = MagicMock(spec=nn.Module)
        quantized_model.parameters = MagicMock(return_value=iter([]))
        quantized_model.buffers = MagicMock(return_value=iter([]))

        example_inputs = (torch.zeros(1, 4),)

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=["position"],
        )

        size = report.compute_size_reduction()

        # max(0, 1) = 1, so ratio = float_bytes / 1
        assert size[ReportMetricKey.QUANTIZED_BYTES.value] == 0.0
        assert size[ReportMetricKey.COMPRESSION_RATIO.value] == float(4 * 2 * 4)


@pytest.mark.unit
class TestInferenceTiming:
    def test_returns_positive_timing_values(
        self,
        report_factory: Callable[..., QuantizationReport],
    ):
        report = report_factory(num_outputs=1, action_keys=["position"])

        timing = report.compute_inference_timing(
            warmup_runs=2,
            benchmark_runs=5,
        )

        assert timing[ReportMetricKey.FLOAT_MS.value] > 0.0
        assert timing[ReportMetricKey.QUANTIZED_MS.value] > 0.0
        assert timing[ReportMetricKey.SPEEDUP.value] > 0.0

    def test_speedup_is_ratio_of_timings(
        self,
        example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
    ):
        example_inputs = example_inputs_factory()

        float_model = MagicMock(spec=nn.Module)
        float_model.return_value = (torch.zeros(2, 2),)

        quantized_model = MagicMock(spec=nn.Module)
        quantized_model.return_value = (torch.zeros(2, 2),)

        report = QuantizationReport(
            float_model=float_model,
            quantized_model=quantized_model,
            example_inputs=example_inputs,
            action_keys=["position"],
        )

        timing = report.compute_inference_timing(
            warmup_runs=1,
            benchmark_runs=3,
        )

        expected_speedup = (
            timing[ReportMetricKey.FLOAT_MS.value]
            / timing[ReportMetricKey.QUANTIZED_MS.value]
        )
        assert abs(timing[ReportMetricKey.SPEEDUP.value] - expected_speedup) < 0.01


@pytest.mark.unit
class TestGenerateReport:
    def test_contains_section_headers(
        self,
        report_factory: Callable[..., QuantizationReport],
    ):
        report = report_factory(num_outputs=1, action_keys=["position"])

        result = report.generate_report()

        assert "Quantization Report" in result
        assert "Operator Coverage" in result
        assert "Output Divergence" in result
        assert "Size Reduction" in result
        assert "Inference Timing" in result

    def test_contains_action_key_names(
        self,
        report_factory: Callable[..., QuantizationReport],
    ):
        action_keys = ["position", "gripper"]
        report = report_factory(num_outputs=2, action_keys=action_keys)

        result = report.generate_report()

        for key in action_keys:
            assert key in result

    def test_contains_compression_ratio(
        self,
        report_factory: Callable[..., QuantizationReport],
    ):
        report = report_factory(num_outputs=1, action_keys=["position"])

        result = report.generate_report()

        assert "Compression ratio" in result
        assert "bytes" in result
