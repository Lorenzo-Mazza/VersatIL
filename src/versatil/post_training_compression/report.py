"""Post-Training-Quantization report: operator coverage, output divergence, size and timing analysis."""

import time

import torch
import torch.nn as nn

from versatil.quantization.constants import (
    FXNodePattern,
    QuantizableOperatorType,
    ReportMetricKey,
)


class QuantizationReport:
    """Analyzes a PTQ model and generates a comparison report."""

    def __init__(
        self,
        float_model: nn.Module,
        quantized_model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        action_keys: list[str],
    ) -> None:
        """Initialize with float and quantized models for comparison.

        Args:
            float_model: Original float32 model.
            quantized_model: Quantized model to compare against.
            example_inputs: Example inputs for running inference.
            action_keys: Ordered list of action output keys.
        """
        self._float_model = float_model
        self._quantized_model = quantized_model
        self._example_inputs = example_inputs
        self._action_keys = action_keys

    def compute_operator_coverage(self) -> dict[str, dict[str, int]]:
        """Count quantized vs total operators in the quantized model's FX graph.

        Inspects the quantized model's graph nodes for conv2d and linear
        operations. A node is considered quantized if any of its args
        references a dequantize operation.

        Returns:
            Dict with keys "conv2d", "linear", each mapping to
            {"quantized": N, "total": M}.
        """
        coverage: dict[str, dict[str, int]] = {
            QuantizableOperatorType.CONV2D.value: {
                ReportMetricKey.QUANTIZED.value: 0,
                ReportMetricKey.TOTAL.value: 0,
            },
            QuantizableOperatorType.LINEAR.value: {
                ReportMetricKey.QUANTIZED.value: 0,
                ReportMetricKey.TOTAL.value: 0,
            },
        }

        if not hasattr(self._quantized_model, "graph"):
            return coverage

        for node in self._quantized_model.graph.nodes:
            target_name = str(node.target)
            operator_type = None
            if QuantizableOperatorType.CONV2D.value in target_name:
                operator_type = QuantizableOperatorType.CONV2D.value
            elif (
                QuantizableOperatorType.LINEAR.value in target_name
                or FXNodePattern.ADDMM.value in target_name
            ):
                operator_type = QuantizableOperatorType.LINEAR.value

            if operator_type is None:
                continue

            coverage[operator_type][ReportMetricKey.TOTAL.value] += 1

            has_dequantize_input = any(
                hasattr(arg, "target")
                and FXNodePattern.DEQUANTIZE.value in str(arg.target)
                for arg in node.args
                if hasattr(arg, "target")
            )
            if has_dequantize_input:
                coverage[operator_type][ReportMetricKey.QUANTIZED.value] += 1

        return coverage

    def compute_output_divergence(self) -> dict[str, dict[str, float]]:
        """Compare float and quantized model outputs.

        Returns:
            Dict keyed by action_key, each mapping to
            {"max_difference": float, "mean_difference": float}.
        """
        with torch.no_grad():
            float_outputs = self._float_model(*self._example_inputs)
            quantized_outputs = self._quantized_model(*self._example_inputs)

        if not isinstance(float_outputs, tuple):
            float_outputs = (float_outputs,)
        if not isinstance(quantized_outputs, tuple):
            quantized_outputs = (quantized_outputs,)

        divergence: dict[str, dict[str, float]] = {}
        for index, key in enumerate(self._action_keys):
            float_tensor = float_outputs[index]
            quantized_tensor = quantized_outputs[index]
            difference = (float_tensor - quantized_tensor).abs()
            divergence[key] = {
                ReportMetricKey.MAX_DIFFERENCE.value: difference.max().item(),
                ReportMetricKey.MEAN_DIFFERENCE.value: difference.mean().item(),
            }
        return divergence

    def compute_size_reduction(self) -> dict[str, float]:
        """Compare model sizes (float32 bytes vs quantized bytes).

        Counts parameter bytes for the float model and parameter + buffer
        bytes for the quantized model (scales and zero points are stored
        as buffers).

        Returns:
            Dict with "float_bytes", "quantized_bytes", "compression_ratio".
        """
        float_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self._float_model.parameters()
        )
        quantized_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self._quantized_model.parameters()
        )
        quantized_bytes += sum(
            buffer.numel() * buffer.element_size()
            for buffer in self._quantized_model.buffers()
        )
        compression_ratio = float_bytes / max(quantized_bytes, 1)
        return {
            ReportMetricKey.FLOAT_BYTES.value: float(float_bytes),
            ReportMetricKey.QUANTIZED_BYTES.value: float(quantized_bytes),
            ReportMetricKey.COMPRESSION_RATIO.value: compression_ratio,
        }

    def compute_inference_timing(
        self,
        warmup_runs: int = 10,
        benchmark_runs: int = 50,
    ) -> dict[str, float]:
        """Compare float vs quantized model inference latency.

        Args:
            warmup_runs: Number of warmup iterations before timing.
            benchmark_runs: Number of iterations to time.

        Returns:
            Dict with "float_milliseconds", "quantized_milliseconds",
            "speedup".
        """
        with torch.no_grad():
            for _ in range(warmup_runs):
                self._float_model(*self._example_inputs)
                self._quantized_model(*self._example_inputs)

            start = time.perf_counter()
            for _ in range(benchmark_runs):
                self._float_model(*self._example_inputs)
            float_time = (time.perf_counter() - start) / benchmark_runs

            start = time.perf_counter()
            for _ in range(benchmark_runs):
                self._quantized_model(*self._example_inputs)
            quantized_time = (time.perf_counter() - start) / benchmark_runs

        return {
            ReportMetricKey.FLOAT_MS.value: float_time * 1000,
            ReportMetricKey.QUANTIZED_MS.value: quantized_time * 1000,
            ReportMetricKey.SPEEDUP.value: float_time / max(quantized_time, 1e-9),
        }

    def generate_report(self) -> str:
        """Generate human-readable report string.

        Returns:
            Formatted string containing operator coverage,
            output divergence, size reduction, and inference timing.
        """
        lines: list[str] = ["Quantization Report", "=" * 40, "\nOperator Coverage:"]
        coverage = self.compute_operator_coverage()
        for operator_type, counts in coverage.items():
            total = counts[ReportMetricKey.TOTAL.value]
            quantized = counts[ReportMetricKey.QUANTIZED.value]
            lines.append(f"  {operator_type}: {quantized}/{total} quantized")
        lines.append("\nOutput Divergence:")
        divergence = self.compute_output_divergence()
        for key, metrics in divergence.items():
            lines.append(
                f"  {key}: "
                f"max={metrics[ReportMetricKey.MAX_DIFFERENCE.value]:.6f}, "
                f"mean={metrics[ReportMetricKey.MEAN_DIFFERENCE.value]:.6f}"
            )
        lines.append("\nSize Reduction:")
        size = self.compute_size_reduction()
        lines.append(
            f"  Float model: {size[ReportMetricKey.FLOAT_BYTES.value]:.0f} bytes"
        )
        lines.append(
            f"  Quantized model: {size[ReportMetricKey.QUANTIZED_BYTES.value]:.0f} bytes"
        )
        lines.append(
            f"  Compression ratio: {size[ReportMetricKey.COMPRESSION_RATIO.value]:.2f}x"
        )

        lines.append("\nInference Timing:")
        timing = self.compute_inference_timing()
        lines.append(f"  Float model: {timing[ReportMetricKey.FLOAT_MS.value]:.2f} ms")
        lines.append(
            f"  Quantized model: {timing[ReportMetricKey.QUANTIZED_MS.value]:.2f} ms"
        )
        lines.append(f"  Speedup: {timing[ReportMetricKey.SPEEDUP.value]:.2f}x")
        return "\n".join(lines)
