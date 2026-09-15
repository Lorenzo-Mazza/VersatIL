"""ExecuTorch XNNPACK deployment backend for .pte artifacts, to deploy policies on mobile Arm and x86 CPUs, ref. https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html."""

import importlib
import logging
from types import ModuleType

import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows.intx.intx_packing_format import (
    IntxPackingFormat,
)

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentArtifact,
    DeploymentBackend,
)
from versatil.post_training_compression.export import _export_with_dynamic_batch
from versatil.quantization.constants import (
    FXNodeOp,
    PT2EBackendName,
    QuantizationMode,
    QuantizationModuleType,
)
from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.schemas.smoothquant import SmoothQuantSchema

logger = logging.getLogger(__name__)


class ExecutorchXNNPACKBackend(DeploymentBackend):
    """Backend that lowers exported programs to ExecuTorch XNNPACK."""

    name = DeploymentBackendName.EXECUTORCH_XNNPACK.value
    artifact_format = ArtifactFormat.EXECUTORCH_PTE
    model_filename = CompressionFilename.EXECUTORCH_MODEL.value
    supported_quantization_modes = (
        QuantizationMode.NONE.value,
        QuantizationMode.PT2E.value,
        QuantizationMode.EAGER.value,
    )
    supported_pt2e_backends = (PT2EBackendName.XNNPACK.value,)

    def __init__(self, max_batch_size: int) -> None:
        """Initialize XNNPACK deployment settings.

        Args:
            max_batch_size: Upper bound for dynamic batch execution in the
                serialized ExecuTorch program.
        """
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1.")
        self.max_batch_size = max_batch_size

    def validate_eager_target(
        self,
        model: nn.Module,
        target: EagerQuantizationModuleTarget,
        module_names: set[str],
        for_conversion: bool = True,
    ) -> None:
        """Check the weight representation and device used for XNNPACK export.

        Args:
            model: Model containing the selected linear or embedding layers.
            target: Quantization schema and module scope selected by the workflow.
            module_names: Fully qualified names of the selected layers.
            for_conversion: Require CPU weights when converting for XNNPACK export.

        Raises:
            ValueError: If the schema, weight format, group size or conversion
                device conflicts with the implemented XNNPACK export path.

        Note:
            Linear layers use dynamic INT8 activations and symmetric INT4 weights.
            Embeddings use symmetric INT2, INT4 or INT8 weight-only quantization.
            Conversion produces unpacked INT8 storage for ExecuTorch lowering.
        """
        config = target.quantize_config
        label = target.label
        if isinstance(target.schema, SmoothQuantSchema):
            raise ValueError(
                f"Target '{label}': SmoothQuantSchema requires the "
                "torch_inductor deployment backend. XNNPACK lowering requires "
                "separate validation."
            )
        if target.module_type == QuantizationModuleType.EMBEDDING:
            self._validate_embedding_config(config=config, label=label)
        elif isinstance(
            config, (Int4WeightOnlyConfig, Int8WeightOnlyConfig, IntxWeightOnlyConfig)
        ):
            raise ValueError(
                f"Target '{label}': XNNPACK linear lowering requires "
                "Int8DynamicActivationIntxWeightConfig with symmetric INT4 "
                "weights and unpacked_to_int8 packing."
            )
        elif isinstance(config, Int8DynamicActivationIntxWeightConfig):
            if config.intx_packing_format != IntxPackingFormat.UNPACKED_TO_INT8:
                raise ValueError(
                    f"Target '{label}': XNNPACK requires unpacked_to_int8 weight packing."
                )
            if config.weight_mapping_type not in (
                MappingType.SYMMETRIC,
                MappingType.SYMMETRIC_NO_CLIPPING_ERR,
            ):
                raise ValueError(
                    f"Target '{label}': XNNPACK requires symmetric weights without affine offsets."
                )
            if config.weight_dtype != torch.int4:
                logger.warning(
                    f"Target '{label}': XNNPACK eager linear weights other than INT4 are unverified by these rules."
                )
            group_size = target.schema.weight_group_size
            if group_size is not None and group_size % 32 != 0:
                raise ValueError(
                    f"Target '{label}': XNNPACK grouped INT4 requires a group size divisible by 32."
                )
        else:
            config_name = f"{type(config).__module__}.{type(config).__qualname__}"
            logger.warning(
                f"Target '{label}': XNNPACK lowering for {config_name} is unverified."
            )
        if for_conversion and any(
            model.get_submodule(name).weight.device.type != "cpu"
            for name in module_names
        ):
            raise ValueError(
                f"Target '{label}': this XNNPACK export workflow requires CPU weights."
            )

    @staticmethod
    def _validate_embedding_config(config: AOBaseConfig, label: str) -> None:
        """Check the weight-only embedding format accepted by ExecuTorch.

        Args:
            config: TorchAO conversion configuration.
            label: Target path used in validation errors.

        Raises:
            ValueError: If the configuration conflicts with the embedding format.
        """
        if not isinstance(config, IntxWeightOnlyConfig):
            raise ValueError(
                f"Target '{label}': XNNPACK embeddings require IntxWeightOnlyConfig."
            )
        if (
            config.version != 2
            or config.intx_packing_format != IntxPackingFormat.UNPACKED_TO_INT8
            or config.mapping_type != MappingType.SYMMETRIC
            or config.weight_dtype not in (torch.int2, torch.int4, torch.int8)
        ):
            raise ValueError(
                f"Target '{label}': XNNPACK embeddings require version=2, symmetric "
                "INT2, INT4 or INT8 weights and unpacked_to_int8 packing."
            )

    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> DeploymentArtifact:
        """Lower a PyTorch module into an ExecuTorch .pte buffer."""
        exported_program = _export_with_dynamic_batch(
            model=model,
            example_inputs=example_inputs,
            max_batch_size=self.max_batch_size,
        )
        model_bytes = self._lower_to_pte_buffer(exported_program=exported_program)
        return DeploymentArtifact(
            converted_model=None,
            example_inputs=example_inputs,
            model_filename=self.model_filename,
            artifact_format=self.artifact_format,
            backend_name=self.name,
            model_bytes=model_bytes,
        )

    @staticmethod
    def _lower_to_pte_buffer(
        exported_program: torch.export.ExportedProgram,
    ) -> bytes:
        """Lower an exported program to an ExecuTorch PTE buffer."""
        executorch_exir = importlib.import_module("executorch.exir")
        xnnpack_partitioner = importlib.import_module(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )  # This avoids a hard dependency on executorch for the entire versatil package, only requiring it when this adapter is used.
        return _lower_exported_program(
            exported_program=exported_program,
            executorch_exir=executorch_exir,
            xnnpack_partitioner=xnnpack_partitioner,
        )


def _lower_exported_program(
    exported_program: torch.export.ExportedProgram,
    executorch_exir: ModuleType,
    xnnpack_partitioner: ModuleType,
) -> bytes:
    """Lower an exported inference program using imported ExecuTorch modules.

    Note:
        Embedding padding and gradient flags control backward computation.
        Removing those arguments preserves lookup values and enables fusion
        into packed quantized embedding operators.
    """
    for node in exported_program.graph.nodes:
        if (
            node.op == FXNodeOp.CALL_FUNCTION.value
            and node.target == torch.ops.aten.embedding.default
        ):
            node.args = node.args[:2]
    exported_program.graph_module.recompile()
    edge_program = executorch_exir.to_edge_transform_and_lower(
        exported_program,
        partitioner=[xnnpack_partitioner.XnnpackPartitioner()],
    )
    executorch_program = edge_program.to_executorch(
        config=executorch_exir.ExecutorchBackendConfig(
            do_quant_fusion_and_const_prop=True
        )
    )
    return bytes(executorch_program.buffer)
