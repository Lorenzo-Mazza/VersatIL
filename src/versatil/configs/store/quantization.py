"""Quantization and compression config registrations."""

from hydra.core.config_store import ConfigStore

from versatil.configs import (
    CompressionTargetConfig,
    EagerQuantizationModuleTargetConfig,
    EagerQuantizationWorkflowConfig,
    ExecutorchXNNPACKBackendConfig,
    Int4WeightOnlyQuantizeConfig,
    Int8DynamicQuantizeConfig,
    PT2EQuantizationModuleTargetConfig,
    PT2EQuantizationWorkflowConfig,
    StructuredPrunerConfig,
    TorchInductorBackendConfig,
    UnstructuredPrunerConfig,
    X86InductorBackendConfig,
    XNNPACKPT2EBackendConfig,
)


def register(cs: ConfigStore) -> None:
    """Store this domain's config nodes.

    Args:
        cs: The global Hydra config store.
    """
    cs.store(
        group="compression/module",
        name="base",
        node=CompressionTargetConfig,
    )
    cs.store(
        group="compression/pruning",
        name="unstructured",
        node=UnstructuredPrunerConfig,
    )
    cs.store(
        group="compression/pruning",
        name="structured",
        node=StructuredPrunerConfig,
    )
    cs.store(
        group="compression/deployment_backend",
        name="torch_inductor",
        node=TorchInductorBackendConfig,
    )
    cs.store(
        group="compression/deployment_backend",
        name="executorch_xnnpack",
        node=ExecutorchXNNPACKBackendConfig,
    )
    cs.store(
        group="quantization/workflow",
        name="pt2e",
        node=PT2EQuantizationWorkflowConfig,
    )
    cs.store(
        group="quantization/workflow",
        name="eager",
        node=EagerQuantizationWorkflowConfig,
    )
    cs.store(
        group="quantization/target",
        name="eager",
        node=EagerQuantizationModuleTargetConfig,
    )
    cs.store(
        group="quantization/target",
        name="pt2e",
        node=PT2EQuantizationModuleTargetConfig,
    )
    cs.store(
        group="quantization/backend",
        name="x86_inductor",
        node=X86InductorBackendConfig,
    )
    cs.store(
        group="quantization/backend",
        name="xnnpack",
        node=XNNPACKPT2EBackendConfig,
    )
    cs.store(
        group="quantization/quantize_config",
        name="int8_dynamic",
        node=Int8DynamicQuantizeConfig,
    )
    cs.store(
        group="quantization/quantize_config",
        name="int4_weight_only",
        node=Int4WeightOnlyQuantizeConfig,
    )
