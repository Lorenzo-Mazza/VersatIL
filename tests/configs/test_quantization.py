"""Tests for versatil.configs.quantization module."""

import hydra
import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from versatil.configs.quantization import (
    DirectQuantizationSchemaConfig,
    EagerQuantizationModuleTargetConfig,
    EagerQuantizationWorkflowConfig,
    Int4WeightOnlyQuantizeConfig,
    Int8DynamicQuantizeConfig,
    PT2EQuantizationModuleTargetConfig,
    PT2EQuantizationWorkflowConfig,
    SmoothQuantSchemaConfig,
    X86InductorBackendConfig,
    XNNPACKPT2EBackendConfig,
)
from versatil.quantization.constants import PT2EBackendName
from versatil.quantization.module_target import (
    EagerQuantizationModuleTarget,
    PT2EQuantizationModuleTarget,
)
from versatil.quantization.pt2e.backends.x86_inductor import X86InductorBackend
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow


@pytest.mark.unit
class TestX86InductorBackendConfig:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    @pytest.mark.parametrize("is_qat", [True, False])
    def test_hydra_instantiates_backend(self, is_dynamic, is_qat):
        config = OmegaConf.structured(
            X86InductorBackendConfig(
                is_dynamic=is_dynamic,
                is_qat=is_qat,
            )
        )

        backend = hydra.utils.instantiate(config)

        assert isinstance(backend, X86InductorBackend)
        assert backend.is_dynamic == is_dynamic
        assert backend.is_qat == is_qat


@pytest.mark.unit
class TestXNNPACKPT2EBackendConfig:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    @pytest.mark.parametrize("is_qat", [True, False])
    @pytest.mark.parametrize("is_per_channel", [True, False])
    def test_hydra_instantiates_backend(
        self,
        is_dynamic: bool,
        is_qat: bool,
        is_per_channel: bool,
    ) -> None:
        config = OmegaConf.structured(
            XNNPACKPT2EBackendConfig(
                is_dynamic=is_dynamic,
                is_qat=is_qat,
                is_per_channel=is_per_channel,
            )
        )

        backend = hydra.utils.instantiate(config)

        assert backend.name == PT2EBackendName.XNNPACK.value
        assert backend.is_dynamic == is_dynamic
        assert backend.is_qat == is_qat
        assert backend.is_per_channel == is_per_channel


@pytest.mark.unit
class TestPT2EQuantizationWorkflowConfig:
    def test_hydra_instantiates_with_default_backend(self):
        config = OmegaConf.structured(PT2EQuantizationWorkflowConfig())

        result = hydra.utils.instantiate(config)

        assert isinstance(result, PT2EQuantizationWorkflow)
        assert isinstance(result.targets[0], PT2EQuantizationModuleTarget)
        assert isinstance(result.pt2e_backend, X86InductorBackend)

    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_propagates_backend_config(self, is_dynamic):
        config = OmegaConf.structured(
            PT2EQuantizationWorkflowConfig(
                targets=[
                    PT2EQuantizationModuleTargetConfig(
                        pt2e_backend=X86InductorBackendConfig(
                            is_dynamic=is_dynamic,
                            is_qat=False,
                        ),
                    ),
                ],
            )
        )

        result = hydra.utils.instantiate(config)

        assert result.pt2e_backend.is_dynamic == is_dynamic
        assert result.pt2e_backend.is_qat is False

    def test_hydra_instantiates_with_xnnpack_backend(self) -> None:
        config = OmegaConf.structured(
            PT2EQuantizationWorkflowConfig(
                targets=[
                    PT2EQuantizationModuleTargetConfig(
                        pt2e_backend=XNNPACKPT2EBackendConfig(
                            is_dynamic=False,
                            is_qat=False,
                            is_per_channel=True,
                        ),
                    ),
                ],
            )
        )

        result = hydra.utils.instantiate(config)

        assert result.pt2e_backend.name == PT2EBackendName.XNNPACK.value
        assert result.pt2e_backend.is_per_channel is True

    def test_rejects_qat_backend_config(self):
        config = OmegaConf.structured(
            PT2EQuantizationWorkflowConfig(
                targets=[
                    PT2EQuantizationModuleTargetConfig(
                        pt2e_backend=X86InductorBackendConfig(
                            is_qat=True,
                        ),
                    ),
                ],
            )
        )

        with pytest.raises(
            InstantiationException,
            match="PT2E QAT configuration is not supported yet.",
        ):
            hydra.utils.instantiate(config)


@pytest.mark.unit
class TestEagerQuantizationWorkflowConfig:
    @pytest.mark.integration
    @pytest.mark.parametrize("is_qat", [False, True])
    def test_hydra_resolves_explicit_direct_schema_for_ptq_and_qat(
        self, is_qat: bool
    ) -> None:
        config = OmegaConf.structured(
            EagerQuantizationWorkflowConfig(
                targets=[
                    EagerQuantizationModuleTargetConfig(
                        module_path="decoder",
                        schema=DirectQuantizationSchemaConfig(
                            base_config=Int4WeightOnlyQuantizeConfig(group_size=64)
                        ),
                    )
                ],
                is_qat=is_qat,
            )
        )

        workflow = hydra.utils.instantiate(config)

        target = workflow.targets[0]
        assert workflow.is_qat is is_qat
        assert target.quantize_config.group_size == 64
        assert target.schema.parameters == {}
        assert target.schema.needs_calibration is False
        conversion = target.schema.conversion_config(is_qat=is_qat)
        if is_qat:
            assert conversion.base_config is target.quantize_config
            assert conversion.step == "convert"
        else:
            assert conversion is target.quantize_config

    @pytest.mark.integration
    @pytest.mark.parametrize("alpha", [0.25, 0.75])
    def test_hydra_preserves_smoothquant_schema_and_base_config(
        self, alpha: float
    ) -> None:
        config = OmegaConf.structured(
            EagerQuantizationWorkflowConfig(
                targets=[
                    EagerQuantizationModuleTargetConfig(
                        module_path="decoder",
                        schema=SmoothQuantSchemaConfig(
                            base_config=Int8DynamicQuantizeConfig(),
                            alpha=alpha,
                        ),
                    )
                ],
                is_qat=False,
            )
        )
        workflow = hydra.utils.instantiate(config)
        target = workflow.targets[0]
        assert target.module_path == "decoder"
        assert target.schema.parameters == {"alpha": str(alpha)}
        assert target.schema.needs_calibration is True
        assert target.quantize_config.version == 2
        assert target.quantize_config.weight_only_decode is False

    def test_hydra_instantiates_with_int4_config(self):
        config = OmegaConf.structured(
            EagerQuantizationWorkflowConfig(
                targets=[
                    EagerQuantizationModuleTargetConfig(
                        quantize_config=Int4WeightOnlyQuantizeConfig(
                            group_size=64,
                        ),
                    ),
                ],
            )
        )

        result = hydra.utils.instantiate(config)

        assert isinstance(result, EagerQuantizationWorkflow)
        assert isinstance(result.targets[0], EagerQuantizationModuleTarget)
        assert result.targets[0].quantize_config.group_size == 64

    def test_hydra_instantiates_qat_variant(self):
        config = OmegaConf.structured(
            EagerQuantizationWorkflowConfig(
                targets=[
                    EagerQuantizationModuleTargetConfig(
                        quantize_config=Int4WeightOnlyQuantizeConfig(),
                    ),
                ],
                is_qat=True,
            )
        )

        result = hydra.utils.instantiate(config)

        assert isinstance(result, EagerQuantizationWorkflow)
        assert result.is_qat is True
