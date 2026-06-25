"""Tests for versatil.configs.quantization module."""

import hydra
import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from versatil.configs.quantization import (
    EagerQuantizationModuleTargetConfig,
    EagerQuantizationWorkflowConfig,
    Int4WeightOnlyQuantizeConfig,
    PT2EQuantizationModuleTargetConfig,
    PT2EQuantizationWorkflowConfig,
    X86InductorBackendConfig,
)
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
