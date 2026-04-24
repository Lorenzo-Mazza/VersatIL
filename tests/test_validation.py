"""Tests for versatil.validation module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from versatil.configs.training import (
    AdamWConfig,
    ParameterGroupConfig,
    TrainingConfig,
)
from versatil.data.constants import ImageNormalizationType, ObsKey, SampleKey
from versatil.data.metadata import CameraMetadata
from versatil.metrics.base import BaseLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.decoders import MoEDecoder
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.encoding.encoders.base import EncoderInput, EncodingMixin
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.feature_meta import FeatureMetadata, FeatureType
from versatil.training.stage import TrainingStage
from versatil.validation import (
    ExperimentValidationError,
    ExperimentValidator,
    validate_experiment,
)


def _flat_features(dims: dict[str, int]) -> dict[str, FeatureMetadata]:
    """Build a FeatureMetadata registry from simple name→int pairs."""
    return {
        name: FeatureMetadata(
            key=name,
            feature_type=FeatureType.FLAT.value,
            dimension=(dim,),
        )
        for name, dim in dims.items()
    }


@pytest.fixture
def mock_encoder_factory() -> Callable[..., MagicMock]:
    """Factory for mock EncodingMixin instances with configurable input keys."""

    def factory(
        input_keys: str | list[str] = "left",
        backbone_name: str | None = None,
    ) -> MagicMock:
        encoder = MagicMock(spec=EncodingMixin)
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        encoder.input_specification = EncoderInput(keys=input_keys)
        encoder.validate_input_metadata.return_value = None
        if backbone_name is not None:
            encoder.backbone_name = backbone_name
        else:
            del encoder.backbone_name
        return encoder

    return factory


@pytest.fixture
def mock_encoding_pipeline_factory(
    mock_encoder_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    """Factory for mock EncodingPipeline instances with configurable encoders."""

    def factory(
        encoders: dict[str, MagicMock] | None = None,
        features: dict[str, FeatureMetadata] | None = None,
    ) -> MagicMock:
        pipeline = MagicMock(spec=EncodingPipeline)
        if encoders is None:
            encoders = {"rgb_encoder": mock_encoder_factory(input_keys="left")}
        pipeline.encoders = encoders
        if features is None:
            features = {
                "visual_features": FeatureMetadata(
                    key="visual_features",
                    feature_type=FeatureType.FLAT.value,
                    dimension=(256,),
                )
            }
        pipeline.get_features.return_value = features
        return pipeline

    return factory


@pytest.fixture
def mock_decoder_factory() -> Callable[..., MagicMock]:
    """Factory for mock ActionDecoder instances with configurable input and head keys."""

    def factory(
        input_keys: list[str] | None = None,
        action_head_keys: list[str] | None = None,
        supports_tokenized_actions: bool = False,
        auxiliary_output_keys: set[str] | None = None,
        decoder_class: type = ActionDecoder,
    ) -> MagicMock:
        decoder = MagicMock(spec=decoder_class)
        if input_keys is None:
            input_keys = ["visual_features"]
        decoder.decoder_input = DecoderInput(keys=input_keys)
        if action_head_keys is None:
            action_head_keys = ["position"]
        head_dict = {}
        for key in action_head_keys:
            head_dict[key] = MagicMock()
        decoder.action_heads = head_dict
        decoder.supports_tokenized_actions = supports_tokenized_actions
        if auxiliary_output_keys is None:
            auxiliary_output_keys = set()
        decoder.get_auxiliary_output_keys.return_value = auxiliary_output_keys
        return decoder

    return factory


@pytest.fixture
def mock_observation_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ObservationSpace with configurable observation keys."""

    def factory(
        observation_keys: dict | None = None,
    ) -> MagicMock:
        observation_space = MagicMock()
        if observation_keys is None:
            camera_meta = MagicMock(spec=CameraMetadata)
            observation_keys = {"left": camera_meta}
        observation_space.observations_metadata = observation_keys
        return observation_space

    return factory


@pytest.fixture
def mock_action_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ActionSpace with configurable action metadata."""

    def factory(
        actions_metadata: dict | None = None,
    ) -> MagicMock:
        action_space = MagicMock()
        if actions_metadata is None:
            position_meta = MagicMock()
            position_meta.requires_prediction_head = True
            actions_metadata = {"position": position_meta}
        action_space.actions_metadata = actions_metadata
        return action_space

    return factory


@pytest.fixture
def mock_loss_factory() -> Callable[..., MagicMock]:
    """Factory for mock BaseLoss instances with configurable required keys."""

    def factory(
        required_keys: set[str] | None = None,
        sub_losses: dict[str, MagicMock] | None = None,
    ) -> MagicMock:
        loss = MagicMock(spec=BaseLoss)
        if required_keys is None:
            required_keys = {"position"}
        loss.get_required_keys.return_value = required_keys
        if sub_losses is None:
            mock_sub = MagicMock(spec=BaseLoss)
            mock_sub.requires_action_space_targets = False
            loss.loss_modules = {"regression": mock_sub}
        else:
            loss.loss_modules = sub_losses
        return loss

    return factory


@pytest.fixture
def mock_algorithm_factory() -> Callable[..., MagicMock]:
    """Factory for mock DecodingAlgorithm instances."""

    def factory(
        is_variational: bool = False,
        predicts_in_action_space: bool = True,
        auxiliary_output_keys: set[str] | None = None,
    ) -> MagicMock:
        if is_variational:
            algorithm = MagicMock(spec=VariationalAlgorithm)
            if auxiliary_output_keys is None:
                auxiliary_output_keys = {
                    LatentKey.POSTERIOR_LATENT.value,
                    LatentKey.POSTERIOR_MU.value,
                    LatentKey.POSTERIOR_LOGVAR.value,
                    LatentKey.PRIOR_LATENT.value,
                    LatentKey.PRIOR_MU.value,
                    LatentKey.PRIOR_LOGVAR.value,
                    LatentKey.PRIOR_PREDICTION.value,
                    LatentKey.PRIOR_TARGET.value,
                }
        else:
            algorithm = MagicMock(spec=DecodingAlgorithm)
            if auxiliary_output_keys is None:
                auxiliary_output_keys = set()
        algorithm.get_auxiliary_output_keys.return_value = auxiliary_output_keys
        algorithm.predicts_in_action_space = predicts_in_action_space
        return algorithm

    return factory


@pytest.fixture
def validator_factory(
    mock_encoding_pipeline_factory: Callable[..., MagicMock],
    mock_algorithm_factory: Callable[..., MagicMock],
    mock_decoder_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    mock_action_space_factory: Callable[..., MagicMock],
    mock_loss_factory: Callable[..., MagicMock],
) -> Callable[..., ExperimentValidator]:
    """Factory for ExperimentValidator instances with all dependencies mocked."""

    def factory(
        encoding_pipeline: MagicMock | None = None,
        algorithm: MagicMock | None = None,
        decoder: MagicMock | None = None,
        observation_space: MagicMock | None = None,
        action_space: MagicMock | None = None,
        loss: MagicMock | None = None,
        is_tokenized: bool = False,
        tokenized_obs_keys: set[str] | None = None,
        image_norm_type: str | None = None,
        training_config: TrainingConfig | None = None,
    ) -> ExperimentValidator:
        return ExperimentValidator(
            encoding_pipeline=encoding_pipeline or mock_encoding_pipeline_factory(),
            algorithm=algorithm or mock_algorithm_factory(),
            decoder=decoder or mock_decoder_factory(),
            observation_space=observation_space or mock_observation_space_factory(),
            action_space=action_space or mock_action_space_factory(),
            loss=loss or mock_loss_factory(),
            is_tokenized=is_tokenized,
            tokenized_obs_keys=tokenized_obs_keys,
            image_norm_type=image_norm_type,
            training_config=training_config,
        )

    return factory


@pytest.fixture
def training_config_with_groups() -> Callable[..., TrainingConfig]:
    """Factory for ``TrainingConfig`` instances with named optimizer groups."""

    def factory(
        stages: list[TrainingStage] | None = None,
        group_names: tuple[str, ...] = ("backbone", "prior"),
    ) -> TrainingConfig:
        return TrainingConfig(
            optimizer=AdamWConfig(
                param_groups=[
                    ParameterGroupConfig(name=name, lr=1e-4) for name in group_names
                ],
            ),
            stages=stages or [],
        )

    return factory


@pytest.mark.unit
class TestExperimentValidatorInit:
    def test_stores_configuration(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_algorithm_factory: Callable[..., MagicMock],
        mock_decoder_factory: Callable[..., MagicMock],
    ):
        pipeline = mock_encoding_pipeline_factory()
        algorithm = mock_algorithm_factory()
        decoder = mock_decoder_factory()
        validator = validator_factory(
            encoding_pipeline=pipeline,
            algorithm=algorithm,
            decoder=decoder,
        )
        assert validator.encoding_pipeline is pipeline
        assert validator.algorithm is algorithm
        assert validator.decoder is decoder

    def test_tokenized_obs_keys_defaults_to_empty_set(
        self,
        validator_factory: Callable[..., ExperimentValidator],
    ):
        validator = validator_factory(tokenized_obs_keys=None)
        assert validator.tokenized_obs_keys == set()

    def test_stores_tokenized_obs_keys_when_provided(
        self,
        validator_factory: Callable[..., ExperimentValidator],
    ):
        keys = {"language_instruction", "gripper_state_obs"}
        validator = validator_factory(tokenized_obs_keys=keys)
        assert validator.tokenized_obs_keys == keys

    def test_initializes_empty_errors_and_warnings(
        self,
        validator_factory: Callable[..., ExperimentValidator],
    ):
        validator = validator_factory()
        assert validator.errors == []
        assert validator.warnings == []


@pytest.mark.unit
class TestValidateAll:
    def test_passes_with_no_errors(
        self,
        validator_factory: Callable[..., ExperimentValidator],
    ):
        validator = validator_factory()
        validator.validate_all()
        assert validator.errors == []

    def test_raises_experiment_validation_error_when_errors_found(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        observation_space = mock_observation_space_factory(
            observation_keys={ObsKey.LANGUAGE.value: MagicMock()}
        )
        validator = validator_factory(
            observation_space=observation_space,
            is_tokenized=False,
        )
        with pytest.raises(
            ExperimentValidationError,
            match=re.escape("Policy validation failed with 1 error(s):"),
        ):
            validator.validate_all()

    def test_skips_loss_key_validation_when_disabled(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        loss = mock_loss_factory(required_keys={"nonexistent_key"})
        validator = validator_factory(loss=loss)
        # Should not raise because validate_loss_keys=False skips the check
        validator.validate_all(validate_loss_keys=False)
        assert validator.errors == []

    def test_runs_loss_key_validation_by_default(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_loss_factory: Callable[..., MagicMock],
        mock_decoder_factory: Callable[..., MagicMock],
    ):
        loss = mock_loss_factory(required_keys={"nonexistent_key"})
        decoder = mock_decoder_factory(action_head_keys=["position"])
        validator = validator_factory(loss=loss, decoder=decoder)
        with pytest.raises(
            ExperimentValidationError,
            match=re.escape("Policy validation failed with 1 error(s):"),
        ):
            validator.validate_all(validate_loss_keys=True)

    def test_logs_warnings_without_raising(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        # Encoder processes "left" but observation space also has "right" uncovered
        encoder = mock_encoder_factory(input_keys="left")
        pipeline = mock_encoding_pipeline_factory(encoders={"rgb_encoder": encoder})
        camera_left = MagicMock(spec=CameraMetadata)
        camera_right = MagicMock(spec=CameraMetadata)
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_left, "right": camera_right}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
        )
        with patch("versatil.validation.logging") as mock_logging:
            validator.validate_all()
        assert len(validator.warnings) == 1
        expected_warning = (
            "Observation space contains keys {'right'} "
            "but no encoder is configured to process them."
        )
        assert validator.warnings[0] == expected_warning
        mock_logging.warning.assert_called_once()

    def test_aggregates_multiple_errors(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
        mock_decoder_factory: Callable[..., MagicMock],
    ):
        # Error 1: encoder requires key not in observation space
        encoder = mock_encoder_factory(input_keys="nonexistent_camera")
        pipeline = mock_encoding_pipeline_factory(encoders={"rgb_encoder": encoder})
        camera_meta = MagicMock(spec=CameraMetadata)
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_meta}
        )
        # Error 2: loss requires key not in decoder heads
        loss = mock_loss_factory(required_keys={"missing_action"})
        decoder = mock_decoder_factory(action_head_keys=["position"])

        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            loss=loss,
            decoder=decoder,
        )
        with pytest.raises(
            ExperimentValidationError,
            match=re.escape("Policy validation failed with 2 error(s):"),
        ):
            validator.validate_all()


@pytest.mark.unit
class TestValidateEncoderObservationConsistency:
    def test_language_without_tokenization_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        observation_space = mock_observation_space_factory(
            observation_keys={ObsKey.LANGUAGE.value: MagicMock()}
        )
        validator = validator_factory(
            observation_space=observation_space,
            is_tokenized=False,
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.errors) == 1
        assert validator.errors[0] == (
            "Language observations are enabled but tokenization is disabled. "
            "Language observations require tokenization to be enabled."
        )

    def test_language_without_tokenization_returns_early(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_observation_space_factory: Callable[..., MagicMock],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        # Encoder requires key "nonexistent" which is not in obs space;
        # but the language-without-tokenization error causes early return
        # so the missing key error should NOT be added
        encoder = mock_encoder_factory(input_keys="nonexistent")
        pipeline = mock_encoding_pipeline_factory(encoders={"encoder": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={ObsKey.LANGUAGE.value: MagicMock()}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            is_tokenized=False,
        )
        validator.validate_encoder_observation_consistency()
        # Only the language error, not the missing encoder key error
        assert len(validator.errors) == 1
        assert validator.errors[0] == (
            "Language observations are enabled but tokenization is disabled. "
            "Language observations require tokenization to be enabled."
        )

    def test_tokenized_camera_stays_available(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        # Camera observations should remain in available_keys even with tokenization
        camera_meta = MagicMock(spec=CameraMetadata)
        encoder = mock_encoder_factory(input_keys="left")
        pipeline = mock_encoding_pipeline_factory(encoders={"rgb_encoder": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            is_tokenized=True,
            tokenized_obs_keys=set(),
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.errors) == 0

    def test_tokenized_non_camera_obs_adds_tokenized_keys(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        # Non-camera obs that IS in tokenized_obs_keys should be consumed;
        # tokenized_observations and is_pad_observation should be added
        proprio_meta = MagicMock()  # Not CameraMetadata => not a camera
        encoder = mock_encoder_factory(
            input_keys=SampleKey.TOKENIZED_OBSERVATIONS.value
        )
        pipeline = mock_encoding_pipeline_factory(encoders={"token_encoder": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={"gripper_state_obs": proprio_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            is_tokenized=True,
            tokenized_obs_keys={"gripper_state_obs"},
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.errors) == 0

    def test_language_enabled_but_not_in_tokenized_keys_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        language_meta = MagicMock()  # Not CameraMetadata
        encoder = mock_encoder_factory(input_keys="some_key")
        pipeline = mock_encoding_pipeline_factory(encoders={"encoder": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={ObsKey.LANGUAGE.value: language_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            is_tokenized=True,
            tokenized_obs_keys={"gripper_state_obs"},
        )
        tokenized_keys = {"gripper_state_obs"}
        validator.validate_encoder_observation_consistency()
        expected_error = (
            f"Language observations are enabled but '{ObsKey.LANGUAGE.value}' is not in "
            f"observation_tokenizer.observation_keys: {tokenized_keys}"
        )
        assert expected_error in validator.errors

    def test_non_tokenized_obs_keeps_all_keys_available(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        camera_meta = MagicMock(spec=CameraMetadata)
        proprio_meta = MagicMock()
        encoder = mock_encoder_factory(input_keys=["left", "gripper_state_obs"])
        pipeline = mock_encoding_pipeline_factory(encoders={"encoder": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={
                "left": camera_meta,
                "gripper_state_obs": proprio_meta,
            }
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            is_tokenized=False,
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.errors) == 0

    def test_encoder_requires_missing_key_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        encoder = mock_encoder_factory(input_keys="depth")
        pipeline = mock_encoding_pipeline_factory(encoders={"depth_encoder": encoder})
        camera_meta = MagicMock(spec=CameraMetadata)
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.errors) == 1
        expected_error = (
            "Encoder 'depth_encoder' requires keys {'depth'} "
            "which are not in observation space. "
            "Available keys: {'left'}. "
            "Please either add them to the observation space or modify encoder configuration."
        )
        assert validator.errors[0] == expected_error

    def test_dinov3_backbone_with_wrong_normalization_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        encoder = mock_encoder_factory(
            input_keys="left",
            backbone_name="DINOv3_ViT_Small",
        )
        pipeline = mock_encoding_pipeline_factory(encoders={"vit_encoder": encoder})
        camera_meta = MagicMock(spec=CameraMetadata)
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            image_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.errors) == 1
        expected_error = (
            f"Encoder 'vit_encoder' uses DINOv3 backbone which requires "
            f"ImageNet normalization, but image_norm_type is set to "
            f"'{ImageNormalizationType.ZERO_TO_ONE.value}'. Set it to 'imagenet'."
        )
        assert validator.errors[0] == expected_error

    def test_dinov3_backbone_with_imagenet_normalization_passes(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        encoder = mock_encoder_factory(
            input_keys="left",
            backbone_name="DINOv3_ViT_Small",
        )
        pipeline = mock_encoding_pipeline_factory(encoders={"vit_encoder": encoder})
        camera_meta = MagicMock(spec=CameraMetadata)
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            image_norm_type=ImageNormalizationType.IMAGENET.value,
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.errors) == 0

    def test_uncovered_observation_keys_produce_warning(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        encoder = mock_encoder_factory(input_keys="left")
        pipeline = mock_encoding_pipeline_factory(encoders={"rgb_encoder": encoder})
        camera_left = MagicMock(spec=CameraMetadata)
        camera_right = MagicMock(spec=CameraMetadata)
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_left, "right": camera_right}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
        )
        validator.validate_encoder_observation_consistency()
        assert len(validator.warnings) == 1
        expected_warning = (
            "Observation space contains keys {'right'} "
            "but no encoder is configured to process them."
        )
        assert validator.warnings[0] == expected_warning

    def test_uncovered_keys_excludes_tokenized_special_keys(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        # tokenized_observations and is_pad_observation should not trigger warnings
        proprio_meta = MagicMock()
        encoder = mock_encoder_factory(
            input_keys=SampleKey.TOKENIZED_OBSERVATIONS.value
        )
        pipeline = mock_encoding_pipeline_factory(encoders={"token_encoder": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={"gripper_state_obs": proprio_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
            is_tokenized=True,
            tokenized_obs_keys={"gripper_state_obs"},
        )
        validator.validate_encoder_observation_consistency()
        # is_pad_observation key is added but excluded from uncovered check
        assert len(validator.warnings) == 0

    def test_encoder_metadata_validation_error_appended(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_observation_space_factory: Callable[..., MagicMock],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        camera_meta = MagicMock()
        encoder = mock_encoder_factory(input_keys="left")
        encoder.validate_input_metadata.return_value = (
            "Expected CameraMetadata for 'left', got MagicMock"
        )
        pipeline = mock_encoding_pipeline_factory(encoders={"rgb": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
        )
        validator.validate_encoder_observation_consistency()
        assert any("Expected CameraMetadata" in e for e in validator.errors)

    def test_encoder_metadata_validation_no_error_when_valid(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_observation_space_factory: Callable[..., MagicMock],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        camera_meta = MagicMock()
        encoder = mock_encoder_factory(input_keys="left")
        encoder.validate_input_metadata.return_value = None
        pipeline = mock_encoding_pipeline_factory(encoders={"rgb": encoder})
        observation_space = mock_observation_space_factory(
            observation_keys={"left": camera_meta}
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            observation_space=observation_space,
        )
        validator.validate_encoder_observation_consistency()
        metadata_errors = [e for e in validator.errors if "CameraMetadata" in e]
        assert len(metadata_errors) == 0


@pytest.mark.unit
class TestValidateDecoderEncoderCompatibility:
    def test_passes_when_decoder_keys_match_encoder_outputs(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_decoder_factory: Callable[..., MagicMock],
    ):
        pipeline = mock_encoding_pipeline_factory(
            features={
                "visual_features": FeatureMetadata(
                    key="visual_features",
                    feature_type=FeatureType.FLAT.value,
                    dimension=(256,),
                ),
                "proprio_features": FeatureMetadata(
                    key="proprio_features",
                    feature_type=FeatureType.FLAT.value,
                    dimension=(64,),
                ),
            }
        )
        decoder = mock_decoder_factory(
            input_keys=["visual_features", "proprio_features"]
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )
        validator.validate_decoder_encoder_compatibility()
        assert len(validator.errors) == 0

    def test_missing_decoder_feature_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_decoder_factory: Callable[..., MagicMock],
    ):
        pipeline = mock_encoding_pipeline_factory(
            features={
                "visual_features": FeatureMetadata(
                    key="visual_features",
                    feature_type=FeatureType.FLAT.value,
                    dimension=(256,),
                ),
            }
        )
        decoder = mock_decoder_factory(
            input_keys=["visual_features", "language_features"]
        )
        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )
        validator.validate_decoder_encoder_compatibility()
        assert len(validator.errors) == 1
        expected_error = (
            "Action decoding network expects input feature 'language_features' "
            "but it's not produced by any encoder or fusion layer. "
            "Available features: ['visual_features']"
        )
        assert validator.errors[0] == expected_error

    def test_calls_validate_feature_types_on_decoder_input(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_decoder_factory: Callable[..., MagicMock],
    ):
        features = {
            "visual_features": FeatureMetadata(
                key="visual_features",
                feature_type=FeatureType.SPATIAL.value,
                dimension=(3, 14, 14),
            ),
        }
        pipeline = mock_encoding_pipeline_factory(features=features)
        decoder = mock_decoder_factory(input_keys=["visual_features"])
        mock_decoder_input = MagicMock()
        mock_decoder_input.keys = ["visual_features"]
        decoder.decoder_input = mock_decoder_input

        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )
        validator.validate_decoder_encoder_compatibility()
        mock_decoder_input.validate_feature_types.assert_called_once_with(
            available_features=features
        )


@pytest.mark.unit
class TestValidateMoEGatingFeature:
    def test_moe_decoder_triggers_gating_validation(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        features = _flat_features({"visual_features": 256})
        pipeline = mock_encoding_pipeline_factory(features=features)
        decoder = MagicMock(spec=MoEDecoder)
        decoder.decoder_input = DecoderInput(keys=["visual_features"])
        decoder.action_heads = {"position": MagicMock()}
        decoder.has_gating_network = True
        decoder.gating_feature_key = "visual_features"
        decoder.supports_tokenized_actions = False
        decoder.__class__ = MoEDecoder

        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )
        validator.validate_decoder_encoder_compatibility()
        assert len(validator.errors) == 0

    def test_moe_gating_key_missing_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        features = _flat_features({"visual_features": 256})
        pipeline = mock_encoding_pipeline_factory(features=features)
        decoder = MagicMock(spec=MoEDecoder)
        decoder.decoder_input = DecoderInput(keys=["visual_features"])
        decoder.action_heads = {"position": MagicMock()}
        decoder.has_gating_network = True
        decoder.gating_feature_key = "nonexistent_feature"
        decoder.supports_tokenized_actions = False
        decoder.__class__ = MoEDecoder

        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )
        validator.validate_decoder_encoder_compatibility()
        assert len(validator.errors) == 1
        expected_error = (
            "MoE decoder gating feature key 'nonexistent_feature' not found. "
            "Available features from encoding pipeline: ['visual_features']. "
            "Algorithm provides latent: No."
        )
        assert validator.errors[0] == expected_error

    def test_moe_without_gating_network_skips_validation(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        features = _flat_features({"visual_features": 256})
        pipeline = mock_encoding_pipeline_factory(features=features)
        decoder = MagicMock(spec=MoEDecoder)
        decoder.decoder_input = DecoderInput(keys=["visual_features"])
        decoder.action_heads = {"position": MagicMock()}
        decoder.has_gating_network = False
        decoder.supports_tokenized_actions = False
        decoder.__class__ = MoEDecoder

        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )
        validator.validate_decoder_encoder_compatibility()
        assert len(validator.errors) == 0

    def test_variational_algorithm_with_posterior_latent_gating_key_passes(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_algorithm_factory: Callable[..., MagicMock],
    ):
        features = _flat_features({"visual_features": 256})
        pipeline = mock_encoding_pipeline_factory(features=features)
        decoder = MagicMock(spec=MoEDecoder)
        decoder.decoder_input = DecoderInput(keys=["visual_features"])
        decoder.action_heads = {"position": MagicMock()}
        decoder.has_gating_network = True
        decoder.gating_feature_key = LatentKey.POSTERIOR_LATENT.value
        decoder.supports_tokenized_actions = False
        decoder.__class__ = MoEDecoder

        algorithm = mock_algorithm_factory(is_variational=True)

        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
            algorithm=algorithm,
        )
        validator.validate_decoder_encoder_compatibility()
        assert len(validator.errors) == 0

    def test_non_variational_algorithm_with_latent_gating_key_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_encoding_pipeline_factory: Callable[..., MagicMock],
        mock_algorithm_factory: Callable[..., MagicMock],
    ):
        features = _flat_features({"visual_features": 256})
        pipeline = mock_encoding_pipeline_factory(features=features)
        decoder = MagicMock(spec=MoEDecoder)
        decoder.decoder_input = DecoderInput(keys=["visual_features"])
        decoder.action_heads = {"position": MagicMock()}
        decoder.has_gating_network = True
        decoder.gating_feature_key = LatentKey.POSTERIOR_LATENT.value
        decoder.supports_tokenized_actions = False
        decoder.__class__ = MoEDecoder

        algorithm = mock_algorithm_factory(is_variational=False)

        validator = validator_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
            algorithm=algorithm,
        )
        validator.validate_decoder_encoder_compatibility()
        assert len(validator.errors) == 1
        expected_error = (
            f"MoE decoder gating feature key '{LatentKey.POSTERIOR_LATENT.value}' not found. "
            f"Available features from encoding pipeline: ['visual_features']. "
            f"Algorithm provides latent: No."
        )
        assert validator.errors[0] == expected_error


@pytest.mark.unit
class TestValidateLossKeys:
    def test_valid_loss_keys_pass(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        decoder = mock_decoder_factory(action_head_keys=["position", "orientation"])
        loss = mock_loss_factory(required_keys={"position", "orientation"})
        validator = validator_factory(decoder=decoder, loss=loss)
        validator.validate_loss_keys()
        assert len(validator.errors) == 0

    def test_invalid_loss_key_produces_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        decoder = mock_decoder_factory(action_head_keys=["position"])
        loss = mock_loss_factory(required_keys={"position", "unknown_key"})
        validator = validator_factory(decoder=decoder, loss=loss)
        validator.validate_loss_keys()
        assert len(validator.errors) == 1
        expected_error = (
            "Loss module references keys {'unknown_key'} that are not "
            "defined in the action space or auxiliary keys. "
            "Valid loss keys: {'position'}. "
            "Please update your loss configuration or decoder."
        )
        assert validator.errors[0] == expected_error

    def test_action_without_prediction_head_is_valid_loss_key(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_action_space_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        # Action with requires_prediction_head=False should still be a valid loss key
        meta_with_head = MagicMock()
        meta_with_head.requires_prediction_head = True
        meta_without_head = MagicMock()
        meta_without_head.requires_prediction_head = False
        action_space = mock_action_space_factory(
            actions_metadata={
                "position": meta_with_head,
                "phase_label": meta_without_head,
            }
        )
        decoder = mock_decoder_factory(action_head_keys=["position"])
        loss = mock_loss_factory(required_keys={"position", "phase_label"})
        validator = validator_factory(
            decoder=decoder,
            action_space=action_space,
            loss=loss,
        )
        validator.validate_loss_keys()
        assert len(validator.errors) == 0

    def test_variational_algorithm_adds_latent_keys(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_algorithm_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        algorithm = mock_algorithm_factory(is_variational=True)
        decoder = mock_decoder_factory(action_head_keys=["position"])
        loss = mock_loss_factory(
            required_keys={
                "position",
                LatentKey.POSTERIOR_MU.value,
                LatentKey.POSTERIOR_LOGVAR.value,
            }
        )
        validator = validator_factory(
            decoder=decoder,
            algorithm=algorithm,
            loss=loss,
        )
        validator.validate_loss_keys()
        assert len(validator.errors) == 0

    def test_moe_decoder_adds_routing_weights_key(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        decoder = mock_decoder_factory(
            action_head_keys=["position"],
            auxiliary_output_keys={
                DecoderOutputKey.ROUTING_WEIGHTS.value,
                DecoderOutputKey.EXPERT_OUTPUTS.value,
            },
        )
        loss = mock_loss_factory(
            required_keys={"position", DecoderOutputKey.ROUTING_WEIGHTS.value}
        )
        validator = validator_factory(decoder=decoder, loss=loss)
        validator.validate_loss_keys()
        assert len(validator.errors) == 0

    def test_mode_act_decoder_adds_routing_weights_key(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        decoder = mock_decoder_factory(
            action_head_keys=["position"],
            auxiliary_output_keys={DecoderOutputKey.ROUTING_WEIGHTS.value},
        )
        loss = mock_loss_factory(
            required_keys={"position", DecoderOutputKey.ROUTING_WEIGHTS.value}
        )
        validator = validator_factory(decoder=decoder, loss=loss)
        validator.validate_loss_keys()
        assert len(validator.errors) == 0

    def test_moe_head_adds_routing_weights_key(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        decoder = mock_decoder_factory(
            action_head_keys=["position"],
            auxiliary_output_keys={DecoderOutputKey.ROUTING_WEIGHTS.value},
        )
        loss = mock_loss_factory(
            required_keys={"position", DecoderOutputKey.ROUTING_WEIGHTS.value}
        )
        validator = validator_factory(decoder=decoder, loss=loss)
        validator.validate_loss_keys()
        assert len(validator.errors) == 0

    def test_free_action_transformer_adds_binary_logits_key(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        decoder = mock_decoder_factory(
            action_head_keys=["position"],
            auxiliary_output_keys={
                DecoderOutputKey.BINARY_LOGITS.value,
                DecoderOutputKey.LATENT_CODES.value,
            },
        )
        loss = mock_loss_factory(
            required_keys={"position", DecoderOutputKey.BINARY_LOGITS.value}
        )
        validator = validator_factory(decoder=decoder, loss=loss)
        validator.validate_loss_keys()
        assert len(validator.errors) == 0

    def test_tokenized_decoder_adds_tokenized_actions_key(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_decoder_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        decoder = mock_decoder_factory(
            action_head_keys=["position"],
            auxiliary_output_keys={SampleKey.TOKENIZED_ACTIONS.value},
        )
        loss = mock_loss_factory(
            required_keys={"position", SampleKey.TOKENIZED_ACTIONS.value}
        )
        validator = validator_factory(decoder=decoder, loss=loss)
        validator.validate_loss_keys()
        assert len(validator.errors) == 0


@pytest.mark.unit
class TestValidateLossAlgorithmCompatibility:
    def test_passes_when_algorithm_predicts_in_action_space(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_algorithm_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        bce_loss = MagicMock(spec=BaseLoss)
        bce_loss.requires_action_space_targets = True
        loss = mock_loss_factory(sub_losses={"gripper": bce_loss})
        algorithm = mock_algorithm_factory(predicts_in_action_space=True)
        validator = validator_factory(algorithm=algorithm, loss=loss)
        validator.validate_loss_algorithm_compatibility()
        assert len(validator.errors) == 0

    def test_errors_when_bce_loss_with_non_action_space_algorithm(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_algorithm_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        bce_loss = MagicMock(spec=BaseLoss)
        bce_loss.requires_action_space_targets = True
        loss = mock_loss_factory(sub_losses={"gripper": bce_loss})
        algorithm = mock_algorithm_factory(predicts_in_action_space=False)
        validator = validator_factory(algorithm=algorithm, loss=loss)
        validator.validate_loss_algorithm_compatibility()
        assert len(validator.errors) == 1
        assert "gripper" in validator.errors[0]
        assert "action space" in validator.errors[0]

    def test_passes_when_regression_loss_with_non_action_space_algorithm(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_algorithm_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        regression_loss = MagicMock(spec=BaseLoss)
        regression_loss.requires_action_space_targets = False
        loss = mock_loss_factory(sub_losses={"regression": regression_loss})
        algorithm = mock_algorithm_factory(predicts_in_action_space=False)
        validator = validator_factory(algorithm=algorithm, loss=loss)
        validator.validate_loss_algorithm_compatibility()
        assert len(validator.errors) == 0

    def test_errors_for_each_incompatible_loss_module(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        mock_algorithm_factory: Callable[..., MagicMock],
        mock_loss_factory: Callable[..., MagicMock],
    ):
        bce_1 = MagicMock(spec=BaseLoss)
        bce_1.requires_action_space_targets = True
        bce_2 = MagicMock(spec=BaseLoss)
        bce_2.requires_action_space_targets = True
        regression = MagicMock(spec=BaseLoss)
        regression.requires_action_space_targets = False
        loss = mock_loss_factory(
            sub_losses={
                "gripper_left": bce_1,
                "gripper_right": bce_2,
                "regression": regression,
            }
        )
        algorithm = mock_algorithm_factory(predicts_in_action_space=False)
        validator = validator_factory(algorithm=algorithm, loss=loss)
        validator.validate_loss_algorithm_compatibility()
        assert len(validator.errors) == 2


class TestValidateExperiment:
    def test_validate_experiment_passes_with_valid_config(self):
        mock_config = MagicMock()
        mock_config.task.observation_space = MagicMock()
        mock_config.task.action_space = MagicMock()
        mock_config.task.dataloader.tokenization.tokenize_observations = False
        mock_config.task.dataloader.tokenization.observation_tokenizer = None
        mock_config.task.dataloader.image_norm_type = (
            ImageNormalizationType.IMAGENET.value
        )
        mock_config.policy.encoding_pipeline = MagicMock(spec=EncodingPipeline)
        mock_config.policy.encoding_pipeline.encoders = {}
        mock_config.policy.algorithm = MagicMock(spec=DecodingAlgorithm)
        mock_config.policy.decoder = MagicMock(spec=ActionDecoder)
        mock_config.policy.decoder.decoder_input = DecoderInput(keys=[])
        mock_config.policy.decoder.action_heads = {}
        mock_config.policy.decoder.supports_tokenized_actions = False
        mock_config.policy.decoder.__class__ = ActionDecoder
        mock_config.policy.loss_module = MagicMock(spec=BaseLoss)
        mock_config.policy.loss_module.get_required_keys.return_value = set()
        mock_config.experiment.validate_loss_keys = True
        mock_config.quantization = None

        # No observation keys => no encoder mismatch errors
        mock_config.task.observation_space.observations_metadata = {}
        mock_config.policy.encoding_pipeline.get_features.return_value = {}

        validate_experiment(config=mock_config)

    def test_extracts_tokenized_obs_keys_from_config(self):
        mock_config = MagicMock()
        mock_config.task.observation_space = MagicMock()
        mock_config.task.action_space = MagicMock()
        mock_config.task.dataloader.tokenization.tokenize_observations = True
        mock_config.task.dataloader.tokenization.observation_tokenizer.observation_keys = [
            "gripper_state_obs",
            "proprio_robot_frame",
        ]
        mock_config.task.dataloader.image_norm_type = None
        mock_config.policy.encoding_pipeline = MagicMock(spec=EncodingPipeline)
        mock_config.policy.encoding_pipeline.encoders = {}
        mock_config.policy.algorithm = MagicMock(spec=DecodingAlgorithm)
        mock_config.policy.decoder = MagicMock(spec=ActionDecoder)
        mock_config.policy.decoder.decoder_input = DecoderInput(keys=[])
        mock_config.policy.decoder.action_heads = {}
        mock_config.policy.decoder.supports_tokenized_actions = False
        mock_config.policy.decoder.__class__ = ActionDecoder
        mock_config.policy.loss_module = MagicMock(spec=BaseLoss)
        mock_config.policy.loss_module.get_required_keys.return_value = set()
        mock_config.experiment.validate_loss_keys = False

        mock_config.task.observation_space.observations_metadata = {}
        mock_config.policy.encoding_pipeline.get_features.return_value = {}

        with patch("versatil.validation.ExperimentValidator") as mock_validator_class:
            mock_validator_instance = MagicMock()
            mock_validator_class.return_value = mock_validator_instance
            validate_experiment(config=mock_config)
            call_kwargs = mock_validator_class.call_args[1]
            assert call_kwargs["is_tokenized"] is True
            assert call_kwargs["tokenized_obs_keys"] == {
                "gripper_state_obs",
                "proprio_robot_frame",
            }

    def test_no_tokenizer_yields_empty_tokenized_keys(self):
        mock_config = MagicMock()
        mock_config.task.observation_space = MagicMock()
        mock_config.task.action_space = MagicMock()
        mock_config.task.dataloader.tokenization.tokenize_observations = True
        mock_config.task.dataloader.tokenization.observation_tokenizer = None
        mock_config.task.dataloader.image_norm_type = None
        mock_config.policy.encoding_pipeline = MagicMock(spec=EncodingPipeline)
        mock_config.policy.encoding_pipeline.encoders = {}
        mock_config.policy.algorithm = MagicMock(spec=DecodingAlgorithm)
        mock_config.policy.decoder = MagicMock(spec=ActionDecoder)
        mock_config.policy.decoder.decoder_input = DecoderInput(keys=[])
        mock_config.policy.decoder.action_heads = {}
        mock_config.policy.decoder.supports_tokenized_actions = False
        mock_config.policy.decoder.__class__ = ActionDecoder
        mock_config.policy.loss_module = MagicMock(spec=BaseLoss)
        mock_config.policy.loss_module.get_required_keys.return_value = set()
        mock_config.experiment.validate_loss_keys = False

        mock_config.task.observation_space.observations_metadata = {}
        mock_config.policy.encoding_pipeline.get_features.return_value = {}

        with patch("versatil.validation.ExperimentValidator") as mock_validator_class:
            mock_validator_instance = MagicMock()
            mock_validator_class.return_value = mock_validator_instance
            validate_experiment(config=mock_config)
            call_kwargs = mock_validator_class.call_args[1]
            assert call_kwargs["tokenized_obs_keys"] == set()

    def test_passes_validate_loss_keys_from_experiment_config(self):
        mock_config = MagicMock()
        mock_config.task.observation_space = MagicMock()
        mock_config.task.action_space = MagicMock()
        mock_config.task.dataloader.tokenization.tokenize_observations = False
        mock_config.task.dataloader.tokenization.observation_tokenizer = None
        mock_config.task.dataloader.image_norm_type = None
        mock_config.policy.encoding_pipeline = MagicMock(spec=EncodingPipeline)
        mock_config.policy.encoding_pipeline.encoders = {}
        mock_config.policy.algorithm = MagicMock(spec=DecodingAlgorithm)
        mock_config.policy.decoder = MagicMock(spec=ActionDecoder)
        mock_config.policy.decoder.decoder_input = DecoderInput(keys=[])
        mock_config.policy.decoder.action_heads = {}
        mock_config.policy.decoder.supports_tokenized_actions = False
        mock_config.policy.decoder.__class__ = ActionDecoder
        mock_config.policy.loss_module = MagicMock(spec=BaseLoss)
        mock_config.policy.loss_module.get_required_keys.return_value = set()
        mock_config.experiment.validate_loss_keys = False

        mock_config.task.observation_space.observations_metadata = {}
        mock_config.policy.encoding_pipeline.get_features.return_value = {}

        with patch("versatil.validation.ExperimentValidator") as mock_validator_class:
            mock_validator_instance = MagicMock()
            mock_validator_class.return_value = mock_validator_instance
            validate_experiment(config=mock_config)
            mock_validator_instance.validate_all.assert_called_once_with(
                validate_loss_keys=False
            )

    def test_forwards_training_config_to_validator(self):
        mock_config = MagicMock()
        mock_config.task.observation_space = MagicMock()
        mock_config.task.action_space = MagicMock()
        mock_config.task.dataloader.tokenization.tokenize_observations = False
        mock_config.task.dataloader.tokenization.observation_tokenizer = None
        mock_config.task.dataloader.image_norm_type = None
        mock_config.policy.encoding_pipeline = MagicMock(spec=EncodingPipeline)
        mock_config.policy.encoding_pipeline.encoders = {}
        mock_config.policy.algorithm = MagicMock(spec=DecodingAlgorithm)
        mock_config.policy.decoder = MagicMock(spec=ActionDecoder)
        mock_config.policy.decoder.decoder_input = DecoderInput(keys=[])
        mock_config.policy.decoder.action_heads = {}
        mock_config.policy.decoder.supports_tokenized_actions = False
        mock_config.policy.decoder.__class__ = ActionDecoder
        mock_config.policy.loss_module = MagicMock(spec=BaseLoss)
        mock_config.policy.loss_module.get_required_keys.return_value = set()
        mock_config.experiment.validate_loss_keys = False

        mock_config.task.observation_space.observations_metadata = {}
        mock_config.policy.encoding_pipeline.get_features.return_value = {}

        sentinel_training = MagicMock(spec=TrainingConfig)
        mock_config.training = sentinel_training

        with patch("versatil.validation.ExperimentValidator") as mock_validator_class:
            mock_validator_class.return_value = MagicMock()
            validate_experiment(config=mock_config)
            assert (
                mock_validator_class.call_args.kwargs["training_config"]
                is sentinel_training
            )


@pytest.mark.unit
class TestValidateStageOrdering:
    def test_duplicate_stage_names_raise(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(name="vae", start_epoch=0),
            TrainingStage(name="vae", start_epoch=1),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages)
        )
        with pytest.raises(
            ExperimentValidationError,
            match=r"Training stage names must be unique: \['vae'\]",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_non_increasing_start_epoch_raises(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(name="first", start_epoch=5),
            TrainingStage(name="second", start_epoch=5),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages)
        )
        with pytest.raises(
            ExperimentValidationError,
            match="strictly increasing start_epoch order",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_overlapping_stage_intervals_raise(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(name="first", start_epoch=0, end_epoch=5),
            TrainingStage(name="second", start_epoch=3),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages)
        )
        with pytest.raises(
            ExperimentValidationError,
            match="intervals must not overlap",
        ):
            validator.validate_all(validate_loss_keys=False)


@pytest.mark.unit
class TestValidateStageGroupReferences:
    def test_unknown_group_raises(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(name="typo", start_epoch=0, trainable_groups=["prioor"]),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match=(
                r"Training stage 'typo' references unknown optimizer groups "
                r"\['prioor'\]"
            ),
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_known_optimizer_group_passes(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(
                name="freeze_backbone", start_epoch=0, frozen_groups=["backbone"]
            ),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages),
        )
        validator.validate_stage_group_references()
        assert validator.errors == []

    def test_unmatched_group_is_always_available(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(
                name="freeze_rest", start_epoch=0, frozen_groups=["unmatched"]
            ),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages),
        )
        validator.validate_stage_group_references()
        assert validator.errors == []


@pytest.mark.unit
class TestValidateStageLossPaths:
    def test_empty_loss_module_with_loss_weights_raises(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {}
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights={"denoising_prior": {"weight": 0.0}},
            )
        ]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match="no tunable weights",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_unknown_loss_key_raises(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {"denoising_prior": {"weight": 0.03}}
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights={"denoising_proir": {"weight": 0.0}},
            )
        ]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match=r"'stage' loss_weights:.+Unknown weight key 'denoising_proir'",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_shape_mismatch_scalar_for_dict_subtree_raises(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {"regression_loss": {"mse_weight": 1.0}}
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights={"regression_loss": 0.5},
            )
        ]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match=r"expects a dict subtree",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_valid_stage_loss_weights_pass(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {"denoising_prior": {"weight": 0.03}}
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights={"denoising_prior": {"weight": 0.0}},
            )
        ]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        validator.validate_stage_loss_paths()
        assert validator.errors == []

    def test_hydra_stage_loss_weights_pass(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {"denoising_prior": {"weight": 0.03}}
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights=OmegaConf.create({"denoising_prior": {"weight": 0.0}}),
            )
        ]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        validator.validate_stage_loss_paths()
        assert validator.errors == []

    def test_shape_mismatch_dict_for_scalar_leaf_raises(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {"denoising_prior": {"weight": 0.03}}
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights={"denoising_prior": {"weight": {"nested": 0.0}}},
            )
        ]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match=r"expects a scalar",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_nested_unknown_key_raises(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {"regression_loss": {"mse_weight": 1.0, "l1_weight": 0.0}}
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights={"regression_loss": {"bogus_weight": 0.0}},
            )
        ]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match=r"Unknown weight key 'bogus_weight'",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_stage_without_loss_weights_is_skipped(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        loss = mock_loss_factory()
        loss.weights = {}
        stages = [TrainingStage(name="stage", start_epoch=0)]
        validator = validator_factory(
            loss=loss,
            training_config=training_config_with_groups(stages=stages),
        )
        validator.validate_stage_loss_paths()
        assert validator.errors == []


@pytest.mark.unit
class TestValidateStageGroupReferencesSweep:
    def test_group_lrs_reference_triggers_unknown_group_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                group_lrs={"missing_group": 1e-3},
            ),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match=r"references unknown optimizer groups \['missing_group'\]",
        ):
            validator.validate_all(validate_loss_keys=False)

    def test_group_weight_decays_reference_triggers_unknown_group_error(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        stages = [
            TrainingStage(
                name="stage",
                start_epoch=0,
                group_weight_decays={"missing_group": 1e-2},
            ),
        ]
        validator = validator_factory(
            training_config=training_config_with_groups(stages=stages),
        )
        with pytest.raises(
            ExperimentValidationError,
            match=r"references unknown optimizer groups \['missing_group'\]",
        ):
            validator.validate_all(validate_loss_keys=False)


@pytest.mark.unit
class TestStageValidationSkippedWithoutTrainingConfig:
    def test_no_training_config_skips_stage_validation(
        self,
        validator_factory: Callable[..., ExperimentValidator],
    ) -> None:
        validator = validator_factory(training_config=None)
        validator.validate_all(validate_loss_keys=False)
        assert validator.errors == []

    def test_empty_stages_skips_stage_validation(
        self,
        validator_factory: Callable[..., ExperimentValidator],
        training_config_with_groups: Callable[..., TrainingConfig],
    ) -> None:
        validator = validator_factory(
            training_config=training_config_with_groups(stages=[])
        )
        validator.validate_all(validate_loss_keys=False)
        assert validator.errors == []
