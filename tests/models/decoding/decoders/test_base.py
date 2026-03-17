"""Tests for versatil.models.decoding.decoders.base module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.constants import FeatureType
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput


class ConcreteDecoder(ActionDecoder):
    """Minimal concrete subclass for testing ActionDecoder methods."""

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {}


class TokenizedConcreteDecoder(ConcreteDecoder):
    """Concrete decoder that supports tokenized actions."""

    supports_tokenized_actions: bool = True


@pytest.fixture
def decoder_input_factory() -> Callable[..., DecoderInput]:
    """Factory for DecoderInput instances."""

    def factory(
        keys: list[str] | None = None,
        required_types: list[str] | None = None,
        raises_for_types: list[str] | None = None,
        requires_actions: bool = False,
        conditioning_key: str | None = None,
        conditioning_required: list[str] | None = None,
        conditioning_one_of_groups: list[list[str]] | None = None,
    ) -> DecoderInput:
        if keys is None:
            keys = ["rgb_features"]
        if required_types is None:
            required_types = []
        if raises_for_types is None:
            raises_for_types = []
        if conditioning_required is None:
            conditioning_required = []
        if conditioning_one_of_groups is None:
            conditioning_one_of_groups = []
        return DecoderInput(
            keys=keys,
            required_types=required_types,
            raises_for_types=raises_for_types,
            requires_actions=requires_actions,
            conditioning_key=conditioning_key,
            conditioning_required=conditioning_required,
            conditioning_one_of_groups=conditioning_one_of_groups,
        )

    return factory


@pytest.fixture
def mock_action_head_factory() -> Callable[..., MagicMock]:
    """Factory for mock action heads compatible with nn.ModuleDict."""

    def factory(output_dim: int = 3) -> MagicMock:
        head = MagicMock(spec=nn.Module)
        head.output_dim = output_dim
        head.set_output_dim = MagicMock(
            side_effect=lambda dim: setattr(head, "output_dim", dim)
        )
        return head

    return factory


@pytest.fixture
def concrete_decoder_factory(
    decoder_input_factory: Callable[..., DecoderInput],
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    mock_action_head_factory: Callable[..., MagicMock],
) -> Callable[..., ConcreteDecoder]:
    """Factory for ConcreteDecoder instances with configurable dependencies."""

    def factory(
        decoder_input: DecoderInput | None = None,
        action_space: MagicMock | None = None,
        observation_space: MagicMock | None = None,
        action_heads: dict[str, MagicMock] | None = None,
        observation_horizon: int = 1,
        prediction_horizon: int = 8,
        device: str = "cpu",
        tokenized: bool = False,
    ) -> ConcreteDecoder:
        if decoder_input is None:
            decoder_input = decoder_input_factory()
        if action_space is None:
            action_space = mock_action_space_factory()
        if observation_space is None:
            observation_space = mock_observation_space_factory()
        if action_heads is None:
            action_heads = {
                key: mock_action_head_factory(
                    output_dim=meta.prediction_dimension,
                )
                for key, meta in action_space.actions_metadata.items()
                if meta.requires_prediction_head
            }
        decoder_class = TokenizedConcreteDecoder if tokenized else ConcreteDecoder
        return decoder_class(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )

    return factory


class TestDecoderInputInitialization:
    @pytest.mark.parametrize("keys", [["rgb", "depth"], ["proprio"]])
    @pytest.mark.parametrize("requires_actions", [True, False])
    @pytest.mark.parametrize("conditioning_key", [None, "language"])
    def test_stores_configuration(
        self,
        decoder_input_factory: Callable[..., DecoderInput],
        keys: list[str],
        requires_actions: bool,
        conditioning_key: str | None,
    ):
        decoder_input = decoder_input_factory(
            keys=keys,
            requires_actions=requires_actions,
            conditioning_key=conditioning_key,
        )
        assert decoder_input.keys == keys
        assert decoder_input.requires_actions is requires_actions
        assert decoder_input.conditioning_key == conditioning_key

    @pytest.mark.parametrize(
        "conditioning_required, expectation",
        [
            (["language"], does_not_raise()),
            (
                ["language", "depth_cond"],
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Missing required conditioning for decoder input: {'depth_cond'}"
                    ),
                ),
            ),
        ],
    )
    def test_conditioning_required_validation(
        self,
        conditioning_required: list[str],
        expectation,
    ):
        with expectation:
            DecoderInput(
                keys=["rgb"],
                conditioning_key="language",
                conditioning_required=conditioning_required,
            )

    @pytest.mark.parametrize(
        "one_of_groups, expectation",
        [
            ([["language", "other"]], does_not_raise()),
            (
                [["other1", "other2"]],
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Exactly one from ['other1', 'other2'] required for decoder input conditioning"
                    ),
                ),
            ),
        ],
    )
    def test_conditioning_one_of_groups_validation(
        self,
        one_of_groups: list[list[str]],
        expectation,
    ):
        with expectation:
            DecoderInput(
                keys=["rgb"],
                conditioning_key="language",
                conditioning_one_of_groups=one_of_groups,
            )


class TestDecoderInputValidateFeatureTypes:
    @pytest.mark.parametrize(
        "required_type, feature_dim, expectation",
        [
            (FeatureType.FLAT.value, 64, does_not_raise()),
            (FeatureType.SEQUENTIAL.value, (10, 64), does_not_raise()),
            (FeatureType.SPATIAL.value, (512, 7, 7), does_not_raise()),
            (
                FeatureType.FLAT.value,
                (512, 7, 7),
                pytest.raises(
                    ValueError,
                    match=f"requires at least one input feature of type '{FeatureType.FLAT.value}'",
                ),
            ),
            (
                FeatureType.SPATIAL.value,
                64,
                pytest.raises(
                    ValueError,
                    match=f"requires at least one input feature of type '{FeatureType.SPATIAL.value}'",
                ),
            ),
        ],
    )
    def test_required_type_validation(
        self,
        decoder_input_factory: Callable[..., DecoderInput],
        required_type: str,
        feature_dim: int | tuple,
        expectation,
    ):
        decoder_input = decoder_input_factory(
            keys=["feat"],
            required_types=[required_type],
        )
        with expectation:
            decoder_input.validate_feature_types(
                available_features_to_dims={"feat": feature_dim},
            )

    @pytest.mark.parametrize(
        "raises_for_type, feature_dim, expectation",
        [
            (FeatureType.SPATIAL.value, 64, does_not_raise()),
            (
                FeatureType.SPATIAL.value,
                (512, 7, 7),
                pytest.raises(
                    ValueError,
                    match="Decoder architecture cannot accept spatial features as input.",
                ),
            ),
            (
                FeatureType.SEQUENTIAL.value,
                (10, 64),
                pytest.raises(
                    ValueError,
                    match="Decoder architecture cannot accept sequential features as input.",
                ),
            ),
            (
                FeatureType.FLAT.value,
                64,
                pytest.raises(
                    ValueError,
                    match="Decoder architecture cannot accept flat features as input.",
                ),
            ),
        ],
    )
    def test_raises_for_type_validation(
        self,
        decoder_input_factory: Callable[..., DecoderInput],
        raises_for_type: str,
        feature_dim: int | tuple,
        expectation,
    ):
        decoder_input = decoder_input_factory(
            keys=["feat"],
            raises_for_types=[raises_for_type],
        )
        with expectation:
            decoder_input.validate_feature_types(
                available_features_to_dims={"feat": feature_dim},
            )


class TestActionDecoderInterface:
    def test_supports_tokenized_actions_default_false(self):
        assert ActionDecoder.supports_tokenized_actions is False


class TestActionDecoderProperties:
    @pytest.mark.parametrize(
        "observation_horizon, expected",
        [
            (1, False),
            (2, True),
            (3, True),
        ],
    )
    def test_has_history(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        observation_horizon: int,
        expected: bool,
    ):
        decoder = concrete_decoder_factory(observation_horizon=observation_horizon)
        assert decoder.has_history is expected

    def test_action_dim_delegates_to_action_space(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(position_dim=12)
        decoder = concrete_decoder_factory(action_space=action_space)
        assert decoder.action_dim == 12
        action_space.get_total_action_dim.assert_called()

    def test_use_gripper_actions_delegates_to_action_space(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(has_gripper=True, gripper_dim=1)
        decoder = concrete_decoder_factory(action_space=action_space)
        assert decoder.use_gripper_actions is True

    @pytest.mark.parametrize(
        "has_gripper, gripper_dim, expected",
        [
            (False, 0, None),
            (True, 1, 1),
        ],
    )
    def test_gripper_dim(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
        has_gripper: bool,
        gripper_dim: int,
        expected: int | None,
    ):
        action_space = mock_action_space_factory(
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        decoder = concrete_decoder_factory(action_space=action_space)
        assert decoder.gripper_dim == expected

    def test_use_orientation_actions(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(
            has_orientation=True, orientation_dim=4
        )
        decoder = concrete_decoder_factory(action_space=action_space)
        assert decoder.use_orientation_actions is True

    @pytest.mark.parametrize(
        "has_orientation, orientation_dim, expected",
        [
            (False, 0, None),
            (True, 4, 4),
        ],
    )
    def test_orientation_dim(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
        has_orientation: bool,
        orientation_dim: int,
        expected: int | None,
    ):
        action_space = mock_action_space_factory(
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
        )
        decoder = concrete_decoder_factory(action_space=action_space)
        assert decoder.orientation_dim == expected

    @pytest.mark.parametrize(
        "position_dim, expected",
        [
            (0, None),
            (3, 3),
        ],
    )
    def test_position_dim(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
        position_dim: int,
        expected: int | None,
    ):
        action_space = mock_action_space_factory(
            position_dim=position_dim,
        )
        decoder = concrete_decoder_factory(action_space=action_space)
        assert decoder.position_dim == expected


class TestActionDecoderSetTokenizer:
    def test_non_tokenized_decoder_ignores_tokenizer(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
    ):
        decoder = concrete_decoder_factory(tokenized=False)
        mock_tokenizer = MagicMock(spec=Tokenizer)
        decoder.set_tokenizer(tokenizer=mock_tokenizer)
        assert decoder.tokenizer is None

    def test_tokenized_decoder_raises_without_tokenizer(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
    ):
        decoder = concrete_decoder_factory(tokenized=True)
        with pytest.raises(
            ValueError,
            match="Tokenizer must be provided for tokenized action decoders.",
        ):
            decoder.set_tokenizer(tokenizer=None)

    def test_tokenized_decoder_stores_tokenizer(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
    ):
        decoder = concrete_decoder_factory(tokenized=True)
        mock_tokenizer = MagicMock(spec=Tokenizer)
        mock_action_tokenizer = MagicMock()
        mock_tokenizer.action_tokenizer = mock_action_tokenizer
        decoder.set_tokenizer(tokenizer=mock_tokenizer)
        assert decoder.tokenizer is mock_action_tokenizer


class TestActionDecoderSetNormalizer:
    def test_stores_normalizer(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
    ):
        decoder = concrete_decoder_factory()
        normalizer = LinearNormalizer()
        decoder.set_normalizer(normalizer=normalizer)
        assert decoder.normalizer is normalizer


class TestActionDecoderValidateActionHeads:
    def test_missing_head_raises(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
        mock_action_head_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
        )
        position_head = mock_action_head_factory(output_dim=3)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Action space requires heads for {'orientation_action'}, but they are not configured. "
                "Configured heads: {'position_action'}"
            ),
        ):
            concrete_decoder_factory(
                action_space=action_space,
                action_heads={"position_action": position_head},
            )

    def test_extra_head_raises(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
        mock_action_head_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(position_dim=3)
        position_head = mock_action_head_factory(output_dim=3)
        extra_head = mock_action_head_factory(output_dim=2)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Action head 'nonexistent_action' not found in action_space.actions_metadata. "
                "Available keys: ['position_action']"
            ),
        ):
            concrete_decoder_factory(
                action_space=action_space,
                action_heads={
                    "position_action": position_head,
                    "nonexistent_action": extra_head,
                },
            )

    def test_dimension_mismatch_raises(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
        mock_action_head_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(position_dim=3)
        # Create head with output_dim=5 and override set_output_dim to be a no-op,
        # so _set_action_head_dimensions cannot correct it to the expected dim=3.
        wrong_dim_head = mock_action_head_factory(output_dim=5)
        wrong_dim_head.set_output_dim = MagicMock()
        with pytest.raises(
            ValueError,
            match="Action head 'position_action' has output_dim=5, but action space requires dim=3",
        ):
            concrete_decoder_factory(
                action_space=action_space,
                action_heads={"position_action": wrong_dim_head},
            )

    def test_valid_heads_pass(
        self,
        concrete_decoder_factory: Callable[..., ConcreteDecoder],
        mock_action_space_factory: Callable[..., MagicMock],
        mock_action_head_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(
            position_dim=3,
            has_gripper=True,
            gripper_dim=1,
        )
        position_head = mock_action_head_factory(output_dim=3)
        gripper_head = mock_action_head_factory(output_dim=1)
        decoder = concrete_decoder_factory(
            action_space=action_space,
            action_heads={
                "position_action": position_head,
                "gripper_action": gripper_head,
            },
        )
        assert "position_action" in decoder.action_heads
        assert "gripper_action" in decoder.action_heads


class TestActionDecoderSetActionHeadDimensions:
    def test_sets_dims_from_action_metadata(
        self,
        mock_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        mock_action_head_factory: Callable[..., MagicMock],
        decoder_input_factory: Callable[..., DecoderInput],
    ):
        action_space = mock_action_space_factory(
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
        )
        position_head = mock_action_head_factory(output_dim=0)
        orientation_head = mock_action_head_factory(output_dim=0)
        ConcreteDecoder(
            decoder_input=decoder_input_factory(),
            observation_space=mock_observation_space_factory(),
            action_space=action_space,
            action_heads={
                "position_action": position_head,
                "orientation_action": orientation_head,
            },
            device="cpu",
            observation_horizon=1,
            prediction_horizon=8,
        )
        position_head.set_output_dim.assert_called_once_with(3)
        orientation_head.set_output_dim.assert_called_once_with(4)

    def test_tokenized_sets_dim_to_one(
        self,
        mock_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        mock_action_head_factory: Callable[..., MagicMock],
        decoder_input_factory: Callable[..., DecoderInput],
    ):
        action_space = mock_action_space_factory(position_dim=3)
        head = mock_action_head_factory(output_dim=0)
        TokenizedConcreteDecoder(
            decoder_input=decoder_input_factory(),
            observation_space=mock_observation_space_factory(),
            action_space=action_space,
            action_heads={"position_action": head},
            device="cpu",
            observation_horizon=1,
            prediction_horizon=8,
        )
        head.set_output_dim.assert_called_once_with(1)
