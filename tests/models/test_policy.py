"""Tests for Policy class - validation, methods, and attributes."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, Mock

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    Cameras,
    GripperType,
    ObsKey,
    ProprioceptiveType,
    SampleKey,
)
from versatil.metrics import (
    ActionReconstructionLoss,
    PhaseActionLoss,
)
from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput
from versatil.models.encoding.encoders.depth.cnn import DepthCNNEncoder
from versatil.models.encoding.encoders.depth.dformerv2 import DFormerEncoder
from versatil.models.encoding.encoders.depth.light_geometric import LightGeometricEncoder
from versatil.models.encoding.encoders.rgb.cnn import CNNEncoder
from versatil.models.encoding.encoders.rgb.conditional_cnn import ConditionalCNNEncoder
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.policy import Policy
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm
from versatil.models.decoding.latent import VAETransformerEncoder, DiffusionPrior
from versatil.models.decoding.constants import LatentKey
from versatil.metrics.components import PriorDenoisingLoss
from tests.conftest import DummyNormalizer


class TestPolicyLossValidation:
    """Test that Policy validates loss keys against action space."""

    def test_valid_loss_keys_pass_validation(self):
        """Test that valid loss configuration passes validation."""

        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=False,
        )

        observation_space = ObservationSpace(
            use_proprio_base_frame=False,
            use_proprio_camera_frame=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        loss = ActionReconstructionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            gripper_bce_weight=1.0,
        )

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        algorithm = MagicMock()

        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=10,
            loss=loss,
            device="cpu",
            validate_loss_keys=True,
        )

        assert policy is not None

    def test_invalid_loss_keys_raise_error(self):
        """Test that invalid loss configuration raises ValueError."""

        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=False,
        )

        observation_space = ObservationSpace(
            use_proprioceptive_data=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        loss = PhaseActionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            phase_ce_weight=1.0,
        )

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        algorithm = MagicMock()

        with pytest.raises(ValueError, match="phase_label.*not defined in the action space"):
            Policy(
                encoding_pipeline=encoding_pipeline,
                algorithm=algorithm,
                decoder=decoder,
                observation_space=observation_space,
                action_space=action_space,
                prediction_horizon=10,
                loss=loss,
                device="cpu",
                validate_loss_keys=True,
            )

    def test_phase_loss_with_phase_action_space(self):
        """Test that phase loss works when action space has phases."""

        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=True,
            number_of_phases=5,
        )

        observation_space = ObservationSpace(
            use_proprioceptive_data=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        loss = PhaseActionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            phase_ce_weight=1.0,
        )

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        algorithm = MagicMock()

        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=10,
            loss=loss,
            device="cpu",
            validate_loss_keys=True,
        )

        assert policy is not None

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled if needed."""

        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=False,
        )

        observation_space = ObservationSpace(
            use_proprioceptive_data=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        loss = PhaseActionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            phase_ce_weight=1.0,
        )

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        algorithm = MagicMock()

        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=10,
            loss=loss,
            device="cpu",
            validate_loss_keys=False,
        )

        assert policy is not None

    def test_vae_keys_valid_when_algorithm_is_variational(self):
        """Test that VAE keys (mu, logvar) are valid when using VariationalAlgorithm."""
        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=False,
        )

        observation_space = ObservationSpace(
            use_proprioceptive_data=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        loss = ActionReconstructionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            gripper_bce_weight=1.0,
            use_vae=True,  # This will add KL divergence loss which uses mu and logvar
        )

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        # Create actual VariationalAlgorithm
        vae_encoder = VAETransformerEncoder(
            latent_dimension=16,
            embedding_dimension=64,
            prediction_horizon=10,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            dropout_rate=0.0,
            device="cpu",
        )
        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=None,  # Auto-creates GaussianPrior
        )

        # Should NOT raise an error because algorithm is VariationalAlgorithm
        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=10,
            loss=loss,
            device="cpu",
            validate_loss_keys=True,
        )

        assert policy is not None

    def test_vae_keys_invalid_when_algorithm_is_not_variational(self):
        """Test that VAE keys (mu, logvar) are invalid when using pure algorithm."""

        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=False,
        )

        observation_space = ObservationSpace(
            use_proprioceptive_data=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        loss = ActionReconstructionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            gripper_bce_weight=1.0,
            use_vae=True,  # This will add KL divergence loss which uses mu and logvar
        )

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        # Pure algorithm (not variational)
        algorithm = BehavioralCloning()

        # Should raise an error because algorithm is not variational but loss uses VAE keys
        with pytest.raises(ValueError, match="mu.*not.*defined"):
            Policy(
                encoding_pipeline=encoding_pipeline,
                algorithm=algorithm,
                decoder=decoder,
                observation_space=observation_space,
                action_space=action_space,
                prediction_horizon=10,
                loss=loss,
                device="cpu",
                validate_loss_keys=True,
            )

    def test_prior_keys_invalid_with_gaussian_prior(self):
        """Test that prior keys (prior_prediction, prior_target) are invalid with GaussianPrior."""
        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=False,
        )

        observation_space = ObservationSpace(
            use_proprioceptive_data=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        # Loss that requires prior keys
        loss = PriorDenoisingLoss(weight=1.0)

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        # VariationalAlgorithm with GaussianPrior (doesn't produce prior keys)
        vae_encoder = VAETransformerEncoder(
            latent_dimension=16,
            embedding_dimension=64,
            prediction_horizon=10,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            dropout_rate=0.0,
            device="cpu",
        )
        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=None,  # Auto-creates GaussianPrior
        )

        # Should raise because GaussianPrior doesn't produce prior keys
        with pytest.raises(ValueError, match="prior_prediction.*not.*defined"):
            Policy(
                encoding_pipeline=encoding_pipeline,
                algorithm=algorithm,
                decoder=decoder,
                observation_space=observation_space,
                action_space=action_space,
                prediction_horizon=10,
                loss=loss,
                device="cpu",
                validate_loss_keys=True,
            )

    def test_prior_keys_valid_with_learned_prior(self):
        """Test that prior keys (prior_prediction, prior_target) are valid with DiffusionPrior."""
        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            task_has_phases=False,
        )

        observation_space = ObservationSpace(
            use_proprioceptive_data=False,
            camera_keys=[Cameras.LEFT.value],
            use_language=False,
        )

        # Loss that requires prior keys
        loss = PriorDenoisingLoss(weight=1.0)

        encoding_pipeline = MagicMock()
        encoding_pipeline.get_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }
        encoding_pipeline.get_final_features_to_dimensions.return_value = {
            "spatial_features": (256, 7, 7)
        }

        decoder = MagicMock()
        decoder.decoder_input.keys = ["spatial_features"]
        decoder.decoder_input.validate_feature_types = MagicMock()

        # VariationalAlgorithm with DiffusionPrior (produces prior keys)
        vae_encoder = VAETransformerEncoder(
            latent_dimension=16,
            embedding_dimension=64,
            prediction_horizon=10,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            dropout_rate=0.0,
            device="cpu",
        )
        diffusion_prior = DiffusionPrior(
            latent_dimension=16,
            conditioning_dim=64,
            output_dim=64,
            hidden_dims=[32, 32],
            num_train_timesteps=10,
            num_inference_steps=3,
            device="cpu",
        )
        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=diffusion_prior,  # Learned prior
        )

        # Should NOT raise because DiffusionPrior produces prior keys
        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=10,
            loss=loss,
            device="cpu",
            validate_loss_keys=True,
        )

        assert policy is not None


@pytest.mark.unit
class TestPolicyAttributes:
    """Test Policy attributes and initialization."""

    def test_policy_device_attribute(self, simple_policy, device):
        """Test policy.device is correctly set."""
        assert simple_policy.device == device
        assert isinstance(simple_policy.device, torch.device)

    def test_policy_normalizer_attribute(self, simple_policy):
        """Test policy has normalizer attribute."""
        assert hasattr(simple_policy, "normalizer")
        assert simple_policy.normalizer is not None

    def test_policy_components_exist(self, simple_policy):
        """Test policy has all expected components."""
        assert hasattr(simple_policy, "encoding_pipeline")
        assert hasattr(simple_policy, "algorithm")
        assert hasattr(simple_policy, "decoder")
        assert hasattr(simple_policy, "loss_module")
        assert hasattr(simple_policy, "observation_space")
        assert hasattr(simple_policy, "action_space")
        assert hasattr(simple_policy, "prediction_horizon")

    def test_policy_action_space_attributes(self, simple_policy, simple_action_space):
        """Test policy stores action space correctly."""
        assert simple_policy.action_space == simple_action_space
        assert simple_policy.action_space.has_position
        assert simple_policy.action_space.has_orientation
        assert simple_policy.action_space.has_gripper

    def test_policy_observation_space_attributes(self, simple_policy, simple_observation_space):
        """Test policy stores observation space correctly."""
        assert simple_policy.observation_space == simple_observation_space


@pytest.mark.unit
class TestPolicyNormalizerMethods:
    """Test Policy normalizer-related methods."""

    def test_set_normalizer(self, simple_policy):
        """Test set_normalizer method."""
        original_normalizer = simple_policy.normalizer
        new_normalizer = DummyNormalizer()
        simple_policy.set_normalizer(new_normalizer)

        assert simple_policy.normalizer.state_dict().keys() == new_normalizer.state_dict().keys()


@pytest.mark.unit
class TestPolicyForwardMethod:
    """Test Policy forward method."""

    def test_forward_returns_predictions(self, simple_policy, synthetic_training_batch):
        """Test forward method returns action predictions."""
        simple_policy.train()

        output = simple_policy.forward(synthetic_training_batch)

        assert isinstance(output, dict)
        assert ProprioceptiveType.POSITION.value in output
        assert ProprioceptiveType.ORIENTATION.value in output
        assert ProprioceptiveType.GRIPPER.value in output

    def test_forward_output_shapes(self, simple_policy, synthetic_training_batch):
        """Test forward output has correct shapes."""
        simple_policy.train()

        output = simple_policy.forward(synthetic_training_batch)

        batch_size = synthetic_training_batch[SampleKey.ACTION.value][ProprioceptiveType.POSITION.value].shape[0]
        pred_horizon = synthetic_training_batch[SampleKey.ACTION.value][ProprioceptiveType.POSITION.value].shape[1]

        assert output[ProprioceptiveType.POSITION.value].shape[0] == batch_size
        assert output[ProprioceptiveType.POSITION.value].shape[1] == pred_horizon


@pytest.mark.unit
class TestPolicyComputeLoss:
    """Test Policy compute_loss method."""

    def test_compute_loss_returns_loss_output(self, simple_policy, synthetic_training_batch):
        """Test compute_loss returns LossOutput."""
        simple_policy.train()

        loss_output = simple_policy.compute_loss(synthetic_training_batch)

        assert hasattr(loss_output, "total_loss")
        assert hasattr(loss_output, "component_losses")

    def test_compute_loss_total_loss_is_scalar(self, simple_policy, synthetic_training_batch):
        """Test total loss is a scalar tensor."""
        simple_policy.train()

        loss_output = simple_policy.compute_loss(synthetic_training_batch)

        assert loss_output.total_loss.dim() == 0
        assert loss_output.total_loss.requires_grad

    def test_compute_loss_components_exist(self, simple_policy, synthetic_training_batch):
        """Test loss components are populated."""
        simple_policy.train()

        loss_output = simple_policy.compute_loss(synthetic_training_batch)

        assert len(loss_output.component_losses) > 0


@pytest.mark.unit
class TestPolicyPredictAction:
    """Test Policy predict_action method."""

    def test_predict_action_inference_mode(self, simple_policy, device):
        """Test predict_action in inference mode."""
        simple_policy.eval()

        obs = {
            "rgb": torch.randn(1, 2, 3, 64, 64, device=device),
            "proprio": torch.randn(1, 2, 7, device=device),
        }

        with torch.no_grad():
            actions = simple_policy.predict_action(obs)

        assert isinstance(actions, dict)

    def test_predict_action_device_movement(self, simple_policy, device):
        """Test predict_action handles device movement automatically."""
        simple_policy.eval()

        obs_cpu = {
            "rgb": torch.randn(1, 2, 3, 64, 64),
            "proprio": torch.randn(1, 2, 7),
        }

        with torch.no_grad():
            actions = simple_policy.predict_action(obs_cpu)

        for key, action_tensor in actions.items():
            assert action_tensor.device.type == device.type

    def test_predict_action_no_gradients(self, simple_policy, device):
        """Test predict_action doesn't compute gradients."""
        simple_policy.eval()

        obs = {
            "rgb": torch.randn(1, 2, 3, 64, 64, device=device),
            "proprio": torch.randn(1, 2, 7, device=device),
        }

        with torch.no_grad():
            actions = simple_policy.predict_action(obs)

            for action_tensor in actions.values():
                assert not action_tensor.requires_grad


@pytest.mark.unit
class TestPolicyDeviceConsistency:
    """Test device consistency across Policy components."""

    def test_all_encoders_on_same_device(self, simple_policy, device):
        """Test all encoders are on policy device."""
        for encoder in simple_policy.encoding_pipeline.encoders.values():
            for param in encoder.parameters():
                assert param.device.type == device.type

    def test_decoder_on_policy_device(self, simple_policy, device):
        """Test decoder is on policy device."""
        for param in simple_policy.decoder.parameters():
            assert param.device.type == device.type

    def test_normalizer_on_policy_device(self, simple_policy, device):
        """Test normalizer is on policy device."""
        for param in simple_policy.normalizer.parameters():
            assert param.device.type == device.type

    def test_policy_to_device_moves_all_components(self, simple_policy):
        """Test policy.to(device) moves all components."""
        target_device = torch.device("cpu")
        simple_policy.to(target_device)

        for param in simple_policy.parameters():
            assert param.device.type == target_device.type


class MockEncodingPipeline(nn.Module):
    """Mock encoding pipeline for testing explainer integration."""

    def __init__(self, encoders: dict[str, nn.Module] | None = None, conditional_encoders: dict[str, nn.Module] | None = None):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders or {})
        self.conditional_encoders = nn.ModuleDict(conditional_encoders or {})
        self.encoder_to_outputs: dict[str, EncoderOutput] = {}

        for name, encoder in self.encoders.items():
            self.encoder_to_outputs[name] = encoder.get_output_specification()
        for name, encoder in self.conditional_encoders.items():
            self.encoder_to_outputs[name] = encoder.get_output_specification()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}

    def get_features_to_dimensions(self) -> dict[str, int | tuple[int, ...]]:
        result = {}
        for encoder_name, output in self.encoder_to_outputs.items():
            for feat_name, dim in output.dimensions.items():
                result[f"{encoder_name}_{feat_name}"] = dim
        return result


def create_minimal_policy_for_explainer(encoding_pipeline: MockEncodingPipeline) -> Policy:
    """Create minimal Policy instance for testing explainer methods."""
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.encoding_pipeline = encoding_pipeline
    return policy


@pytest.mark.unit
class TestPolicyExplainerIntegration:
    """Test Policy methods for explainer integration."""

    def test_get_vision_encoder_modules_with_cnn(self):
        """Test that get_vision_encoder_modules() returns CNNEncoder."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"rgb_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        vision_encoders = policy.get_vision_encoder_modules()
        assert "rgb_encoder" in vision_encoders
        assert vision_encoders["rgb_encoder"] is encoder
        assert hasattr(vision_encoders["rgb_encoder"], "backbone")

    def test_get_vision_encoder_modules_with_depth_cnn(self):
        """Test that get_vision_encoder_modules() returns DepthCNNEncoder."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"depth_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        vision_encoders = policy.get_vision_encoder_modules()
        assert "depth_encoder" in vision_encoders
        assert hasattr(vision_encoders["depth_encoder"], "backbone")

    def test_get_vision_encoder_modules_with_conditional_cnn(self):
        """Test that get_vision_encoder_modules() returns ConditionalCNNEncoder."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language",
            condition_dim=512,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(conditional_encoders={"conditional_rgb_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        vision_encoders = policy.get_vision_encoder_modules()
        assert "conditional_rgb_encoder" in vision_encoders
        assert hasattr(vision_encoders["conditional_rgb_encoder"], "layer4")

    def test_get_vision_encoder_modules_with_dformer(self):
        """Test that get_vision_encoder_modules() returns DFormerEncoder."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant="S",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"dformer_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        vision_encoders = policy.get_vision_encoder_modules()
        assert "dformer_encoder" in vision_encoders
        assert hasattr(vision_encoders["dformer_encoder"], "stages")

    def test_get_vision_encoder_modules_with_light_geometric(self):
        """Test that get_vision_encoder_modules() returns LightGeometricEncoder."""
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"light_geo_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        vision_encoders = policy.get_vision_encoder_modules()
        assert "light_geo_encoder" in vision_encoders
        assert hasattr(vision_encoders["light_geo_encoder"], "attention_block")

    def test_get_vision_encoder_modules_raises_when_no_vision_encoders(self):
        """Test that get_vision_encoder_modules() raises error when no vision encoders exist."""
        class DummyEncoder(Encoder):
            def __init__(self):
                spec = EncoderInput(keys=["state"])
                super().__init__(input_specification=spec)
                self.output_dim = 10

            def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                return {"state_features": torch.zeros(1, self.output_dim)}

            def get_output_specification(self) -> EncoderOutput:
                return EncoderOutput(features=["state_features"], dimensions={"state_features": self.output_dim})

        encoder = DummyEncoder()
        encoding_pipeline = MockEncodingPipeline(encoders={"state_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        with pytest.raises(RuntimeError, match="No compatible vision encoders found"):
            policy.get_vision_encoder_modules()

    def test_get_gradcam_target_layers_for_cnn(self):
        """Test get_gradcam_target_layers() returns correct layers for CNNEncoder."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"rgb_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        target_layers = policy.get_gradcam_target_layers("rgb_encoder")
        assert len(target_layers) == 1
        assert isinstance(target_layers[0], nn.Module)

    def test_get_gradcam_target_layers_for_dformer(self):
        """Test get_gradcam_target_layers() returns correct layers for DFormerEncoder."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant="S",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"dformer_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        target_layers = policy.get_gradcam_target_layers("dformer_encoder")
        assert len(target_layers) == 1
        assert isinstance(target_layers[0], nn.Module)

    def test_get_gradcam_target_layers_for_conditional_cnn(self):
        """Test get_gradcam_target_layers() returns correct layers for ConditionalCNNEncoder."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language",
            condition_dim=512,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(conditional_encoders={"conditional_rgb_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        target_layers = policy.get_gradcam_target_layers("conditional_rgb_encoder")
        assert len(target_layers) == 1
        assert isinstance(target_layers[0], nn.Module)

    def test_get_gradcam_target_layers_for_light_geometric(self):
        """Test get_gradcam_target_layers() returns correct layers for LightGeometricEncoder."""
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"light_geo_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        target_layers = policy.get_gradcam_target_layers("light_geo_encoder")
        assert len(target_layers) == 1
        assert isinstance(target_layers[0], nn.Module)

    def test_get_gradcam_target_layers_raises_for_invalid_encoder(self):
        """Test get_gradcam_target_layers() raises error for non-existent encoder."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(encoders={"rgb_encoder": encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        with pytest.raises(ValueError, match="not found or not a vision encoder"):
            policy.get_gradcam_target_layers("nonexistent_encoder")

    def test_get_camera_to_encoder_mapping(self):
        """Test get_camera_to_encoder_mapping() returns correct mappings."""
        rgb_encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        depth_encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        encoding_pipeline = MockEncodingPipeline(
            encoders={"rgb_encoder": rgb_encoder, "depth_encoder": depth_encoder}
        )
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        mapping = policy.get_camera_to_encoder_mapping()
        assert Cameras.LEFT.value in mapping
        assert Cameras.DEPTH.value in mapping
        assert mapping[Cameras.LEFT.value] == "rgb_encoder"
        assert mapping[Cameras.DEPTH.value] == "depth_encoder"

    def test_get_camera_to_encoder_mapping_excludes_non_camera_encoders(self):
        """Test that proprioceptive/language encoders are excluded from camera mapping."""
        class ProprioEncoder(Encoder):
            def __init__(self):
                spec = EncoderInput(keys=["robot_state"])
                super().__init__(input_specification=spec)
                self.output_dim = 10
                self.backbone = nn.Identity()

            def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                return {"proprio_features": torch.zeros(1, self.output_dim)}

            def get_output_specification(self) -> EncoderOutput:
                return EncoderOutput(features=["proprio_features"], dimensions={"proprio_features": self.output_dim})

        rgb_encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone="timm/resnet18.a1_in1k",
            pretrained=False,
        )
        proprio_encoder = ProprioEncoder()

        encoding_pipeline = MockEncodingPipeline(encoders={"rgb_encoder": rgb_encoder, "proprio_encoder": proprio_encoder})
        policy = create_minimal_policy_for_explainer(encoding_pipeline)

        mapping = policy.get_camera_to_encoder_mapping()
        assert Cameras.LEFT.value in mapping
        assert "robot_state" not in mapping


@pytest.mark.unit
class TestPolicySetTokenizer:
    """Test Policy.set_tokenizer() propagation."""

    def test_set_tokenizer_propagates_to_pipeline(self):
        """Test tokenizer propagates to encoding pipeline."""
        encoding_pipeline = MagicMock()
        encoding_pipeline.set_tokenizer = Mock()
        encoding_pipeline.get_features_to_dimensions.return_value = {"features": 256}
        encoding_pipeline.get_final_features_to_dimensions.return_value = {"features": 256}

        decoder = MagicMock()
        decoder.decoder_input.keys = ["features"]
        decoder.decoder_input.validate_feature_types = MagicMock()
        decoder.set_tokenizer = Mock()

        algorithm = MagicMock()

        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=MagicMock(),
            action_space=MagicMock(),
            prediction_horizon=10,
            loss=None,
            device="cpu",
            validate_loss_keys=False,
        )

        tokenizer = Mock()
        policy.set_tokenizer(tokenizer)

        encoding_pipeline.set_tokenizer.assert_called_once_with(tokenizer)

    def test_set_tokenizer_propagates_to_decoder(self):
        """Test tokenizer propagates to decoder."""
        encoding_pipeline = MagicMock()
        encoding_pipeline.set_tokenizer = Mock()
        encoding_pipeline.get_features_to_dimensions.return_value = {"features": 256}
        encoding_pipeline.get_final_features_to_dimensions.return_value = {"features": 256}

        decoder = MagicMock()
        decoder.decoder_input.keys = ["features"]
        decoder.decoder_input.validate_feature_types = MagicMock()
        decoder.set_tokenizer = Mock()

        algorithm = MagicMock()

        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=MagicMock(),
            action_space=MagicMock(),
            prediction_horizon=10,
            loss=None,
            device="cpu",
            validate_loss_keys=False,
        )

        tokenizer = Mock()
        policy.set_tokenizer(tokenizer)

        decoder.set_tokenizer.assert_called_once_with(tokenizer)

    def test_set_tokenizer_none(self):
        """Test setting None tokenizer sets attribute but doesn't propagate."""
        encoding_pipeline = MagicMock()
        encoding_pipeline.set_tokenizer = Mock()
        encoding_pipeline.get_features_to_dimensions.return_value = {"features": 256}
        encoding_pipeline.get_final_features_to_dimensions.return_value = {"features": 256}

        decoder = MagicMock()
        decoder.decoder_input.keys = ["features"]
        decoder.decoder_input.validate_feature_types = MagicMock()
        decoder.set_tokenizer = Mock()

        algorithm = MagicMock()

        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=MagicMock(),
            action_space=MagicMock(),
            prediction_horizon=10,
            loss=None,
            device="cpu",
            validate_loss_keys=False,
        )

        policy.set_tokenizer(None)

        assert policy.tokenizer is None
        encoding_pipeline.set_tokenizer.assert_not_called()
        decoder.set_tokenizer.assert_not_called()
