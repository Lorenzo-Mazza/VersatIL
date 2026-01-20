"""End-to-end training tests with simulated data."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from versatil.workspace import Workspace
from versatil.training.callbacks import EMACallback, ConfusionMatrixCallback
from versatil.data.constants import (
    OBSERVATION_KEY,
    ACTION_KEY,
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    PHASE_LABEL_KEY,
    OrientationRepresentation,
    GripperType,
    Cameras,
)
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.policy import Policy
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.encoding.encoders.rgb.cnn import CNNEncoder
from versatil.models.encoding.encoders.constants import RGBBackboneType, PoolingMethod
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from versatil.models.decoding.decoders.factory.act import ACT
from versatil.models.decoding.decoders.factory.phase_act import PhaseACT
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.latent.posterior.transformer_encoder import VAETransformerEncoder
from versatil.models.decoding.constants import MU_KEY, LOGVAR_KEY
from versatil.metrics.composite import ActionReconstructionLoss, PhaseActionLoss
from tests.conftest import DummyNormalizer


@pytest.mark.slow
class TestWorkspaceInitialization:
    """Test Workspace initialization."""

    def test_workspace_creation(self, mock_main_config):
        workspace = Workspace(mock_main_config)

        assert workspace.config == mock_main_config
        assert workspace.exp_name == "test_experiment"
        assert workspace.output_dir.exists()
        assert workspace.policy is None
        assert workspace.lightning_policy is None
        assert workspace.trainer is None

    def test_seed_setting(self, mock_main_config):
        workspace = Workspace(mock_main_config)

        torch.manual_seed(42)
        expected_tensor = torch.rand(3, 3)

        torch.manual_seed(42)
        actual_tensor = torch.rand(3, 3)

        assert torch.allclose(expected_tensor, actual_tensor)


@pytest.mark.slow
class TestWorkspaceCallbacks:
    """Test Workspace callback creation."""

    def test_ema_callback_created_when_enabled(self, mock_main_config):
        mock_main_config.training.use_ema = True
        mock_main_config.training.ema_power = 0.75

        workspace = Workspace(mock_main_config)
        callbacks = workspace._create_callbacks()

        ema_callbacks = [cb for cb in callbacks if isinstance(cb, EMACallback)]
        assert len(ema_callbacks) == 1
        assert ema_callbacks[0].power == 0.75

    def test_ema_callback_not_created_when_disabled(self, mock_main_config):
        mock_main_config.training.use_ema = False

        workspace = Workspace(mock_main_config)
        callbacks = workspace._create_callbacks()

        ema_callbacks = [cb for cb in callbacks if isinstance(cb, EMACallback)]
        assert len(ema_callbacks) == 0

    def test_confusion_matrix_callback_for_phase_models(self, mock_main_config):
        mock_main_config.task.action_space.predict_task_phases = True

        workspace = Workspace(mock_main_config)
        callbacks = workspace._create_callbacks()

        cm_callbacks = [cb for cb in callbacks if isinstance(cb, ConfusionMatrixCallback)]
        assert len(cm_callbacks) == 1

    def test_no_confusion_matrix_callback_for_non_phase_models(self, mock_main_config):
        mock_main_config.task.action_space.predict_task_phases = False

        workspace = Workspace(mock_main_config)
        callbacks = workspace._create_callbacks()

        cm_callbacks = [cb for cb in callbacks if isinstance(cb, ConfusionMatrixCallback)]
        assert len(cm_callbacks) == 0

    def test_checkpoint_callbacks_created(self, mock_main_config):
        workspace = Workspace(mock_main_config)
        callbacks = workspace._create_callbacks()

        checkpoint_callbacks = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
        assert len(checkpoint_callbacks) == 2

        best_callbacks = [cb for cb in checkpoint_callbacks if cb.monitor == "val_loss"]
        assert len(best_callbacks) == 1
        assert best_callbacks[0].save_top_k == 3

        latest_callbacks = [cb for cb in checkpoint_callbacks if cb.save_last]
        assert len(latest_callbacks) == 1


@pytest.mark.slow
class TestWorkspaceLogger:
    """Test Workspace logger creation."""

    def test_wandb_logger_disabled_when_configured(self, mock_main_config):
        mock_main_config.experiment.use_wandb = False

        workspace = Workspace(mock_main_config)
        logger = workspace._create_logger()

        assert logger is None

    @patch.dict("os.environ", {"WANDB_API_KEY": ""}, clear=True)
    def test_wandb_logger_disabled_without_api_key(self, mock_main_config):
        mock_main_config.experiment.use_wandb = True

        workspace = Workspace(mock_main_config)
        logger = workspace._create_logger()

        assert logger is None
        assert not mock_main_config.experiment.use_wandb


@pytest.mark.slow
class TestWorkspaceStrategy:
    """Test distributed training strategy."""

    def test_auto_strategy_when_not_distributed(self, mock_main_config):
        mock_main_config.experiment.distributed = False

        workspace = Workspace(mock_main_config)
        strategy = workspace._create_strategy()

        assert strategy == "auto"

    def test_ddp_strategy_when_distributed(self, mock_main_config):
        mock_main_config.experiment.distributed = True

        workspace = Workspace(mock_main_config)
        strategy = workspace._create_strategy()

        assert isinstance(strategy, DDPStrategy)


class DummyDataset:
    """Simple picklable dataset for testing."""
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class DummyDataLoader:
    """Simple picklable dataloader for testing."""
    def __init__(self, dataset, batches):
        self.dataset = dataset
        self.batches = batches
        self.__pl_saved_kwargs = {}
        self.__pl_saved_arg_names = []

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


@pytest.mark.slow
@pytest.mark.integration
class TestEndToEndTraining:
    """End-to-end training tests with mocked data."""

    @pytest.fixture
    def mock_dataloaders(self, synthetic_training_batch):
        train_dataset = DummyDataset(length=10)
        val_dataset = DummyDataset(length=5)

        mock_train_loader = DummyDataLoader(
            dataset=train_dataset,
            batches=[synthetic_training_batch] * 3
        )

        mock_val_loader = DummyDataLoader(
            dataset=val_dataset,
            batches=[synthetic_training_batch] * 2
        )

        return mock_train_loader, mock_val_loader

    @pytest.fixture
    def mock_normalizer(self):
        return LinearNormalizer()

    def test_complete_training_workflow(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders,
        mock_normalizer,
    ):
        mock_train_loader, mock_val_loader = mock_dataloaders

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: self._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, mock_normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                workspace.run()

        assert workspace.policy == simple_policy
        assert workspace.lightning_policy is not None
        assert workspace.trainer is not None

        assert workspace.trainer.max_epochs == mock_main_config.training.num_epochs

    def test_checkpoint_saving(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders,
        mock_normalizer,
    ):
        mock_train_loader, mock_val_loader = mock_dataloaders

        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: self._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, mock_normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                workspace.run()

        assert workspace.output_dir.exists()

    def test_ema_callback_integration(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders,
        mock_normalizer,
    ):
        mock_train_loader, mock_val_loader = mock_dataloaders

        mock_main_config.training.use_ema = True
        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: self._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, mock_normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                workspace.run()

        ema_callbacks = [
            cb for cb in workspace.trainer.callbacks if isinstance(cb, EMACallback)
        ]
        assert len(ema_callbacks) == 1

    @staticmethod
    def _set_data_attrs(workspace, train_loader, val_loader, normalizer):
        workspace.train_loader = train_loader
        workspace.val_loader = val_loader
        workspace.normalizer = normalizer
        workspace.gripper_class_weights = None


@pytest.mark.slow
class TestWorkspacePrediction:
    """Test prediction functionality."""

    def test_predict_without_training_raises_error(self, mock_main_config):
        workspace = Workspace(mock_main_config)

        obs_dict = {"dummy": torch.randn(1, 3, 64, 64)}

        with pytest.raises(RuntimeError, match="Policy not initialized"):
            workspace.predict(obs_dict)

    def test_predict_with_trained_policy(
        self,
        mock_main_config,
        simple_policy,
        device,
    ):
        workspace = Workspace(mock_main_config)

        workspace.config.experiment.device = device
        workspace.policy = simple_policy
        workspace.lightning_policy = MagicMock()
        workspace.lightning_policy.policy = simple_policy

        workspace.trainer = MagicMock()
        workspace.trainer.callbacks = []

        obs_dict = {
            "rgb": torch.randn(1, 2, 3, 64, 64, device=device),
            "proprio": torch.randn(1, 2, 7, device=device),
        }

        output = workspace.predict(obs_dict)

        assert output is not None
        assert isinstance(output, dict)




@pytest.mark.integration
@pytest.mark.requires_gpu
class TestACTPolicyEndToEnd:
    """End-to-end test for ACT policy with real ResNet18 encoder."""

    @pytest.fixture
    def act_observation_space(self):
        return ObservationSpace(
            camera_keys=[Cameras.LEFT.value],
            use_proprioceptive_data=False,
            use_language=False,
            use_gripper_state=False,
            gripper_type=GripperType.BINARY.value,
        )

    @pytest.fixture
    def act_action_space(self):
        return ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            orientation_repr=OrientationRepresentation.QUATERNION.value,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            predict_in_camera_frame=False,
            deltas_as_actions=False,
            denoise_actions=False,
            task_has_phases=False,
        )

    @pytest.fixture
    def resnet18_encoder(self, device):
        """Real ResNet18 CNN encoder for RGB images."""

        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.NONE.value,
            use_group_norm=True,
            pretrained=False,
            frozen=False,
            image_height=224,
            image_width=224,
        )
        return encoder.to(device)

    @pytest.fixture
    def encoding_pipeline_act(self, resnet18_encoder, device):
        """Real encoding pipeline for ACT tests."""

        encoders = nn.ModuleDict({"rgb": resnet18_encoder})
        encoder_outputs = {"rgb": resnet18_encoder.get_output_specification()}
        fusion_stages = nn.ModuleList([])

        feature_keys_to_dims = {
            "rgb_image": (512, 7, 7),
        }

        pipeline = EncodingPipeline.__new__(EncodingPipeline)
        nn.Module.__init__(pipeline)
        pipeline.encoders = encoders
        pipeline.conditional_encoders = nn.ModuleDict()
        pipeline.fusion_stages = fusion_stages
        pipeline.encoders_to_outputs = encoder_outputs
        pipeline._feature_keys_to_dims = feature_keys_to_dims
        pipeline._consumed_features = set()  # Initialize consumed features tracker

        def _flatten_observation_dict(self, observation):
            return observation
        pipeline._flatten_observation_dict = _flatten_observation_dict.__get__(
            pipeline, EncodingPipeline
        )

        return pipeline.to(device)

    @pytest.fixture
    def mock_observations_act(self, device):
        """Mock RGB observations matching ResNet18 input requirements."""
        batch_size = 2
        return {
            Cameras.LEFT.value: torch.randn(
                batch_size, 3, 224, 224, device=device
            )
        }

    @pytest.fixture
    def mock_actions_act(self, act_action_space, device):
        """Mock action dictionary for ACT."""
        batch_size = 2
        prediction_horizon = 10
        actions = {
            POSITION_ACTION_KEY: torch.randn(
                batch_size, prediction_horizon, act_action_space.position_dim, device=device
            ),
            ORIENTATION_ACTION_KEY: torch.randn(
                batch_size, prediction_horizon, act_action_space.orientation_dim, device=device
            ),
            GRIPPER_ACTION_KEY: torch.randint(
                0, 2, (batch_size, prediction_horizon, 1), device=device
            ).float(),
        }
        return actions

    @pytest.fixture
    def act_policy(
        self,
        encoding_pipeline_act,
        act_observation_space,
        act_action_space,
        device
    ):
        """ACT policy with real ResNet18 encoder."""
        embedding_dim = 256
        prediction_horizon = 10
        observation_horizon = 1

        action_heads = {}
        if act_action_space.has_position:
            action_heads[POSITION_ACTION_KEY] = ActionHead(
                input_dim=embedding_dim,
                output_dim=act_action_space.position_dim,
                blocks=[],
            ).to(device)
        if act_action_space.has_orientation:
            action_heads[ORIENTATION_ACTION_KEY] = ActionHead(
                input_dim=embedding_dim,
                output_dim=act_action_space.orientation_dim,
                blocks=[],
            ).to(device)
        if act_action_space.has_gripper:
            action_heads[GRIPPER_ACTION_KEY] = ActionHead(
                input_dim=embedding_dim,
                output_dim=act_action_space.gripper_dim,
                blocks=[],
            ).to(device)

        decoder = ACT(
            input_keys=["rgb_image"],
            action_space=act_action_space,
            action_heads=action_heads,
            observation_space=act_observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=str(device),
            embedding_dimension=embedding_dim,
            number_of_heads=8,
            feedforward_dimension=2048,
            number_of_encoder_layers=4,
            number_of_decoder_layers=6,
        ).to(device)

        loss = ActionReconstructionLoss(
            action_keys=[POSITION_ACTION_KEY, ORIENTATION_ACTION_KEY, GRIPPER_ACTION_KEY],
            mse_weight=1.0,
            gripper_bce_weight=1.0,
            use_vae=False,
            kl_weight=0.0,
        )

        algorithm = BehavioralCloning()

        policy = Policy(
            encoding_pipeline=encoding_pipeline_act,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=act_observation_space,
            action_space=act_action_space,
            prediction_horizon=prediction_horizon,
            loss=loss,
            device=str(device),
            validate_loss_keys=True,
        )

        policy.normalizer = DummyNormalizer()
        policy.to(device)
        return policy

    @pytest.fixture
    def act_training_batch(self, mock_observations_act, mock_actions_act):
        """Complete training batch for ACT."""
        return {
            OBSERVATION_KEY: mock_observations_act,
            ACTION_KEY: mock_actions_act,
        }

    def test_act_policy_forward_pass(self, act_policy, act_training_batch, device):
        """Test ACT policy forward pass works."""
        act_policy.train()

        output = act_policy.forward(act_training_batch)

        assert POSITION_ACTION_KEY in output
        assert ORIENTATION_ACTION_KEY in output
        assert GRIPPER_ACTION_KEY in output

        assert output[POSITION_ACTION_KEY].device.type == device.type
        assert output[ORIENTATION_ACTION_KEY].device.type == device.type
        assert output[GRIPPER_ACTION_KEY].device.type == device.type

    def test_act_policy_loss_computation(self, act_policy, act_training_batch):
        """Test ACT policy loss computation."""
        act_policy.train()

        loss_output = act_policy.compute_loss(act_training_batch)

        assert loss_output.total_loss is not None
        assert loss_output.total_loss.requires_grad
        assert loss_output.total_loss.item() >= 0

    def test_act_policy_backward_pass(self, act_policy, act_training_batch):
        """Test ACT policy backward pass works."""
        act_policy.train()

        loss_output = act_policy.compute_loss(act_training_batch)

        loss_output.total_loss.backward()

        for name, param in act_policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient not computed for {name}"

    def test_act_policy_inference(self, act_policy, mock_observations_act, device):
        """Test ACT policy inference mode."""
        act_policy.eval()

        with torch.no_grad():
            actions = act_policy.predict_action(mock_observations_act)

        assert isinstance(actions, dict)
        assert POSITION_ACTION_KEY in actions
        assert ORIENTATION_ACTION_KEY in actions
        assert GRIPPER_ACTION_KEY in actions

        assert actions[POSITION_ACTION_KEY].device.type == device.type

    def test_act_policy_device_consistency(self, act_policy, device):
        """Test all policy components are on correct device."""
        assert act_policy.device == device

        for module in act_policy.encoding_pipeline.encoders.values():
            for param in module.parameters():
                assert param.device.type == device.type

        for param in act_policy.decoder.parameters():
            assert param.device.type == device.type

    def test_act_policy_training_step_shapes(self, act_policy, act_training_batch):
        """Test output shapes match expected dimensions."""
        act_policy.train()

        output = act_policy.forward(act_training_batch)

        batch_size = act_training_batch[ACTION_KEY][POSITION_ACTION_KEY].shape[0]
        pred_horizon = act_training_batch[ACTION_KEY][POSITION_ACTION_KEY].shape[1]

        assert output[POSITION_ACTION_KEY].shape == (batch_size, pred_horizon, 3)
        assert output[ORIENTATION_ACTION_KEY].shape == (batch_size, pred_horizon, 4)
        assert output[GRIPPER_ACTION_KEY].shape == (batch_size, pred_horizon, 1)



@pytest.mark.integration
@pytest.mark.requires_gpu
class TestPhaseACTEndToEnd:
    """End-to-end test for PhaseACT decoder with real ResNet18 encoder."""

    @pytest.fixture
    def phase_act_action_space(self):
        return ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            orientation_repr=OrientationRepresentation.QUATERNION.value,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            predict_in_camera_frame=False,
            deltas_as_actions=False,
            denoise_actions=False,
            task_has_phases=True,
            number_of_phases=3,
        )

    @pytest.fixture
    def phase_act_observation_space(self):
        return ObservationSpace(
            camera_keys=[Cameras.LEFT.value],
            use_proprioceptive_data=False,
            use_language=False,
            use_gripper_state=False,
            gripper_type=GripperType.BINARY.value,
        )

    @pytest.fixture
    def resnet18_encoder_phase(self, device):
        """Real ResNet18 CNN encoder for PhaseACT."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.NONE.value,
            use_group_norm=True,
            pretrained=False,
            frozen=False,
            image_height=224,
            image_width=224,
        )
        return encoder.to(device)

    @pytest.fixture
    def encoding_pipeline_phase(self, resnet18_encoder_phase, device):
        """Real encoding pipeline for PhaseACT tests."""

        encoders = nn.ModuleDict({"rgb": resnet18_encoder_phase})
        encoder_outputs = {"rgb": resnet18_encoder_phase.get_output_specification()}
        fusion_stages = nn.ModuleList([])

        feature_keys_to_dims = {
            "rgb_image": (512, 7, 7),
        }

        pipeline = EncodingPipeline.__new__(EncodingPipeline)
        nn.Module.__init__(pipeline)
        pipeline.encoders = encoders
        pipeline.conditional_encoders = nn.ModuleDict()
        pipeline.fusion_stages = fusion_stages
        pipeline.encoders_to_outputs = encoder_outputs
        pipeline._feature_keys_to_dims = feature_keys_to_dims
        pipeline._consumed_features = set()  # Initialize consumed features tracker

        def _flatten_observation_dict(self, observation):
            return observation
        pipeline._flatten_observation_dict = _flatten_observation_dict.__get__(
            pipeline, EncodingPipeline
        )

        return pipeline.to(device)

    @pytest.fixture
    def mock_observations_phase(self, device):
        """Mock RGB observations for PhaseACT."""
        batch_size = 2
        return {
            Cameras.LEFT.value: torch.randn(
                batch_size, 3, 224, 224, device=device
            )
        }

    @pytest.fixture
    def mock_actions_phase(self, phase_act_action_space, device):
        """Mock action dictionary for PhaseACT including phase labels."""
        batch_size = 2
        prediction_horizon = 10
        actions = {
            POSITION_ACTION_KEY: torch.randn(
                batch_size, prediction_horizon, phase_act_action_space.position_dim, device=device
            ),
            ORIENTATION_ACTION_KEY: torch.randn(
                batch_size, prediction_horizon, phase_act_action_space.orientation_dim, device=device
            ),
            GRIPPER_ACTION_KEY: torch.randint(
                0, 2, (batch_size, prediction_horizon, 1), device=device
            ).float(),
            PHASE_LABEL_KEY: torch.randint(
                0, phase_act_action_space.number_of_phases,
                (batch_size, prediction_horizon, 1),
                device=device,
                dtype=torch.long
            ),
        }
        return actions

    @pytest.fixture
    def phase_act_policy(
        self,
        encoding_pipeline_phase,
        phase_act_observation_space,
        phase_act_action_space,
        device
    ):
        """PhaseACT policy with real ResNet18 encoder and MoE routing."""
        embedding_dim = 256
        prediction_horizon = 10
        observation_horizon = 1

        action_heads = {}

        action_heads[PHASE_LABEL_KEY] = ActionHead(
            input_dim=embedding_dim,
            output_dim=phase_act_action_space.number_of_phases,
            blocks=[],
        ).to(device)

        action_heads[POSITION_ACTION_KEY] = MoEHead(
            base_expert_config={
                "_target_": "versatil.models.decoding.action_heads.head.ActionHead",
                "input_dim": embedding_dim,
                "output_dim": phase_act_action_space.position_dim,
                "blocks": [],
            },
            num_experts=phase_act_action_space.number_of_phases,
            output_dim=phase_act_action_space.position_dim,
            gating_input_dim=None,
            device=str(device),
        )

        action_heads[ORIENTATION_ACTION_KEY] = ActionHead(
            input_dim=embedding_dim,
            output_dim=phase_act_action_space.orientation_dim,
            blocks=[],
        ).to(device)

        action_heads[GRIPPER_ACTION_KEY] = MoEHead(
            base_expert_config={
                "_target_": "versatil.models.decoding.action_heads.head.ActionHead",
                "input_dim": embedding_dim,
                "output_dim": 1,
                "blocks": [],
            },
            num_experts=phase_act_action_space.number_of_phases,
            output_dim=1,
            gating_input_dim=None,
            device=str(device),
        )

        decoder = PhaseACT(
            input_keys=["rgb_image"],
            action_space=phase_act_action_space,
            action_heads=action_heads,
            observation_space=phase_act_observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=str(device),
            embedding_dimension=embedding_dim,
            number_of_heads=8,
            feedforward_dimension=2048,
            number_of_encoder_layers=4,
            number_of_decoder_layers=6,
            phase_routing_key=PHASE_LABEL_KEY,
        ).to(device)

        loss = PhaseActionLoss(
            action_keys=[POSITION_ACTION_KEY, ORIENTATION_ACTION_KEY, GRIPPER_ACTION_KEY],
            mse_weight=1.0,
            gripper_bce_weight=1.0,
            phase_ce_weight=1.0,
            use_vae=False,
            kl_weight=0.0,
        )

        algorithm = BehavioralCloning()

        policy = Policy(
            encoding_pipeline=encoding_pipeline_phase,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=phase_act_observation_space,
            action_space=phase_act_action_space,
            prediction_horizon=prediction_horizon,
            loss=loss,
            device=str(device),
            validate_loss_keys=True,
        )

        policy.normalizer = DummyNormalizer()
        policy.to(device)
        return policy

    @pytest.fixture
    def phase_act_training_batch(self, mock_observations_phase, mock_actions_phase):
        """Complete training batch for PhaseACT."""
        return {
            OBSERVATION_KEY: mock_observations_phase,
            ACTION_KEY: mock_actions_phase,
        }

    def test_phase_act_forward_pass(self, phase_act_policy, phase_act_training_batch, device):
        """Test PhaseACT policy forward pass works."""
        phase_act_policy.train()

        output = phase_act_policy.forward(phase_act_training_batch)

        assert POSITION_ACTION_KEY in output
        assert ORIENTATION_ACTION_KEY in output
        assert GRIPPER_ACTION_KEY in output
        assert PHASE_LABEL_KEY in output
        assert f"{POSITION_ACTION_KEY}_routing_weights" in output
        assert f"{GRIPPER_ACTION_KEY}_routing_weights" in output

        assert output[POSITION_ACTION_KEY].device.type == device.type
        assert output[PHASE_LABEL_KEY].device.type == device.type

    def test_phase_act_loss_computation(self, phase_act_policy, phase_act_training_batch):
        """Test PhaseACT policy loss computation."""
        phase_act_policy.train()

        loss_output = phase_act_policy.compute_loss(phase_act_training_batch)

        assert loss_output.total_loss is not None
        assert loss_output.total_loss.requires_grad
        assert loss_output.total_loss.item() >= 0

    def test_phase_act_backward_pass(self, phase_act_policy, phase_act_training_batch):
        """Test PhaseACT policy backward pass works."""
        phase_act_policy.train()

        loss_output = phase_act_policy.compute_loss(phase_act_training_batch)

        loss_output.total_loss.backward()

        for name, param in phase_act_policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_phase_act_inference(self, phase_act_policy, mock_observations_phase, device):
        """Test PhaseACT policy inference mode."""
        phase_act_policy.eval()

        with torch.no_grad():
            actions = phase_act_policy.predict_action(mock_observations_phase)

        assert isinstance(actions, dict)
        assert POSITION_ACTION_KEY in actions
        assert ORIENTATION_ACTION_KEY in actions
        assert GRIPPER_ACTION_KEY in actions
        assert PHASE_LABEL_KEY in actions

        assert actions[POSITION_ACTION_KEY].device.type == device.type

    def test_phase_act_moe_routing(self, phase_act_policy, phase_act_training_batch):
        """Test that MoE routing weights are produced correctly."""
        phase_act_policy.train()

        output = phase_act_policy.forward(phase_act_training_batch)

        position_routing = output[f"{POSITION_ACTION_KEY}_routing_weights"]
        gripper_routing = output[f"{GRIPPER_ACTION_KEY}_routing_weights"]

        batch_size = phase_act_training_batch[ACTION_KEY][POSITION_ACTION_KEY].shape[0]
        pred_horizon = phase_act_training_batch[ACTION_KEY][POSITION_ACTION_KEY].shape[1]
        num_experts = 3

        assert position_routing.shape == (batch_size, pred_horizon, num_experts)
        assert gripper_routing.shape == (batch_size, pred_horizon, num_experts)

        assert torch.allclose(
            position_routing.sum(dim=-1),
            torch.ones(batch_size, pred_horizon, device=position_routing.device),
            atol=1e-5
        )


@pytest.mark.integration
@pytest.mark.requires_gpu
class TestACTPolicyWithVAEEndToEnd:
    """End-to-end test for ACT policy with VAE latent encoder at algorithm level."""

    @pytest.fixture
    def vae_observation_space(self):
        return ObservationSpace(
            camera_keys=[Cameras.LEFT.value],
            use_proprioceptive_data=False,
            use_language=False,
            use_gripper_state=False,
            gripper_type=GripperType.BINARY.value,
        )

    @pytest.fixture
    def vae_action_space(self):
        return ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            orientation_repr=OrientationRepresentation.QUATERNION.value,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            gripper_dim=1,
            predict_in_camera_frame=False,
            deltas_as_actions=False,
            denoise_actions=False,
            task_has_phases=False,
        )

    @pytest.fixture
    def resnet18_encoder_vae(self, device):
        """Real ResNet18 CNN encoder for RGB images."""

        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.NONE.value,
            use_group_norm=True,
            pretrained=False,
            frozen=False,
            image_height=224,
            image_width=224,
        )
        return encoder.to(device)

    @pytest.fixture
    def encoding_pipeline_vae(self, resnet18_encoder_vae, device):
        """Real encoding pipeline for VAE tests."""

        encoders = nn.ModuleDict({"rgb": resnet18_encoder_vae})
        encoder_outputs = {"rgb": resnet18_encoder_vae.get_output_specification()}
        fusion_stages = nn.ModuleList([])

        feature_keys_to_dims = {
            "rgb_image": (512, 7, 7),
        }

        pipeline = EncodingPipeline.__new__(EncodingPipeline)
        nn.Module.__init__(pipeline)
        pipeline.encoders = encoders
        pipeline.conditional_encoders = nn.ModuleDict()
        pipeline.fusion_stages = fusion_stages
        pipeline.encoders_to_outputs = encoder_outputs
        pipeline._feature_keys_to_dims = feature_keys_to_dims
        pipeline._consumed_features = set()  # Initialize consumed features tracker

        def _flatten_observation_dict(self, observation):
            return observation
        pipeline._flatten_observation_dict = _flatten_observation_dict.__get__(
            pipeline, EncodingPipeline
        )

        return pipeline.to(device)

    @pytest.fixture
    def vae_latent_encoder(self, device):
        """Real VAETransformerEncoder for testing."""

        encoder = VAETransformerEncoder(
            embedding_dimension=256,
            latent_dimension=32,
            prediction_horizon=10,
            device=str(device),
            number_of_heads=8,
            feedforward_dimension=512,
            number_of_encoder_layers=4,
            use_proprioceptive=False,
        )
        return encoder.to(device)

    @pytest.fixture
    def mock_observations_vae(self, device):
        """Mock RGB observations matching ResNet18 input requirements."""
        batch_size = 2
        return {
            Cameras.LEFT.value: torch.randn(
                batch_size, 3, 224, 224, device=device
            )
        }

    @pytest.fixture
    def mock_actions_vae(self, vae_action_space, device):
        """Mock action dictionary."""
        batch_size = 2
        prediction_horizon = 10
        actions = {
            POSITION_ACTION_KEY: torch.randn(
                batch_size, prediction_horizon, vae_action_space.position_dim, device=device
            ),
            ORIENTATION_ACTION_KEY: torch.randn(
                batch_size, prediction_horizon, vae_action_space.orientation_dim, device=device
            ),
            GRIPPER_ACTION_KEY: torch.randint(
                0, 2, (batch_size, prediction_horizon, 1), device=device
            ).float(),
        }
        return actions

    @pytest.fixture
    def vae_act_policy(
        self,
        encoding_pipeline_vae,
        vae_observation_space,
        vae_action_space,
        vae_latent_encoder,
        device
    ):
        """ACT policy with VAE encoder at algorithm level."""

        embedding_dim = 256
        prediction_horizon = 10
        observation_horizon = 1

        # Create action heads
        action_heads = {}
        if vae_action_space.has_position:
            action_heads[POSITION_ACTION_KEY] = ActionHead(
                input_dim=embedding_dim,
                output_dim=vae_action_space.position_dim,
                blocks=[],
            ).to(device)
        if vae_action_space.has_orientation:
            action_heads[ORIENTATION_ACTION_KEY] = ActionHead(
                input_dim=embedding_dim,
                output_dim=vae_action_space.orientation_dim,
                blocks=[],
            ).to(device)
        if vae_action_space.has_gripper:
            action_heads[GRIPPER_ACTION_KEY] = ActionHead(
                input_dim=embedding_dim,
                output_dim=vae_action_space.gripper_dim,
                blocks=[],
            ).to(device)

        # Create ACT decoder (without VAE - VAE is now in algorithm)
        # Note: LATENT_KEY is not in input_keys because it's provided by algorithm at runtime
        decoder = ACT(
            input_keys=["rgb_image"],
            action_space=vae_action_space,
            action_heads=action_heads,
            observation_space=vae_observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=str(device),
            embedding_dimension=embedding_dim,
            number_of_heads=8,
            feedforward_dimension=2048,
            number_of_encoder_layers=4,
            number_of_decoder_layers=6,
        ).to(device)

        # Create BC algorithm with VAE encoder
        algorithm = BehavioralCloning(latent_encoder=vae_latent_encoder)

        # Create loss that expects VAE outputs
        loss = ActionReconstructionLoss(
            action_keys=[POSITION_ACTION_KEY, ORIENTATION_ACTION_KEY, GRIPPER_ACTION_KEY],
            mse_weight=1.0,
            gripper_bce_weight=1.0,
            use_vae=True,
            kl_weight=0.1,
        )

        # Create policy
        policy = Policy(
            encoding_pipeline=encoding_pipeline_vae,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=vae_observation_space,
            action_space=vae_action_space,
            prediction_horizon=prediction_horizon,
            loss=loss,
            device=str(device),
            validate_loss_keys=True,
        )

        policy.normalizer = DummyNormalizer()
        policy.to(device)
        return policy

    @pytest.fixture
    def vae_training_batch(self, mock_observations_vae, mock_actions_vae):
        """Complete training batch."""
        return {
            OBSERVATION_KEY: mock_observations_vae,
            ACTION_KEY: mock_actions_vae,
        }

    def test_vae_policy_forward_pass(self, vae_act_policy, vae_training_batch, device):
        """Test that VAE encoder produces latent features consumed by decoder."""

        vae_act_policy.train()

        output = vae_act_policy.forward(vae_training_batch)

        # Check action predictions
        assert POSITION_ACTION_KEY in output
        assert ORIENTATION_ACTION_KEY in output
        assert GRIPPER_ACTION_KEY in output

        # Check VAE latent variables are in output (for loss computation)
        assert MU_KEY in output
        assert LOGVAR_KEY in output

        # Check device consistency
        assert output[POSITION_ACTION_KEY].device.type == device.type
        assert output[MU_KEY].device.type == device.type
        assert output[LOGVAR_KEY].device.type == device.type

    def test_vae_policy_loss_computation(self, vae_act_policy, vae_training_batch):
        """Test that loss includes KL divergence term from VAE."""
        vae_act_policy.train()

        loss_output = vae_act_policy.compute_loss(vae_training_batch)

        # Check total loss
        assert loss_output.total_loss is not None
        assert loss_output.total_loss.requires_grad
        assert loss_output.total_loss.item() >= 0

        # Check that KL loss is computed (key is 'kl/kl_divergence')
        assert "kl/kl_divergence" in loss_output.component_losses
        assert loss_output.component_losses["kl/kl_divergence"].item() >= 0

    def test_vae_policy_backward_pass(self, vae_act_policy, vae_training_batch):
        """Test that gradients flow through VAE encoder and decoder."""
        vae_act_policy.train()

        loss_output = vae_act_policy.compute_loss(vae_training_batch)

        loss_output.total_loss.backward()

        # Check gradients in VAE encoder
        for name, param in vae_act_policy.algorithm.latent_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient in VAE encoder for {name}"

        # Check gradients in decoder
        for name, param in vae_act_policy.decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient in decoder for {name}"

    def test_vae_policy_inference_samples_from_prior(self, vae_act_policy, mock_observations_vae, device):
        """Test that inference samples from VAE prior (not posterior)."""

        vae_act_policy.eval()

        with torch.no_grad():
            actions = vae_act_policy.predict_action(mock_observations_vae)

        # Check action predictions
        assert isinstance(actions, dict)
        assert POSITION_ACTION_KEY in actions
        assert ORIENTATION_ACTION_KEY in actions
        assert GRIPPER_ACTION_KEY in actions

        # Should NOT have mu/logvar during inference (sampling from prior)
        assert MU_KEY not in actions
        assert LOGVAR_KEY not in actions

        assert actions[POSITION_ACTION_KEY].device.type == device.type

    def test_vae_latent_shapes(self, vae_act_policy, vae_training_batch):
        """Test that VAE latent outputs have correct shapes."""

        vae_act_policy.train()

        output = vae_act_policy.forward(vae_training_batch)

        batch_size = vae_training_batch[ACTION_KEY][POSITION_ACTION_KEY].shape[0]

        # Note: LATENT_KEY is consumed by decoder and not in output
        # Only MU and LOGVAR are preserved for loss computation

        # VAE latent space shape (z dimension)
        assert output[MU_KEY].shape == (batch_size, 32)
        assert output[LOGVAR_KEY].shape == (batch_size, 32)

    def test_vae_deterministic_with_seed(self, vae_act_policy, mock_observations_vae):
        """Test that VAE sampling is deterministic with same seed."""
        vae_act_policy.eval()

        # Sample twice with same seed
        torch.manual_seed(42)
        with torch.no_grad():
            actions1 = vae_act_policy.predict_action(mock_observations_vae)

        torch.manual_seed(42)
        with torch.no_grad():
            actions2 = vae_act_policy.predict_action(mock_observations_vae)

        # Actions should be identical
        assert torch.allclose(actions1[POSITION_ACTION_KEY], actions2[POSITION_ACTION_KEY])
        assert torch.allclose(actions1[ORIENTATION_ACTION_KEY], actions2[ORIENTATION_ACTION_KEY])
        assert torch.allclose(actions1[GRIPPER_ACTION_KEY], actions2[GRIPPER_ACTION_KEY])


@pytest.mark.slow
@pytest.mark.integration
class TestHyperparameterTuning:
    """Test hyperparameter tuning functionality."""

    @pytest.fixture
    def mock_dataloaders_tuning(self, synthetic_training_batch):
        """Mock dataloaders with batch_size attribute for tuning."""
        train_dataset = DummyDataset(length=20)
        val_dataset = DummyDataset(length=10)

        mock_train_loader = DummyDataLoader(
            dataset=train_dataset,
            batches=[synthetic_training_batch] * 5
        )
        mock_train_loader.batch_size = 4  # Add batch_size attribute

        mock_val_loader = DummyDataLoader(
            dataset=val_dataset,
            batches=[synthetic_training_batch] * 3
        )
        mock_val_loader.batch_size = 4

        return mock_train_loader, mock_val_loader

    def test_tuning_disabled_by_default(self, mock_main_config):
        """Test that tuning is disabled by default."""
        assert mock_main_config.training.tune_lr is False
        assert mock_main_config.training.tune_batch_size is False

    def test_tuning_skipped_when_disabled(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders_tuning,
    ):
        """Test that tuning is skipped when both flags are False."""
        mock_train_loader, mock_val_loader = mock_dataloaders_tuning
        normalizer = LinearNormalizer()

        mock_main_config.training.tune_lr = False
        mock_main_config.training.tune_batch_size = False
        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: TestEndToEndTraining._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                # Mock Tuner to verify it's not called
                with patch("versatil.workspace.Tuner") as mock_tuner:
                    workspace.run()

                    # Tuner should not be instantiated when tuning is disabled
                    mock_tuner.assert_not_called()

    def test_tuning_skipped_in_distributed_mode(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders_tuning,
    ):
        """Test that tuning is skipped in distributed training mode."""
        mock_train_loader, mock_val_loader = mock_dataloaders_tuning
        normalizer = LinearNormalizer()

        mock_main_config.training.tune_lr = True
        mock_main_config.training.tune_batch_size = True
        mock_main_config.experiment.distributed = True
        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: TestEndToEndTraining._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                # Mock Tuner to verify tuning is skipped
                with patch("versatil.workspace.Tuner") as mock_tuner:
                    workspace.run()

                    # Tuner should not be instantiated in distributed mode
                    mock_tuner.assert_not_called()

    def test_batch_size_tuning_enabled(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders_tuning,
    ):
        """Test that batch size tuning is called when enabled."""
        mock_train_loader, mock_val_loader = mock_dataloaders_tuning
        normalizer = LinearNormalizer()

        mock_main_config.training.tune_batch_size = True
        mock_main_config.training.tune_lr = False
        mock_main_config.experiment.distributed = False
        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: TestEndToEndTraining._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                # Mock Tuner and scale_batch_size
                with patch("versatil.workspace.Tuner") as mock_tuner_class:
                    mock_tuner = MagicMock()
                    mock_tuner_class.return_value = mock_tuner

                    workspace.run()

                    # Verify Tuner was created
                    mock_tuner_class.assert_called_once_with(workspace.trainer)

                    # Verify scale_batch_size was called
                    mock_tuner.scale_batch_size.assert_called_once()
                    call_kwargs = mock_tuner.scale_batch_size.call_args[1]
                    assert call_kwargs["mode"] == "power"
                    assert call_kwargs["steps_per_trial"] == 3
                    assert call_kwargs["max_trials"] == 25

    def test_learning_rate_tuning_enabled(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders_tuning,
    ):
        """Test that learning rate tuning is called when enabled."""
        mock_train_loader, mock_val_loader = mock_dataloaders_tuning
        normalizer = LinearNormalizer()

        mock_main_config.training.tune_lr = True
        mock_main_config.training.tune_batch_size = False
        mock_main_config.experiment.distributed = False
        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: TestEndToEndTraining._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                # Mock Tuner and lr_find
                with patch("versatil.workspace.Tuner") as mock_tuner_class:
                    mock_tuner = MagicMock()
                    mock_lr_finder = MagicMock()
                    mock_lr_finder.suggestion.return_value = 1e-3
                    mock_tuner.lr_find.return_value = mock_lr_finder
                    mock_tuner_class.return_value = mock_tuner

                    workspace.run()

                    # Verify Tuner was created
                    mock_tuner_class.assert_called_once_with(workspace.trainer)

                    # Verify lr_find was called
                    mock_tuner.lr_find.assert_called_once()
                    call_kwargs = mock_tuner.lr_find.call_args[1]
                    assert call_kwargs["min_lr"] == 1e-8
                    assert call_kwargs["max_lr"] == 1.0
                    assert call_kwargs["num_training"] == 100

    def test_both_tuning_enabled(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders_tuning,
    ):
        """Test that both batch size and LR tuning can run together."""
        mock_train_loader, mock_val_loader = mock_dataloaders_tuning
        normalizer = LinearNormalizer()

        mock_main_config.training.tune_lr = True
        mock_main_config.training.tune_batch_size = True
        mock_main_config.experiment.distributed = False
        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: TestEndToEndTraining._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                # Mock Tuner and both tuning methods
                with patch("versatil.workspace.Tuner") as mock_tuner_class:
                    mock_tuner = MagicMock()
                    mock_lr_finder = MagicMock()
                    mock_lr_finder.suggestion.return_value = 5e-4
                    mock_tuner.lr_find.return_value = mock_lr_finder
                    mock_tuner_class.return_value = mock_tuner

                    workspace.run()

                    # Verify both tuning methods were called
                    mock_tuner.scale_batch_size.assert_called_once()
                    mock_tuner.lr_find.assert_called_once()

    def test_tuning_updates_config(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders_tuning,
    ):
        """Test that tuned values are saved to config."""
        mock_train_loader, mock_val_loader = mock_dataloaders_tuning
        normalizer = LinearNormalizer()

        mock_main_config.training.tune_lr = True
        mock_main_config.training.tune_batch_size = True
        mock_main_config.experiment.distributed = False
        mock_main_config.training.num_epochs = 1

        workspace = Workspace(mock_main_config)

        tuned_lr = 3e-4
        tuned_batch_size = 16

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: TestEndToEndTraining._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                # Mock Tuner
                with patch("versatil.workspace.Tuner") as mock_tuner_class:
                    mock_tuner = MagicMock()

                    # Mock LR finder
                    mock_lr_finder = MagicMock()
                    mock_lr_finder.suggestion.return_value = tuned_lr
                    mock_tuner.lr_find.return_value = mock_lr_finder

                    # Mock batch size tuning - update dataloader batch_size
                    def update_batch_size(*args, **kwargs):
                        workspace.train_loader.batch_size = tuned_batch_size
                    mock_tuner.scale_batch_size.side_effect = update_batch_size

                    mock_tuner_class.return_value = mock_tuner

                    workspace.run()

                    # Verify config was updated
                    assert workspace.config.training.optimizer.lr == tuned_lr

    def test_tuner_integration_with_callbacks(
        self,
        mock_main_config,
        simple_policy,
        mock_dataloaders_tuning,
    ):
        """Test that tuning works alongside other callbacks."""
        mock_train_loader, mock_val_loader = mock_dataloaders_tuning
        normalizer = LinearNormalizer()

        mock_main_config.training.tune_lr = True
        mock_main_config.training.use_ema = True
        mock_main_config.training.swa_lrs = 0.05
        mock_main_config.training.swa_epoch_start = 0.8
        mock_main_config.experiment.distributed = False
        mock_main_config.training.num_epochs = 10  # Need at least 10 epochs for SWA to start at epoch 8

        workspace = Workspace(mock_main_config)

        with patch.object(workspace, "_setup_data") as mock_setup_data:
            mock_setup_data.side_effect = lambda: TestEndToEndTraining._set_data_attrs(
                workspace, mock_train_loader, mock_val_loader, normalizer
            )

            with patch("versatil.workspace.instantiate", return_value=simple_policy):
                # Mock Tuner
                with patch("versatil.workspace.Tuner") as mock_tuner_class:
                    mock_tuner = MagicMock()
                    mock_lr_finder = MagicMock()
                    mock_lr_finder.suggestion.return_value = 1e-3
                    mock_tuner.lr_find.return_value = mock_lr_finder
                    mock_tuner_class.return_value = mock_tuner

                    workspace.run()

                    # Verify tuning happened
                    mock_tuner.lr_find.assert_called_once()

                    # Verify callbacks were still created

                    ema_callbacks = [cb for cb in workspace.trainer.callbacks if isinstance(cb, EMACallback)]
                    swa_callbacks = [cb for cb in workspace.trainer.callbacks if isinstance(cb, StochasticWeightAveraging)]

                    assert len(ema_callbacks) == 1
                    assert len(swa_callbacks) == 1
