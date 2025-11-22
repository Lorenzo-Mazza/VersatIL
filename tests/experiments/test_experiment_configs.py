"""Comprehensive tests for experiment YAML configurations.

This test suite validates that all experiment configurations:
1. Load correctly via Hydra
2. Instantiate all components successfully
3. Create functional policies
4. Can perform forward/backward passes
5. Have consistent encoder-decoder feature mappings
6. Have valid loss configurations
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from refactoring.data.constants import (
    ACTION_KEY,
    Cameras,
    POSITION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    OBSERVATION_KEY,
    IS_PAD_ACTION_KEY,
)
from refactoring.data.normalization.normalizer import LinearNormalizer
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.models.policy import Policy

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def get_all_experiment_configs():
    """Discover all top-level experiment YAML files."""
    experiment_files = []
    for yaml_file in EXPERIMENTS_DIR.glob("*.yaml"):
        if yaml_file.is_file():
            experiment_files.append(yaml_file.stem)
    return sorted(experiment_files)


def create_dummy_normalizer(config: DictConfig) -> LinearNormalizer:
    """Create a dummy normalizer with identity normalization for all keys.

    Args:
        config: Experiment configuration

    Returns:
        LinearNormalizer with identity parameters for all observation and action keys
    """
    from refactoring.data.normalization.normalizer import SingleFieldLinearNormalizer

    obs_space = instantiate(config.task.observation_space)
    action_space = instantiate(config.task.action_space)

    normalizer = LinearNormalizer()

    if hasattr(obs_space, 'camera_keys') and obs_space.camera_keys:
        for camera_key in obs_space.camera_keys:
            normalizer[camera_key] = SingleFieldLinearNormalizer.create_identity()

    if hasattr(obs_space, 'use_depth') and obs_space.use_depth:
        normalizer[Cameras.DEPTH.value] = SingleFieldLinearNormalizer.create_identity()

    if hasattr(obs_space, 'use_proprio_base_frame') and obs_space.use_proprio_base_frame:
        normalizer['proprio_robot_frame'] = SingleFieldLinearNormalizer.create_identity()

    if hasattr(obs_space, 'use_proprio_camera_frame') and obs_space.use_proprio_camera_frame:
        normalizer['proprio_camera_frame'] = SingleFieldLinearNormalizer.create_identity()

    if action_space.has_position:
        normalizer[POSITION_ACTION_KEY] = SingleFieldLinearNormalizer.create_identity()

    if action_space.has_gripper and action_space.gripper_type != 'binary':
        normalizer[GRIPPER_ACTION_KEY] = SingleFieldLinearNormalizer.create_identity()

    return normalizer


def create_dummy_tokenizer(config: DictConfig, device: str = 'cpu') -> Tokenizer | None:
    """Create a dummy tokenizer if the config requires action tokenization.

    Args:
        config: Experiment configuration
        device: Device for tokenizer

    Returns:
        Tokenizer if tokenization is enabled, None otherwise
    """
    if not hasattr(config.task.dataloader, 'tokenization'):
        return None

    tokenization_config = config.task.dataloader.tokenization
    if not tokenization_config.tokenize_actions:
        return None

    action_space = instantiate(config.task.action_space)
    prediction_horizon = config.task.prediction_horizon
    total_action_dim = action_space.get_total_action_dim()

    tokenizer_obj = Tokenizer(device=torch.device(device))
    dummy_action_chunks = np.random.randn(100, prediction_horizon, total_action_dim) * 2 - 1
    tokenizer_obj.fit_action_tokenizer(
        action_chunks=dummy_action_chunks,
        use_pretrained_weights=tokenization_config.use_pretrained_action_tokenizer,
    )
    sample_actions = torch.randn(1, prediction_horizon, total_action_dim, device=device)
    tokenizer_obj.tokenize({ACTION_KEY: sample_actions})

    # Also fit proprio tokenizer if observation space uses proprioception
    observation_space = instantiate(config.task.observation_space)
    if observation_space.use_proprio_robot_frame or observation_space.use_proprio_camera_frame:
        # Determine proprio dim based on observation space
        proprio_dim = 0
        if action_space.has_position:
            proprio_dim += action_space.position_dim
        if action_space.has_orientation:
            proprio_dim += action_space.get_orientation_dim()
        if action_space.has_gripper:
            proprio_dim += 1

        dummy_proprio_chunks = np.random.randn(100, prediction_horizon, proprio_dim) * 2 - 1
        if observation_space.use_proprio_robot_frame:
            tokenizer_obj.fit_proprio_tokenizer(
                proprio_chunks=dummy_proprio_chunks,
                key='proprio_robot_frame',
                use_pretrained_weights=False,
            )
        if observation_space.use_proprio_camera_frame:
            tokenizer_obj.fit_proprio_tokenizer(
                proprio_chunks=dummy_proprio_chunks,
                key='proprio_camera_frame',
                use_pretrained_weights=False,
            )

    return tokenizer_obj


def create_dummy_batch(config: DictConfig, batch_size: int = 2) -> Dict[str, torch.Tensor]:
    """Create a dummy batch of data matching the config's observation and action spaces.

    Args:
        config: Experiment configuration
        batch_size: Batch size for dummy data

    Returns:
        Dictionary containing observations and actions
    """
    observation_horizon = config.task.observation_horizon
    prediction_horizon = config.task.prediction_horizon

    obs_space = instantiate(config.task.observation_space)
    action_space = instantiate(config.task.action_space)

    image_h = config.task.dataloader.image_height
    image_w = config.task.dataloader.image_width

    batch = {OBSERVATION_KEY: {}, "action": {}}

    if hasattr(obs_space, 'camera_keys') and obs_space.camera_keys:
        for camera_key in obs_space.camera_keys:
            if camera_key == Cameras.DEPTH.value:
                channels = 1
            else:
                channels = 3
            batch[OBSERVATION_KEY][camera_key] = torch.randn(
                batch_size, observation_horizon, channels, image_h, image_w
            )

    if hasattr(obs_space, 'use_depth') and obs_space.use_depth:
        if Cameras.DEPTH.value not in batch[OBSERVATION_KEY]:
            batch[OBSERVATION_KEY][Cameras.DEPTH.value] = torch.randn(
                batch_size, observation_horizon, 1, image_h, image_w
            )

    if hasattr(obs_space, 'use_proprio_base_frame') and obs_space.use_proprio_base_frame:
        batch[OBSERVATION_KEY]['proprio_robot_frame'] = torch.randn(
            batch_size, observation_horizon, action_space.position_dim
        )

    if hasattr(obs_space, 'use_proprio_camera_frame') and obs_space.use_proprio_camera_frame:
        batch[OBSERVATION_KEY]['proprio_camera_frame'] = torch.randn(
            batch_size, observation_horizon, action_space.position_dim
        )

    if hasattr(obs_space, 'use_gripper_state') and obs_space.use_gripper_state:
        batch[OBSERVATION_KEY]['gripper_state'] = torch.randn(
            batch_size, observation_horizon, 1
        )

    if action_space.has_position:
        batch["action"][POSITION_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.position_dim
        )

    if action_space.has_gripper:
        if action_space.gripper_type == 'binary':
            batch["action"][GRIPPER_ACTION_KEY] = torch.randint(
                0, 2, (batch_size, prediction_horizon, 1), dtype=torch.float32
            )
        else:
            batch["action"][GRIPPER_ACTION_KEY] = torch.randn(
                batch_size, prediction_horizon, 1
            )

    if action_space.task_has_phases:
        num_phases = action_space.number_of_phases
        batch["action"]["phase_label"] = torch.randint(
            0, num_phases, (batch_size, prediction_horizon, 1), dtype=torch.long
        )

    batch[IS_PAD_ACTION_KEY] = torch.zeros(batch_size, prediction_horizon, dtype=torch.bool)

    return batch


@pytest.mark.unit
class TestExperimentConfigLoading:
    """Test that all experiment configs load correctly via Hydra."""

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_config_loads(self, experiment_name):
        """Test that experiment config loads without errors."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)
            assert cfg is not None
            assert 'experiment' in cfg
            assert 'task' in cfg
            assert 'training' in cfg
            assert 'policy' in cfg

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_config_has_required_sections(self, experiment_name):
        """Test that config has all required top-level sections."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            required_sections = [
                'experiment', 'task', 'training', 'policy', 'inference'
            ]
            for section in required_sections:
                assert section in cfg, f"Missing required section: {section}"

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_task_section_structure(self, experiment_name):
        """Test that task section has required fields."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            assert 'observation_space' in cfg.task
            assert 'action_space' in cfg.task
            assert 'dataset_schema' in cfg.task
            assert 'dataloader' in cfg.task
            assert 'observation_horizon' in cfg.task
            assert 'prediction_horizon' in cfg.task

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_policy_section_structure(self, experiment_name):
        """Test that policy section has required fields."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            assert '_target_' in cfg.policy
            assert cfg.policy._target_ == 'refactoring.models.policy.Policy'
            assert 'encoding_pipeline' in cfg.policy
            assert 'decoder' in cfg.policy
            assert 'algorithm' in cfg.policy
            assert 'loss' in cfg.policy


@pytest.mark.unit
class TestComponentInstantiation:
    """Test that all config components can be instantiated."""

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_observation_space_instantiation(self, experiment_name):
        """Test that observation space instantiates correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)
            obs_space = instantiate(cfg.task.observation_space)
            assert obs_space is not None
            assert hasattr(obs_space, 'camera_keys')

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_action_space_instantiation(self, experiment_name):
        """Test that action space instantiates correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)
            action_space = instantiate(cfg.task.action_space)
            assert action_space is not None
            assert hasattr(action_space, 'has_position')
            assert hasattr(action_space, 'has_gripper')

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_encoding_pipeline_instantiation(self, experiment_name):
        """Test that encoding pipeline instantiates correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            pipeline = instantiate(cfg.policy.encoding_pipeline)
            assert pipeline is not None
            assert hasattr(pipeline, 'encoders')
            assert hasattr(pipeline, 'get_features_to_dimensions')

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_decoder_instantiation(self, experiment_name):
        """Test that decoder instantiates correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            obs_space = instantiate(cfg.task.observation_space)
            action_space = instantiate(cfg.task.action_space)

            decoder_cfg = OmegaConf.to_container(cfg.policy.decoder, resolve=True)
            decoder_cfg['observation_space'] = obs_space
            decoder_cfg['action_space'] = action_space
            decoder_cfg['observation_horizon'] = cfg.task.observation_horizon
            decoder_cfg['prediction_horizon'] = cfg.task.prediction_horizon
            decoder_cfg['device'] = 'cpu'

            decoder = instantiate(decoder_cfg)
            assert decoder is not None
            assert hasattr(decoder, 'decoder_input')

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_algorithm_instantiation(self, experiment_name):
        """Test that algorithm instantiates correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            algorithm = instantiate(cfg.policy.algorithm)
            assert algorithm is not None

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_loss_instantiation(self, experiment_name):
        """Test that loss module instantiates correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            loss = instantiate(cfg.policy.loss)
            assert loss is not None
            assert callable(loss)


@pytest.mark.unit
class TestPolicyCreation:
    """Test that complete policies can be created from configs."""

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_policy_instantiation(self, experiment_name):
        """Test that policy instantiates from config."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)
            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)
            assert policy is not None
            assert isinstance(policy, Policy)

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_policy_has_required_components(self, experiment_name):
        """Test that policy has all required components."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)
            assert hasattr(policy, 'encoding_pipeline')
            assert hasattr(policy, 'decoder')
            assert hasattr(policy, 'algorithm')
            assert hasattr(policy, 'loss_module')
            assert hasattr(policy, 'observation_space')
            assert hasattr(policy, 'action_space')

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_encoder_decoder_feature_compatibility(self, experiment_name):
        """Test that decoder's required features are provided by encoder."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)

            available_features = set(policy.encoding_pipeline.get_features_to_dimensions().keys())
            required_features = set(policy.decoder.decoder_input.keys)

            missing_features = required_features - available_features
            assert len(missing_features) == 0, (
                f"Decoder requires features not provided by encoder: {missing_features}\n"
                f"Available: {available_features}\n"
                f"Required: {required_features}"
            )


@pytest.mark.unit
class TestPolicyForwardPass:
    """Test that policies can perform forward passes."""

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_forward_pass_training(self, experiment_name):
        """Test forward pass in training mode."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)
            policy.train()

            batch = create_dummy_batch(cfg, batch_size=2)
            batch = {k: v.to('cpu') if torch.is_tensor(v) else v for k, v in batch.items()}

            try:
                output = policy(batch)
                assert output is not None
                # For tokenized decoders, check for action_tokens; otherwise check for position
                has_tokenized_output = 'action_tokens' in output
                has_continuous_output = POSITION_ACTION_KEY in output or 'position' in output
                assert has_tokenized_output or has_continuous_output, f"Output keys: {output.keys()}"
            except Exception as e:
                pytest.fail(f"Forward pass failed for {experiment_name}: {e}")

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_predict_action(self, experiment_name):
        """Test predict_action method (inference mode)."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)
            policy.eval()

            batch = create_dummy_batch(cfg, batch_size=1)

            with torch.no_grad():
                try:
                    actions = policy.predict_action(batch[OBSERVATION_KEY])
                    assert actions is not None
                    assert isinstance(actions, dict)
                except Exception as e:
                    pytest.fail(f"Predict action failed for {experiment_name}: {e}")

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_loss_computation(self, experiment_name):
        """Test that loss can be computed."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)
            policy.train()

            batch = create_dummy_batch(cfg, batch_size=2)

            try:
                loss_output = policy.compute_loss(batch)

                assert loss_output is not None
                assert hasattr(loss_output, 'total_loss')
                assert torch.is_tensor(loss_output.total_loss)
                assert loss_output.total_loss.requires_grad
            except Exception as e:
                pytest.fail(f"Loss computation failed for {experiment_name}: {e}")


@pytest.mark.unit
class TestGradientFlow:
    """Test that gradients flow correctly through the policy."""

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_backward_pass(self, experiment_name):
        """Test that backward pass works without errors."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)
            policy.train()

            batch = create_dummy_batch(cfg, batch_size=2)

            try:
                loss_output = policy.compute_loss(batch)

                loss_output.total_loss.backward()

                has_gradients = False
                for param in policy.parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_gradients = True
                        break

                assert has_gradients, "No gradients were computed during backward pass"
            except Exception as e:
                pytest.fail(f"Backward pass failed for {experiment_name}: {e}")

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_optimizer_step(self, experiment_name):
        """Test that optimizer can perform a step."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg.experiment.device = 'cpu'

            # Override language encoder to use smaller model for tests
            if 'encoding_pipeline' in cfg.policy and 'encoders' in cfg.policy.encoding_pipeline:
                if 'tokenizer' in cfg.policy.encoding_pipeline.encoders:
                    cfg.policy.encoding_pipeline.encoders.tokenizer.language_model_name = 'bert-base-uncased'

            policy: Policy = instantiate(cfg.policy)
            normalizer = create_dummy_normalizer(cfg)
            policy.set_normalizer(normalizer)
            tokenizer = create_dummy_tokenizer(cfg, device='cpu')
            if tokenizer is not None:
                policy.set_tokenizer(tokenizer)
            policy.train()

            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

            batch = create_dummy_batch(cfg, batch_size=2)

            try:
                optimizer.zero_grad()
                loss_output = policy.compute_loss(batch)
                loss_output.total_loss.backward()
                optimizer.step()
            except Exception as e:
                pytest.fail(f"Optimizer step failed for {experiment_name}: {e}")


@pytest.mark.unit
class TestConfigInterpolation:
    """Test that Hydra interpolations resolve correctly."""

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_no_unresolved_interpolations(self, experiment_name):
        """Test that all interpolations in config resolve."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            cfg_container = OmegaConf.to_container(cfg, resolve=True)
            assert cfg_container is not None

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_device_interpolation(self, experiment_name):
        """Test that device interpolations work correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            assert '${experiment.device}' not in str(cfg.policy.device)
            assert cfg.policy.device in ['cpu', 'cuda', 'mps']

    @pytest.mark.parametrize("experiment_name", get_all_experiment_configs())
    def test_horizon_interpolations(self, experiment_name):
        """Test that horizon interpolations resolve correctly."""
        with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
            cfg = compose(config_name=experiment_name)

            assert isinstance(cfg.task.observation_horizon, int)
            assert isinstance(cfg.task.prediction_horizon, int)
            assert cfg.task.observation_horizon > 0
            assert cfg.task.prediction_horizon > 0
