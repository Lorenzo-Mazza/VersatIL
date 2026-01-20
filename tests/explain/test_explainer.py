"""Tests for explainer module - gradient-based explanations with real components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hydra.utils import instantiate

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import Cameras, POSITION_ACTION_KEY
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from versatil.models.decoding.decoders.factory.act import ACT
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.explain import (
    PolicyExplainerWrapper,
    compute_gradcam_for_policy,
    compute_saliency_maps,
    create_target_layers_getter_from_policy,
    show_cam_on_image,
)
from versatil.models.policy import Policy
from versatil.metrics.components import RegressionLoss


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def action_space():
    """Create action space for testing."""
    return ActionSpace(
        has_position=True,
        position_dim=3,
        has_orientation=False,
        has_gripper=False,
    )




@pytest.mark.unit
@pytest.mark.parametrize("encoder_type,encoder_params", [
    ("cnn", {"backbone": "timm/resnet18.a1_in1k", "pretrained": False}),
    ("depth_cnn", {"backbone": "timm/resnet18.a1_in1k", "pretrained": False}),
    ("dformer", {"variant": "S", "pretrained": False}),
    ("light_geometric", {"pretrained": False}),
])
def test_policy_explainer_with_vision_encoder_types(encoder_type, encoder_params, action_space, device):
    """Test explainer with different vision encoder types."""
    if encoder_type == "cnn":
        encoder_config = {
            "_target_": "versatil.models.encoding.encoders.rgb.cnn.CNNEncoder",
            "input_keys": Cameras.LEFT.value,
            "pooling_method": "none",
            **encoder_params
        }
        obs_space = ObservationSpace(camera_keys=[Cameras.LEFT.value])
        obs = {Cameras.LEFT.value: torch.randn(1, 3, 224, 224, device=device)}
        encoder_dim = (512, 7, 7)
        feature_key = "cnn_encoder_image"
    elif encoder_type == "depth_cnn":
        encoder_config = {
            "_target_": "versatil.models.encoding.encoders.depth.cnn.DepthCNNEncoder",
            "input_keys": Cameras.DEPTH.value,
            "pooling_method": "none",
            **encoder_params
        }
        obs_space = ObservationSpace(camera_keys=[Cameras.DEPTH.value])
        obs = {Cameras.DEPTH.value: torch.randn(1, 1, 224, 224, device=device)}
        encoder_dim = (512, 7, 7)
        feature_key = "depth_cnn_encoder_depth"
    elif encoder_type == "dformer":
        encoder_config = {
            "_target_": "versatil.models.encoding.encoders.depth.dformerv2.DFormerEncoder",
            "input_keys": [Cameras.LEFT.value, Cameras.DEPTH.value],
            "pooling_method": "none",
            **encoder_params
        }
        obs_space = ObservationSpace(camera_keys=[Cameras.LEFT.value, Cameras.DEPTH.value])
        obs = {
            Cameras.LEFT.value: torch.randn(1, 3, 224, 224, device=device),
            Cameras.DEPTH.value: torch.randn(1, 1, 224, 224, device=device),
        }
        encoder_dim = (256, 56, 56)
        feature_key = "dformer_encoder_rgbd"
    elif encoder_type == "light_geometric":
        encoder_config = {
            "_target_": "versatil.models.encoding.encoders.depth.light_geometric.LightGeometricEncoder",
            "input_keys": [Cameras.LEFT.value, Cameras.DEPTH.value],
            "pooling_method": "none",
            **encoder_params
        }
        obs_space = ObservationSpace(camera_keys=[Cameras.LEFT.value, Cameras.DEPTH.value])
        obs = {
            Cameras.LEFT.value: torch.randn(1, 3, 224, 224, device=device),
            Cameras.DEPTH.value: torch.randn(1, 1, 224, 224, device=device),
        }
        encoder_dim = (256, 56, 56)
        feature_key = "light_geometric_encoder_rgbd"

    # Instantiate encoder from config before passing to pipeline
    encoder = instantiate(encoder_config)
    encoding_pipeline = EncodingPipeline(encoders={f"{encoder_type}_encoder": encoder})

    embedding_dim = 256
    action_heads = {
        POSITION_ACTION_KEY: ActionHead(
            input_dim=embedding_dim,
            output_dim=3,
            blocks=[],
        )
    }

    decoder = ACT(
        input_keys=[feature_key],
        action_space=action_space,
        action_heads=action_heads,
        observation_space=obs_space,
        observation_horizon=1,
        prediction_horizon=10,
        device=str(device),
        embedding_dimension=embedding_dim,
        number_of_heads=4,
        feedforward_dimension=512,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        dropout_rate=0.1,
    )

    algorithm = BehavioralCloning()
    loss = RegressionLoss(action_keys=[POSITION_ACTION_KEY])

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=obs_space,
        action_space=action_space,
        prediction_horizon=10,
        loss=loss,
        device=str(device),
        validate_loss_keys=True,
    )
    policy.to(device)
    policy.eval()

    for key in obs.keys():
        policy.normalizer.params_dict[key] = torch.nn.ParameterDict({
            'scale': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            'offset': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
            'input_stats': torch.nn.ParameterDict({
                'min': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'max': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
                'mean': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'std': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            })
        })

    heatmaps = compute_gradcam_for_policy(policy, obs)

    assert isinstance(heatmaps, dict)
    assert len(heatmaps) > 0
    for camera_key, heatmap in heatmaps.items():
        assert isinstance(heatmap, torch.Tensor)
        assert heatmap.dim() >= 2


@pytest.mark.unit
def test_explainer_with_mixed_encoders(action_space, device):
    """Test explainer with mix of vision and non-vision encoders."""
    rgb_encoder_config = {
        "_target_": "versatil.models.encoding.encoders.rgb.cnn.CNNEncoder",
        "input_keys": Cameras.LEFT.value,
        "backbone": "timm/resnet18.a1_in1k",
        "pretrained": False,
        "pooling_method": "none",
    }

    proprio_encoder_config = {
        "_target_": "versatil.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder",
        "input_keys": "robot_state",
        "hidden_dims": [128, 128],
        "output_dim": 64,
    }

    # Instantiate encoders from configs
    rgb_encoder = instantiate(rgb_encoder_config)
    proprio_encoder = instantiate(proprio_encoder_config)

    encoding_pipeline = EncodingPipeline(
        encoders={
            "rgb_encoder": rgb_encoder,
            "proprio_encoder": proprio_encoder,
        }
    )

    obs_space = ObservationSpace(
        camera_keys=[Cameras.LEFT.value],
        use_proprioceptive_data=True,
        use_proprio_base_frame=True,
    )

    embedding_dim = 256
    action_heads = {
        POSITION_ACTION_KEY: ActionHead(
            input_dim=embedding_dim,
            output_dim=3,
            blocks=[],
        )
    }

    decoder = ACT(
        input_keys=["rgb_encoder_image", "proprio_encoder_proprio"],
        action_space=action_space,
        action_heads=action_heads,
        observation_space=obs_space,
        observation_horizon=1,
        prediction_horizon=10,
        device=str(device),
        embedding_dimension=embedding_dim,
        number_of_heads=4,
        feedforward_dimension=512,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        dropout_rate=0.1,
    )

    algorithm = BehavioralCloning()
    loss = RegressionLoss(action_keys=[POSITION_ACTION_KEY])

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=obs_space,
        action_space=action_space,
        prediction_horizon=10,
        loss=loss,
        device=str(device),
        validate_loss_keys=True,
    )
    policy.to(device)
    policy.eval()

    obs = {
        Cameras.LEFT.value: torch.randn(1, 3, 224, 224, device=device),
        "robot_state": torch.randn(1, 7, device=device),
    }

    for key in obs.keys():
        policy.normalizer.params_dict[key] = torch.nn.ParameterDict({
            'scale': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            'offset': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
            'input_stats': torch.nn.ParameterDict({
                'min': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'max': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
                'mean': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'std': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            })
        })

    heatmaps = compute_gradcam_for_policy(policy, obs)

    assert isinstance(heatmaps, dict)
    assert Cameras.LEFT.value in heatmaps
    assert "robot_state" not in heatmaps


@pytest.mark.unit
def test_explainer_with_conditional_encoder(action_space, device):
    """Test explainer with conditional (FiLM) encoder."""
    language_encoder_config = {
        "_target_": "versatil.models.encoding.encoders.language.language.LanguageEncoder",
        "input_keys": "language_instruction",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "feature_extraction_method": "cls_token",
        "pretrained": True,
        "frozen": True,
    }

    conditional_encoder_config = {
        "_target_": "versatil.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder",
        "input_keys": Cameras.LEFT.value,
        "condition_key": "language_encoder_language",
        "condition_dim": 384,
        "backbone": "timm/resnet18.a1_in1k",
        "pretrained": False,
        "pooling_method": "none",
    }

    # Instantiate encoders from configs
    language_encoder = instantiate(language_encoder_config)
    conditional_encoder = instantiate(conditional_encoder_config)

    encoding_pipeline = EncodingPipeline(
        encoders={
            "language_encoder": language_encoder,
            "conditional_rgb_encoder": conditional_encoder,
        }
    )

    obs_space = ObservationSpace(
        camera_keys=[Cameras.LEFT.value],
        use_language=True,
    )

    action_heads = {
        POSITION_ACTION_KEY: ActionHead(
            input_dim=256,
            output_dim=3,
            blocks=[],
        )
    }

    decoder = ACT(
        input_keys=["conditional_rgb_encoder_image"],
        action_space=action_space,
        action_heads=action_heads,
        observation_space=obs_space,
        observation_horizon=1,
        prediction_horizon=10,
        device=str(device),
        embedding_dimension=256,
        number_of_heads=4,
        feedforward_dimension=512,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        dropout_rate=0.1,
    )

    algorithm = BehavioralCloning()
    loss = RegressionLoss(action_keys=[POSITION_ACTION_KEY])

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=obs_space,
        action_space=action_space,
        prediction_horizon=10,
        loss=loss,
        device=str(device),
        validate_loss_keys=True,
    )
    policy.to(device)
    policy.eval()

    obs = {
        Cameras.LEFT.value: torch.randn(1, 3, 224, 224, device=device),
        "language_instruction": ["test instruction"],
    }

    for key in obs.keys():
        if key != "language_instruction":
            policy.normalizer.params_dict[key] = torch.nn.ParameterDict({
                'scale': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
                'offset': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'input_stats': torch.nn.ParameterDict({
                    'min': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                    'max': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
                    'mean': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                    'std': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
                })
            })

    heatmaps = compute_gradcam_for_policy(policy, obs)

    assert isinstance(heatmaps, dict)
    assert Cameras.LEFT.value in heatmaps


@pytest.mark.unit
def test_wrapper_forward_with_real_policy(action_space, device):
    """Test PolicyExplainerWrapper with real policy."""
    encoder_config = {
        "_target_": "versatil.models.encoding.encoders.rgb.cnn.CNNEncoder",
        "input_keys": Cameras.LEFT.value,
        "backbone": "timm/resnet18.a1_in1k",
        "pretrained": False,
        "pooling_method": "none",
    }

    # Instantiate encoder from config
    encoder = instantiate(encoder_config)
    encoding_pipeline = EncodingPipeline(encoders={"rgb_encoder": encoder})

    obs_space = ObservationSpace(camera_keys=[Cameras.LEFT.value])

    action_heads = {
        POSITION_ACTION_KEY: ActionHead(
            input_dim=256,
            output_dim=3,
            blocks=[],
        )
    }

    decoder = ACT(
        input_keys=["rgb_encoder_image"],
        action_space=action_space,
        action_heads=action_heads,
        observation_space=obs_space,
        observation_horizon=1,
        prediction_horizon=10,
        device=str(device),
        embedding_dimension=256,
        number_of_heads=4,
        feedforward_dimension=512,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        dropout_rate=0.1,
    )

    algorithm = BehavioralCloning()
    loss = RegressionLoss(action_keys=[POSITION_ACTION_KEY])

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=obs_space,
        action_space=action_space,
        prediction_horizon=10,
        loss=loss,
        device=str(device),
        validate_loss_keys=True,
    )
    policy.to(device)
    policy.eval()

    obs = {Cameras.LEFT.value: torch.randn(1, 3, 224, 224, device=device)}

    for key in obs.keys():
        policy.normalizer.params_dict[key] = torch.nn.ParameterDict({
            'scale': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            'offset': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
            'input_stats': torch.nn.ParameterDict({
                'min': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'max': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
                'mean': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'std': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            })
        })

    wrapper = PolicyExplainerWrapper(policy)

    result = wrapper.forward(obs)

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert POSITION_ACTION_KEY in result[0]


@pytest.mark.unit
def test_target_layers_getter_creation(action_space, device):
    """Test create_target_layers_getter_from_policy with real policy."""
    encoder_config = {
        "_target_": "versatil.models.encoding.encoders.rgb.cnn.CNNEncoder",
        "input_keys": Cameras.LEFT.value,
        "backbone": "timm/resnet18.a1_in1k",
        "pretrained": False,
        "pooling_method": "none",
    }

    # Instantiate encoder from config
    encoder = instantiate(encoder_config)
    encoding_pipeline = EncodingPipeline(encoders={"rgb_encoder": encoder})

    obs_space = ObservationSpace(camera_keys=[Cameras.LEFT.value])

    action_heads = {
        POSITION_ACTION_KEY: ActionHead(
            input_dim=256,
            output_dim=3,
            blocks=[],
        )
    }

    decoder = ACT(
        input_keys=["rgb_encoder_image"],
        action_space=action_space,
        action_heads=action_heads,
        observation_space=obs_space,
        observation_horizon=1,
        prediction_horizon=10,
        device=str(device),
        embedding_dimension=256,
        number_of_heads=4,
        feedforward_dimension=512,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        dropout_rate=0.1,
    )

    algorithm = BehavioralCloning()
    loss = RegressionLoss(action_keys=[POSITION_ACTION_KEY])

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=obs_space,
        action_space=action_space,
        prediction_horizon=10,
        loss=loss,
        device=str(device),
        validate_loss_keys=True,
    )
    policy.to(device)

    target_layers_getter = create_target_layers_getter_from_policy(policy)

    assert callable(target_layers_getter)

    layers = target_layers_getter(Cameras.LEFT.value)
    assert isinstance(layers, list)
    assert len(layers) > 0
    assert all(isinstance(layer, nn.Module) for layer in layers)

    with pytest.raises(ValueError):
        target_layers_getter("nonexistent_camera")


@pytest.mark.unit
def test_saliency_maps_with_real_policy(action_space, device):
    """Test compute_saliency_maps with real policy."""
    encoder_config = {
        "_target_": "versatil.models.encoding.encoders.rgb.cnn.CNNEncoder",
        "input_keys": Cameras.LEFT.value,
        "backbone": "timm/resnet18.a1_in1k",
        "pretrained": False,
        "pooling_method": "none",
    }

    # Instantiate encoder from config
    encoder = instantiate(encoder_config)
    encoding_pipeline = EncodingPipeline(encoders={"rgb_encoder": encoder})

    obs_space = ObservationSpace(camera_keys=[Cameras.LEFT.value])

    action_heads = {
        POSITION_ACTION_KEY: ActionHead(
            input_dim=256,
            output_dim=3,
            blocks=[],
        )
    }

    decoder = ACT(
        input_keys=["rgb_encoder_image"],
        action_space=action_space,
        action_heads=action_heads,
        observation_space=obs_space,
        observation_horizon=1,
        prediction_horizon=10,
        device=str(device),
        embedding_dimension=256,
        number_of_heads=4,
        feedforward_dimension=512,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        dropout_rate=0.1,
    )

    algorithm = BehavioralCloning()
    loss = RegressionLoss(action_keys=[POSITION_ACTION_KEY])

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=obs_space,
        action_space=action_space,
        prediction_horizon=10,
        loss=loss,
        device=str(device),
        validate_loss_keys=True,
    )
    policy.to(device)
    policy.eval()

    obs = {Cameras.LEFT.value: torch.randn(1, 3, 224, 224, device=device)}

    for key in obs.keys():
        policy.normalizer.params_dict[key] = torch.nn.ParameterDict({
            'scale': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            'offset': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
            'input_stats': torch.nn.ParameterDict({
                'min': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'max': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
                'mean': torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                'std': torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False),
            })
        })

    wrapper = PolicyExplainerWrapper(policy)

    def output_selector(predictions):
        return predictions[POSITION_ACTION_KEY].sum()

    saliency_maps = compute_saliency_maps(
        model=wrapper,
        observation=obs,
        camera_names=[Cameras.LEFT.value],
        output_selector=output_selector,
    )

    assert isinstance(saliency_maps, dict)
    assert Cameras.LEFT.value in saliency_maps
    assert isinstance(saliency_maps[Cameras.LEFT.value], torch.Tensor)


@pytest.mark.unit
class TestShowCamOnImage:
    """Test show_cam_on_image visualization function."""

    def test_returns_uint8_array(self):
        """Test that function returns uint8 numpy array."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.random.rand(224, 224).astype(np.float32)

        result = show_cam_on_image(img, mask)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_raises_for_invalid_image_range(self):
        """Test that function raises error for image values > 1."""
        img = np.random.rand(224, 224, 3) * 255
        mask = np.random.rand(224, 224).astype(np.float32)

        with pytest.raises(ValueError, match="should be np.float32 in the range"):
            show_cam_on_image(img, mask)

    def test_raises_for_invalid_image_weight(self):
        """Test that function raises error for invalid image weight."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.random.rand(224, 224).astype(np.float32)

        with pytest.raises(ValueError, match="image_weight should be in the range"):
            show_cam_on_image(img, mask, image_weight=1.5)

        with pytest.raises(ValueError, match="image_weight should be in the range"):
            show_cam_on_image(img, mask, image_weight=-0.1)

    def test_image_weight_affects_output(self):
        """Test that different image weights produce different outputs."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.random.rand(224, 224).astype(np.float32)

        result1 = show_cam_on_image(img, mask, image_weight=0.2)
        result2 = show_cam_on_image(img, mask, image_weight=0.8)

        assert not np.array_equal(result1, result2)

    def test_rgb_vs_bgr_mode(self):
        """Test that RGB and BGR modes produce different outputs."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.random.rand(224, 224).astype(np.float32)

        result_bgr = show_cam_on_image(img, mask, use_rgb=False)
        result_rgb = show_cam_on_image(img, mask, use_rgb=True)

        assert not np.array_equal(result_bgr, result_rgb)
