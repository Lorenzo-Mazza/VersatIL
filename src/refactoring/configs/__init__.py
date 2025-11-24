from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from refactoring.configs.data.augmentations import AugmentationPipelineConfig
from refactoring.configs.data.dataloader import DataLoaderConfig
from refactoring.configs.data.tokenizer import TokenizationConfig, ActionTokenizationConfig, ObservationTokenizationConfig
from refactoring.configs.decoding.action_head import ActionHeadConfig, \
    MixtureOfExpertsHeadConfig, ActionHeadBlockConfig, AttentionBlockConfig, MLPBlockConfig, ResidualBlockConfig
from refactoring.configs.decoding.algorithm import DecodingAlgorithmConfig, DiffusionConfig, BehavioralCloningConfig, FlowMatchingConfig, \
    VariationalAlgorithmConfig
from refactoring.configs.decoding.decoder import (
    ACTConfig,
    DecodingNetworkConfig,
    FreeTransformerConfig,
    MixtureOfExpertsDecoderConfig, FASTGPTDecoderConfig, FASTDETRDecoderConfig, MoEFreeTransformerConfig,
)
from refactoring.configs.decoding.latent import LatentActionEncoderConfig, LatentPriorConfig, VAETransformerEncoderConfig, GaussianPriorConfig, \
    DiffusionPriorConfig
from refactoring.configs.encoding.encoder import (
    DepthCNNEncoderConfig,
    EncoderConfig,
    LanguageEncoderConfig,
    ProprioEncoderConfig, DFormerEncoderConfig, EmbedderConfig,
)
from refactoring.configs.encoding.fusion import FusionConfig, ConcatFusionConfig, AttentionFusionConfig, MLPFusionConfig, SpatialFusionConfig
from refactoring.configs.encoding.image import ImageEncoderConfig, CNNEncoderConfig, ViTEncoderConfig
from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.configs.loss import CompositeLossConfig, PhaseActionLossConfig, ActionReconstructionLossConfig, RegressionLossConfig, BaseLossConfig, \
    GripperLossConfig, KLDivergenceLossConfig, BinaryKLDivergenceLossConfig, TrajectoryLengthLossConfig, TrajectorySmoothnessConfig, \
    PhaseClassificationLossConfig, ActionTokenLossConfig
from refactoring.configs.main import MainConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.data.task import TaskSpaceConfig, ActionSpaceConfig, ObservationSpaceConfig
from refactoring.configs.training import (
    OptimizerConfig,
    ParameterGroupConfig,
    TrainingConfig, AdamWConfig, AdamConfig, SGDConfig,
)
from refactoring.data.constants import (
    Cameras,
    GripperType,
    GRIPPER_ACTION_KEY,
    LANGUAGE_KEY,
    OrientationRepresentation,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY, TOKENIZED_OBSERVATIONS_KEY, GRIPPER_STATE_OBS_KEY, TokenizerType,
)
from refactoring.data.tokenization import ActionTokenizer
from refactoring.metrics.components import ActionTokenLoss
from refactoring.models.encoding.encoders.constants import RGBBackboneType, PoolingMethod, LanguageEncoderType
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType, PositionalEncodingType
from refactoring.models.layers.normalization.constants import NormalizationType
from refactoring.training.constants import Float32MatmulPrecision, PrecisionType

__all__ = ["MainConfig","ExperimentConfig", "TrainingConfig", "OptimizerConfig", "ParameterGroupConfig",
           "TaskSpaceConfig", "PolicyConfig", "EncoderConfig",
           "ImageEncoderConfig", "DepthCNNEncoderConfig", "ProprioEncoderConfig", "LanguageEncoderConfig",
           "DecodingNetworkConfig", "ACTConfig", "FreeTransformerConfig", "MixtureOfExpertsDecoderConfig",
           "InferenceConfig", "DataLoaderConfig",
           "DecodingAlgorithmConfig", "BehavioralCloningConfig", "DiffusionConfig", "FlowMatchingConfig", "VariationalAlgorithmConfig",
           "LatentActionEncoderConfig", "LatentPriorConfig", "VAETransformerEncoderConfig", "GaussianPriorConfig", "DiffusionPriorConfig",
           "ActionHeadConfig", "MixtureOfExpertsHeadConfig",
           "ActionHeadBlockConfig", "AttentionBlockConfig", "MLPBlockConfig", "ResidualBlockConfig",
           "FusionConfig", "ConcatFusionConfig", "AttentionFusionConfig", "MLPFusionConfig", "SpatialFusionConfig",
           "CompositeLossConfig", "PhaseActionLossConfig", "ActionReconstructionLossConfig", "RegressionLossConfig", "BaseLossConfig",
           "GripperLossConfig", "KLDivergenceLossConfig", "BinaryKLDivergenceLossConfig", "TrajectoryLengthLossConfig", "TrajectorySmoothnessConfig",
           "PhaseClassificationLossConfig",
           ]


def register_resolvers():
    """Register custom OmegaConf resolvers for enum access in YAML configs.

    This allows using ${cameras:LEFT} in YAML to get Cameras.LEFT.value.
    """
    if not OmegaConf.has_resolver("cameras"):
        OmegaConf.register_new_resolver("cameras", lambda name: Cameras[name].value)
    if not OmegaConf.has_resolver("gripper"):
        OmegaConf.register_new_resolver("gripper", lambda name: GripperType[name].value)
    if not OmegaConf.has_resolver("orientation"):
        OmegaConf.register_new_resolver("orientation", lambda name: OrientationRepresentation[name].value)
    if not OmegaConf.has_resolver("rgb_backbone"):
        OmegaConf.register_new_resolver("rgb_backbone", lambda name: RGBBackboneType[name].value)
    if not OmegaConf.has_resolver("precision"):
        OmegaConf.register_new_resolver("precision", lambda name: PrecisionType[name].value)
    if not OmegaConf.has_resolver("float32_matmul"):
        OmegaConf.register_new_resolver("float32_matmul", lambda name: Float32MatmulPrecision[name].value)
    if not OmegaConf.has_resolver("pooling_method"):
        OmegaConf.register_new_resolver("pooling_method", lambda name: PoolingMethod[name].value)
    if not OmegaConf.has_resolver("language_model"):
        OmegaConf.register_new_resolver("language_model", lambda name: LanguageEncoderType[name].value)
    if not OmegaConf.has_resolver("activation_function"):
        OmegaConf.register_new_resolver("activation_function", lambda name: ActivationFunction[name].value)
    if not OmegaConf.has_resolver("normalization"):
        OmegaConf.register_new_resolver("normalization", lambda name: NormalizationType[name].value)
    if not OmegaConf.has_resolver("attention"):
        OmegaConf.register_new_resolver("attention", lambda name: AttentionType[name].value)
    if not OmegaConf.has_resolver("pos_encoding"):
        OmegaConf.register_new_resolver("pos_encoding", lambda name: PositionalEncodingType[name].value)
    if not OmegaConf.has_resolver("tokenizer_type"):
        OmegaConf.register_new_resolver("tokenizer_type", lambda name: TokenizerType[name].value)

    if not OmegaConf.has_resolver("action_key"):
        action_key_map = {
            "POSITION": POSITION_ACTION_KEY,
            "ORIENTATION": ORIENTATION_ACTION_KEY,
            "GRIPPER": GRIPPER_ACTION_KEY,
        }
        OmegaConf.register_new_resolver("action_key", lambda name: action_key_map[name])
    if not OmegaConf.has_resolver("obs_key"):
        obs_key_map = {
            "PROPRIO_CAMERA_FRAME": PROPRIO_OBS_CAMERA_FRAME_KEY,
            "PROPRIO_ROBOT_FRAME": PROPRIO_OBS_ROBOT_FRAME_KEY,
            "LANGUAGE": LANGUAGE_KEY,
            "GRIPPER_STATE_OBS_KEY": GRIPPER_STATE_OBS_KEY,
            "TOKENIZED_OBSERVATIONS_KEY": TOKENIZED_OBSERVATIONS_KEY
        }
        OmegaConf.register_new_resolver("obs_key", lambda name: obs_key_map[name])


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="config", node=MainConfig)

    cs.store(group="experiment", name="base", node=ExperimentConfig)
    cs.store(group="inference", name="base", node=InferenceConfig)
    cs.store(group="task", name="base", node=TaskSpaceConfig)
    cs.store(group="task/dataloader", name="base", node=DataLoaderConfig)
    cs.store(group="task/dataloader/image_augmentations", name="base", node=AugmentationPipelineConfig)
    cs.store(group="task/dataloader/tokenization", name="base", node=TokenizationConfig)
    cs.store(group="task/dataloader/tokenization/action", name="base", node=ActionTokenizationConfig)
    cs.store(group="task/dataloader/tokenization/observation", name="base", node=ObservationTokenizationConfig)
    cs.store(group="task/action_space", name="base", node=ActionSpaceConfig)
    cs.store(group="task/observation_space", name="base", node=ObservationSpaceConfig)
    cs.store(group="training", name="base", node=TrainingConfig)
    cs.store(group="training/optimizer", name="base", node=OptimizerConfig)
    cs.store(group="training/optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="training/optimizer", name="adam", node=AdamConfig)
    cs.store(group="training/optimizer", name="sgd", node=SGDConfig)
    cs.store(group="training/optimizer/parameter_group", name="base", node=ParameterGroupConfig)

    cs.store(group="policy", name="base", node=PolicyConfig)
    cs.store(group="policy/algorithm", name="base", node=DecodingAlgorithmConfig)
    cs.store(group="policy/algorithm", name="bc", node=BehavioralCloningConfig)
    cs.store(group="policy/algorithm", name="diffusion", node=DiffusionConfig)
    cs.store(group="policy/algorithm", name="flowmatching", node=FlowMatchingConfig)
    cs.store(group="policy/algorithm", name="variational", node=VariationalAlgorithmConfig)
    cs.store(group="policy/algorithm/posterior", name="base", node=LatentActionEncoderConfig)
    cs.store(group="policy/algorithm/prior", name="base", node=LatentPriorConfig)
    cs.store(group="policy/algorithm/posterior", name="vae", node=VAETransformerEncoderConfig)
    cs.store(group="policy/algorithm/prior", name="gaussian", node=GaussianPriorConfig)
    cs.store(group="policy/algorithm/prior", name="diffusion", node=DiffusionPriorConfig)

    cs.store(group="policy/loss", name="composite", node=CompositeLossConfig)
    cs.store(group="policy/loss", name="phase_action", node=PhaseActionLossConfig)
    cs.store(group="policy/loss", name="action_reconstruction", node=ActionReconstructionLossConfig)
    cs.store(group="policy/loss", name="regression", node=RegressionLossConfig)
    cs.store(group="policy/loss", name="base", node=BaseLossConfig)
    cs.store(group="policy/loss", name="gripper", node=GripperLossConfig)
    cs.store(group="policy/loss", name="kl", node=KLDivergenceLossConfig)
    cs.store(group="policy/loss", name="binary_kl", node=BinaryKLDivergenceLossConfig)
    cs.store(group="policy/loss", name="traj_len", node=TrajectoryLengthLossConfig)
    cs.store(group="policy/loss", name="traj_smooth", node=TrajectorySmoothnessConfig)
    cs.store(group="policy/loss", name="phase_classification", node=PhaseClassificationLossConfig)
    cs.store(group="policy/loss", name="token_loss", node=ActionTokenLossConfig)


    cs.store(group="policy/encoding_pipeline", name="base", node=ImageEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder", name="image", node=ImageEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder/image", name="cnn", node=CNNEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder/image", name="vit", node=ViTEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder", name="depth_cnn", node=DepthCNNEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder", name="dformer", node=DFormerEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder", name="proprio", node=ProprioEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder", name="language", node=LanguageEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder", name="dformer", node=DFormerEncoderConfig)
    cs.store(group="policy/encoding_pipeline/encoder", name="embedder", node=EmbedderConfig)
    cs.store(group="policy/encoding_pipeline/fusion", name="base", node=FusionConfig)
    cs.store(group="policy/encoding_pipeline/fusion", name="concat", node=ConcatFusionConfig)
    cs.store(group="policy/encoding_pipeline/fusion", name="attention", node=AttentionFusionConfig)
    cs.store(group="policy/encoding_pipeline/fusion", name="mlp", node=MLPFusionConfig)
    cs.store(group="policy/encoding_pipeline/fusion", name="spatial", node=SpatialFusionConfig)

    cs.store(group="policy/decoder", name="base", node=DecodingNetworkConfig)
    cs.store(group="policy/decoder", name="act", node=ACTConfig)
    cs.store(group="policy/decoder", name="gpt", node=FASTGPTDecoderConfig)
    cs.store(group="policy/decoder", name="fastdetr", node=FASTDETRDecoderConfig)
    cs.store(group="policy/decoder", name="free_transformer", node=FreeTransformerConfig)
    cs.store(group="policy/decoder", name="moe_free_transformer", node=MoEFreeTransformerConfig)
    cs.store(group="policy/decoder", name="moe", node=MixtureOfExpertsDecoderConfig)
    cs.store(group="policy/decoder/action_head", name="base", node=ActionHeadConfig)
    cs.store(group="policy/decoder/action_head", name="moe", node=MixtureOfExpertsHeadConfig)
    cs.store(group="policy/decoder/head_block", name="base", node=ActionHeadBlockConfig)
    cs.store(group="policy/decoder/head_block", name="mlp", node=MLPBlockConfig)
    cs.store(group="policy/decoder/head_block", name="attention", node=AttentionBlockConfig)
    cs.store(group="policy/decoder/head_block", name="residual", node=ResidualBlockConfig)

# Register resolvers on module import
register_resolvers()
register_configs()