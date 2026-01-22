from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from versatil.configs.data.augmentations import AugmentationPipelineConfig
from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.data.raw import (
    DatasetSchemaConfig,
    CsvDatasetSchemaConfig,
    Hdf5DatasetSchemaConfig,
    DatasetMetadataConfig,
)
from versatil.configs.data.metadata import (
    ObservationMetadataConfig,
    PositionObservationMetadataConfig,
    OrientationObservationMetadataConfig,
    GripperObservationMetadataConfig,
    CameraMetadataConfig,
    PrecomputedActionMetadataConfig,
    PositionActionMetadataConfig,
    OrientationActionMetadataConfig,
    GripperActionMetadataConfig,
)
from versatil.configs.data.tokenizer import (
    TokenizationConfig,
    ActionTokenizationConfig,
    ObservationTokenizationConfig,
)
from versatil.configs.decoding.action_head import (
    ActionHeadConfig,
    MixtureOfExpertsHeadConfig,
    ActionHeadBlockConfig,
    AttentionBlockConfig,
    MLPBlockConfig,
    ResidualBlockConfig,
)
from versatil.configs.decoding.algorithm import (
    DecodingAlgorithmConfig,
    DiffusionConfig,
    BehavioralCloningConfig,
    FlowMatchingConfig,
    VariationalAlgorithmConfig,
)
from versatil.configs.decoding.decoder import (
    ACTConfig,
    ConditionalUNetDecoderConfig,
    DecodingNetworkConfig,
    DiTDecoderConfig,
    FreeTransformerConfig,
    LACTConfig,
    MixtureOfExpertsDecoderConfig,
    FASTGPTDecoderConfig,
    FASTDETRDecoderConfig,
    MoEFreeTransformerConfig,
    PhaseACTConfig,
    ActionTransformerConfig,
)
from versatil.configs.decoding.latent import (
    PosteriorLatentEncoderConfig,
    PriorLatentEncoderConfig,
    VAETransformerEncoderConfig,
    GaussianPriorConfig,
    DenoisingTransformerPriorConfig,
    PriorTransformerEncoderConfig,
    VampPriorConfig,
)
from versatil.configs.encoding.encoder import (
    DepthCNNEncoderConfig,
    EncoderConfig,
    LanguageEncoderConfig,
    ProprioEncoderConfig,
    DFormerEncoderConfig,
)
from versatil.configs.encoding.fusion import (
    FusionConfig,
    ConcatFusionConfig,
    AttentionFusionConfig,
    MLPFusionConfig,
    SpatialFusionConfig,
)
from versatil.configs.encoding.image import (
    ImageEncoderConfig,
    CNNEncoderConfig,
    ViTEncoderConfig, ConditionalCNNEncoderConfig,
)
from versatil.configs.experiment import ExperimentConfig
from versatil.configs.inference import InferenceConfig
from versatil.configs.loss import (
    CompositeLossConfig,
    PhaseActionLossConfig,
    ActionReconstructionLossConfig,
    RegressionLossConfig,
    BaseLossConfig,
    GaussianEntropyLossConfig,
    GripperLossConfig,
    KLDivergenceLossConfig,
    BinaryKLDivergenceLossConfig,
    TrajectoryLengthLossConfig,
    TrajectorySmoothnessConfig,
    PhaseClassificationLossConfig,
    ActionTokenLossConfig,
    MoELossConfig,
    MaximumMeanDiscrepancyLossConfig,
    BinaryMaximumMeanDiscrepancyLossConfig,
    FixedVarianceGaussianNLLossConfig,
    FixedVarianceGripperMixtureNLLossConfig,
    PriorDenoisingLossConfig,
)
from versatil.configs.main import MainConfig
from versatil.configs.policy import PolicyConfig
from versatil.configs.data.task import (
    TaskSpaceConfig,
    ActionSpaceConfig,
    ObservationSpaceConfig,
)
from versatil.configs.training import (
    OptimizerConfig,
    ParameterGroupConfig,
    TrainingConfig,
    AdamWConfig,
    AdamConfig,
    SGDConfig,
)
from versatil.data.constants import (
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    GripperType,
    GRIPPER_ACTION_KEY,
    ObsKey,
    OrientationRepresentation,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    ProprioKey,
    TokenizerType,
    KinematicsNormalizationType,
    ImageNormalizationType,
)
from versatil.models.decoding.constants import (
    ACTION_LOGITS_KEY,
    LATENT_KEY,
    LatentKey,
    MoERoutingType, DenoisingAlgorithm,
)
from versatil.models.encoding.encoders.constants import (
    RGBBackboneType,
    PoolingMethod,
    LanguageEncoderType,
    BatchNormHandling
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, ConditioningType, PositionalEncodingType
from versatil.metrics.constants import MetadataKey
from versatil.models.layers.diffusion_process import SchedulerType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.training.constants import Float32MatmulPrecision, PrecisionType

__all__ = [
    "MainConfig",
    "ExperimentConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "ParameterGroupConfig",
    "TaskSpaceConfig",
    "PolicyConfig",
    "EncoderConfig",
    "ImageEncoderConfig",
    "DepthCNNEncoderConfig",
    "ProprioEncoderConfig",
    "LanguageEncoderConfig",
    "DecodingNetworkConfig",
    "ACTConfig",
    "ConditionalUNetDecoderConfig",
    "DiTDecoderConfig",
    "FreeTransformerConfig",
    "LACTConfig",
    "MixtureOfExpertsDecoderConfig",
    "InferenceConfig",
    "DataLoaderConfig",
    "DecodingAlgorithmConfig",
    "BehavioralCloningConfig",
    "DiffusionConfig",
    "FlowMatchingConfig",
    "VariationalAlgorithmConfig",
    "PosteriorLatentEncoderConfig",
    "PriorLatentEncoderConfig",
    "VAETransformerEncoderConfig",
    "GaussianPriorConfig",
    "DenoisingTransformerPriorConfig",
    "VampPriorConfig",
    "ActionHeadConfig",
    "MixtureOfExpertsHeadConfig",
    "ActionHeadBlockConfig",
    "AttentionBlockConfig",
    "MLPBlockConfig",
    "ResidualBlockConfig",
    "FusionConfig",
    "ConcatFusionConfig",
    "AttentionFusionConfig",
    "MLPFusionConfig",
    "SpatialFusionConfig",
    "CompositeLossConfig",
    "PhaseActionLossConfig",
    "ActionReconstructionLossConfig",
    "RegressionLossConfig",
    "BaseLossConfig",
    "GripperLossConfig",
    "KLDivergenceLossConfig",
    "BinaryKLDivergenceLossConfig",
    "TrajectoryLengthLossConfig",
    "TrajectorySmoothnessConfig",
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
        OmegaConf.register_new_resolver(
            "orientation", lambda name: OrientationRepresentation[name].value
        )
    if not OmegaConf.has_resolver("rgb_backbone"):
        OmegaConf.register_new_resolver(
            "rgb_backbone", lambda name: RGBBackboneType[name].value
        )
    if not OmegaConf.has_resolver("batch_norm_handling"):
        OmegaConf.register_new_resolver(
            "batch_norm_handling", lambda name: BatchNormHandling[name].value
        )
    if not OmegaConf.has_resolver("precision"):
        OmegaConf.register_new_resolver(
            "precision", lambda name: PrecisionType[name].value
        )
    if not OmegaConf.has_resolver("float32_matmul"):
        OmegaConf.register_new_resolver(
            "float32_matmul", lambda name: Float32MatmulPrecision[name].value
        )
    if not OmegaConf.has_resolver("pooling_method"):
        OmegaConf.register_new_resolver(
            "pooling_method", lambda name: PoolingMethod[name].value
        )
    if not OmegaConf.has_resolver("language_model"):
        OmegaConf.register_new_resolver(
            "language_model", lambda name: LanguageEncoderType[name].value
        )
    if not OmegaConf.has_resolver("activation_function"):
        OmegaConf.register_new_resolver(
            "activation_function", lambda name: ActivationFunction[name].value
        )
    if not OmegaConf.has_resolver("normalization"):
        OmegaConf.register_new_resolver(
            "normalization", lambda name: NormalizationType[name].value
        )
    if not OmegaConf.has_resolver("attention"):
        OmegaConf.register_new_resolver(
            "attention", lambda name: AttentionType[name].value
        )
    if not OmegaConf.has_resolver("pos_encoding"):
        OmegaConf.register_new_resolver(
            "pos_encoding", lambda name: PositionalEncodingType[name].value
        )
    if not OmegaConf.has_resolver("tokenizer_type"):
        OmegaConf.register_new_resolver(
            "tokenizer_type", lambda name: TokenizerType[name].value
        )
    if not OmegaConf.has_resolver("normalization_type"):
        OmegaConf.register_new_resolver(
            "kinematics_norm_type", lambda name: KinematicsNormalizationType[name].value
        )
    if not OmegaConf.has_resolver("image_normalization_type"):
        OmegaConf.register_new_resolver(
            "image_norm_type", lambda name: ImageNormalizationType[name].value
        )
    if not OmegaConf.has_resolver("action_key"):
        action_key_map = {
            "POSITION": POSITION_ACTION_KEY,
            "ORIENTATION": ORIENTATION_ACTION_KEY,
            "GRIPPER": GRIPPER_ACTION_KEY,
            "ACTION_TOKENS": ACTION_LOGITS_KEY,
        }
        OmegaConf.register_new_resolver("action_key", lambda name: action_key_map[name])
    if not OmegaConf.has_resolver("obs_key"):
        OmegaConf.register_new_resolver("obs_key", lambda name: ObsKey[name].value)
    if not OmegaConf.has_resolver("moe_routing_type"):
        OmegaConf.register_new_resolver(
            "moe_routing_type", lambda name: MoERoutingType[name].value
        )
    if not OmegaConf.has_resolver("coordinate_system"):
        OmegaConf.register_new_resolver(
            "coordinate_system", lambda name: CoordinateSystem[name].value
        )
    if not OmegaConf.has_resolver("gripper_range"):
        OmegaConf.register_new_resolver(
            "gripper_range", lambda name: BinaryGripperRange[name].value
        )
    if not OmegaConf.has_resolver("proprio_key"):
        OmegaConf.register_new_resolver(
            "proprio_key", lambda name: ProprioKey[name].value
        )
    if not OmegaConf.has_resolver("latent_key"):
        OmegaConf.register_new_resolver(
            "latent_key", lambda name: LatentKey[name].value
        )
    if not OmegaConf.has_resolver("scheduler_type"):
        OmegaConf.register_new_resolver(
            "scheduler_type", lambda name: SchedulerType[name].value
        )
    if not OmegaConf.has_resolver("denoising_algorithm"):
        OmegaConf.register_new_resolver(
            "denoising_algorithm", lambda name: DenoisingAlgorithm[name].value
        )
    if not OmegaConf.has_resolver("conditioning_type"):
        OmegaConf.register_new_resolver(
            "conditioning_type", lambda name: ConditioningType[name].value
        )
    if not OmegaConf.has_resolver("metadata_key"):
        OmegaConf.register_new_resolver(
            "metadata_key", lambda name: MetadataKey[name].value
        )

def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="config", node=MainConfig)

    cs.store(group="experiment", name="base", node=ExperimentConfig)
    cs.store(group="inference", name="base", node=InferenceConfig)
    cs.store(group="task", name="base", node=TaskSpaceConfig)
    cs.store(
        group="task/dataset_schema/zarr_meta", name="base", node=DatasetMetadataConfig
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="base",
        node=ObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="position",
        node=PositionObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="orientation",
        node=OrientationObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="gripper",
        node=GripperObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/camera",
        name="base",
        node=CameraMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="base",
        node=PrecomputedActionMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="position",
        node=PositionActionMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="orientation",
        node=OrientationActionMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="gripper",
        node=GripperActionMetadataConfig,
    )
    cs.store(group="task/dataset_schema", name="base", node=DatasetSchemaConfig)
    cs.store(group="task/dataset_schema", name="hdf5", node=Hdf5DatasetSchemaConfig)
    cs.store(group="task/dataset_schema", name="csv", node=CsvDatasetSchemaConfig)
    cs.store(group="task/dataloader", name="base", node=DataLoaderConfig)
    cs.store(
        group="task/dataloader/image_augmentations",
        name="base",
        node=AugmentationPipelineConfig,
    )
    cs.store(group="task/dataloader/tokenization", name="base", node=TokenizationConfig)
    cs.store(
        group="task/dataloader/tokenization/action",
        name="base",
        node=ActionTokenizationConfig,
    )
    cs.store(
        group="task/dataloader/tokenization/observation",
        name="base",
        node=ObservationTokenizationConfig,
    )
    cs.store(group="task/action_space", name="base", node=ActionSpaceConfig)
    cs.store(group="task/observation_space", name="base", node=ObservationSpaceConfig)
    cs.store(group="training", name="base", node=TrainingConfig)
    cs.store(group="training/optimizer", name="base", node=OptimizerConfig)
    cs.store(group="training/optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="training/optimizer", name="adam", node=AdamConfig)
    cs.store(group="training/optimizer", name="sgd", node=SGDConfig)
    cs.store(
        group="training/optimizer/parameter_group",
        name="base",
        node=ParameterGroupConfig,
    )

    cs.store(group="policy", name="base", node=PolicyConfig)
    cs.store(group="policy/algorithm", name="base", node=DecodingAlgorithmConfig)
    cs.store(group="policy/algorithm", name="bc", node=BehavioralCloningConfig)
    cs.store(group="policy/algorithm", name="diffusion_process", node=DiffusionConfig)
    cs.store(group="policy/algorithm", name="flow", node=FlowMatchingConfig)
    cs.store(
        group="policy/algorithm", name="variational", node=VariationalAlgorithmConfig
    )
    cs.store(
        group="policy/algorithm/posterior",
        name="base",
        node=PosteriorLatentEncoderConfig,
    )
    cs.store(group="policy/algorithm/prior", name="base", node=PriorLatentEncoderConfig)
    cs.store(
        group="policy/algorithm/posterior",
        name="transformerencoder",
        node=VAETransformerEncoderConfig,
    )
    cs.store(group="policy/algorithm/prior", name="gaussian", node=GaussianPriorConfig)
    cs.store(
        group="policy/algorithm/prior",
        name="transformerencoder",
        node=PriorTransformerEncoderConfig,
    )
    cs.store(group="policy/algorithm/prior", name="vamp", node=VampPriorConfig)
    cs.store(
        group="policy/algorithm/prior",
        name="dit",
        node=DenoisingTransformerPriorConfig,
    )

    cs.store(group="policy/loss", name="composite", node=CompositeLossConfig)
    cs.store(group="policy/loss", name="phase_action", node=PhaseActionLossConfig)
    cs.store(
        group="policy/loss",
        name="action_reconstruction",
        node=ActionReconstructionLossConfig,
    )
    cs.store(group="policy/loss", name="regression", node=RegressionLossConfig)
    cs.store(group="policy/loss", name="base", node=BaseLossConfig)
    cs.store(group="policy/loss", name="gripper", node=GripperLossConfig)
    cs.store(group="policy/loss", name="entropy", node=GaussianEntropyLossConfig)
    cs.store(group="policy/loss", name="kl", node=KLDivergenceLossConfig)
    cs.store(group="policy/loss", name="mmd", node=MaximumMeanDiscrepancyLossConfig)
    cs.store(
        group="policy/loss",
        name="binary_mmd",
        node=BinaryMaximumMeanDiscrepancyLossConfig,
    )
    cs.store(group="policy/loss", name="binary_kl", node=BinaryKLDivergenceLossConfig)
    cs.store(group="policy/loss", name="traj_len", node=TrajectoryLengthLossConfig)
    cs.store(group="policy/loss", name="traj_smooth", node=TrajectorySmoothnessConfig)
    cs.store(
        group="policy/loss",
        name="phase_classification",
        node=PhaseClassificationLossConfig,
    )
    cs.store(group="policy/loss", name="token_loss", node=ActionTokenLossConfig)
    cs.store(group="policy/loss", name="moe_loss", node=MoELossConfig)
    cs.store(
        group="policy/loss",
        name="fv_gaussian_nll",
        node=FixedVarianceGaussianNLLossConfig,
    )
    cs.store(
        group="policy/loss",
        name="fv_bernoulli_nll",
        node=FixedVarianceGripperMixtureNLLossConfig,
    )
    cs.store(
        group="policy/loss",
        name="denoising_prior",
        node=PriorDenoisingLossConfig,
    )

    cs.store(group="policy/encoding_pipeline", name="base", node=ImageEncoderConfig)
    cs.store(
        group="policy/encoding_pipeline/encoder", name="image", node=ImageEncoderConfig
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/image",
        name="cnn",
        node=CNNEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/image",
        name="conditional_cnn",
        node=ConditionalCNNEncoderConfig,
    )

    cs.store(
        group="policy/encoding_pipeline/encoder/image",
        name="vit",
        node=ViTEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="depth_cnn",
        node=DepthCNNEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="dformer",
        node=DFormerEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="proprio",
        node=ProprioEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="language",
        node=LanguageEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="dformer",
        node=DFormerEncoderConfig,
    )
    cs.store(group="policy/encoding_pipeline/fusion", name="base", node=FusionConfig)
    cs.store(
        group="policy/encoding_pipeline/fusion", name="concat", node=ConcatFusionConfig
    )
    cs.store(
        group="policy/encoding_pipeline/fusion",
        name="attention",
        node=AttentionFusionConfig,
    )
    cs.store(group="policy/encoding_pipeline/fusion", name="mlp", node=MLPFusionConfig)
    cs.store(
        group="policy/encoding_pipeline/fusion",
        name="spatial",
        node=SpatialFusionConfig,
    )

    cs.store(group="policy/decoder", name="base", node=DecodingNetworkConfig)
    cs.store(group="policy/decoder", name="act", node=ACTConfig)
    cs.store(group="policy/decoder", name="phase_act", node=PhaseACTConfig)
    cs.store(
        group="policy/decoder",
        name="simple_action_transformer",
        node=ActionTransformerConfig,
    )

    cs.store(group="policy/decoder", name="gpt", node=FASTGPTDecoderConfig)
    cs.store(group="policy/decoder", name="fastdetr", node=FASTDETRDecoderConfig)
    cs.store(
        group="policy/decoder", name="free_transformer", node=FreeTransformerConfig
    )
    cs.store(
        group="policy/decoder",
        name="moe_free_transformer",
        node=MoEFreeTransformerConfig,
    )
    cs.store(group="policy/decoder", name="moe", node=MixtureOfExpertsDecoderConfig)
    cs.store(group="policy/decoder", name="lact_decoder", node=LACTConfig)
    cs.store(group="policy/decoder", name="dit", node=DiTDecoderConfig)
    cs.store(group="policy/decoder", name="unet", node=ConditionalUNetDecoderConfig)
    cs.store(group="policy/decoder/action_head", name="base", node=ActionHeadConfig)
    cs.store(
        group="policy/decoder/action_head", name="moe", node=MixtureOfExpertsHeadConfig
    )
    cs.store(group="policy/decoder/head_block", name="base", node=ActionHeadBlockConfig)
    cs.store(group="policy/decoder/head_block", name="mlp", node=MLPBlockConfig)
    cs.store(
        group="policy/decoder/head_block", name="attention", node=AttentionBlockConfig
    )
    cs.store(
        group="policy/decoder/head_block", name="residual", node=ResidualBlockConfig
    )


# Register resolvers on module import
register_resolvers()
register_configs()
