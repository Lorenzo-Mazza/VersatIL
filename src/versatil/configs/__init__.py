"""Configuration and resolver centralized store for OmegaConf."""

import os
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from versatil.configs.data.augmentations import AugmentationPipelineConfig
from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.data.metadata import (
    CameraMetadataConfig,
    GripperActionMetadataConfig,
    GripperObservationMetadataConfig,
    ObservationMetadataConfig,
    OrientationActionMetadataConfig,
    OrientationObservationMetadataConfig,
    PositionActionMetadataConfig,
    PositionObservationMetadataConfig,
    PrecomputedActionMetadataConfig,
)
from versatil.configs.data.raw import (
    CsvDatasetSchemaConfig,
    DatasetMetadataConfig,
    DatasetSchemaConfig,
    Hdf5DatasetSchemaConfig,
    LeRobotDatasetSchemaConfig,
    SyntheticDatasetSchemaConfig,
    DatasetMetadataConfig,
)
from versatil.configs.data.task import (
    ActionSpaceConfig,
    ObservationSpaceConfig,
    TaskSpaceConfig,
)
from versatil.configs.data.tokenizer import (
    ActionTokenizationConfig,
    ObservationTokenizationConfig,
    TokenizationConfig,
)
from versatil.configs.decoding.action_head import (
    ActionHeadBlockConfig,
    ActionHeadConfig,
    AttentionBlockConfig,
    GaussianHeadConfig,
    MixtureOfExpertsHeadConfig,
    MLPBlockConfig,
    ResidualBlockConfig,
)
from versatil.configs.decoding.algorithm import (
    BehavioralCloningConfig,
    DecodingAlgorithmConfig,
    DiffusionConfig,
    FlowMatchingConfig,
    VariationalAlgorithmConfig,
)
from versatil.configs.decoding.decoder import (
    ACTConfig,
    ActionTransformerConfig,
    ConditionalActionUNetConfig,
    DecodingNetworkConfig,
    DiffusionActionTransformerConfig,
    DiscreteDETRActionTransformerConfig,
    DiTBlockActionTransformerConfig,
    FreeActionTransformerConfig,
    GPTActionTransformerConfig,
    LACTConfig,
    MixtureOfDensitiesActionTransformerConfig,
    MixtureOfExpertsDecoderConfig,
    MoEFreeActionTransformerConfig,
    PhaseACTConfig,
    Pi0DecoderConfig,
    SmolVLADecoderConfig,
)
from versatil.configs.decoding.latent import (
    DiTPriorConfig,
    GaussianPriorConfig,
    PosteriorLatentEncoderConfig,
    PriorLatentEncoderConfig,
    PriorTransformerEncoderConfig,
    VAETransformerEncoderConfig,
    VampPriorConfig,
)
from versatil.configs.encoding.encoder import (
    ConditionalCNNEncoderConfig,
    DFormerEncoderConfig,
    EncoderConfig,
    FlatRGBEncoderConfig,
    GeometricRGBDEncoderConfig,
    ImageEncoderConfig,
    LanguageEncoderConfig,
    PaliGemmaEncoderConfig,
    ProprioEncoderConfig,
    SmolVLMEncoderConfig,
    SpatialDepthEncoderConfig,
    SpatialRGBEncoderConfig,
    TwoTowerVLMEncoderConfig,
)
from versatil.configs.encoding.fusion import (
    AttentionFusionConfig,
    ConcatFusionConfig,
    FusionConfig,
    MLPFusionConfig,
    SpatialFusionConfig,
)
from versatil.configs.encoding.pipeline import EncodingPipelineConfig
from versatil.configs.experiment import ExperimentConfig
from versatil.configs.inference import InferenceConfig
from versatil.configs.loss import (
    ActionTokenLossConfig,
    BaseLossConfig,
    BinaryKLDivergenceLossConfig,
    BinaryMaximumMeanDiscrepancyLossConfig,
    CompositeLossConfig,
    GaussianEntropyLossConfig,
    GaussianMixtureNLLossConfig,
    GripperLossConfig,
    GripperMixtureNLLossConfig,
    KLDivergenceLossConfig,
    LatentOptimalTransportLossConfig,
    MaximumMeanDiscrepancyLossConfig,
    MoELossConfig,
    OptimalTransportLossConfig,
    PhaseClassificationLossConfig,
    PriorDenoisingLossConfig,
    RegressionLossConfig,
    TrajectoryLengthLossConfig,
    TrajectorySmoothnessConfig,
    VICLatentLossConfig,
)
from versatil.configs.main import MainConfig
from versatil.configs.policy import PolicyConfig
from versatil.configs.post_training_compression import (
    BasePrunerConfig,
    CompressionTargetConfig,
    PostTrainingCompressorConfig,
    PreparationConfig,
    StructuredPrunerConfig,
    UnstructuredPrunerConfig,
)
from versatil.configs.quantization import (
    BasePT2EBackendConfig,
    Int4WeightOnlyQuantizeConfig,
    Int8DynamicQuantizeConfig,
    PT2EStrategyConfig,
    QuantizeApiStrategyConfig,
    X86InductorBackendConfig,
)
from versatil.configs.training import (
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    SGDConfig,
    TrainingConfig,
)
from versatil.data.constants import (
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    DatasetType,
    GripperType,
    ImageNormalizationType,
    KinematicsNormalizationType,
    ObsKey,
    OrientationRepresentation,
    ProprioKey,
    RawCameraKey,
    SampleKey,
    TokenizerType,
    TokenPaddingStrategy,
)
from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.metrics.constants import MetadataKey
from versatil.metrics.kernels import KernelType
from versatil.models.decoding.constants import (
    DecoderOutputKey,
    DenoisingAlgorithm,
    DiTType,
    LatentKey,
    MoERoutingType,
    TimeConditioning,
)
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    ImageTextModelType,
    LanguageEncoderType,
    PoolingMethod,
    RGBBackboneType,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import (
    AttentionType,
    ConditioningType,
    PositionalEncodingType,
)
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.post_training_compression.constants import PrunableLayerType
from versatil.quantization.constants import QuantizationBackend
from versatil.training.constants import (
    CompileMode,
    Float32MatmulPrecision,
    PrecisionType,
)

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
    "SpatialDepthEncoderConfig",
    "SpatialRGBEncoderConfig",
    "FlatRGBEncoderConfig",
    "ProprioEncoderConfig",
    "LanguageEncoderConfig",
    "DecodingNetworkConfig",
    "ACTConfig",
    "ConditionalActionUNetConfig",
    "Pi0DecoderConfig",
    "SmolVLADecoderConfig",
    "DiTBlockActionTransformerConfig",
    "DiffusionActionTransformerConfig",
    "FreeActionTransformerConfig",
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
    "DiTPriorConfig",
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
    "VICLatentLossConfig",
    "RegressionLossConfig",
    "BaseLossConfig",
    "GripperLossConfig",
    "KLDivergenceLossConfig",
    "BinaryKLDivergenceLossConfig",
    "TrajectoryLengthLossConfig",
    "TrajectorySmoothnessConfig",
    "PhaseClassificationLossConfig",
    "BasePT2EBackendConfig",
    "BasePrunerConfig",
    "Int4WeightOnlyQuantizeConfig",
    "Int8DynamicQuantizeConfig",
    "CompressionTargetConfig",
    "PT2EStrategyConfig",
    "PostTrainingCompressorConfig",
    "PreparationConfig",
    "QuantizeApiStrategyConfig",
    "StructuredPrunerConfig",
    "UnstructuredPrunerConfig",
    "X86InductorBackendConfig",
    "SyntheticDatasetSchemaConfig",
]


def register_resolvers():
    """Register custom OmegaConf resolvers for enum access in YAML configs.

    This allows using ${cameras:LEFT} in YAML to get Cameras.LEFT.value.
    """
    if not OmegaConf.has_resolver("cameras"):
        OmegaConf.register_new_resolver("cameras", lambda name: Cameras[name].value)
    if not OmegaConf.has_resolver("raw_camera"):
        OmegaConf.register_new_resolver(
            "raw_camera", lambda name: RawCameraKey[name].value
        )
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
    if not OmegaConf.has_resolver("vlm_model"):
        OmegaConf.register_new_resolver(
            "vlm_model", lambda name: ImageTextModelType[name].value
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
    if not OmegaConf.has_resolver("kinematics_norm_type"):
        OmegaConf.register_new_resolver(
            "kinematics_norm_type", lambda name: KinematicsNormalizationType[name].value
        )
    if not OmegaConf.has_resolver("image_norm_type"):
        OmegaConf.register_new_resolver(
            "image_norm_type", lambda name: ImageNormalizationType[name].value
        )
    if not OmegaConf.has_resolver("obs_key"):
        OmegaConf.register_new_resolver("obs_key", lambda name: ObsKey[name].value)
    if not OmegaConf.has_resolver("sample_key"):
        OmegaConf.register_new_resolver(
            "sample_key", lambda name: SampleKey[name].value
        )
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
    if not OmegaConf.has_resolver("dit_type"):
        OmegaConf.register_new_resolver("dit_type", lambda name: DiTType[name].value)
    if not OmegaConf.has_resolver("time_conditioning"):
        OmegaConf.register_new_resolver(
            "time_conditioning", lambda name: TimeConditioning[name].value
        )
    if not OmegaConf.has_resolver("timestep_sampler"):
        OmegaConf.register_new_resolver(
            "timestep_sampler", lambda name: TimestepSampler[name].value
        )
    if not OmegaConf.has_resolver("dataset_type"):
        OmegaConf.register_new_resolver(
            "dataset_type", lambda name: DatasetType[name].value
        )
    if not OmegaConf.has_resolver("kernel_type"):
        OmegaConf.register_new_resolver(
            "kernel_type", lambda name: KernelType[name].value
        )
    if not OmegaConf.has_resolver("token_padding"):
        OmegaConf.register_new_resolver(
            "token_padding", lambda name: TokenPaddingStrategy[name].value
        )
    if not OmegaConf.has_resolver("synthetic_task"):
        OmegaConf.register_new_resolver(
            "synthetic_task", lambda name: SyntheticTaskName[name].value
        )

    if not OmegaConf.has_resolver("compile_mode"):
        OmegaConf.register_new_resolver(
            "compile_mode", lambda name: CompileMode[name].value
        )
    if not OmegaConf.has_resolver("quantization_backend"):
        OmegaConf.register_new_resolver(
            "quantization_backend", lambda name: QuantizationBackend[name].value
        )
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env", lambda key, default=None: os.environ.get(key, default)
        )
    if not OmegaConf.has_resolver("checkpoint_dir"):
        OmegaConf.register_new_resolver(
            "checkpoint_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_CHECKPOINT_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("zarr_dir"):
        OmegaConf.register_new_resolver(
            "zarr_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_ZARR_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("cache_dir"):
        OmegaConf.register_new_resolver(
            "cache_dir",
            lambda: os.environ.get(
                "VERSATIL_CACHE_DIR", str(Path.home() / ".cache" / "versatil")
            ),
        )
    if not OmegaConf.has_resolver("pretrained_dir"):
        OmegaConf.register_new_resolver(
            "pretrained_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_PRETRAINED_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("bowel_retraction_dir"):
        OmegaConf.register_new_resolver(
            "bowel_retraction_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_BOWEL_RETRACTION_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("libero_hdf5_dir"):
        OmegaConf.register_new_resolver(
            "libero_hdf5_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_LIBERO_HDF5_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("libero_lerobot_dir"):
        OmegaConf.register_new_resolver(
            "libero_lerobot_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_LIBERO_LEROBOT_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("libero_plus_lerobot_dir"):
        OmegaConf.register_new_resolver(
            "libero_plus_lerobot_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_LIBERO_PLUS_LEROBOT_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("metaworld_lerobot_dir"):
        OmegaConf.register_new_resolver(
            "metaworld_lerobot_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_METAWORLD_LEROBOT_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("prunable_layer"):
        OmegaConf.register_new_resolver(
            "prunable_layer",
            lambda name: PrunableLayerType[name].value,
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
    cs.store(
        group="task/dataset_schema", name="lerobot", node=LeRobotDatasetSchemaConfig
    )
    cs.store(group="task/dataset_schema", name="base", node=DatasetSchemaConfig)
    cs.store(group="task/dataset_schema", name="hdf5", node=Hdf5DatasetSchemaConfig)
    cs.store(group="task/dataset_schema", name="csv", node=CsvDatasetSchemaConfig)
    cs.store(
        group="task/dataset_schema", name="synthetic", node=SyntheticDatasetSchemaConfig
    )
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
    cs.store(group="training/optimizer", name="adamw_schema", node=AdamWConfig)
    cs.store(group="training/optimizer", name="adam_schema", node=AdamConfig)
    cs.store(group="training/optimizer", name="sgd_schema", node=SGDConfig)
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
        node=DiTPriorConfig,
    )

    cs.store(group="policy/loss", name="composite", node=CompositeLossConfig)
    cs.store(group="policy/loss", name="regression", node=RegressionLossConfig)
    cs.store(group="policy/loss", name="base", node=BaseLossConfig)
    cs.store(group="policy/loss", name="gripper", node=GripperLossConfig)
    cs.store(group="policy/loss", name="entropy", node=GaussianEntropyLossConfig)
    cs.store(group="policy/loss", name="kl", node=KLDivergenceLossConfig)
    cs.store(group="policy/loss", name="vic_latent", node=VICLatentLossConfig)
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
        name="gripper_mixture_nll",
        node=GripperMixtureNLLossConfig,
    )
    cs.store(
        group="policy/loss",
        name="denoising_prior",
        node=PriorDenoisingLossConfig,
    )
    cs.store(
        group="policy/loss",
        name="gaussian_mixture_nll",
        node=GaussianMixtureNLLossConfig,
    )
    cs.store(
        group="policy/loss",
        name="optimal_transport",
        node=OptimalTransportLossConfig,
    )
    cs.store(
        group="policy/loss",
        name="latent_optimal_transport",
        node=LatentOptimalTransportLossConfig,
    )
    cs.store(
        group="policy/encoding_pipeline",
        name="base",
        node=EncodingPipelineConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder", name="image", node=ImageEncoderConfig
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/image",
        name="spatial",
        node=SpatialRGBEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/image",
        name="conditional_cnn",
        node=ConditionalCNNEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/image",
        name="flat",
        node=FlatRGBEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/vlm",
        name="two_tower_vlm",
        node=TwoTowerVLMEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/vlm",
        name="paligemma",
        node=PaliGemmaEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/vlm",
        name="smolvlm",
        node=SmolVLMEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="depth_spatial",
        node=SpatialDepthEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="dformer",
        node=DFormerEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder",
        name="geometric_rgbd",
        node=GeometricRGBDEncoderConfig,
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

    cs.store(group="policy/decoder", name="gpt", node=GPTActionTransformerConfig)
    cs.store(
        group="policy/decoder",
        name="discrete_detr",
        node=DiscreteDETRActionTransformerConfig,
    )
    cs.store(
        group="policy/decoder",
        name="free_transformer",
        node=FreeActionTransformerConfig,
    )
    cs.store(
        group="policy/decoder",
        name="moe_free_transformer",
        node=MoEFreeActionTransformerConfig,
    )
    cs.store(group="policy/decoder", name="moe", node=MixtureOfExpertsDecoderConfig)
    cs.store(group="policy/decoder", name="lact_decoder", node=LACTConfig)
    cs.store(
        group="policy/decoder", name="dit_block", node=DiTBlockActionTransformerConfig
    )
    cs.store(
        group="policy/decoder",
        name="diffusion_act",
        node=DiffusionActionTransformerConfig,
    )
    cs.store(group="policy/decoder", name="unet", node=ConditionalActionUNetConfig)
    cs.store(group="policy/decoder", name="smolvla", node=SmolVLADecoderConfig)
    cs.store(group="policy/decoder", name="pi0", node=Pi0DecoderConfig)
    cs.store(group="policy/decoder/action_head", name="base", node=ActionHeadConfig)
    cs.store(
        group="policy/decoder/action_head", name="gaussian", node=GaussianHeadConfig
    )
    cs.store(
        group="policy/decoder/action_head", name="moe", node=MixtureOfExpertsHeadConfig
    )
    cs.store(
        group="policy/decoder",
        name="mode_act",
        node=MixtureOfDensitiesActionTransformerConfig,
    )
    cs.store(group="policy/decoder/head_block", name="base", node=ActionHeadBlockConfig)
    cs.store(group="policy/decoder/head_block", name="mlp", node=MLPBlockConfig)
    cs.store(
        group="policy/decoder/head_block", name="attention", node=AttentionBlockConfig
    )
    cs.store(
        group="policy/decoder/head_block", name="residual", node=ResidualBlockConfig
    )

    cs.store(
        name="post_training_compression",
        node=PostTrainingCompressorConfig,
    )
    cs.store(
        group="compression/module",
        name="base",
        node=CompressionTargetConfig,
    )
    cs.store(
        group="compression/pruning",
        name="unstructured",
        node=UnstructuredPrunerConfig,
    )
    cs.store(
        group="compression/pruning",
        name="structured",
        node=StructuredPrunerConfig,
    )
    cs.store(
        group="quantization/strategy",
        name="pt2e",
        node=PT2EStrategyConfig,
    )
    cs.store(
        group="quantization/strategy",
        name="quantize_api",
        node=QuantizeApiStrategyConfig,
    )
    cs.store(
        group="quantization/backend",
        name="x86_inductor",
        node=X86InductorBackendConfig,
    )
    cs.store(
        group="quantization/quantize_config",
        name="int8_dynamic",
        node=Int8DynamicQuantizeConfig,
    )
    cs.store(
        group="quantization/quantize_config",
        name="int4_weight_only",
        node=Int4WeightOnlyQuantizeConfig,
    )


# Register resolvers on module import
register_resolvers()
register_configs()
