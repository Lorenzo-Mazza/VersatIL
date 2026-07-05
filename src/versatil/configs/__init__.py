"""Configuration and resolver centralized store for OmegaConf."""

import os
from pathlib import Path

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from versatil_constants.tso import TSOObsKey

from versatil.common.set_cache_dir import resolve_cache_directory
from versatil.configs.adaptation import LoRAAdaptationConfig
from versatil.configs.data.augmentations import AugmentationPipelineConfig
from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.data.metadata import (
    CameraMetadataConfig,
    DepthCameraMetadataConfig,
    GripperActionMetadataConfig,
    GripperObservationMetadataConfig,
    ObservationMetadataConfig,
    OrientationActionMetadataConfig,
    OrientationObservationMetadataConfig,
    PositionActionMetadataConfig,
    PositionObservationMetadataConfig,
    PrecomputedActionMetadataConfig,
    RGBCameraMetadataConfig,
)
from versatil.configs.data.raw import (
    CsvDatasetSchemaConfig,
    DatasetMetadataConfig,
    DatasetSchemaConfig,
    Hdf5DatasetSchemaConfig,
    LeRobotDatasetSchemaConfig,
    SyntheticDatasetSchemaConfig,
)
from versatil.configs.data.task import (
    ActionSpaceConfig,
    ObservationSpaceConfig,
    TaskSpaceConfig,
)
from versatil.configs.data.tokenizer import (
    ActionDiscretizerConfig,
    ActionTokenIdMappingConfig,
    ActionTokenizationConfig,
    ObservationTokenizationConfig,
    TokenizationConfig,
)
from versatil.configs.decoding.action_head import (
    ActionHeadBlockConfig,
    ActionHeadConfig,
    AdaNormBlockConfig,
    AttentionBlockConfig,
    ConditionalActionHeadConfig,
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
    AutoregressiveVLAConfig,
    ConditionalActionUNetConfig,
    DecodingNetworkConfig,
    DiffusionActionTransformerConfig,
    DiTBlockActionTransformerConfig,
    GPTActionTransformerConfig,
    LACTConfig,
    MixtureOfDensitiesActionTransformerConfig,
    MixtureOfExpertsDecoderConfig,
    OpenVLAOFTConfig,
    PhaseACTConfig,
    Pi0DecoderConfig,
    SmolVLADecoderConfig,
)
from versatil.configs.decoding.latent import (
    CodebookPriorConfig,
    DiTPriorConfig,
    GaussianPriorConfig,
    PosteriorLatentEncoderConfig,
    PriorLatentEncoderConfig,
    PriorTransformerEncoderConfig,
    UniformCodebookPriorConfig,
    VAETransformerEncoderConfig,
    VampPriorConfig,
    VQPosteriorEncoderConfig,
)
from versatil.configs.deployment import DeploymentConfig
from versatil.configs.encoding.encoder import (
    ConditionalCNNEncoderConfig,
    DFormerEncoderConfig,
    DinoV2SigLIPRGBEncoderConfig,
    EncoderConfig,
    FlatRGBEncoderConfig,
    GeometricRGBDEncoderConfig,
    ImageEncoderConfig,
    LanguageEncoderConfig,
    ProprioEncoderConfig,
    SpatialDepthEncoderConfig,
    SpatialRGBEncoderConfig,
    VLMEncoderConfig,
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
from versatil.configs.explainability import (
    ExplainabilityConfig,
    ExplanationWriterConfig,
)
from versatil.configs.inference_client import InferenceClientConfig
from versatil.configs.loss import (
    ActionTokenLossConfig,
    BaseLossConfig,
    BinaryKLDivergenceLossConfig,
    BinaryMaximumMeanDiscrepancyLossConfig,
    CompositeLossConfig,
    ConditionalMaximumMeanDiscrepancyLossConfig,
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
    PosteriorGeometryLossConfig,
    PriorDenoisingLossConfig,
    RegressionLossConfig,
    RelaxedConditionalLatentOptimalTransportLossConfig,
    TrajectoryLengthLossConfig,
    TrajectorySmoothnessConfig,
    VICLatentLossConfig,
    VQCommitmentLossConfig,
    VQPriorCrossEntropyLossConfig,
)
from versatil.configs.main import MainConfig
from versatil.configs.policy import PolicyConfig
from versatil.configs.post_training_compression import (
    BasePrunerConfig,
    CompressionTargetConfig,
    ExecutorchXNNPACKBackendConfig,
    PostTrainingCompressorConfig,
    PreparationConfig,
    StructuredPrunerConfig,
    TorchInductorBackendConfig,
    UnstructuredPrunerConfig,
)
from versatil.configs.quantization import (
    BasePT2EBackendConfig,
    EagerQuantizationModuleTargetConfig,
    EagerQuantizationWorkflowConfig,
    Int4WeightOnlyQuantizeConfig,
    Int8DynamicQuantizeConfig,
    PT2EQuantizationModuleTargetConfig,
    PT2EQuantizationWorkflowConfig,
    X86InductorBackendConfig,
    XNNPACKPT2EBackendConfig,
)
from versatil.configs.training import (
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    SGDConfig,
    TrainingConfig,
    TrainingStageConfig,
)
from versatil.data.constants import (
    ActionComputationMethod,
    ActionDiscretizerType,
    ActionTokenIdMappingType,
    BinaryGripperRange,
    BinningStrategy,
    Cameras,
    CoordinateSystem,
    DatasetType,
    GripperType,
    ImageNormalizationType,
    KinematicsNormalizationType,
    MetadataPassthroughSource,
    ObsKey,
    OrientationRepresentation,
    ProprioKey,
    RawCameraKey,
    SampleKey,
    SyntheticObsKey,
    TokenizerType,
    TokenPaddingStrategy,
)
from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.metrics.constants import MetadataKey
from versatil.metrics.kernels import KernelType
from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.decoding.constants import (
    DenoisingAlgorithm,
    DiTType,
    GMMInitStrategy,
    LatentKey,
    MixtureSamplingMode,
    MoERoutingType,
    TimeConditioning,
)
from versatil.models.decoding.generative_language_models.constants import (
    PRISMATIC_LLM_BACKBONES,
    PaliGemmaModelType,
    PrismaticLLMBackboneType,
    PrismaticModelType,
    SmolVLMModelType,
)
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    ImageTextModelType,
    LanguageEncoderType,
    PoolingMethod,
    RGBBackboneType,
)
from versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2 import (
    DFormerPretrainedWeights,
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
from versatil.quantization.constants import PT2EBackendName
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
    "TrainingStageConfig",
    "TaskSpaceConfig",
    "PolicyConfig",
    "LoRAAdaptationConfig",
    "EncoderConfig",
    "ImageEncoderConfig",
    "SpatialDepthEncoderConfig",
    "SpatialRGBEncoderConfig",
    "FlatRGBEncoderConfig",
    "DinoV2SigLIPRGBEncoderConfig",
    "ProprioEncoderConfig",
    "LanguageEncoderConfig",
    "DecodingNetworkConfig",
    "ACTConfig",
    "ConditionalActionUNetConfig",
    "AutoregressiveVLAConfig",
    "OpenVLAOFTConfig",
    "Pi0DecoderConfig",
    "SmolVLADecoderConfig",
    "DiTBlockActionTransformerConfig",
    "DiffusionActionTransformerConfig",
    "LACTConfig",
    "MixtureOfExpertsDecoderConfig",
    "DeploymentConfig",
    "ExplanationWriterConfig",
    "InferenceClientConfig",
    "ExplainabilityConfig",
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
    "ConditionalActionHeadConfig",
    "MixtureOfExpertsHeadConfig",
    "ActionHeadBlockConfig",
    "AdaNormBlockConfig",
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
    "PosteriorGeometryLossConfig",
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
    "EagerQuantizationModuleTargetConfig",
    "Int4WeightOnlyQuantizeConfig",
    "Int8DynamicQuantizeConfig",
    "PT2EQuantizationModuleTargetConfig",
    "CompressionTargetConfig",
    "EagerQuantizationWorkflowConfig",
    "PT2EQuantizationWorkflowConfig",
    "PostTrainingCompressorConfig",
    "PreparationConfig",
    "StructuredPrunerConfig",
    "UnstructuredPrunerConfig",
    "X86InductorBackendConfig",
    "XNNPACKPT2EBackendConfig",
    "SyntheticDatasetSchemaConfig",
]


def _stage_split_epoch(num_epochs: int | float | str, fraction: float | str) -> int:
    """Resolve a valid epoch boundary for two-stage training configs.

    Args:
        num_epochs: Total number of configured training epochs.
        fraction: Fraction of training budget assigned before the split.

    Returns:
        Integer epoch where the second stage should start.

    Raises:
        ValueError: If ``num_epochs`` is not positive or ``fraction`` is outside
            the open interval ``(0, 1)``.
    """
    total_epochs = int(float(num_epochs))
    split_fraction = float(fraction)
    if total_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}.")
    if split_fraction <= 0.0 or split_fraction >= 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}.")
    if total_epochs == 1:
        return 1
    split_epoch = int(total_epochs * split_fraction)
    return min(max(split_epoch, 1), total_epochs - 1)


def _multiply_resolver(
    left: int | float | str, right: int | float | str
) -> int | float:
    """Multiply numeric OmegaConf resolver inputs.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        Integer product for integer operands, otherwise floating-point product.
    """
    left_value = float(left)
    right_value = float(right)
    product = left_value * right_value
    if _is_integer_resolver_value(value=left) and _is_integer_resolver_value(
        value=right
    ):
        return int(product)
    return product


def _integer_multiply_resolver(
    left: int | float | str, right: int | float | str
) -> int:
    """Multiply numeric OmegaConf resolver inputs and return an integer."""
    return int(float(left) * float(right))


def _is_integer_resolver_value(value: int | float | str) -> bool:
    """Return whether a resolver input represents an integer value."""
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    normalized_value = value.strip()
    if normalized_value.startswith("-"):
        normalized_value = normalized_value[1:]
    return normalized_value.isdecimal()


def _action_space_prediction_dimension_resolver(action_space: DictConfig) -> int:
    """Return total predicted action dimension from an action-space config.

    Args:
        action_space: Action-space config containing ``actions_metadata``.

    Returns:
        Total prediction dimension for actions that require prediction heads.
    """
    total_dimension = 0
    for metadata in action_space.actions_metadata.values():
        if metadata.get("requires_prediction_head", True):
            total_dimension += int(metadata.prediction_dimension)
    return total_dimension


def register_resolvers():
    """Register custom OmegaConf resolvers for enum access in YAML configs.

    This allows using ${cameras:LEFT} in YAML to get Cameras.LEFT.value.
    """
    if not OmegaConf.has_resolver("cameras"):
        OmegaConf.register_resolver("cameras", lambda name: Cameras[name].value)
    if not OmegaConf.has_resolver("raw_camera"):
        OmegaConf.register_resolver("raw_camera", lambda name: RawCameraKey[name].value)
    if not OmegaConf.has_resolver("gripper"):
        OmegaConf.register_resolver("gripper", lambda name: GripperType[name].value)
    if not OmegaConf.has_resolver("orientation"):
        OmegaConf.register_resolver(
            "orientation", lambda name: OrientationRepresentation[name].value
        )
    if not OmegaConf.has_resolver("action_computation"):
        OmegaConf.register_resolver(
            "action_computation", lambda name: ActionComputationMethod[name].value
        )
    if not OmegaConf.has_resolver("rgb_backbone"):
        OmegaConf.register_resolver(
            "rgb_backbone", lambda name: RGBBackboneType[name].value
        )
    if not OmegaConf.has_resolver("batch_norm_handling"):
        OmegaConf.register_resolver(
            "batch_norm_handling", lambda name: BatchNormHandling[name].value
        )
    if not OmegaConf.has_resolver("precision"):
        OmegaConf.register_resolver("precision", lambda name: PrecisionType[name].value)
    if not OmegaConf.has_resolver("lora_target_modules"):
        OmegaConf.register_resolver(
            "lora_target_modules", lambda name: LoRATargetModulePreset[name].value
        )
    if not OmegaConf.has_resolver("float32_matmul"):
        OmegaConf.register_resolver(
            "float32_matmul", lambda name: Float32MatmulPrecision[name].value
        )
    if not OmegaConf.has_resolver("pooling_method"):
        OmegaConf.register_resolver(
            "pooling_method", lambda name: PoolingMethod[name].value
        )
    if not OmegaConf.has_resolver("language_model"):
        OmegaConf.register_resolver(
            "language_model", lambda name: LanguageEncoderType[name].value
        )
    if not OmegaConf.has_resolver("vlm_model"):
        OmegaConf.register_resolver(
            "vlm_model", lambda name: ImageTextModelType[name].value
        )
    if not OmegaConf.has_resolver("smolvlm_model"):
        OmegaConf.register_resolver(
            "smolvlm_model", lambda name: SmolVLMModelType[name].value
        )
    if not OmegaConf.has_resolver("paligemma_model"):
        OmegaConf.register_resolver(
            "paligemma_model", lambda name: PaliGemmaModelType[name].value
        )
    if not OmegaConf.has_resolver("prismatic_model"):
        OmegaConf.register_resolver(
            "prismatic_model", lambda name: PrismaticModelType[name].value
        )
    if not OmegaConf.has_resolver("prismatic_llm_model"):
        OmegaConf.register_resolver(
            "prismatic_llm_model",
            lambda name: PRISMATIC_LLM_BACKBONES[PrismaticLLMBackboneType[name]],
        )
    if not OmegaConf.has_resolver("activation_function"):
        OmegaConf.register_resolver(
            "activation_function", lambda name: ActivationFunction[name].value
        )
    if not OmegaConf.has_resolver("normalization"):
        OmegaConf.register_resolver(
            "normalization", lambda name: NormalizationType[name].value
        )
    if not OmegaConf.has_resolver("attention"):
        OmegaConf.register_resolver("attention", lambda name: AttentionType[name].value)
    if not OmegaConf.has_resolver("pos_encoding"):
        OmegaConf.register_resolver(
            "pos_encoding", lambda name: PositionalEncodingType[name].value
        )
    if not OmegaConf.has_resolver("tokenizer_type"):
        OmegaConf.register_resolver(
            "tokenizer_type", lambda name: TokenizerType[name].value
        )
    if not OmegaConf.has_resolver("action_discretizer"):
        OmegaConf.register_resolver(
            "action_discretizer", lambda name: ActionDiscretizerType[name].value
        )
    if not OmegaConf.has_resolver("binning_strategy"):
        OmegaConf.register_resolver(
            "binning_strategy", lambda name: BinningStrategy[name].value
        )
    if not OmegaConf.has_resolver("dformer_weights"):
        OmegaConf.register_resolver(
            "dformer_weights", lambda name: DFormerPretrainedWeights[name].value
        )
    if not OmegaConf.has_resolver("action_token_id_mapping"):
        OmegaConf.register_resolver(
            "action_token_id_mapping",
            lambda name: ActionTokenIdMappingType[name].value,
        )
    if not OmegaConf.has_resolver("kinematics_norm_type"):
        OmegaConf.register_resolver(
            "kinematics_norm_type", lambda name: KinematicsNormalizationType[name].value
        )
    if not OmegaConf.has_resolver("image_norm_type"):
        OmegaConf.register_resolver(
            "image_norm_type", lambda name: ImageNormalizationType[name].value
        )
    if not OmegaConf.has_resolver("obs_key"):
        OmegaConf.register_resolver("obs_key", lambda name: ObsKey[name].value)
    if not OmegaConf.has_resolver("sample_key"):
        OmegaConf.register_resolver("sample_key", lambda name: SampleKey[name].value)
    if not OmegaConf.has_resolver("moe_routing_type"):
        OmegaConf.register_resolver(
            "moe_routing_type", lambda name: MoERoutingType[name].value
        )
    if not OmegaConf.has_resolver("coordinate_system"):
        OmegaConf.register_resolver(
            "coordinate_system", lambda name: CoordinateSystem[name].value
        )
    if not OmegaConf.has_resolver("gripper_range"):
        OmegaConf.register_resolver(
            "gripper_range", lambda name: BinaryGripperRange[name].value
        )
    if not OmegaConf.has_resolver("proprio_key"):
        OmegaConf.register_resolver("proprio_key", lambda name: ProprioKey[name].value)
    if not OmegaConf.has_resolver("latent_key"):
        OmegaConf.register_resolver("latent_key", lambda name: LatentKey[name].value)
    if not OmegaConf.has_resolver("scheduler_type"):
        OmegaConf.register_resolver(
            "scheduler_type", lambda name: SchedulerType[name].value
        )
    if not OmegaConf.has_resolver("denoising_algorithm"):
        OmegaConf.register_resolver(
            "denoising_algorithm", lambda name: DenoisingAlgorithm[name].value
        )
    if not OmegaConf.has_resolver("conditioning_type"):
        OmegaConf.register_resolver(
            "conditioning_type", lambda name: ConditioningType[name].value
        )
    if not OmegaConf.has_resolver("metadata_key"):
        OmegaConf.register_resolver(
            "metadata_key", lambda name: MetadataKey[name].value
        )
    if not OmegaConf.has_resolver("metadata_passthrough_source"):
        OmegaConf.register_resolver(
            "metadata_passthrough_source",
            lambda name: MetadataPassthroughSource[name].value,
        )
    if not OmegaConf.has_resolver("dit_type"):
        OmegaConf.register_resolver("dit_type", lambda name: DiTType[name].value)
    if not OmegaConf.has_resolver("time_conditioning"):
        OmegaConf.register_resolver(
            "time_conditioning", lambda name: TimeConditioning[name].value
        )
    if not OmegaConf.has_resolver("timestep_sampler"):
        OmegaConf.register_resolver(
            "timestep_sampler", lambda name: TimestepSampler[name].value
        )
    if not OmegaConf.has_resolver("dataset_type"):
        OmegaConf.register_resolver(
            "dataset_type", lambda name: DatasetType[name].value
        )
    if not OmegaConf.has_resolver("kernel_type"):
        OmegaConf.register_resolver("kernel_type", lambda name: KernelType[name].value)
    if not OmegaConf.has_resolver("mixture_sampling"):
        OmegaConf.register_resolver(
            "mixture_sampling", lambda name: MixtureSamplingMode[name].value
        )
    if not OmegaConf.has_resolver("gmm_init"):
        OmegaConf.register_resolver(
            "gmm_init", lambda name: GMMInitStrategy[name].value
        )
    if not OmegaConf.has_resolver("token_padding"):
        OmegaConf.register_resolver(
            "token_padding", lambda name: TokenPaddingStrategy[name].value
        )
    if not OmegaConf.has_resolver("synthetic_task"):
        OmegaConf.register_resolver(
            "synthetic_task", lambda name: SyntheticTaskName[name].value
        )
    if not OmegaConf.has_resolver("synthetic_obs_key"):
        OmegaConf.register_resolver(
            "synthetic_obs_key", lambda name: SyntheticObsKey[name].value
        )
    if not OmegaConf.has_resolver("tso_obs_key"):
        OmegaConf.register_resolver("tso_obs_key", lambda name: TSOObsKey[name].value)

    if not OmegaConf.has_resolver("compile_mode"):
        OmegaConf.register_resolver(
            "compile_mode", lambda name: CompileMode[name].value
        )
    if not OmegaConf.has_resolver("quantization_backend"):
        OmegaConf.register_resolver(
            "quantization_backend", lambda name: PT2EBackendName[name].value
        )
    if not OmegaConf.has_resolver("torch_dtype"):
        OmegaConf.register_resolver("torch_dtype", lambda name: getattr(torch, name))
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_resolver(
            "env", lambda key, default=None: os.environ.get(key, default)
        )
    if not OmegaConf.has_resolver("dataset_dir"):
        OmegaConf.register_resolver(
            "dataset_dir",
            lambda env_var, subpath="": str(
                Path(os.environ.get(env_var, ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("checkpoint_dir"):
        OmegaConf.register_resolver(
            "checkpoint_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_CHECKPOINT_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("zarr_dir"):
        OmegaConf.register_resolver(
            "zarr_dir",
            lambda subpath="": str(
                Path(os.environ.get("VERSATIL_ZARR_DIR", ".")) / subpath
            ),
        )
    if not OmegaConf.has_resolver("cache_dir"):
        OmegaConf.register_resolver(
            "cache_dir",
            lambda: str(resolve_cache_directory()),
        )
    if not OmegaConf.has_resolver("prunable_layer"):
        OmegaConf.register_resolver(
            "prunable_layer",
            lambda name: PrunableLayerType[name].value,
        )
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_resolver(
            "mul",
            _multiply_resolver,
        )
    if not OmegaConf.has_resolver("int_mul"):
        OmegaConf.register_resolver(
            "int_mul",
            _integer_multiply_resolver,
        )
    if not OmegaConf.has_resolver("action_space_prediction_dimension"):
        OmegaConf.register_resolver(
            "action_space_prediction_dimension",
            _action_space_prediction_dimension_resolver,
        )
    if not OmegaConf.has_resolver("stage_split_epoch"):
        OmegaConf.register_resolver("stage_split_epoch", _stage_split_epoch)


def register_configs() -> None:
    """Register Hydra config groups in the global ConfigStore.

    Registrations live in per-domain modules under ``versatil.configs.store``;
    they import config classes from this package, so they are imported here
    at call time, after the package is fully initialized.
    """
    from versatil.configs.store import (  # noqa: PLC0415
        policy,
        quantization,
        root,
        task,
        training,
    )

    cs = ConfigStore.instance()
    root.register(cs)
    task.register(cs)
    policy.register(cs)
    training.register(cs)
    quantization.register(cs)


# Register resolvers on module import
register_resolvers()
register_configs()
