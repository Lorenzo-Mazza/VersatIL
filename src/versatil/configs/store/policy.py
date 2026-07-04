"""Policy, encoder, decoder, algorithm, and loss config registrations."""

from hydra.core.config_store import ConfigStore

from versatil.configs import (
    ACTConfig,
    ActionHeadBlockConfig,
    ActionHeadConfig,
    ActionTokenLossConfig,
    ActionTransformerConfig,
    AdaNormBlockConfig,
    AttentionBlockConfig,
    AttentionFusionConfig,
    AutoregressiveVLAConfig,
    BaseLossConfig,
    BehavioralCloningConfig,
    BinaryKLDivergenceLossConfig,
    BinaryMaximumMeanDiscrepancyLossConfig,
    CodebookPriorConfig,
    CompositeLossConfig,
    ConcatFusionConfig,
    ConditionalActionHeadConfig,
    ConditionalActionUNetConfig,
    ConditionalCNNEncoderConfig,
    ConditionalMaximumMeanDiscrepancyLossConfig,
    DecodingAlgorithmConfig,
    DecodingNetworkConfig,
    DFormerEncoderConfig,
    DiffusionActionTransformerConfig,
    DiffusionConfig,
    DinoV2SigLIPRGBEncoderConfig,
    DiTBlockActionTransformerConfig,
    DiTPriorConfig,
    EncodingPipelineConfig,
    FlatRGBEncoderConfig,
    FlowMatchingConfig,
    FusionConfig,
    GaussianEntropyLossConfig,
    GaussianHeadConfig,
    GaussianMixtureNLLossConfig,
    GaussianPriorConfig,
    GeometricRGBDEncoderConfig,
    GPTActionTransformerConfig,
    GripperLossConfig,
    GripperMixtureNLLossConfig,
    ImageEncoderConfig,
    KLDivergenceLossConfig,
    LACTConfig,
    LanguageEncoderConfig,
    LatentOptimalTransportLossConfig,
    LoRAAdaptationConfig,
    MaximumMeanDiscrepancyLossConfig,
    MixtureOfDensitiesActionTransformerConfig,
    MixtureOfExpertsDecoderConfig,
    MixtureOfExpertsHeadConfig,
    MLPBlockConfig,
    MLPFusionConfig,
    MoELossConfig,
    OpenVLAOFTConfig,
    OptimalTransportLossConfig,
    PhaseACTConfig,
    PhaseClassificationLossConfig,
    Pi0DecoderConfig,
    PolicyConfig,
    PosteriorGeometryLossConfig,
    PosteriorLatentEncoderConfig,
    PriorDenoisingLossConfig,
    PriorLatentEncoderConfig,
    PriorTransformerEncoderConfig,
    ProprioEncoderConfig,
    RegressionLossConfig,
    RelaxedConditionalLatentOptimalTransportLossConfig,
    ResidualBlockConfig,
    SmolVLADecoderConfig,
    SpatialDepthEncoderConfig,
    SpatialFusionConfig,
    SpatialRGBEncoderConfig,
    TrajectoryLengthLossConfig,
    TrajectorySmoothnessConfig,
    UniformCodebookPriorConfig,
    VAETransformerEncoderConfig,
    VampPriorConfig,
    VariationalAlgorithmConfig,
    VICLatentLossConfig,
    VLMEncoderConfig,
    VQCommitmentLossConfig,
    VQPosteriorEncoderConfig,
    VQPriorCrossEntropyLossConfig,
)


def register(cs: ConfigStore) -> None:
    """Store this domain's config nodes.

    Args:
        cs: The global Hydra config store.
    """
    cs.store(group="policy", name="base", node=PolicyConfig)
    cs.store(group="policy/adaptation/lora", name="base", node=LoRAAdaptationConfig)
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
        group="policy/algorithm/posterior",
        name="vq_encoder",
        node=VQPosteriorEncoderConfig,
    )
    cs.store(
        group="policy/algorithm/prior",
        name="uniform_codebook",
        node=UniformCodebookPriorConfig,
    )
    cs.store(
        group="policy/algorithm/prior",
        name="codebook",
        node=CodebookPriorConfig,
    )
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
    cs.store(
        group="policy/loss",
        name="posterior_geometry",
        node=PosteriorGeometryLossConfig,
    )
    cs.store(group="policy/loss", name="vq_commitment", node=VQCommitmentLossConfig)
    cs.store(
        group="policy/loss",
        name="vq_prior_ce",
        node=VQPriorCrossEntropyLossConfig,
    )
    cs.store(group="policy/loss", name="mmd", node=MaximumMeanDiscrepancyLossConfig)
    cs.store(
        group="policy/loss",
        name="conditional_mmd",
        node=ConditionalMaximumMeanDiscrepancyLossConfig,
    )
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
        group="policy/loss",
        name="relaxed_conditional_latent_ot",
        node=RelaxedConditionalLatentOptimalTransportLossConfig,
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
        group="policy/encoding_pipeline/encoder/image",
        name="dinov2_siglip",
        node=DinoV2SigLIPRGBEncoderConfig,
    )
    cs.store(
        group="policy/encoding_pipeline/encoder/vlm",
        name="vlm_encoder",
        node=VLMEncoderConfig,
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
        name="autoregressive_vla_config",
        node=AutoregressiveVLAConfig,
    )
    cs.store(
        group="policy/decoder",
        name="openvla_oft_config",
        node=OpenVLAOFTConfig,
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
        group="policy/decoder/action_head",
        name="conditional",
        node=ConditionalActionHeadConfig,
    )
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
    cs.store(group="policy/decoder/head_block", name="adanorm", node=AdaNormBlockConfig)
