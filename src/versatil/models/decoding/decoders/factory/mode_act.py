"""Mixture of Densities Action Transformer (MODE-ACT) for multi-modal action prediction.

This module implements a Mixture Density Network style transformer decoder that predicts
multiple mixture components for each action, enabling multi-modal action distributions.
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_heads.gaussian import GaussianHead
from versatil.models.decoding.constants import (
    DecoderOutputKey,
    GMMInitStrategy,
    MixtureSamplingMode,
)
from versatil.models.decoding.decoders import DecoderInput
from versatil.models.decoding.decoders.parallel_transformer import (
    BaseParallelTransformerDecoder,
)
from versatil.models.feature_meta import FeatureType
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.mlp import MLP
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.bidirectional_decoder import (
    BidirectionalDecoder,
)


class MixtureOfDensitiesActionTransformer(BaseParallelTransformerDecoder):
    """Mixture Density Network Transformer for multi-modal action prediction.

    Note:
        This architecture combines a transformer decoder with K expert action heads
        to predict mixture density parameters. For the mixture weight computation,
        either a mode learnable query token or an external feature token are used.
    """

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        action_heads: dict[str, ActionHead],
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        number_of_layers: int = 6,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        num_mixture_components: int = 8,
        gating_hidden_dims: list[int] | None = None,
        gating_activation: str = ActivationFunction.SILU.value,
        gating_dropout: float = 0.1,
        gating_normalization: bool = True,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_feature_key: str | None = None,
        gmm_init_strategy: str = GMMInitStrategy.KMEANS_PLUS_PLUS.value,
        inference_sampling_mode: str = MixtureSamplingMode.STOCHASTIC_MEAN.value,
    ) -> None:
        """Initialize MODE-ACT decoder.

        Args:
            input_keys: Feature keys from encoding pipeline.
            action_space: Action space configuration.
            action_heads: Base action heads to clone K times.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of action timesteps to predict.
            device: Device to place model on.
            embedding_dimension: Transformer embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of key-value heads for GQA.
            feedforward_dimension: FFN hidden dimension.
            number_of_layers: Number of decoder layers.
            activation: Activation function.
            normalization_type: Normalization type.
            attention_type: Attention type.
            dropout_rate: Dropout rate.
            attention_dropout: Attention dropout rate.
            positional_encoding_type: Positional encoding type.
            num_mixture_components: Number of mixture components (K).
            gating_hidden_dims: Hidden dimensions for gating MLP.
            gating_activation: Activation for gating MLP.
            gating_dropout: Dropout rate in gating MLP.
            gating_normalization: Whether to normalize gating input.
            temperature: Temperature for softmax scaling.
            learnable_temperature: Whether temperature is learnable.
            gating_feature_key: If set, use this feature for gating instead of mode embedding.
            gmm_init_strategy: Strategy for initializing GMM component means.
            inference_sampling_mode: How to sample from the mixture at inference.
                DETERMINISTIC: argmax component, return mean.
                STOCHASTIC_MEAN: multinomial component, return mean (no noise).
                STOCHASTIC_SAMPLE: multinomial component, add Gaussian noise.
        """
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.SPATIAL.value],
            requires_actions=False,
        )
        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        self.number_of_layers = number_of_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.feedforward_dimension = feedforward_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.positional_encoding_type = positional_encoding_type
        self.num_mixture_components = num_mixture_components
        self.gating_feature_key = gating_feature_key
        if gmm_init_strategy not in [s.value for s in GMMInitStrategy]:
            raise ValueError(
                f"Unknown gmm_init_strategy: {gmm_init_strategy}. "
                f"Expected one of {[s.value for s in GMMInitStrategy]}"
            )
        self.gmm_init_strategy = gmm_init_strategy
        if inference_sampling_mode not in [m.value for m in MixtureSamplingMode]:
            raise ValueError(
                f"Unknown inference_sampling_mode: {inference_sampling_mode}. "
                f"Expected one of {[m.value for m in MixtureSamplingMode]}"
            )
        self.inference_sampling_mode = inference_sampling_mode
        self.action_keys = list(self.action_heads.keys())
        self._build_transformer_components()
        self._build_mixture_heads()
        self._build_gating_network(
            input_dimension=embedding_dimension,
            hidden_dimensions=gating_hidden_dims,
            activation=gating_activation,
            dropout=gating_dropout,
            normalization=gating_normalization,
        )

        if learnable_temperature:
            self.temperature = nn.Parameter(
                torch.tensor(temperature, dtype=torch.float32), requires_grad=True
            )
        else:
            self.register_buffer(
                "temperature", torch.tensor(temperature, dtype=torch.float32)
            )

        self.to(self.device)

    def get_auxiliary_output_keys(self) -> set[str]:
        """MoDEACT produces routing weights for mixture density prediction."""
        keys = super().get_auxiliary_output_keys()
        keys.add(DecoderOutputKey.ROUTING_WEIGHTS.value)
        return keys

    def _build_transformer_components(self) -> None:
        """Build core transformer encoder-decoder and positional encodings."""
        self.input_sequence_builder = self._build_parallel_input_sequence_builder()
        self.action_queries = nn.Parameter(
            torch.randn(self.prediction_horizon, self.embedding_dimension)
        )  # (prediction_horizon, embedding_dimension)
        self.mode_query = nn.Parameter(
            torch.randn(1, self.embedding_dimension)
        )  # (1, embedding_dimension)
        self.action_decoder = BidirectionalDecoder(
            number_of_layers=self.number_of_layers,
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.number_of_heads,
            number_of_key_value_heads=self.number_of_key_value_heads,
            feedforward_dimension=self.feedforward_dimension,
            dropout=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            activation=self.activation,
            normalization_type=self.normalization_type,
            attention_type=self.attention_type,
            positional_encoding_type=self.positional_encoding_type,
        )

    def _build_mixture_heads(self) -> None:
        """Clone each action head K times for mixture components.

        Note:
            Gaussian clones get a per-timestep temporal bias so a bias-only
            initialization can hold a full trajectory per component.
        """
        self.mixture_heads: nn.ModuleDict = nn.ModuleDict()
        for action_key, head in self.action_heads.items():
            cloned_heads = []
            for _ in range(self.num_mixture_components):
                cloned_head = copy.deepcopy(head)
                for module in cloned_head.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
                if isinstance(cloned_head, GaussianHead):
                    cloned_head.enable_temporal_bias(horizon=self.prediction_horizon)
                cloned_heads.append(cloned_head)
            self.mixture_heads[action_key] = nn.ModuleList(cloned_heads)

    def _build_gating_network(
        self,
        input_dimension: int,
        hidden_dimensions: list[int] | None,
        activation: str,
        dropout: float,
        normalization: bool,
    ) -> None:
        """Build gating MLP for computing mixture weights.

        Args:
            input_dimension: Input feature dimension.
            hidden_dimensions: List of hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout rate.
            normalization: Whether to apply layer normalization before MLP.
        """
        if hidden_dimensions is None or len(hidden_dimensions) == 0:
            hidden_dimensions = [input_dimension // 2]

        layers: list[nn.Module] = []
        if normalization:
            layers.append(nn.LayerNorm(input_dimension))

        gating_mlp = MLP(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dim=self.num_mixture_components,
            activation_function=ActivationFunction(activation).to_torch_activation(),
            dropout=dropout,
        )
        layers.append(gating_mlp)
        self.gating_network = nn.Sequential(*layers)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Set normalizer and initialize GMM components from output statistics.

        Note:
            Initialization always runs in chunk space; the strategy only
            selects how the trajectory centers are spread (k-means++ over the
            candidate chunks or uniform envelope interpolation). Candidates
            are the demonstrated chunk subsamples carried by the normalizer,
            or constant trajectories sampled uniformly from the normalized
            action range when no subsample is available.

        Args:
            normalizer: Normalizer with fitted statistics.
        """
        super().set_normalizer(normalizer)
        self._initialize_gating_network()
        gaussian_keys = [
            action_key
            for action_key in self.action_keys
            if action_key in normalizer.params_dict
            and isinstance(self.action_heads[action_key], GaussianHead)
        ]
        if not gaussian_keys:
            return
        chunk_samples = self._collect_chunk_samples(
            normalizer=normalizer, action_keys=gaussian_keys
        )
        if chunk_samples is None:
            chunk_samples = self._constant_chunk_candidates(
                normalizer=normalizer, action_keys=gaussian_keys
            )
        self._initialize_gaussian_mixture(
            normalizer=normalizer,
            action_keys=gaussian_keys,
            chunk_samples=chunk_samples,
        )

    def _constant_chunk_candidates(
        self,
        normalizer: LinearNormalizer,
        action_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Fallback to constant-trajectory candidates when no chunk subsample exists.

        Single actions sampled uniformly from the normalized range are
        broadcast over the prediction horizon, so a per-step candidate is
        simply a chunk of constant value. Covers only the bounding box of the
        action distribution, not its modal structure.

        Args:
            normalizer: Normalizer with fitted statistics.
            action_keys: Gaussian action keys to build candidates for.

        Returns:
            Dict of ``(N, H, D)`` constant candidate chunks per key.
        """
        candidates: dict[str, torch.Tensor] = {}
        for action_key in action_keys:
            stats = normalizer.get_output_stats(key=action_key)
            single_actions = self._sample_uniform_candidates(
                data_min=stats["min"],
                data_max=stats["max"],
                num_candidates=1000,
                out_dim=stats["min"].shape[0],
            )  # (N, D)
            candidates[action_key] = single_actions.unsqueeze(1).expand(
                -1, self.prediction_horizon, -1
            )
        return candidates

    def _collect_chunk_samples(
        self,
        normalizer: LinearNormalizer,
        action_keys: list[str],
    ) -> dict[str, torch.Tensor] | None:
        """Gather aligned normalized chunk subsamples for the given keys.

        Args:
            normalizer: Normalizer possibly carrying chunk subsamples.
            action_keys: Gaussian action keys to collect chunks for.

        Returns:
            Dict of ``(N, H, D)`` normalized chunks per key, or ``None`` when
            any key lacks a chunk subsample.

        Raises:
            ValueError: If chunk counts differ across keys or the chunk horizon
                does not match the decoder prediction horizon.
        """
        chunk_samples: dict[str, torch.Tensor] = {}
        for action_key in action_keys:
            chunks = normalizer[action_key].get_output_sample_chunks()
            if chunks is None or chunks.numel() == 0:
                return None
            chunk_samples[action_key] = chunks
        chunk_counts = {chunks.shape[0] for chunks in chunk_samples.values()}
        if len(chunk_counts) != 1:
            raise ValueError(
                f"Chunk subsamples are not aligned across action keys: "
                f"{ {key: chunks.shape[0] for key, chunks in chunk_samples.items()} }"
            )
        chunk_horizons = {chunks.shape[1] for chunks in chunk_samples.values()}
        if chunk_horizons != {self.prediction_horizon}:
            raise ValueError(
                f"Chunk subsample horizon {chunk_horizons} does not match "
                f"prediction_horizon {self.prediction_horizon}."
            )
        return chunk_samples

    def _initialize_gaussian_mixture(
        self,
        normalizer: LinearNormalizer,
        action_keys: list[str],
        chunk_samples: dict[str, torch.Tensor],
    ) -> None:
        """Initialize components as trajectories selected in chunk space.

        Note:
            Chunks are clamped to the normalized data range (they are stored
            un-winsorized). With the k-means++ strategy, flattened chunks are
            concatenated across keys so k-means++ selects joint demonstrated
            trajectory modes. With the uniform strategy, component ``k``
            interpolates the per-timestep envelope of the chunk sample at
            fraction ``k / (K - 1)``, using the same fraction for every key.
            Each selected trajectory is written in-place into the component's
            temporal bias, giving time-varying initial means.

        Args:
            normalizer: Normalizer with fitted statistics.
            action_keys: Gaussian action keys, ordered as concatenated.
            chunk_samples: Aligned ``(N, H, D)`` normalized chunks per key.
        """
        output_stats_by_key = {
            action_key: normalizer.get_output_stats(key=action_key)
            for action_key in action_keys
        }
        clamped_chunks: dict[str, torch.Tensor] = {}
        for action_key in action_keys:
            stats = output_stats_by_key[action_key]
            chunks = chunk_samples[action_key].to(
                dtype=stats["min"].dtype, device=stats["min"].device
            )
            clamped_chunks[action_key] = chunks.clamp(
                min=stats["min"], max=stats["max"]
            )
        if self.gmm_init_strategy == GMMInitStrategy.KMEANS_PLUS_PLUS.value:
            centers_by_key = self._compute_chunk_kmeans_centers(
                clamped_chunks=clamped_chunks, action_keys=action_keys
            )
        else:
            centers_by_key = self._compute_chunk_uniform_centers(
                clamped_chunks=clamped_chunks
            )
        for action_key in action_keys:
            stats = output_stats_by_key[action_key]
            expert_logvar = self._chunk_logvar(
                data_min=stats["min"], data_max=stats["max"]
            )
            for k, head in enumerate(self.mixture_heads[action_key]):
                self._initialize_single_gaussian_head(
                    head=head,
                    logvar=expert_logvar,
                    temporal_mean=centers_by_key[action_key][k],
                )

    def _compute_chunk_kmeans_centers(
        self,
        clamped_chunks: dict[str, torch.Tensor],
        action_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Select K demonstrated chunks via joint k-means++ across keys.

        Args:
            clamped_chunks: Aligned ``(N, H, D)`` normalized chunks per key.
            action_keys: Gaussian action keys, ordered as concatenated.

        Returns:
            Per-key ``(K, H, D)`` trajectory centers with shared component
            indices across keys.
        """
        candidate_points = torch.cat(
            [clamped_chunks[key].flatten(start_dim=1) for key in action_keys],
            dim=1,
        )  # (N, H * total_dim)
        centers = self._compute_kmeans_plus_plus_centers(
            candidate_points=candidate_points,
            number_of_mixture_components=self.num_mixture_components,
        )
        centers_by_key: dict[str, torch.Tensor] = {}
        column_offset = 0
        for action_key in action_keys:
            horizon, out_dim = clamped_chunks[action_key].shape[1:]
            width = horizon * out_dim
            centers_by_key[action_key] = centers[
                :, column_offset : column_offset + width
            ].view(self.num_mixture_components, horizon, out_dim)
            column_offset += width
        return centers_by_key

    def _compute_chunk_uniform_centers(
        self,
        clamped_chunks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Spread K trajectory centers across the chunk sample envelope.

        Component ``k`` interpolates between the per-timestep minimum and
        maximum of the chunk sample at fraction ``k / (K - 1)``. The fraction
        is shared across keys, so component indices stay in correspondence.

        Args:
            clamped_chunks: Aligned ``(N, H, D)`` normalized chunks per key.

        Returns:
            Per-key ``(K, H, D)`` trajectory centers.
        """
        return {
            action_key: self._compute_uniform_centers(
                data_min=chunks.amin(dim=0),
                data_max=chunks.amax(dim=0),
                number_of_mixture_components=self.num_mixture_components,
            )
            for action_key, chunks in clamped_chunks.items()
        }

    def _chunk_logvar(
        self,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
    ) -> torch.Tensor:
        """Initial per-dimension logvar tempered by the prediction horizon.

        Args:
            data_min: Per-dimension normalized minimum.
            data_max: Per-dimension normalized maximum.

        Returns:
            ``(D,)`` logvar tensor equal to ``2 log(range / 2) + log(horizon)``.
        """
        expert_sigma = ((data_max - data_min) / 2.0).clamp(min=1e-6)
        return 2 * torch.log(expert_sigma) + math.log(self.prediction_horizon)

    def _initialize_gating_network(self) -> None:
        """Zero-initialize gating network for uniform initial mixture weights."""
        for module in self.gating_network.modules():
            if isinstance(module, MLP):
                final_layer = module.layers[-1]
                if isinstance(final_layer, nn.Linear):
                    nn.init.zeros_(final_layer.weight)
                    nn.init.zeros_(final_layer.bias)

    @staticmethod
    def _compute_kmeans_plus_plus_centers(
        candidate_points: torch.Tensor,
        number_of_mixture_components: int,
    ) -> torch.Tensor:
        """Compute K centers using k-means++ initialization over a candidate pool.

        With a uniformly-sampled pool the picked centers span the bounding box but
        may fall in empty regions of the action distribution. With a pool of real
        action samples the picked centers land on actual data (and, for well-
        separated modes, concentrate one center per mode).

        Note: adapted from https://github.com/ziyadsheeba/qfat/blob/main/src/qfat/models/qfat.py

        Args:
            candidate_points: ``(N, out_dim)`` tensor of points to select centers from.
            number_of_mixture_components: Number of mixture components ``K``.

        Returns:
            Tensor of shape ``(K, out_dim)`` with selected centers.
        """
        num_candidates = candidate_points.shape[0]
        if num_candidates == 0:
            raise ValueError(
                "Cannot initialize k-means++ centers from zero candidates."
            )
        first_center_idx = torch.randint(0, num_candidates, (1,)).item()
        selected_centers = candidate_points[first_center_idx].unsqueeze(0)
        for _ in range(1, number_of_mixture_components):
            squared_distances = torch.cdist(
                candidate_points, selected_centers, p=2
            ).pow(2)
            distance_to_nearest_center, _ = torch.min(squared_distances, dim=1)
            distance_sum = distance_to_nearest_center.sum()
            if distance_sum <= 0:
                next_center_idx = selected_centers.shape[0] % num_candidates
            else:
                selection_probabilities = distance_to_nearest_center / distance_sum
                next_center_idx = torch.multinomial(selection_probabilities, 1).item()
            selected_centers = torch.cat(
                [selected_centers, candidate_points[next_center_idx].unsqueeze(0)],
                dim=0,
            )
        return selected_centers

    @staticmethod
    def _sample_uniform_candidates(
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        num_candidates: int,
        out_dim: int,
    ) -> torch.Tensor:
        """Sample ``num_candidates`` points uniformly from the bounding box.

        Fallback candidate pool used only when no data sample is available on the
        normalizer. Covers the hyperrectangle evenly, so it ignores the modal
        structure of the action distribution.
        """
        candidate_points = torch.empty(
            (num_candidates, out_dim), device=data_min.device
        )
        for dim in range(out_dim):
            candidate_points[:, dim] = torch.empty(
                num_candidates, device=data_min.device
            ).uniform_(data_min[dim].item(), data_max[dim].item())
        return candidate_points

    @staticmethod
    def _compute_uniform_centers(
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        number_of_mixture_components: int,
    ) -> torch.Tensor:
        """Compute K centers interpolating uniformly between two bounds.

        Args:
            data_min: Lower bound, ``(D,)`` per-step stats or ``(H, D)``
                chunk envelopes.
            data_max: Upper bound, same shape as ``data_min``.
            number_of_mixture_components: Number of mixture components.

        Returns:
            Tensor of shape ``(K, *data_min.shape)`` with uniformly spread
            centers.
        """
        centers = []
        for k in range(number_of_mixture_components):
            alpha = (
                k / (number_of_mixture_components - 1)
                if number_of_mixture_components > 1
                else 0.5
            )
            center = data_min + alpha * (data_max - data_min)
            centers.append(center)
        return torch.stack(centers, dim=0)

    @staticmethod
    def _initialize_single_gaussian_head(
        head: GaussianHead,
        logvar: torch.Tensor,
        temporal_mean: torch.Tensor,
    ) -> None:
        """Bias-only initialization of a single Gaussian component head.

        Projection weights and the constant bias are zeroed so the initial
        mean equals the temporal bias trajectory. The head's ``max_logvar``
        is raised when the requested logvar exceeds it, so horizon-tempered
        variances are not silently clamped away.

        Args:
            head: GaussianHead to initialize.
            logvar: ``(D,)`` logvar written to the logvar projection bias.
            temporal_mean: ``(H, D)`` trajectory written to the temporal bias.
        """
        with torch.no_grad():
            nn.init.zeros_(head.output_proj.weight)
            nn.init.zeros_(head._logvar_proj.weight)
            head.output_proj.bias.zero_()
            head.temporal_bias.copy_(temporal_mean)
            max_requested_logvar = float(logvar.max().item())
            if max_requested_logvar > head.max_logvar:
                head.max_logvar = max_requested_logvar
            head._logvar_proj.bias.copy_(
                logvar.clamp(min=head.min_logvar, max=head.max_logvar)
            )

    def _forward_mixture(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Core forward pass returning mixture outputs.

        Returns:
            Dictionary containing:
                - {action_key}_{output_key}: Stacked outputs (B, T, K, D) for each head
                - routing_weights: Mixture weights (B, K)
        """
        obs_tokens, obs_padding_mask = self._build_parallel_observation_tokens(
            input_sequence_builder=self.input_sequence_builder,
            features=features,
            add_positional_encodings=True,
        )  # (B, observation_token_count, embedding_dimension), (B, observation_token_count)
        batch_size = obs_tokens.shape[0]
        mode_query_expanded = self._expand_parallel_query_tensor(
            query=self.mode_query,
            batch_size=batch_size,
        )  # (B, 1, embedding_dimension)
        action_queries_expanded = self._expand_parallel_query_tensor(
            query=self.action_queries,
            batch_size=batch_size,
        )  # (B, prediction_horizon, embedding_dimension)
        all_queries = torch.cat(
            [mode_query_expanded, action_queries_expanded], dim=1
        )  # (B, T+1, emb)
        attended = self.action_decoder(
            hidden_states=all_queries,
            encoded_features=obs_tokens,
            query_padding_mask=None,
            memory_padding_mask=obs_padding_mask,
        )  # (B, prediction_horizon + 1, embedding_dimension)
        mode_embedding = attended[:, 0, :]  # (B, emb)
        action_embeddings = attended[:, 1:, :]  # (B, T, emb)
        if self.gating_feature_key is not None:
            gating_input = features[self.gating_feature_key]
        else:
            gating_input = mode_embedding

        gating_logits = self.gating_network(gating_input)  # (B, K)
        routing_weights = F.softmax(gating_logits / self.temperature, dim=-1)  # (B, K)
        predictions = self._apply_mixture_heads(action_embeddings)
        predictions[DecoderOutputKey.ROUTING_WEIGHTS.value] = routing_weights
        return predictions

    def _apply_mixture_heads(
        self, action_embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Apply K mixture heads to action embeddings and stack outputs.

        Args:
            action_embeddings: (B, T, embedding_dimension)

        Returns:
            Dictionary with stacked outputs. For GaussianHead:
                {action_key}_mean: (B, T, K, D)
                {action_key}_logvar: (B, T, K, D)
            For other heads:
                {action_key}: (B, T, K, D)
        """
        predictions: dict[str, torch.Tensor] = {}
        for action_key in self.action_keys:
            component_outputs: list[dict[str, torch.Tensor] | torch.Tensor] = []
            for head in self.mixture_heads[action_key]:
                output = head(action_embeddings)
                component_outputs.append(output)

            base_head = self.action_heads[action_key]
            if isinstance(base_head, GaussianHead):
                for output_key in component_outputs[0]:
                    stacked = torch.stack(
                        [comp[output_key] for comp in component_outputs],
                        dim=2,  # (B, T, Num Mixture, Action Dimension)
                    )
                    predictions[f"{action_key}_{output_key}"] = stacked
            else:
                stacked = torch.stack(component_outputs, dim=2)
                predictions[action_key] = stacked

        return predictions

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass dispatching based on training vs inference mode.

        During training (actions provided): returns raw mixture outputs for loss computation.
        During inference (actions=None): returns sampled actions from mixture.

        Args:
            features: Dictionary of encoded features.
            actions: Ground truth actions (used only to distinguish training from inference).

        Returns:
            During training: Dict with mixture outputs (mean, logvar, routing_weights).
            During inference: Dict with sampled actions (B, T, D) for each action key.
        """
        if actions is None:
            return self._sample_from_mixture_outputs(features)
        return self._forward_mixture(features)

    def _sample_from_mixture_outputs(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample from mixture distribution for inference.

        Args:
            features: Dictionary of encoded features.

        Returns:
            Dictionary with sampled actions (B, T, D) for each action key.
        """
        with torch.no_grad():
            output = self._forward_mixture(features)
            routing_weights = output[DecoderOutputKey.ROUTING_WEIGHTS.value]
            sampled_predictions: dict[str, torch.Tensor] = {}

            for action_key in self.action_keys:
                base_head = self.action_heads[action_key]
                if isinstance(base_head, GaussianHead):
                    mean = output[f"{action_key}_{DecoderOutputKey.MEAN.value}"]
                    logvar = output[f"{action_key}_{DecoderOutputKey.LOGVAR.value}"]
                    sampled_predictions[action_key] = (
                        self._sample_from_gaussian_mixture(
                            mean=mean,
                            logvar=logvar,
                            routing_weights=routing_weights,
                            sampling_mode=self.inference_sampling_mode,
                        )
                    )
                else:
                    stacked = output[action_key]
                    sampled_predictions[action_key] = self._sample_from_mixture(
                        stacked=stacked,
                        routing_weights=routing_weights,
                        sampling_mode=self.inference_sampling_mode,
                    )

            return sampled_predictions

    @staticmethod
    def _sample_from_gaussian_mixture(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        routing_weights: torch.Tensor,
        sampling_mode: str = MixtureSamplingMode.STOCHASTIC_MEAN.value,
    ) -> torch.Tensor:
        """Sample from Gaussian mixture using routing weights.

        Args:
            mean: (B, T, K, D)
            logvar: (B, T, K, D)
            routing_weights: (B, K)
            sampling_mode: Component selection and noise strategy.

        Returns:
            Sampled actions (B, T, D)
        """
        batch_size = mean.shape[0]
        match sampling_mode:
            case MixtureSamplingMode.DETERMINISTIC.value:
                component_indices = torch.argmax(routing_weights, dim=-1)
            case (
                MixtureSamplingMode.STOCHASTIC_MEAN.value
                | MixtureSamplingMode.STOCHASTIC_SAMPLE.value
            ):
                component_indices = torch.multinomial(
                    routing_weights, num_samples=1
                ).squeeze(-1)
            case _:
                raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        batch_indices = torch.arange(batch_size, device=mean.device)
        selected_mean = mean[batch_indices, :, component_indices, :]  # (B, T, D)
        if sampling_mode == MixtureSamplingMode.STOCHASTIC_SAMPLE.value:
            selected_logvar = logvar[batch_indices, :, component_indices, :]
            std = torch.exp(0.5 * selected_logvar)  # (B, T, D)
            return selected_mean + std * torch.randn_like(selected_mean)
        else:
            return selected_mean

    @staticmethod
    def _sample_from_mixture(
        stacked: torch.Tensor,
        routing_weights: torch.Tensor,
        sampling_mode: str = MixtureSamplingMode.STOCHASTIC_MEAN.value,
    ) -> torch.Tensor:
        """Sample from mixture using routing weights (non-Gaussian heads).

        Args:
            stacked: (B, T, K, D)
            routing_weights: (B, K)
            sampling_mode: Component selection strategy.

        Returns:
            Selected outputs (B, T, D)
        """
        batch_size = stacked.shape[0]
        if sampling_mode == MixtureSamplingMode.DETERMINISTIC.value:
            component_indices = torch.argmax(routing_weights, dim=-1)
        else:
            component_indices = torch.multinomial(
                routing_weights, num_samples=1
            ).squeeze(-1)
        batch_indices = torch.arange(batch_size, device=stacked.device)
        return stacked[batch_indices, :, component_indices, :]
