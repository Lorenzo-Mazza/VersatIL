"""Mixture of Densities Action Transformer (MODE-ACT) for multi-modal action prediction.

This module implements a Mixture Density Network style transformer decoder that predicts
multiple mixture components for each action, enabling multi-modal action distributions.
"""

import copy

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
        """Clone each action head K times for mixture components."""
        self.mixture_heads: nn.ModuleDict = nn.ModuleDict()
        for action_key, head in self.action_heads.items():
            cloned_heads = []
            for _ in range(self.num_mixture_components):
                cloned_head = copy.deepcopy(head)
                for module in cloned_head.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
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

        Args:
            normalizer: Normalizer with fitted statistics.
        """
        super().set_normalizer(normalizer)
        self._initialize_gating_network()
        for action_key in self.action_keys:
            if action_key not in normalizer.params_dict:
                continue
            field_normalizer = normalizer[action_key]
            output_stats = field_normalizer.get_output_stats()
            candidate_sample = field_normalizer.get_output_sample()
            base_head = self.action_heads[action_key]
            if isinstance(base_head, GaussianHead):
                self._initialize_gaussian_mixture(
                    action_key=action_key,
                    output_stats=output_stats,
                    candidate_sample=candidate_sample,
                )

    def _initialize_gating_network(self) -> None:
        """Zero-initialize gating network for uniform initial mixture weights."""
        for module in self.gating_network.modules():
            if isinstance(module, MLP):
                final_layer = module.layers[-1]
                if isinstance(final_layer, nn.Linear):
                    nn.init.zeros_(final_layer.weight)
                    nn.init.zeros_(final_layer.bias)

    def _initialize_gaussian_mixture(
        self,
        action_key: str,
        output_stats: dict[str, torch.Tensor],
        candidate_sample: torch.Tensor | None = None,
    ) -> None:
        """Initialize Gaussian mixture heads from output statistics.

        Args:
            action_key: Key for the action head.
            output_stats: Dict with "min", "max", "std" tensors from normalizer.get_output_stats().
            candidate_sample: Optional (N, out_dim) tensor of normalized action samples
                to use as the candidate pool for k-means++. When supplied, k-means++
                picks centers from actual action data, so centers land near true data
                modes. When ``None``, the candidate pool is sampled uniformly from
                ``[data_min, data_max]`` and reflects only the bounding box, not the
                modal structure.
        """
        data_min = output_stats["min"]
        data_max = output_stats["max"]
        out_dim = data_min.shape[0]
        if self.gmm_init_strategy == GMMInitStrategy.KMEANS_PLUS_PLUS.value:
            if candidate_sample is not None and candidate_sample.numel() > 0:
                candidate_points = candidate_sample.to(
                    dtype=data_min.dtype, device=data_min.device
                )
            else:
                candidate_points = self._sample_uniform_candidates(
                    data_min=data_min,
                    data_max=data_max,
                    num_candidates=1000,
                    out_dim=out_dim,
                )
            centers = self._compute_kmeans_plus_plus_centers(
                candidate_points=candidate_points,
                number_of_mixture_components=self.num_mixture_components,
            )
        else:
            centers = self._compute_uniform_centers(
                data_min=data_min,
                data_max=data_max,
                number_of_mixture_components=self.num_mixture_components,
            )

        data_range = data_max - data_min
        # QFAT-style fixed variance: sigma = half-range, regardless of K.
        # For [-1, 1] normalized data this gives sigma = 1, so every component
        # has non-trivial responsibility for every data point at init.
        expert_sigma = (data_range / 2.0).clamp(min=1e-6)
        expert_logvar = 2 * torch.log(expert_sigma)
        for k, head in enumerate(self.mixture_heads[action_key]):
            self._initialize_single_gaussian_head(
                head=head, mean=centers[k], logvar=expert_logvar
            )

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
        first_center_idx = torch.randint(0, num_candidates, (1,)).item()
        selected_centers = candidate_points[first_center_idx].unsqueeze(0)
        for _ in range(1, number_of_mixture_components):
            squared_distances = torch.cdist(
                candidate_points, selected_centers, p=2
            ).pow(2)
            distance_to_nearest_center, _ = torch.min(squared_distances, dim=1)
            selection_probabilities = (
                distance_to_nearest_center / distance_to_nearest_center.sum()
            )
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
        """Compute K centers using uniform spread.

        Args:
            data_min: Min values per dimension from output stats.
            data_max: Max values per dimension from output stats.
            number_of_mixture_components: Number of mixture components.

        Returns:
            Tensor of shape (K, out_dim) with uniformly spread centers.
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
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> None:
        """Initialize a single Gaussian head with given mean and logvar.

        Args:
            head: GaussianHead to initialize.
            mean: Mean tensor for output_proj bias.
            logvar: Logvar tensor for _logvar_proj bias.
        """
        with torch.no_grad():
            nn.init.zeros_(head.output_proj.weight)
            nn.init.zeros_(head._logvar_proj.weight)
            head.output_proj.bias.copy_(mean)
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
