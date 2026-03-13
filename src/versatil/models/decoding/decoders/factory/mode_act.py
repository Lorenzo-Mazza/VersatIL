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
from versatil.models.decoding.constants import DecoderOutputKey, GMMInitStrategy, FeatureType
from versatil.models.decoding.decoders import ActionDecoder, DecoderInput
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.mlp import MLP
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding2D,
)
from versatil.models.layers.transformer import BidirectionalDecoder
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder


class MixtureOfDensitiesActionTransformer(ActionDecoder):
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
        deterministic_inference: bool = True,
    ):
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
            deterministic_inference: If True, use argmax for component selection and return
                mean without noise. If False, sample component via multinomial and add
                Gaussian noise. Defaults to True for reproducible inference.
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
        )

        self.embedding_dimension = embedding_dimension
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
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
        self.gmm_init_strategy = gmm_init_strategy
        self.deterministic_inference = deterministic_inference
        self.action_keys = list(self.action_heads.keys())
        self._build_transformer_components()
        self._build_mixture_heads()
        self._build_gating_network(
            input_dim=embedding_dimension,
            hidden_dims=gating_hidden_dims,
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

    def _build_transformer_components(self) -> None:
        """Build core transformer encoder-decoder and positional encodings."""
        image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=self.embedding_dimension, normalize=True
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension
            )
        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dim=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=image_positional_encoding,
            flat_positional_encoding_layer=LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )
        self.action_queries = nn.Parameter(
            torch.randn(self.prediction_horizon, self.embedding_dimension)
        )
        self.mode_query = nn.Parameter(torch.randn(1, self.embedding_dimension))
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
        input_dim: int,
        hidden_dims: list[int] | None,
        activation: str,
        dropout: float,
        normalization: bool,
    ) -> None:
        """Build gating MLP for computing mixture weights.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout rate.
            normalization: Whether to apply layer normalization before MLP.
        """
        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [input_dim // 2]

        layers: list[nn.Module] = []
        if normalization:
            layers.append(nn.LayerNorm(input_dim))

        gating_mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
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
            output_stats = normalizer[action_key].get_output_stats()
            base_head = self.action_heads[action_key]
            if isinstance(base_head, GaussianHead):
                self._initialize_gaussian_mixture(
                    action_key=action_key,
                    output_stats=output_stats,
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
    ) -> None:
        """Initialize Gaussian mixture heads from output statistics.

        Args:
            action_key: Key for the action head.
            output_stats: Dict with "min", "max", "std" tensors from normalizer.get_output_stats().
        """
        data_min = output_stats["min"]
        data_max = output_stats["max"]
        out_dim = data_min.shape[0]
        if self.gmm_init_strategy == GMMInitStrategy.KMEANS_PLUS_PLUS.value:
            centers = self._compute_kmeans_plus_plus_centers(data_min=data_min, data_max=data_max,
                                                             number_of_mixture_components=self.num_mixture_components, out_dim=out_dim)
        else:
            centers = self._compute_uniform_centers(data_min=data_min, data_max=data_max, number_of_mixture_components=self.num_mixture_components)

        data_range = data_max - data_min
        expert_sigma = data_range / self.num_mixture_components
        expert_logvar = 2 * torch.log(expert_sigma.clamp(min=1e-6))
        for k, head in enumerate(self.mixture_heads[action_key]):
            self._initialize_single_gaussian_head(head=head, mean=centers[k], logvar=expert_logvar)

    @staticmethod
    def _compute_kmeans_plus_plus_centers(
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        number_of_mixture_components: int,
        out_dim: int,
    ) -> torch.Tensor:
        """Compute K centers using k-means++ initialization.
        
        Note: from https://github.com/ziyadsheeba/qfat/blob/main/src/qfat/models/qfat.py

        Args:
            data_min: Min values per dimension from output stats.
            data_max: Max values per dimension from output stats.
            number_of_mixture_components: Number of mixture components.
            out_dim: Output dimension.

        Returns:
            Tensor of shape (K, out_dim) with k-means++ initialized centers.
        """
        # Generate candidate points uniformly within the data range for each dimension
        num_candidates = 1000
        candidate_points = torch.empty((num_candidates, out_dim), device=data_min.device)
        for dim in range(out_dim):
            candidate_points[:, dim] = torch.empty(
                num_candidates, device=data_min.device
            ).uniform_(data_min[dim].item(), data_max[dim].item())

        # Pick the first center randomly from the candidate pool
        first_center_idx = torch.randint(0, num_candidates, (1,)).item()
        selected_centers = candidate_points[first_center_idx].unsqueeze(0)
        # Select remaining centers with probability proportional to squared distance
        # from the nearest existing center (k-means++ initialization)
        for _ in range(1, number_of_mixture_components):
            squared_distances = torch.cdist(candidate_points, selected_centers, p=2).pow(2)
            distance_to_nearest_center, _ = torch.min(squared_distances, dim=1)
            selection_probabilities = distance_to_nearest_center / distance_to_nearest_center.sum()
            next_center_idx = torch.multinomial(selection_probabilities, 1).item()
            selected_centers = torch.cat(
                [selected_centers, candidate_points[next_center_idx].unsqueeze(0)], dim=0
            )
        return selected_centers

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
            alpha = k / (number_of_mixture_components - 1) if number_of_mixture_components > 1 else 0.5
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
            head._logvar_proj.bias.copy_(logvar.clamp(min=head.min_logvar, max=head.max_logvar))

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
        obs_tokens, obs_pos_encodings, obs_padding_mask = self.input_sequence_builder(
            features
        )
        obs_tokens = obs_tokens + obs_pos_encodings
        batch_size = obs_tokens.shape[0]
        mode_query_expanded = self.mode_query.unsqueeze(0).expand(batch_size, -1, -1) # (B, 1, emb)
        action_queries_expanded = self.action_queries.unsqueeze(0).expand(
            batch_size, -1, -1 # (B, T, emb
        )
        all_queries = torch.cat([mode_query_expanded, action_queries_expanded], dim=1) # (B, T+1, emb)
        attended = self.action_decoder(
            hidden_states=all_queries,
            encoded_features=obs_tokens,
            query_padding_mask=None,
            memory_padding_mask=obs_padding_mask,
        )
        mode_embedding = attended[:, 0, :] # (B, emb)
        action_embeddings = attended[:, 1:, :] # (B, T, emb)
        if self.gating_feature_key is not None:
            gating_input = features[self.gating_feature_key]
        else:
            gating_input = mode_embedding

        gating_logits = self.gating_network(gating_input)
        routing_weights = F.softmax(gating_logits / self.temperature, dim=-1)
        predictions = self._apply_mixture_heads(action_embeddings)
        predictions[DecoderOutputKey.ROUTING_WEIGHTS.value] = routing_weights
        return predictions

    def _apply_mixture_heads(
        self, action_embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Apply K mixture heads to action embeddings and stack outputs.

        Args:
            action_embeddings: (B, T, embedding_dim)

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
                for output_key in component_outputs[0].keys():
                    stacked = torch.stack(
                        [comp[output_key] for comp in component_outputs], dim=2 # (B, T, Num Mixture, Action Dimension)
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
                    sampled_predictions[action_key] = self._sample_from_gaussian_mixture(
                        mean=mean,
                        logvar=logvar,
                        routing_weights=routing_weights,
                        deterministic=self.deterministic_inference,
                    )
                else:
                    stacked = output[action_key]
                    sampled_predictions[action_key] = self._sample_from_mixture(
                        stacked=stacked,
                        routing_weights=routing_weights,
                        deterministic=self.deterministic_inference,
                    )

            return sampled_predictions

    @staticmethod
    def _sample_from_gaussian_mixture(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        routing_weights: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Sample from Gaussian mixture using routing weights.

        Args:
            mean: (B, T, K, D)
            logvar: (B, T, K, D)
            routing_weights: (B, K)
            deterministic: If True, use argmax for component selection and return mean.
                If False, sample component via multinomial and add Gaussian noise.

        Returns:
            Sampled actions (B, T, D)
        """
        batch_size = mean.shape[0]
        if deterministic:
            component_indices = torch.argmax(routing_weights, dim=-1)
        else:
            component_indices = torch.multinomial(routing_weights, num_samples=1).squeeze(-1)
        batch_indices = torch.arange(batch_size, device=mean.device)
        selected_mean = mean[batch_indices, :, component_indices, :]
        if deterministic:
            return selected_mean
        selected_logvar = logvar[batch_indices, :, component_indices, :]
        std = torch.exp(0.5 * selected_logvar)
        eps = torch.randn_like(selected_mean)
        return selected_mean + std * eps

    @staticmethod
    def _sample_from_mixture(
        stacked: torch.Tensor,
        routing_weights: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Sample from mixture using routing weights (Bernoulli heads).

        Args:
            stacked: (B, T, K, D)
            routing_weights: (B, K)
            deterministic: If True, use argmax for component selection.
                If False, sample component via multinomial.

        Returns:
            Selected outputs (B, T, D)
        """
        batch_size = stacked.shape[0]
        if deterministic:
            component_indices = torch.argmax(routing_weights, dim=-1)
        else:
            component_indices = torch.multinomial(routing_weights, num_samples=1).squeeze(-1)
        batch_indices = torch.arange(batch_size, device=stacked.device)
        return stacked[batch_indices, :, component_indices, :]