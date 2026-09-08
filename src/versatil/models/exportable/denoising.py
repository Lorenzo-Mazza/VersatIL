"""Policy export with explicit diffusion and flow sampling inputs."""

import torch

from versatil.models.decoding.algorithm.base import resolve_feature_reference
from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.exportable.base import ExportablePolicy
from versatil.models.exportable.metadata import (
    NoiseInput,
    PolicyExportMetadata,
    SamplingInput,
)
from versatil.models.policy import Policy


class ExportableDenoisingPolicy(ExportablePolicy):
    """Export flow or diffusion inference with explicit initial and per-step noise."""

    @classmethod
    def from_policy(cls, policy: Policy) -> "ExportableDenoisingPolicy":
        """Describe noise tensors using the policy's action dimensions and schedule.

        Args:
            policy: Initialized policy using Diffusion or FlowMatching.

        Returns:
            Wrapper sharing the policy modules and declaring the sampling inputs.
        """
        action_shapes = {
            key: (
                policy.prediction_horizon,
                policy.action_space.actions_metadata[key].prediction_dimension,
            )
            for key in policy.output_keys
        }
        noise_inputs = [
            NoiseInput(name=f"{SamplingInput.INITIAL_NOISE.value}.{key}", shape=shape)
            for key, shape in action_shapes.items()
        ]
        if (
            isinstance(policy.algorithm, Diffusion)
            and policy.algorithm.inference_schedule.stochastic
        ):
            step_count = len(policy.algorithm.inference_schedule.timesteps)
            noise_inputs.extend(
                NoiseInput(
                    name=f"{SamplingInput.STEP_NOISE.value}.{key}",
                    shape=(step_count, *shape),
                )
                for key, shape in action_shapes.items()
            )
        return cls(
            encoding_pipeline=policy.encoding_pipeline,
            algorithm=policy.algorithm,
            decoder=policy.decoder,
            observation_keys=policy.input_keys,
            action_keys=policy.output_keys,
            export_metadata=PolicyExportMetadata(noise_inputs=tuple(noise_inputs)),
        )

    def _predict(
        self,
        features: dict[str, torch.Tensor],
        sampling_inputs: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Run the denoising algorithm using the graph's noise arguments.

        Args:
            features: Encoded observation features for each denoising step.
            sampling_inputs: Initial noise for each action key, followed by DDPM
                step noise when required by the export metadata.

        Returns:
            Normalized actions shaped ``(batch, horizon, action_dimension)``.

        Note:
            Noise uses the encoded features' device and floating-point precision,
            matching the algorithm's native sampling path.
        """
        _, device, dtype = resolve_feature_reference(features=features)
        sampling_inputs = tuple(
            noise.to(device=device, dtype=dtype)  # (batch, ..., horizon, dimension)
            for noise in sampling_inputs
        )
        action_count = len(self._action_keys)
        initial_noise = dict(
            zip(self._action_keys, sampling_inputs[:action_count], strict=True)
        )
        if isinstance(self.algorithm, Diffusion):
            step_inputs = sampling_inputs[action_count:]
            step_noise = (
                dict(zip(self._action_keys, step_inputs, strict=True))
                if step_inputs
                else None
            )
            return self.algorithm.predict_from_noise(
                network=self.decoder,
                features=features,
                initial_noise=initial_noise,
                step_noise=step_noise,
            )  # (batch, horizon, action_dimension)
        return self.algorithm.predict_from_noise(
            network=self.decoder,
            features=features,
            initial_noise=initial_noise,
        )  # (batch, horizon, action_dimension)
