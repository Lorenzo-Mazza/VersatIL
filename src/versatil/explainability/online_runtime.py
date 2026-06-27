"""Policy runtime used by online explainability runs."""

from pathlib import Path

import torch

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.inference.policy_runtime.base import PolicyRuntime
from versatil.training.constants import PrecisionType


class ExplainabilityPolicyRuntime(PolicyRuntime):
    """Runs policy inference with the checkpoint already loaded by the explainer."""

    def __init__(
        self,
        checkpoint_loader: FloatCheckpointLoader,
        checkpoint_name: str,
    ) -> None:
        """Initialize the runtime from the explainer checkpoint loader.

        Args:
            checkpoint_loader: Loader owned by ``ExplainabilityRunner``. Reusing
                it keeps attribution and action prediction on the same policy
                instance.
            checkpoint_name: Checkpoint filename used to build the client
                identifier registered with the environment server.
        """
        self._precision = str(checkpoint_loader.config.experiment.precision)
        client_identifier = str(
            Path(checkpoint_loader.checkpoint_path) / Path(checkpoint_name).stem
        )
        super().__init__(
            checkpoint_loader=checkpoint_loader,
            client_identifier=client_identifier,
        )

    def run_inference(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run action prediction for the already-preprocessed inference batch.

        Args:
            obs_dict: Observation window produced by ``InferenceClient`` after
                transport parsing, image transforms, and observation buffering.

        Returns:
            Policy action tensors keyed by action component.
        """
        with (
            torch.autocast(
                device_type=self.device.type,
                dtype=PrecisionType(self._precision).get_model_dtype(),
            ),
            torch.no_grad(),
        ):
            return self.policy.predict_action(obs_dict=obs_dict)
