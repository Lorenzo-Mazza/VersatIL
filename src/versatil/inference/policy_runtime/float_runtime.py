"""Floating-point policy inference runtime."""

import logging
from pathlib import Path

import torch

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.inference.policy_runtime.base import (
    PolicyRuntime,
    initialize_inference_seed,
)
from versatil.training.constants import (
    CheckpointFilename,
    PrecisionType,
)


class FloatPolicyRuntime(PolicyRuntime):
    """Inference runtime for floating-point policy checkpoints."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
        seed: int | None = None,
        compile_model: bool = True,
    ) -> None:
        """Initialize the float policy runtime.

        Args:
            device: Device to load the model onto.
            checkpoint_path: Path to the checkpoint directory.
            checkpoint_name: Name of the checkpoint file.
            seed: Explicit inference seed. ``None`` draws a fresh seed from
                operating-system entropy.
            compile_model: Whether to compile the policy with
                torch.compile for optimized inference.
        """
        self._compile_model = compile_model
        checkpoint_loader = FloatCheckpointLoader(
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
        )
        super().__init__(
            checkpoint_loader=checkpoint_loader,
            client_identifier=str(Path(checkpoint_path) / Path(checkpoint_name).stem),
        )
        self._prepare_inference_model()
        self._seed = initialize_inference_seed(seed=seed)

    def _prepare_inference_model(self) -> None:
        """Apply inference-only model preparation."""
        self._precision = str(self.config.experiment.precision)
        precision_type = PrecisionType(self._precision)
        if precision_type.should_convert_model():
            self._policy = self._policy.to(precision_type.get_model_dtype())

        if self._compile_model:
            self._policy = torch.compile(self._policy)
            logging.info("Compiled policy with torch.compile")

        logging.info("Policy inference model ready.")

    def run_inference(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run policy inference with autocast and no_grad.

        Args:
            obs_dict: Observation dictionary for the policy.

        Returns:
            Action dictionary from policy.predict_action.
        """
        with (
            torch.autocast(
                device_type=self.device.type,
                dtype=PrecisionType(self._precision).get_model_dtype(),
            ),
            torch.no_grad(),
        ):
            return self._policy.predict_action(obs_dict=obs_dict)
