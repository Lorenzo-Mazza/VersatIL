"""Quantization-Aware-Trained policy checkpoint restoration."""

import os

import torch

from versatil.checkpoint_loading.base import (
    BaseCheckpointLoader,
    strip_compiled_prefixes,
    versatil_checkpoint_safe_globals,
)
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.export import build_example_inputs
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.training.constants import (
    CheckpointFilename,
    CheckpointKey,
)
from versatil.training.lightning_policy import LightningPolicy


class _QATCheckpointLoader(BaseCheckpointLoader):
    """Restore a QAT policy checkpoint, which needs specific handling of the fake-quantized layers inserted in its saved weights."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str,
        quantization: BaseQuantizationWorkflow,
    ) -> None:
        """Initialize the QAT checkpoint loader."""
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self._checkpoint_name = checkpoint_name
        self._quantization = quantization
        self._load_model()

    def _load_model(self) -> None:
        """Prepare fake quantization before loading QAT-trained weights."""
        config_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.CONFIG.value
        )
        self._config = self._load_config(config_path=config_path)
        self._policy = self._config.policy
        tokenizer_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.TOKENIZER_DIR.value
        )
        self._tokenizer = self._load_tokenizer(tokenizer_path=tokenizer_path)
        if self._tokenizer is not None:
            self._tokenizer.to(self._device)
            self._policy.set_tokenizer(self._tokenizer)

        self._policy.to(self._device).eval()
        self._policy.device = self._device
        self._policy.encoding_pipeline.set_output_dtype(torch.float32)
        self._materialize_lazy_modules()
        self._quantization.prepare_model(model=self._policy)
        checkpoint_file = os.path.join(self._checkpoint_path, self._checkpoint_name)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}.")
        with torch.serialization.safe_globals(versatil_checkpoint_safe_globals()):
            checkpoint = torch.load(
                checkpoint_file,
                map_location=self._device,
                weights_only=True,
            )
        lightning_module = LightningPolicy(
            policy=self._policy,
            training_config=self._config.training,
        )
        state_dict_key = CheckpointKey.STATE_DICT.value
        checkpoint_state = strip_compiled_prefixes(checkpoint[state_dict_key])
        lightning_module.load_state_dict(checkpoint_state, strict=False)
        self._validate_checkpoint_loading(
            checkpoint_state_dict=checkpoint_state,
            model_state_dict=lightning_module.state_dict(),
        )

    def _materialize_lazy_modules(self) -> None:
        """Run one eager pass before QAT prepare mutates Linear modules."""
        exportable = ExportablePolicy.from_policy(policy=self._policy)
        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=self.observation_space,
            observation_horizon=self.observation_horizon,
            tokenizer=self.tokenizer,
        )
        example_inputs = tuple(tensor.to(self._device) for tensor in example_inputs)
        with torch.no_grad():
            exportable(*example_inputs)
