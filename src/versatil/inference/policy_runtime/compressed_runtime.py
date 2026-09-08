"""Compressed policy inference runtime."""

import logging

import hydra
import torch
import torch.nn as nn

from versatil.checkpoint_loading.compressed_policy import CompressedCheckpointLoader
from versatil.common.tensor_ops import to_device
from versatil.data.processing.transform import (
    detokenize_actions,
    normalize_observation,
    tokenize_observation,
    unnormalize_actions,
)
from versatil.inference.policy_runtime.base import PolicyRuntime
from versatil.inference.policy_runtime.executorch_adapter import ExecuTorchModuleAdapter
from versatil.models.exportable.metadata import PredictionOutput
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionMetadataKey,
    QuantizationWorkflow,
)
from versatil.quantization.pt2e.backends.base import BasePT2EBackend


class CompressedPolicyRuntime(PolicyRuntime):
    """Inference runtime for compressed policy checkpoints."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        compile_model: bool = True,
    ) -> None:
        """Initialize the compressed policy runtime.

        Args:
            device: Device to load the model onto.
            checkpoint_path: Path to the compressed checkpoint directory
                containing compression_metadata.json, the model artifact
                (Torch Export ``.pt2`` or ExecuTorch ``.pte``), and
                normalizer.pt.
            compile_model: Whether to compile the model with
                torch.compile before inference. For PT2E models on
                CPU this enables inductor int8 kernel fusion and is
                essential for performance. For other workflows it
                applies standard inductor compilation.
        """
        self._compile_model = compile_model
        self._compressed_model: nn.Module | None = None
        checkpoint_loader = CompressedCheckpointLoader(
            device=device,
            checkpoint_path=checkpoint_path,
        )
        super().__init__(
            checkpoint_loader=checkpoint_loader,
            client_identifier=checkpoint_path,
        )
        self._compressed_model = self._load_artifact_model(
            model_path=checkpoint_loader.model_path,
            workflow=checkpoint_loader.workflow,
        )

    def _load_artifact_model(
        self,
        model_path: str,
        workflow: str | None,
    ) -> nn.Module:
        """Load the runtime model for the saved artifact format."""
        artifact_format = self.checkpoint_loader.artifact_format
        if artifact_format == ArtifactFormat.TORCH_EXPORT_PT2.value:
            exported_program = torch.export.load(model_path)
            model = exported_program.module()
            backend = self._load_backend(workflow=workflow)
            if backend is not None:
                self._validate_device(backend=backend)
            if workflow != QuantizationWorkflow.PT2E.value:
                # PT2E graphs bake CPU device metadata into the export and
                # must stay on CPU; other workflows follow the runtime device.
                model = model.to(self.device)
            if self._compile_model and self._should_compile(
                workflow=workflow,
                device=self.device,
            ):
                model = self._compile_model_for_inference(
                    model=model,
                    backend=backend,
                )
            return model
        if artifact_format == ArtifactFormat.EXECUTORCH_PTE.value:
            if self.device.type != "cpu":
                raise ValueError(
                    "ExecuTorch XNNPACK artifacts support CPU inference only, "
                    f"got '{self.device}'."
                )
            return ExecuTorchModuleAdapter(model_path=model_path)
        raise ValueError(
            f"Unsupported compression artifact format '{artifact_format}'."
        )

    def _load_backend(
        self,
        workflow: str | None,
    ) -> BasePT2EBackend | None:
        """Reconstruct the PT2E backend for compile-environment activation.

        Args:
            workflow: The quantization workflow from metadata.

        Returns:
            Instantiated backend, or None if not a PT2E model.

        Raises:
            ValueError: If a PT2E artifact carries no instantiable backend
                node in its metadata.
        """
        if workflow != QuantizationWorkflow.PT2E.value:
            return None
        backend_config = self.checkpoint_loader.metadata.get(
            CompressionMetadataKey.PT2E_BACKEND.value
        )
        if backend_config is None:
            raise ValueError(
                "PT2E artifact metadata carries no pt2e_backend node; "
                "recompress the checkpoint with the current release."
            )
        backend = hydra.utils.instantiate(backend_config)
        if isinstance(backend, BasePT2EBackend):
            return backend
        raise ValueError(
            "Persisted pt2e_backend metadata did not instantiate a "
            f"PT2E backend, got {type(backend).__name__}."
        )

    @staticmethod
    def _should_compile(
        workflow: str | None,
        device: torch.device,
    ) -> bool:
        """Decide whether to compile based on workflow and device.

        CUDA + `quantize_()` models must not be compiled because
        `torch.compile` lowers int8 matmuls to torch._int_mm, which
        requires M > 16 on CUDA (cuBLAS constraint). Since inference
        batch size is unknown at load time (1 on real robot, N in
        sim), compilation is skipped entirely for this combination.

        See: https://github.com/pytorch/ao/issues/2376

        Args:
            workflow: The quantization workflow from metadata.
            device: Target inference device.

        Returns:
            True if the model should be compiled.
        """
        if workflow == QuantizationWorkflow.EAGER.value and device.type == "cuda":
            logging.warning(
                "Skipping torch.compile for eager quantized model on CUDA. "
                "torch._int_mm requires batch > 16 on CUDA which "
                "cannot be guaranteed at inference time. "
                "See https://github.com/pytorch/ao/issues/2376"
            )
            return False
        return True

    def _validate_device(self, backend: BasePT2EBackend) -> None:
        """Validate that the device is supported by the backend.

        Args:
            backend: The PT2E backend instance.

        Raises:
            ValueError: If the device is not supported.
        """
        if self.device.type not in backend.supported_device_types:
            raise ValueError(
                f"Backend {type(backend).__name__} supports devices "
                f"{backend.supported_device_types}, got '{self.device}'."
            )

    @staticmethod
    def _compile_model_for_inference(
        model: nn.Module,
        backend: BasePT2EBackend | None,
    ) -> nn.Module:
        """Compile model with torch.compile, using backend env if available.

        For PT2E models, activates the backend environment
        permanently because torch.compile is lazy — the actual
        inductor compilation happens on the first forward pass.

        Args:
            model: The model to compile.
            backend: PT2E backend for env setup, or None.

        Returns:
            Compiled model.
        """
        if backend is not None:
            backend.activate_environment()
            compiled = torch.compile(model)
            logging.info(f"Compiled PT2E model with {type(backend).__name__} backend")
        else:
            compiled = torch.compile(model)
            logging.info("Compiled model with inductor backend")
        return compiled

    def run_inference(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run the artifact and reconstruct actions using its export metadata.

        Args:
            obs_dict: Observation dictionary for the policy.

        Returns:
            Unnormalized action dictionary.

        Note:
            Observation processing follows the checkpoint's normalizer and
            tokenizer. The export metadata supplies additional noise inputs
            and selects action-token decoding when required. B denotes batch
            size, H the action horizon, D each action's dimension and L the
            generated token sequence length.
        """
        obs_dict = to_device(obs_dict, device=self.device)  # (B, ...)
        normalized_obs = normalize_observation(
            observation=obs_dict,
            normalizer=self.checkpoint_loader.normalizer,
            observation_space=self.observation_space,
        )  # (B, ...)
        tokenizer = self.tokenizer
        if tokenizer is not None and tokenizer.observation_tokenizer is not None:
            normalized_obs = tokenize_observation(
                observation=normalized_obs,
                obs_tokenizer=tokenizer.observation_tokenizer,
                batched=True,
            )  # (B, ...)
        observation_tensors = tuple(normalized_obs[key] for key in self.input_keys)
        model_inputs = self.checkpoint_loader.export_metadata.prepare_inputs(
            observations=observation_tensors
        )  # (B, ...)
        output_tensors = self._run_compressed_model(
            model_inputs=model_inputs,
        )  # actions: (B, H, D); tokens: (B, L)
        if (
            self.checkpoint_loader.export_metadata.output
            == PredictionOutput.ACTION_TOKENS
        ):
            if len(output_tensors) != 1:
                raise ValueError(
                    "Action-token artifacts must return one token tensor; "
                    f"received {len(output_tensors)} tensors."
                )
            self._validate_token_output(
                action_tokens=output_tensors[0],
                batch_size=observation_tensors[0].shape[0],
            )
            if tokenizer is None or tokenizer.action_tokenizer is None:
                raise ValueError(
                    "Action-token artifacts require a saved action tokenizer."
                )
            normalized_actions = detokenize_actions(
                action_tokens=output_tensors[0],
                action_tokenizer=tokenizer.action_tokenizer,
                action_space=self.action_space,
            )  # (B, L) -> (B, H, D)
            normalized_actions = to_device(
                normalized_actions, device=self.device
            )  # (B, H, D)
        else:
            if len(output_tensors) != len(self.output_keys):
                raise ValueError(
                    f"Compressed model returned {len(output_tensors)} tensors, "
                    f"but metadata declares {len(self.output_keys)} output keys."
                )
            normalized_actions = {
                key: output_tensors[index] for index, key in enumerate(self.output_keys)
            }
        return unnormalize_actions(
            normalized_actions=normalized_actions,
            normalizer=self.checkpoint_loader.normalizer,
            action_space=self.action_space,
        )  # (B, H, D)

    @staticmethod
    def _validate_token_output(action_tokens: torch.Tensor, batch_size: int) -> None:
        """Check the token tensor before action-token decoding.

        Args:
            action_tokens: Generated IDs with shape (batch, token_length) and
                dtype torch.int32 or torch.int64.
            batch_size: Number of observation samples processed by the artifact.

        Raises:
            ValueError: If token dtype, rank or batch size violates these
                requirements.
        """
        if action_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "Action-token artifacts require torch.int32 or torch.int64 token "
                f"outputs, got {action_tokens.dtype}."
            )
        if action_tokens.ndim != 2:
            raise ValueError(
                "Action-token outputs require shape (batch, token_length), "
                f"got {tuple(action_tokens.shape)}."
            )
        if action_tokens.shape[0] != batch_size:
            raise ValueError(
                f"Action-token output batch size {action_tokens.shape[0]} "
                f"must match observation batch size {batch_size}."
            )

    def _run_compressed_model(
        self,
        model_inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Execute the artifact with the inputs declared by its export metadata.

        Args:
            model_inputs: Tensors in the exported graph's argument order.

        Returns:
            Graph output tensors in the saved output-key order.

        Raises:
            RuntimeError: If loading the compressed model is still required.

        Note:
            B denotes batch size, H action horizon, D action dimension and L token
            length. Outputs have shape (B, H, D) for actions or (B, L) for tokens.
        """
        if self._compressed_model is None:
            raise RuntimeError("Compressed model has not been loaded.")
        with torch.no_grad():
            if (
                self.checkpoint_loader.artifact_format
                == ArtifactFormat.EXECUTORCH_PTE.value
            ):
                output_tensors = self._compressed_model(
                    model_inputs
                )  # actions: each (B, H, D); tokens: (B, L)
            else:
                output_tensors = self._compressed_model(
                    *model_inputs
                )  # actions: each (B, H, D); tokens: (B, L)

        if isinstance(output_tensors, torch.Tensor):
            return (output_tensors,)
        return tuple(output_tensors)

    @property
    def input_keys(self) -> list[str]:
        """Get compressed model input key ordering."""
        return self.checkpoint_loader.input_keys

    @property
    def output_keys(self) -> list[str]:
        """Get compressed model output key ordering."""
        return self.checkpoint_loader.output_keys
