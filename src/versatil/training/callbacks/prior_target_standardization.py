"""Callback for fitting learned-prior latent target standardization."""

import logging
from collections.abc import Iterable

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from versatil.common.tensor_ops import to_device
from versatil.data.constants import SampleKey
from versatil.models.decoding.latent.prior.latent_standardizer import LatentStandardizer
from versatil.models.policy import Policy
from versatil.training.constants import PrecisionType
from versatil.training.lightning_policy import LightningPolicy


class PriorTargetStandardizationCallback(Callback):
    """Fit latent target statistics for learned denoising priors."""

    def __init__(
        self,
        max_batches: int | None = None,
        require_posterior_frozen: bool = True,
    ) -> None:
        """Initialize the callback.

        Args:
            max_batches: Maximum train batches to scan when fitting stats. ``None``
                scans the full train dataloader.
            require_posterior_frozen: If true, fit only after the posterior encoder
                has no trainable parameters.

        Raises:
            ValueError: If ``max_batches`` is not positive when provided.
        """
        super().__init__()
        if max_batches is not None and max_batches <= 0:
            raise ValueError(
                f"PriorTargetStandardizationCallback max_batches must be positive, "
                f"got {max_batches}."
            )
        self.max_batches = max_batches
        self.require_posterior_frozen = require_posterior_frozen

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Fit stats if the initial training regime is already prior-only.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module wrapping the policy.
        """
        self._fit_if_ready(trainer=trainer, pl_module=pl_module)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Fit stats when staged training first makes the prior trainable.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module wrapping the policy.
        """
        self._fit_if_ready(trainer=trainer, pl_module=pl_module)

    @staticmethod
    def _require_lightning_policy(pl_module: pl.LightningModule) -> LightningPolicy:
        """Validate and return the expected Lightning policy wrapper.

        Args:
            pl_module: Lightning module passed by the trainer.

        Returns:
            The narrowed ``LightningPolicy`` instance.

        Raises:
            TypeError: If ``pl_module`` is not a ``LightningPolicy``.
        """
        if not isinstance(pl_module, LightningPolicy):
            raise TypeError(
                "PriorTargetStandardizationCallback requires a LightningPolicy module, "
                f"got {type(pl_module).__name__}."
            )
        return pl_module

    def _fit_if_ready(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Fit standardizer stats if runtime state satisfies the fit contract.

        Args:
            trainer: Active Lightning trainer providing the training precision.
            pl_module: Lightning module wrapping the policy.

        Raises:
            RuntimeError: If fitting is required but no train dataloader is available.
        """
        lightning_policy = self._require_lightning_policy(pl_module)
        policy = lightning_policy.policy
        algorithm = policy.algorithm
        prior = algorithm.prior
        posterior_encoder = algorithm.posterior_encoder
        standardizer = prior.latent_standardizer
        if not self._should_fit(
            standardizer=standardizer,
            prior=prior,
            posterior_encoder=posterior_encoder,
        ):
            return

        train_dataloader = lightning_policy.train_dataloader()
        if train_dataloader is None:
            raise RuntimeError(
                "PriorTargetStandardizationCallback requires a training dataloader "
                "before fitting latent statistics."
            )

        training_modes = self._capture_training_modes(policy)
        try:
            policy.eval()
            mean, std, count = self._collect_statistics(
                policy=policy,
                algorithm=algorithm,
                standardizer=standardizer,
                train_dataloader=train_dataloader,
                device=lightning_policy.device,
                precision_type=PrecisionType(str(trainer.precision)),
            )
        finally:
            self._restore_training_modes(training_modes)

        standardizer.set_stats(mean=mean, std=std)
        logging.info(
            f"Fitted DiT prior latent standardizer from {count} posterior targets."
        )

    def _should_fit(
        self,
        standardizer: LatentStandardizer,
        prior: torch.nn.Module,
        posterior_encoder: torch.nn.Module,
    ) -> bool:
        """Return whether latent statistics should be fitted now.

        Args:
            standardizer: Latent standardizer owned by the prior.
            prior: Prior module.
            posterior_encoder: Posterior encoder module.

        Returns:
            True when stats are enabled, unfitted, prior-trainable, and the
            posterior freeze requirement is satisfied.
        """
        if not standardizer.enabled:
            return False
        if bool(standardizer.is_fitted.item()):
            return False
        if not self._has_trainable_parameters(prior):
            return False
        return not (
            self.require_posterior_frozen
            and self._has_trainable_parameters(posterior_encoder)
        )

    @staticmethod
    def _has_trainable_parameters(module: torch.nn.Module) -> bool:
        """Return whether any module parameter requires gradients.

        Args:
            module: Module to inspect.

        Returns:
            True if at least one parameter is trainable.
        """
        return any(parameter.requires_grad for parameter in module.parameters())

    @staticmethod
    def _capture_training_modes(
        policy: Policy,
    ) -> list[tuple[torch.nn.Module, bool]]:
        """Capture module training modes before the stats pass.

        Args:
            policy: Policy whose module modes should be restored after fitting.

        Returns:
            Pairs of module and its previous ``training`` value.
        """
        return [(module, module.training) for module in policy.modules()]

    @staticmethod
    def _restore_training_modes(
        training_modes: list[tuple[torch.nn.Module, bool]],
    ) -> None:
        """Restore module training modes after the stats pass.

        Args:
            training_modes: Pairs returned by ``_capture_training_modes``.
        """
        for module, was_training in training_modes:
            module.train(was_training)

    def _collect_statistics(
        self,
        policy: Policy,
        algorithm: torch.nn.Module,
        standardizer: torch.nn.Module,
        train_dataloader: Iterable,
        device: torch.device,
        precision_type: PrecisionType,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Stream posterior targets and compute latent mean/std.

        Args:
            policy: Policy used to encode observations.
            algorithm: Variational algorithm with posterior and prior modules.
            standardizer: Latent standardizer receiving the fitted stats.
            train_dataloader: Dataloader yielding policy training batches.
            device: Device used for the stats pass.
            precision_type: Trainer precision; mixed precisions run the stats
                pass under the same autocast as training so the fitted stats
                match the latents the prior sees.

        Returns:
            Tuple of mean, standard deviation, and number of observed targets.

        Raises:
            RuntimeError: If target dimensionality is wrong or no targets are seen.
        """
        latent_dimension = int(standardizer.latent_dimension)
        sum_latents = torch.zeros(latent_dimension, device=device, dtype=torch.float64)
        sum_squared_latents = torch.zeros_like(sum_latents)
        count = torch.zeros((), device=device, dtype=torch.float64)

        with torch.no_grad(), precision_type.autocast(device_type=device.type):
            for batch_index, batch in enumerate(train_dataloader):
                if self.max_batches is not None and batch_index >= self.max_batches:
                    break
                observations, actions = self._split_batch(batch=batch)
                observations = to_device(observations, device=device)
                actions = to_device(actions, device=device)
                features = policy._build_algorithm_features(observation=observations)
                posterior_output = algorithm.posterior_encoder.encode(
                    actions=actions,
                    observations=features,
                )
                target_latents = algorithm.prior.build_training_target(posterior_output)
                if target_latents.shape[-1] != latent_dimension:
                    raise RuntimeError(
                        "Prior target latent dimension mismatch: expected "
                        f"{latent_dimension}, got {target_latents.shape[-1]}."
                    )
                flattened_latents = target_latents.detach().reshape(
                    -1, latent_dimension
                )
                flattened_latents = flattened_latents.to(
                    device=device,
                    dtype=torch.float64,
                )
                sum_latents += flattened_latents.sum(dim=0)
                sum_squared_latents += flattened_latents.square().sum(dim=0)
                count += flattened_latents.shape[0]

        self._sync_distributed_stats(
            sum_latents=sum_latents,
            sum_squared_latents=sum_squared_latents,
            count=count,
        )
        if count.item() == 0:
            raise RuntimeError(
                "PriorTargetStandardizationCallback could not fit latent statistics "
                "from an empty training dataloader."
            )
        mean = sum_latents / count
        variance = (sum_squared_latents / count) - mean.square()
        std = variance.clamp(min=standardizer.epsilon**2).sqrt()
        return mean, std, int(count.item())

    @staticmethod
    def _split_batch(
        batch: dict[str, dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Split a policy batch into observation and action dictionaries.

        Args:
            batch: Training batch using the standard sample-key structure.

        Returns:
            Tuple of observation dict and action dict.

        Raises:
            RuntimeError: If required batch keys are missing.
        """
        if (
            SampleKey.OBSERVATION.value not in batch
            or SampleKey.ACTION.value not in batch
        ):
            raise RuntimeError(
                "PriorTargetStandardizationCallback expects batches with "
                f"'{SampleKey.OBSERVATION.value}' and '{SampleKey.ACTION.value}' keys."
            )
        return batch[SampleKey.OBSERVATION.value], batch[SampleKey.ACTION.value]

    @staticmethod
    def _sync_distributed_stats(
        sum_latents: torch.Tensor,
        sum_squared_latents: torch.Tensor,
        count: torch.Tensor,
    ) -> None:
        """Synchronize streaming statistics across distributed workers.

        Args:
            sum_latents: Running latent sums.
            sum_squared_latents: Running squared latent sums.
            count: Running target count.
        """
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        torch.distributed.all_reduce(sum_latents, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            sum_squared_latents,
            op=torch.distributed.ReduceOp.SUM,
        )
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
