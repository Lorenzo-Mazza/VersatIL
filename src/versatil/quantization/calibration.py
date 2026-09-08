"""Representative observation batches for quantization calibration."""

from collections.abc import Iterator
from itertools import islice

import torch
from torch.utils.data import DataLoader

from versatil.data.constants import SampleKey
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.models.policy import Policy
from versatil.post_training_compression.policy_context import PolicyContext


class CalibrationDataProvider:
    """Yield named model observations from a VersatIL dataloader.

    Note:
        The dataset must apply the policy's normalization and tokenization.
        This provider selects observations and moves them to the chosen device,
        preserving their values, shapes and dtypes.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        observation_keys: list[str],
        num_calibration_steps: int = 128,
        device: torch.device | None = None,
    ) -> None:
        """Initialize calibration data provider.

        Args:
            dataloader: VersatIL dataloader yielding normalized, tokenized samples.
            observation_keys: Model input keys to select from each observation.
            num_calibration_steps: Maximum number of calibration batches.
            device: Device for calibration tensors. Defaults to CPU.

        Raises:
            ValueError: If the batch limit is less than one.
        """
        if num_calibration_steps < 1:
            raise ValueError(
                f"num_calibration_steps must be positive, got {num_calibration_steps}."
            )
        self._dataloader = dataloader
        self._observation_keys = tuple(observation_keys)
        self._num_calibration_steps = num_calibration_steps
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield selected observations on the configured device.

        Yields:
            A dictionary of normalized and tokenized model input tensors.

        Raises:
            ValueError: If the dataloader is empty or a required input is missing.

        Note:
            B denotes batch size; remaining dimensions depend on the input key.
        """
        consumed_batches = 0
        for batch in islice(self._dataloader, self._num_calibration_steps):
            observation = batch[SampleKey.OBSERVATION.value]
            missing_keys = [
                key for key in self._observation_keys if key not in observation
            ]
            if missing_keys:
                raise ValueError(
                    f"Calibration batch is missing observations: {missing_keys}."
                )
            consumed_batches += 1
            yield {
                key: observation[key].to(device=self.device)  # (B, ...)
                for key in self._observation_keys
            }
        if consumed_batches == 0:
            raise ValueError("Calibration dataloader yielded no observation batches.")


def build_calibration_data(
    context: PolicyContext,
    observation_keys: list[str],
    num_calibration_steps: int,
    device: torch.device,
) -> CalibrationDataProvider:
    """Build calibration batches from a checkpoint's training dataset.

    Args:
        context: Loaded policy, training configuration and fitted tokenizer.
        observation_keys: Model input keys to include in each batch.
        num_calibration_steps: Maximum number of batches to consume.
        device: Device on which the observed model will execute.

    Returns:
        Observation batches using the checkpoint's normalizer and tokenizer.

    Raises:
        ValueError: If the batch limit is less than one.

    Note:
        Image augmentation and batch shuffling are disabled. Dataset construction
        and decoding errors propagate to the workflow requesting calibration.
    """
    if num_calibration_steps < 1:
        raise ValueError(
            f"num_calibration_steps must be positive, got {num_calibration_steps}."
        )
    dataset = EpisodicDataset(
        zarr_path=context.config.task.dataset_schema.zarr_path,
        action_space=context.config.task.action_space,
        observation_space=context.observation_space,
        dataloader_config=context.config.task.dataloader,
        pred_horizon=context.config.task.prediction_horizon,
        obs_horizon=context.observation_horizon,
        train=True,
        seed=context.config.experiment.seed,
        augment_images=False,
    )
    dataset.set_normalizer(normalizer=context.policy.normalizer)
    dataset.set_tokenizer(tokenizer=context.tokenizer)
    calibration_loader = DataLoader(
        dataset=dataset,
        batch_size=context.config.task.dataloader.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return CalibrationDataProvider(
        dataloader=calibration_loader,
        observation_keys=observation_keys,
        num_calibration_steps=num_calibration_steps,
        device=device,
    )


def calibrate_policy(policy: Policy, calibration: CalibrationDataProvider) -> int:
    """Run complete policy predictions to update installed quantization observers.

    Args:
        policy: Prepared policy in evaluation mode. Observers must be installed
            before calling this function.
        calibration: Normalized and tokenized observations on the policy device.

    Returns:
        Number of observation batches processed through the prediction algorithm.

    Raises:
        ValueError: If the policy is in training mode or calibration data is empty.

    Note:
        Gradient recording is disabled. Each batch runs the full denoising or
        generation sequence configured on the policy. B denotes batch size; the
        prediction dimensions depend on the policy's action representation.
    """
    if policy.training:
        raise ValueError(
            "Policy calibration requires evaluation mode. Call policy.eval() first."
        )
    batches = 0
    with torch.no_grad():
        for observation in calibration:
            policy.predict_from_processed_observation(
                observation=observation
            )  # (B, ...)
            batches += 1
    if batches == 0:
        raise ValueError("Calibration data yielded no observation batches.")
    return batches
