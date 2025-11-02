"""Composite loss classes that combine multiple loss components."""


import torch
import torch.nn as nn

from refactoring.data.constants import (
    POSITION_ACTION_KEY,
    GripperType,
)
from refactoring.metrics.base import BaseLoss, LossOutput
from refactoring.metrics.components import (
    GripperLoss,
    KLDivergenceLoss,
    PhaseClassificationLoss,
    RegressionLoss,
    TrajectoryLengthLoss,
    TrajectorySmoothness,
)
from refactoring.metrics.constants import LossModuleName


class CompositeLoss(BaseLoss):
    """Composite loss that combines multiple loss modules with weights.

    This loss module orchestrates multiple sub-losses and combines them
    with configurable weights. It's useful for complex training objectives
    that involve multiple loss terms.
    """

    def __init__(
        self,
        loss_modules: dict[str, BaseLoss],
        weights: dict[str, float] | None = None,
    ):
        """Initialize composite loss.

        Args:
            loss_modules: Dictionary of loss module names to loss instances
            weights: Optional dictionary of weights for each loss module
        """
        super().__init__()
        self.loss_modules = nn.ModuleDict(loss_modules)
        self.weights = weights or dict.fromkeys(loss_modules.keys(), 1.0)

    def get_required_keys(self) -> set[str]:
        """Get required target keys by recursively collecting from all sub-modules.

        Returns:
            Union of all required keys from all sub-modules
        """
        required_keys = set()
        for loss_module in self.loss_modules.values():
            required_keys.update(loss_module.get_required_keys())
        return required_keys

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute weighted sum of all loss modules.

        Args:
            predictions: Model output dictionary
            targets: Ground truth dictionary
            is_pad: Optional padding mask

        Returns:
            LossOutput with total weighted loss and all component losses
        """
        device = next(iter(predictions.values())).device
        total_loss = torch.tensor(0.0, device=device)
        all_component_losses = {}
        all_metadata = {}

        for name, loss_module in self.loss_modules.items():
            loss_output = loss_module(predictions, targets, is_pad)
            weight = self.weights.get(name, 1.0)
            total_loss = total_loss + weight * loss_output.total_loss

            for comp_name, comp_value in loss_output.component_losses.items():
                prefixed_name = f"{name}/{comp_name}"
                all_component_losses[prefixed_name] = comp_value

            all_metadata.update(loss_output.metadata)

        return LossOutput(
            total_loss=total_loss,
            component_losses=all_component_losses,
            metadata=all_metadata,
        )


class ActionReconstructionLoss(BaseLoss):
    """Complete loss for action reconstruction in ACT-style models.

    Combines regression losses for position/orientation, gripper loss,
    VAE KL divergence, and optional trajectory regularization.
    """

    def __init__(
        self,
        action_keys: list[str] | None = None,
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        gripper_bce_weight: float = 1.0,
        kl_weight: float = 0.0001,
        length_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        gripper_type: str = GripperType.BINARY.value,
        use_vae: bool = False,
    ):
        """Initialize action reconstruction loss.

        Args:
            action_keys: List of action keys (default: position and orientation)
            mse_weight: Weight for MSE loss on continuous actions
            l1_weight: Weight for L1 loss on continuous actions
            gripper_bce_weight: Weight for gripper BCE loss
            kl_weight: Weight for VAE KL divergence
            length_weight: Weight for trajectory length loss
            smoothness_weight: Weight for trajectory smoothness loss
            gripper_type: Type of gripper ('binary' or 'continuous')
            use_vae: Whether to include KL divergence loss
        """
        super().__init__()

        if action_keys is None:
            action_keys = [POSITION_ACTION_KEY]

        loss_modules: dict[str, BaseLoss] = {}

        if len(action_keys) > 0:
            loss_modules[LossModuleName.REGRESSION.value] = RegressionLoss(
                action_keys=action_keys,
                mse_weight=mse_weight,
                l1_weight=l1_weight,
            )

        loss_modules[LossModuleName.GRIPPER.value] = GripperLoss(
            gripper_type=gripper_type,
            bce_weight=gripper_bce_weight,
            mse_weight=mse_weight,
        )

        if use_vae:
            loss_modules[LossModuleName.KL.value] = KLDivergenceLoss(weight=kl_weight)

        if length_weight > 0:
            loss_modules[LossModuleName.LENGTH.value] = TrajectoryLengthLoss(
                weight=length_weight, action_key=POSITION_ACTION_KEY
            )

        if smoothness_weight > 0:
            loss_modules[LossModuleName.SMOOTHNESS.value] = TrajectorySmoothness(
                weight=smoothness_weight, action_key=POSITION_ACTION_KEY
            )

        self.composite = CompositeLoss(loss_modules=loss_modules)

    def get_required_keys(self) -> set[str]:
        """Get required target keys from the composite loss.

        Returns:
            Union of all required keys from all sub-losses
        """
        return self.composite.get_required_keys()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute action reconstruction loss.

        Args:
            predictions: Dictionary with predicted actions
            targets: Dictionary with ground truth actions
            is_pad: Optional padding mask

        Returns:
            LossOutput with total loss and all components
        """
        return self.composite(predictions, targets, is_pad)  # type: ignore[no-any-return]


class PhaseActionLoss(BaseLoss):
    """Loss for phase-conditioned action prediction (PhaseACT models).

    Combines action reconstruction loss with phase classification loss.
    """

    def __init__(
        self,
        action_keys: list[str] | None = None,
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        gripper_bce_weight: float = 1.0,
        kl_weight: float = 0.0001,
        length_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        phase_ce_weight: float = 1.0,
        phase_entropy_weight: float = 0.0,
        label_smoothing: float = 0.0,
        gripper_type: str = GripperType.BINARY.value,
        use_vae: bool = False,
    ):
        """Initialize phase action loss.

        Args:
            action_keys: List of action keys
            mse_weight: Weight for MSE loss on continuous actions
            l1_weight: Weight for L1 loss on continuous actions
            gripper_bce_weight: Weight for gripper BCE loss
            kl_weight: Weight for VAE KL divergence
            length_weight: Weight for trajectory length loss
            smoothness_weight: Weight for trajectory smoothness loss
            phase_ce_weight: Weight for phase cross-entropy loss
            phase_entropy_weight: Weight for phase entropy regularization
            label_smoothing: Label smoothing for phase classification
            gripper_type: Type of gripper
            use_vae: Whether to include KL divergence loss
        """
        super().__init__()

        loss_modules = {
            LossModuleName.ACTION.value: ActionReconstructionLoss(
                action_keys=action_keys,
                mse_weight=mse_weight,
                l1_weight=l1_weight,
                gripper_bce_weight=gripper_bce_weight,
                kl_weight=kl_weight,
                length_weight=length_weight,
                smoothness_weight=smoothness_weight,
                gripper_type=gripper_type,
                use_vae=use_vae,
            ),
            LossModuleName.PHASE.value: PhaseClassificationLoss(
                cross_entropy_weight=phase_ce_weight,
                entropy_weight=phase_entropy_weight,
                label_smoothing=label_smoothing,
            ),
        }

        self.composite = CompositeLoss(loss_modules=loss_modules)

    def get_required_keys(self) -> set[str]:
        """Get required target keys from the composite loss.

        Returns:
            Union of all required keys from all sub-losses
        """
        return self.composite.get_required_keys()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute phase action loss.

        Args:
            predictions: Dictionary with predicted actions and phases
            targets: Dictionary with ground truth actions and phases
            is_pad: Optional padding mask

        Returns:
            LossOutput with total loss and all components
        """
        return self.composite(predictions, targets, is_pad)  # type: ignore[no-any-return]


