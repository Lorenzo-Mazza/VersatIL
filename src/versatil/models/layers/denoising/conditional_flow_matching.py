"""Conditional flow matching utilities."""

import torch


def _pad_time_like_tensor(
    time: torch.Tensor | float | int,
    tensor: torch.Tensor,
) -> torch.Tensor | float | int:
    """Reshape a batch time vector for broadcasting with a sample tensor."""
    if isinstance(time, float | int):
        return time
    return time.reshape(-1, *([1] * (tensor.dim() - 1)))


class ConditionalFlowMatcher:
    """Implements the conditional flow matching interpolation API used by VersatIL.

    The path convention is ``t=0`` at source noise ``x0`` and ``t=1`` at target
    data ``x1``. For a batch time vector ``t`` and samples with arbitrary
    trailing dimensions, this class builds the Gaussian bridge
    ``x_t = t * x1 + (1 - t) * x0 + sigma * epsilon`` and the target velocity
    ``u_t = x1 - x0``.
    """

    def __init__(self, sigma: float = 0.0):
        """Initialize the matcher with a constant bridge noise standard deviation."""
        self.sigma = sigma

    def compute_mu_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the bridge mean ``mu_t = t * x1 + (1 - t) * x0``.

        Args:
            x0: Source samples, usually Gaussian noise, with shape ``(B, ...)``.
            x1: Target samples with the same shape as ``x0``.
            t: Per-sample times with shape ``(B,)``.
        """
        padded_time = _pad_time_like_tensor(time=t, tensor=x0)
        return padded_time * x1 + (1 - padded_time) * x0

    def compute_sigma_t(self, t: torch.Tensor) -> float:
        """Return the constant path standard deviation at time ``t``."""
        del t
        return self.sigma

    def compute_lambda(self, t: torch.Tensor) -> float:
        """Compute the score weighting coefficient used by score-CFM variants.

        VersatIL currently trains the velocity target directly, but this method
        preserves the standard matcher API for callers that expect the
        ``2 * sigma_t / sigma**2`` score weight.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``x_t`` from the conditional Gaussian bridge.

        Args:
            x0: Source samples, usually Gaussian noise.
            x1: Target samples.
            t: Per-sample times.
            epsilon: Standard Gaussian noise with the same shape as ``x0``.
        """
        mu_t = self.compute_mu_t(x0=x0, x1=x1, t=t)
        sigma_t = _pad_time_like_tensor(time=self.compute_sigma_t(t), tensor=x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the conditional velocity target ``u_t = x1 - x0``.

        The independent CFM path has a constant velocity along each source-target
        pair, so ``t`` and ``xt`` are accepted for API symmetry but do not affect
        the returned target.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sample standard Gaussian noise with the same shape as ``tensor``."""
        return torch.randn_like(tensor)

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor | None = None,
        return_noise: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Sample the objects needed for one CFM training target.

        If ``t`` is omitted, times are sampled uniformly in ``[0, 1]``. Returns
        ``(t, x_t, u_t)`` by default, or ``(t, x_t, u_t, epsilon)`` when
        ``return_noise`` is enabled.
        """
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        if len(t) != x0.shape[0]:
            raise ValueError(
                f"Time batch size {len(t)} must match sample batch size {x0.shape[0]}."
            )
        epsilon = self.sample_noise_like(x0)
        xt = self.sample_xt(x0=x0, x1=x1, t=t, epsilon=epsilon)
        conditional_flow = self.compute_conditional_flow(x0=x0, x1=x1, t=t, xt=xt)
        if return_noise:
            return t, xt, conditional_flow, epsilon
        return t, xt, conditional_flow
