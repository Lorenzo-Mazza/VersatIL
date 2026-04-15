# Algorithms

Algorithms define the **learning paradigm** -- how the policy is trained and how it generates actions at inference time. They are decoupled from the decoder architecture: the algorithm orchestrates the decoder without knowing its internals. Certain pairings are naturally constrained by their mathematical formulation (e.g., timestep-conditioned decoders require a generative algorithm that provides timesteps).

All algorithms inherit from `DecodingAlgorithm` and implement these methods:

| Method | Purpose | Actions required? |
|--------|---------|-------------------|
| `forward()` | Training pass | Yes (ground-truth actions) |
| `predict()` | Inference pass | No |
| `get_targets()` | Provide regression targets for the loss module | Yes |

```python
class DecodingAlgorithm(nn.Module, abc.ABC):

    @abstractmethod
    def forward(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def predict(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]: ...

    def get_targets(
        self,
        algorithm_output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        ground_truth_actions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return the correct regression targets for the loss module.
        Default returns ground-truth actions (correct for BC).
        """
        return ground_truth_actions

    @property
    def predicts_in_action_space(self) -> bool:
        """Whether the network output lives in the action space."""
        return True
```

The `forward()` / `predict()` contract ensures a clean separation: training logic (noise injection, flow interpolation, latent encoding) lives in the algorithm, while the neural network computation lives in the decoder.

### Algorithm Targets

Different algorithms predict different quantities. The loss module must compare predictions against the correct target, not raw ground-truth actions. `Policy.compute_loss()` calls `algorithm.get_targets()` to obtain the right regression target:

| Algorithm | `get_targets()` returns | `predicts_in_action_space` |
|---|---|---|
| `BehavioralCloning` | Ground-truth actions | `True` |
| `FlowMatching` | Velocity field | `False` |
| `Diffusion` (epsilon) | Noise | `False` |
| `Diffusion` (sample) | Denoised sample | `True` |
| `Diffusion` (velocity) | Velocity | `False` |

The `predicts_in_action_space` property enables loss-algorithm compatibility validation. Classification losses (e.g., BCE for gripper) require action-space predictions -- pairing them with Flow Matching (which predicts velocity fields) is caught at initialization by `ExperimentValidator`.

---

## BehavioralCloning

The simplest algorithm. Directly predicts actions from observations via supervised learning. Both `forward()` and `predict()` delegate to the decoder network without modification.

```python
class BehavioralCloning(DecodingAlgorithm):

    def forward(self, network, features, actions=None):
        return network(features=features, actions=actions)

    def predict(self, network, features):
        return network(features=features, actions=None)
```


---

## Diffusion

Generative modeling via Denoising Score Matching. Trains the network to denoise actions at various noise levels, then generates actions through iterative denoising at inference.

**Training (`forward`):**

1. Sample random timesteps `t` from `[0, num_train_timesteps]`
2. Add noise to ground-truth actions: `x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon`
3. Pass noisy actions + timesteps to the decoder
4. Compute target based on `prediction_type` (epsilon, sample, or velocity)

**Inference (`predict`):**

1. Initialize actions from pure noise `x_T ~ N(0, I)`
2. Iteratively denoise using the learned model for `num_inference_steps` steps
3. Return the final denoised actions `x_0`

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scheduler_type` | `"ddim"` | Scheduler type (`"ddpm"` or `"ddim"`) |
| `num_train_timesteps` | `100` | Diffusion steps during training |
| `num_inference_steps` | `10` | Denoising steps during inference |
| `beta_schedule` | `"squaredcos_cap_v2"` | Noise schedule shape |
| `prediction_type` | `"epsilon"` | Network prediction target (`"epsilon"`, `"sample"`, `"velocity"`) |

!!! note "Encoder caching"
    When used with `DiTBlockActionTransformer`, the diffusion algorithm automatically enables encoder caching during inference. The observation encoder runs once and its output is reused across all denoising steps.

---

## FlowMatching

Generative modeling via Continuous Normalizing Flows. Trains the network to predict velocity fields that transport samples from noise to actions.

**Training (`forward`):**

1. Sample time `t` from `[0, 1]` using the configured timestep sampler
2. Interpolate between noise `x_0` and ground-truth actions `x_1` at time `t`
3. Compute the conditional velocity field `u_t`
4. Train the network to predict `u_t`

**Inference (`predict`):**

1. Initialize from noise `z ~ N(0, I)`
2. Integrate the learned velocity field from `t=0` to `t=1` using an ODE solver
3. Return the final trajectory

**Key parameters:**

| Parameter | Default | Description                                                               |
|-----------|---------|---------------------------------------------------------------------------|
| `sigma` | `0.0` | Noise level for CFM (0 = deterministic optimal transport)                 |
| `num_inference_steps` | `10` | Number of ODE integration steps                                           |
| `ode_solver` | `"euler"` | Solver type (`"euler"`, `"heun"`, `"rk4"`)                                |
| `timestep_sampler` | `"beta"` | Sampling strategy (`"uniform"`, `"logit_normal"`, `"beta"`)               |
| `beta_alpha` / `beta_beta` | `1.5` / `1.0` | Shape parameters for Beta distribution sampler                            |
| `max_timestep` | `0.999` | Upper bound for timestep sampling                                         |
| `reverse_flow_convention` | `False` | When `True`, reverses the flow convention (noise at `t=1`, data at `t=0`) |

!!! tip "Timestep sampling"
    The `beta` sampler (from Pi0) biases training towards later timesteps where the signal-to-noise ratio is higher, improving sample quality. The `logit_normal` sampler provides similar control via `logit_mean` and `logit_std`.

---

## VariationalAlgorithm

A compositional wrapper that adds variational inference to **any** base algorithm. Instead of predicting actions directly from observations, it introduces a latent variable `z` that captures multi-modal action distributions:

```
p(a|s) = integral p(a|z,s) p(z|s) dz
```

Where:

- `p(a|z,s)` is the base algorithm's decoder
- `q(z|a,s)` is the posterior encoder (training only)
- `p(z|s)` is the prior (inference)

### Architecture

```python
VariationalAlgorithm(
    base_algorithm=<any DecodingAlgorithm>,
    posterior_encoder=<PosteriorLatentEncoder>,
    prior=<PriorLatentEncoder | None>,
    sampling_from_prior_probability=0.0,
)
```

The wrapper composes three independent components:

| Component | Role | Available at |
|-----------|------|-------------|
| **Posterior encoder** `q(z\|a,s)` | Encodes ground-truth actions into latent `z` | Training only |
| **Prior** `p(z\|s)` | Samples latent `z` without access to actions | Training + Inference |
| **Base algorithm** | Decodes actions given `z` and observations | Training + Inference |

### Training Flow

```
                    observations + actions
                           |
               +-----------+-----------+
               |                       |
     Posterior q(z|a,s)          Prior p(z|s)
               |                       |
           z_posterior             z_prior
               |                       |
               +-------select----------+
                          |
                   features + z
                          |
                   Base Algorithm
                          |
                     predictions
```

1. **Posterior encoding:** `z ~ q(z|a,s)` -- encode **ground-truth** actions into latent space
2. **Prior training:** train the prior to match posterior samples (for learned priors)
3. **Stochastic mixing:** with probability `sampling_from_prior_probability`, use `z` from the prior instead of the posterior during training
4. **Decoding:** augment features with `z` and delegate to the base algorithm

During validation, the prior sample is always used for action decoding (matching inference behavior), while the posterior is still computed for loss (KL term).

### Inference Flow

```
          observations
               |
         Prior p(z|s)
               |
           z_sampled
               |
        features + z
               |
        Base Algorithm
               |
          predictions
```

1. **Sample latent:** `z ~ p(z|s)` from the prior
2. **Decode:** pass `features + z` to the base algorithm's `forward()` with `actions=None`

### Posterior Encoder

All posterior encoders inherit from `PosteriorLatentEncoder`, which defines the `encode(actions, observations)` interface. Custom posterior encoders can be created by subclassing it.

**`VAETransformerEncoder`** -- the built-in posterior encoder. A transformer encoder that processes action chunks and observation tokens, using a learnable CLS token to predict the mean and log-variance of a conditional Gaussian posterior. The latent `z` is sampled via the reparameterization trick.

Supports a `deterministic` mode (no reparameterization) for use with non-KL regularizers such as MMD or Optimal Transport losses.

### Prior Types

All priors inherit from `PriorLatentEncoder`, which defines `forward(target_latents, observations)` for training and `sample_prior(batch_size, observations)` for inference. Custom priors can be created by subclassing it — any learned network that maps observations to a latent distribution can serve as a prior, enabling probabilistic student-teacher schemes where the prior learns to approximate the posterior without access to actions.

| Prior | Type | Description |
|-------|------|-------------|
| `GaussianPrior` | Fixed | Standard `N(0, I)`. No trainable parameters. Default when `prior=None`. |
| `PriorTransformerEncoder` | Learned | Transformer encoder conditioned on observations. Predicts `mu` and `logvar` of a conditional Gaussian `p(z\|s)`. |
| `DiTPrior` | Learned | DiT-style transformer trained via diffusion or flow matching to denoise latent samples. Supports both `"diffusion"` and `"flow_matching"` algorithm types. |
| `VampPrior` | Learned | Variational Mixture of Posteriors. Defines `K` learnable pseudo-inputs passed through the posterior encoder to form a mixture-of-Gaussians prior. |

!!! info "GaussianPrior auto-creation"
    If `prior=None` is passed, a `GaussianPrior` is automatically created with the same `latent_dimension` as the posterior encoder.

!!! warning "VampPrior initialization"
    `VampPrior` requires access to the posterior encoder to compute mixture components. The `VariationalAlgorithm` automatically calls `prior.set_encoder(posterior_encoder)` during initialization.

### Example Combinations

**BC + VAE (standard VAE, as in ACT):**

```python
VariationalAlgorithm(
    base_algorithm=BehavioralCloning(),
    posterior_encoder=VAETransformerEncoder(...),
    prior=None,  # Auto-creates GaussianPrior N(0, I)
)
```

**BC + Learned Conditional Prior:**

```python
VariationalAlgorithm(
    base_algorithm=BehavioralCloning(),
    posterior_encoder=VAETransformerEncoder(...),
    prior=PriorTransformerEncoder(...),
)
```

**Flow Matching + DiT Prior:**

```python
VariationalAlgorithm(
    base_algorithm=FlowMatching(...),
    posterior_encoder=VAETransformerEncoder(...),
    prior=DiTPrior(algorithm_type="flow_matching", ...),
)
```

**Diffusion + VampPrior:**

```python
VariationalAlgorithm(
    base_algorithm=Diffusion(...),
    posterior_encoder=VAETransformerEncoder(...),
    prior=VampPrior(num_components=10, ...),
)
```

### Source Locations

| Component | Path |
|-----------|------|
| `DecodingAlgorithm` | `src/versatil/models/decoding/algorithm/base.py` |
| `BehavioralCloning` | `src/versatil/models/decoding/algorithm/behavior_cloning.py` |
| `Diffusion` | `src/versatil/models/decoding/algorithm/diffusion.py` |
| `FlowMatching` | `src/versatil/models/decoding/algorithm/flow_matching.py` |
| `VariationalAlgorithm` | `src/versatil/models/decoding/algorithm/variational.py` |
| `VAETransformerEncoder` | `src/versatil/models/decoding/latent/posterior/transformer_encoder.py` |
| `GaussianPrior` | `src/versatil/models/decoding/latent/prior/gaussian_prior.py` |
| `PriorTransformerEncoder` | `src/versatil/models/decoding/latent/prior/transformer_encoder.py` |
| `DiTPrior` | `src/versatil/models/decoding/latent/prior/dit_prior.py` |
| `VampPrior` | `src/versatil/models/decoding/latent/prior/vamp_prior.py` |
