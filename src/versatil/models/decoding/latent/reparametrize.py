import torch


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparametrization trick from Bayes by backprop: sample from N(mu, var) and still do backpropagation.

    Args:
        mu: Mean of the latent distribution, shape (batch, latent_dim).
        logvar: Log variance of the latent distribution, shape (batch, latent_dim).

    Returns:
        Sampled latent vector, shape (batch, latent_dim).
    """
    std = (logvar / 2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps
