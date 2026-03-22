"""VAE-based deep generative sampling.

Based on Sec. 2.1.1 (Deep Generative Models) of
Gao et al., 2025:
    "VAE maps observed data points into a latent space
     while simultaneously learning the underlying data
     structure and probabilistic characteristics."

Reference
---------
- Wan et al., 2017 (Ref [19])

This module provides a skeleton VAE that can be trained
on minority-class images and then sampled to produce
synthetic data for balancing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _Encoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden: int, latent: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(
        self, latent: int, hidden: int, output_dim: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent, hidden)
        self.fc_out = nn.Linear(hidden, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))


class VAESampler:
    """Variational Autoencoder for minority oversampling.

    Train on flattened minority-class samples, then draw
    new synthetic vectors from the learned latent space.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dim : int
        Hidden layer width for encoder / decoder.
    lr : float
        Learning rate for Adam optimizer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for VAE training.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self._encoder: _Encoder | None = None
        self._decoder: _Decoder | None = None
        self._input_dim: int = 0

    def fit(
        self, data: torch.Tensor,
        device: torch.device | None = None,
    ) -> VAESampler:
        """Train the VAE on ``data`` (N, D) tensor.

        Parameters
        ----------
        data : torch.Tensor
            Flattened sample matrix, shape ``(N, D)``.
        device : torch.device, optional
            Training device.  Defaults to CPU.
        """
        torch.manual_seed(self.seed)
        dev = device or torch.device("cpu")

        self._input_dim = data.shape[1]
        self._encoder = _Encoder(
            self._input_dim, self.hidden_dim,
            self.latent_dim,
        ).to(dev)
        self._decoder = _Decoder(
            self.latent_dim, self.hidden_dim,
            self._input_dim,
        ).to(dev)

        params = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
        )
        opt = torch.optim.Adam(params, lr=self.lr)

        loader = DataLoader(
            TensorDataset(data.to(dev)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(self.epochs):
            for (batch,) in loader:
                mu, logvar = self._encoder(batch)
                z = _reparameterize(mu, logvar)
                recon = self._decoder(z)

                loss = _vae_loss(
                    recon, batch, mu, logvar,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self

    @torch.no_grad()
    def sample(
        self, n: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Draw ``n`` synthetic samples from the prior.

        Returns shape ``(n, input_dim)``.
        """
        if self._decoder is None:
            raise RuntimeError("Call .fit() first.")

        dev = device or torch.device("cpu")
        z = torch.randn(
            n, self.latent_dim, device=dev,
        )
        return self._decoder(z)


def _reparameterize(
    mu: torch.Tensor, logvar: torch.Tensor,
) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy(
        recon, target, reduction="sum",
    )
    kld = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
    )
    return bce + kld
