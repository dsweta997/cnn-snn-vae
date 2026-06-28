"""
Spiking Variational Autoencoder (VAE) for event-camera data.

The VAE uses purely spatial time-distributed convolutions (Conv3d with
temporal kernel = 1) so that the T time-bins of an event frame are processed
independently, preserving the temporal structure of the spike representation.

Latent space
------------
The posterior q(z|x) is a Bernoulli distribution whose rate r_q is obtained
by averaging the encoder output across T and then replicating it over T to
sample a binary spike tensor z_q via SampledSpikeAct.  The prior p(z) is
parameterised by r_p = sigmoid(W·ε), ε ~ N(0, I).  Training minimises:

    L = MSE(x_recon, x) + λ · MMD(r_q, r_p)

This is the InfoVAE / WAE formulation with an MMD penalty instead of KL.

Architecture (NMNIST default)
------------------------------
Encoder : tdConv blocks  1 → 32 → 64 → 128 → 256 → 512
          (each block: Conv3d + tdBatchNorm + LIFSpike, spatial stride 2)
Latent  : tdLinear  512·H'·W' → latent_dim  (+ tdBatchNorm + LIFSpike)
          gaussian_sample : r_q, r_p → z_q  (SampledSpikeAct)
Decoder : tdLinear  latent_dim → 512·H'·W'
          tdConvTranspose blocks (mirror of encoder)
          final tdConvTranspose  → 1 output channel

For PokerDVS the channel plan is  1 → 64 → 128 and the bottleneck spatial
size is 16×16 = 256 rather than 2×2 = 4.  These differences are controlled
by the `hidden_dims` and `bottleneck_hw` constructor arguments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import tdConv, tdConvTranspose, tdLinear, tdBatchNorm, LIFSpike, SampledSpikeAct
from .losses import MMD_loss


class VAE(nn.Module):
    """
    Spiking VAE with time-distributed conv/linear blocks and an MMD prior.

    Args:
        hidden_dims (list[int]): Channel progression for the encoder
            (e.g. [32,64,128,256,512] for NMNIST, [64,128] for PokerDVS).
        latent_dim (int): Dimensionality of the latent code per time step.
        n_steps (int): Number of time bins T in the input.
        bottleneck_hw (tuple[int,int]): Spatial size (H', W') of the encoder
            output just before the latent linear.  Determined by the input
            resolution and the number of stride-2 conv blocks.
        decoder_output_padding (int | tuple): output_padding for the
            intermediate decoder transpose-conv stages.  Use 0 for NMNIST
            (odd input → intermediate feature maps are odd) and (1,1,0) for
            PokerDVS (even input → need +1 to recover original resolution).
        in_channels (int): Input channels (1 for single-polarity frames).
        distance_lambda (float): Weight λ for the MMD term in the loss.
        mmd_type (str): 'linear' or 'rbf' kernel for MMD_loss.
        device (str): Device string used for latent sampling.
    """

    def __init__(
        self,
        hidden_dims: list[int]      = None,
        latent_dim:  int            = 64,
        n_steps:     int            = 16,
        bottleneck_hw: tuple        = (2, 2),
        decoder_output_padding: int | tuple = 0,
        in_channels: int            = 1,
        distance_lambda: float      = 0.001,
        mmd_type:    str            = "rbf",
        device:      str            = "cuda:0",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.hidden_dims            = list(hidden_dims)
        self.latent_dim             = latent_dim
        self.n_steps                = n_steps
        self.bottleneck_hw          = bottleneck_hw
        self.decoder_output_padding = decoder_output_padding
        self.device                 = device
        self.distance_lambda        = distance_lambda
        self.mmd_type               = mmd_type
        self.p                      = 0   # scheduling placeholder

        bh, bw  = bottleneck_hw
        flat_sz = hidden_dims[-1] * bh * bw   # flattened spatial size at bottleneck

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layers = []
        in_ch      = in_channels
        for h_dim in hidden_dims:
            enc_layers.append(
                tdConv(
                    in_ch, h_dim,
                    kernel_size=3, stride=2, padding=1, bias=True,
                    bn=tdBatchNorm(h_dim),
                    spike=LIFSpike(),
                )
            )
            in_ch = h_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Time-distributed projection to latent space
        self.before_latent_layer = tdLinear(
            flat_sz, latent_dim, bias=True,
            bn=tdBatchNorm(latent_dim),
            spike=LIFSpike(),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # Time-distributed expansion back to spatial grid
        self.decoder_input = tdLinear(
            latent_dim, flat_sz, bias=True,
            bn=tdBatchNorm(flat_sz),
            spike=LIFSpike(),
        )

        dec_hidden = list(reversed(hidden_dims))  # e.g. [512,256,...,32]

        # Intermediate upsampling stages (all but the last hidden dim)
        dec_layers = []
        for i in range(len(dec_hidden) - 1):
            dec_layers.append(
                tdConvTranspose(
                    dec_hidden[i], dec_hidden[i + 1],
                    kernel_size=3, stride=2, padding=1,
                    output_padding=decoder_output_padding,
                    bias=True,
                    bn=tdBatchNorm(dec_hidden[i + 1]),
                    spike=LIFSpike(),
                )
            )
        self.decoder = nn.Sequential(*dec_layers)

        # Final stage: last hidden dim → 1 output channel (no BN / spike)
        self.final_layer = tdConvTranspose(
            dec_hidden[-1], out_channels=1,
            kernel_size=3, stride=2, padding=1,
            output_padding=(1, 1, 0),   # always needed for the final spatial recovery
            bias=True, bn=None, spike=None,
        )

        # ── Prior and MMD ─────────────────────────────────────────────────────
        # r_p = sigmoid(W·ε),  ε ~ N(0, I)  — parameterises the prior rates
        self.sample_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.mmd_loss = MMD_loss(kernel_type=mmd_type)

    # ── Core forward pass ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, scheduled: bool = False):
        """
        Args:
            x (Tensor): Input spike tensor, shape (N, 1, H, W, T).
            scheduled (bool): Unused flag kept for API compatibility.

        Returns:
            x_recon    (Tensor): Reconstruction,  shape (N, 1, H, W, T).
            r_q        (Tensor): Posterior rates,  shape (N, latent_dim).
            r_p        (Tensor): Prior rates,      shape (N, latent_dim).
            sampled_z_q(Tensor): Spike samples,    shape (N, latent_dim, T).
        """
        sampled_z_q, r_q, r_p = self.encode(x)
        x_recon = self.decode(sampled_z_q)
        return x_recon, r_q, r_p, sampled_z_q

    def encode(self, x: torch.Tensor):
        """Encode input to latent spike samples and rate codes."""
        x = self.encoder(x)                              # (N, C, H', W', T)
        x = torch.flatten(x, start_dim=1, end_dim=3)    # (N, C·H'·W', T)
        latent_x = self.before_latent_layer(x)          # (N, latent_dim, T)
        return self.gaussian_sample(latent_x, latent_x.shape[0])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent spike tensor back to event-frame space."""
        bh, bw = self.bottleneck_hw
        result = self.decoder_input(z)                  # (N, flat_sz, T)
        result = result.view(
            result.shape[0], self.hidden_dims[-1], bh, bw, self.n_steps
        )                                               # (N, C, H', W', T)
        result = self.decoder(result)                   # (N, C', H'', W'', T)
        result = self.final_layer(result)               # (N, 1, H, W, T)
        return result

    def gaussian_sample(
        self,
        latent_x:   torch.Tensor | None = None,
        batch_size: int | None           = None,
        **_kwargs,
    ):
        """
        Build rate codes and sample a binary spike tensor.

        q-path (latent_x provided):
            r_p = sigmoid(W·ε),  ε ~ N(0,I)        -- prior rates for MMD
            r_q = mean_T(latent_x)                   -- posterior rates
            z_q = SampledSpikeAct(tile(r_q, T))     -- spike samples

        p-path (latent_x is None, batch_size required):
            r_p = sigmoid(W·ε)
            z_p = SampledSpikeAct(tile(r_p, T))     -- samples from prior

        Returns:
            (z, r_q, r_p) where r_q / r_p are (N, latent_dim) tensors
            (r_q = None in the p-path).
        """
        if latent_x is not None:
            eps = torch.randn(batch_size, self.latent_dim, device=self.device)
            r_p = self.sample_layer(eps)                              # (N, D)

            r_q_tiled = latent_x.mean(-1, keepdim=True).expand_as(latent_x)  # (N,D,T)
            sampled_z_q = SampledSpikeAct.apply(r_q_tiled)

            r_q = latent_x.mean(-1)                                   # (N, D)
            return sampled_z_q, r_q, r_p

        # p-path: unconditional sampling from the prior
        eps = torch.randn(batch_size, self.latent_dim, device=self.device)
        r_p = self.sample_layer(eps)
        r_p_tiled = r_p.unsqueeze(-1).expand(-1, -1, self.n_steps)   # (N,D,T)
        z_p = SampledSpikeAct.apply(r_p_tiled)
        return z_p, None, None

    def sample(self, batch_size: int = 32):
        """Draw unconditional samples from the prior p(z)."""
        z_p, _, _ = self.gaussian_sample(batch_size=batch_size)
        x_recon   = self.decode(z_p)
        return x_recon, z_p

    # ── Loss and utilities ───────────────────────────────────────────────────

    def loss_function_mmd(
        self,
        input_img:  torch.Tensor,
        recons_img: torch.Tensor,
        r_q:        torch.Tensor,
        r_p:        torch.Tensor,
    ) -> dict:
        """
        Compute total training loss.

        L = MSE(recon, input) + distance_lambda * MMD²(r_q, r_p)

        Returns:
            dict with keys 'loss', 'Reconstruction_Loss', 'Distance_Loss'.
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        mmd_val     = self.mmd_loss(r_q, r_p)
        total       = recons_loss + self.distance_lambda * mmd_val
        return {
            "loss":               total,
            "Reconstruction_Loss": recons_loss,
            "Distance_Loss":       mmd_val,
        }

    def weight_clipper(self):
        """Clamp all parameters to [-4, 4] to prevent weight explosion."""
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4.0, 4.0)

    def update_p(self, epoch: int, max_epoch: int):
        """Linearly schedule self.p from 0.1 to 0.3 over training."""
        self.p = 0.2 * epoch / max_epoch + 0.1