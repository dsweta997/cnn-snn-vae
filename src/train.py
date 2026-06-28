"""
VAE training and evaluation loops.

train_one_epoch  -- one epoch of VAE training with backprop
test_one_epoch   -- one epoch of VAE evaluation (no_grad)
train_model      -- full training loop over N epochs, with checkpointing
                    and final histogram generation
"""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import VAE
from .utils import (
    AverageMeter,
    accumulate_running_mean,
    init_stats_dict,
    save_epoch_images,
    save_mean_z_heatmap,
)


def train_one_epoch(
    network:        VAE,
    trainloader:    torch.utils.data.DataLoader,
    optimizer:      torch.optim.Optimizer,
    epoch:          int,
    max_epoch:      int,
    stats_dict:     dict,
    checkpoint_dir: str,
    device:         str = "cuda:0",
) -> float:
    """
    Run one training epoch.

    Iterates over the trainloader, computes VAE loss (MSE + λ·MMD),
    backpropagates, updates the network, and accumulates stats.

    Side effects:
        - Appends per-epoch scalars to stats_dict['per_epoch'].
        - Appends per-batch latent distribution vectors to stats_dict['histogram_cache'].
        - Saves an input/reconstruction image grid at the last batch.
        - Saves a mean latent heatmap after each epoch.

    Args:
        network        : VAE model (in train mode after this call).
        trainloader    : DataLoader yielding (frame, label) tuples.
        optimizer      : Optimiser (e.g. AdamW).
        epoch          : Current epoch index (1-based).
        max_epoch      : Total number of training epochs.
        stats_dict     : Dictionary produced by init_stats_dict().
        checkpoint_dir : Root directory for saving images and histograms.
        device         : Device string, e.g. 'cuda:0' or 'cpu'.

    Returns:
        Average total loss for this epoch.
    """
    loss_meter   = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter   = AverageMeter()

    mean_r_q        = 0
    mean_r_p        = 0
    mean_sampled_z_q = 0

    network.train()

    for batch_idx, (real_img, labels) in enumerate(trainloader):
        optimizer.zero_grad()

        real_img = real_img.to(device, non_blocking=True)
        labels   = labels.to(device,   non_blocking=True)

        # Add channel dim: (N,H,W,T) → (N,1,H,W,T)
        spike_input = real_img.unsqueeze(1)
        x_recon, r_q, r_p, sampled_z_q = network(spike_input, scheduled=True)

        losses = network.loss_function_mmd(spike_input, x_recon, r_q, r_p)
        losses["loss"].backward()
        optimizer.step()

        # Track losses
        loss_meter.update(losses["loss"].detach().cpu().item())
        recons_meter.update(losses["Reconstruction_Loss"].detach().cpu().item())
        dist_meter.update(losses["Distance_Loss"].detach().cpu().item())

        # Running means for diagnostic logging
        mean_r_q        = accumulate_running_mean(mean_r_q,        r_q,        batch_idx)
        mean_r_p        = accumulate_running_mean(mean_r_p,        r_p,        batch_idx)
        mean_sampled_z_q = accumulate_running_mean(mean_sampled_z_q, sampled_z_q, batch_idx)

        # Cache per-batch latent distribution for the final histogram
        with torch.no_grad():
            per_batch_mean   = sampled_z_q.mean(0)      # (latent_dim, T)
            distribution_vec = per_batch_mean.sum(-1)   # (latent_dim,)
            stats_dict["histogram_cache"].append(distribution_vec.detach().cpu())

        # Save a reconstruction grid at the last batch of the epoch
        if batch_idx == len(trainloader) - 1:
            save_epoch_images(
                spike_input[0][0], x_recon[0][0],
                str(labels[0].item()), epoch, checkpoint_dir,
                n_timesteps=network.n_steps,
            )

    print(
        f"Train [{epoch}/{max_epoch}]  "
        f"Loss: {loss_meter.avg:.5f}  "
        f"Recon: {recons_meter.avg:.5f}  "
        f"MMD: {dist_meter.avg:.5f}"
    )

    stats_dict["per_epoch"]["loss"].append(loss_meter.avg)
    stats_dict["per_epoch"]["recons_loss"].append(recons_meter.avg)
    stats_dict["per_epoch"]["distance"].append(dist_meter.avg)
    stats_dict["per_epoch"]["mean_r_q"].append(float(mean_r_q.mean()))
    stats_dict["per_epoch"]["mean_r_p"].append(float(mean_r_p.mean()))

    save_mean_z_heatmap(mean_sampled_z_q, epoch, checkpoint_dir)

    return loss_meter.avg


@torch.no_grad()
def test_one_epoch(
    network:        VAE,
    testloader:     torch.utils.data.DataLoader,
    epoch:          int,
    max_epoch:      int,
    stats_dict:     dict,
    checkpoint_dir: str,
    device:         str = "cuda:0",
) -> float:
    """
    Run one evaluation epoch (no gradients).

    Side effects:
        - Appends per-epoch scalars to stats_dict['per_epoch_test'].
        - Appends latent distribution vectors to stats_dict['test_histogram_cache'].
        - Saves a reconstruction image and mean latent heatmap at batch 0.

    Args:
        (same as train_one_epoch; no optimizer needed)

    Returns:
        Average total loss for this epoch.
    """
    loss_meter   = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter   = AverageMeter()

    mean_r_q        = 0
    mean_r_p        = 0
    mean_sampled_z_q = 0

    network.eval()

    hist_epoch_dir = os.path.join(checkpoint_dir, "hist_test")
    os.makedirs(hist_epoch_dir, exist_ok=True)

    for batch_idx, (real_img, labels) in enumerate(testloader):
        real_img = real_img.to(device, non_blocking=True)
        labels   = labels.to(device,   non_blocking=True)

        spike_input = real_img.unsqueeze(1)
        x_recon, r_q, r_p, sampled_z_q = network(spike_input, scheduled=False)

        losses = network.loss_function_mmd(spike_input, x_recon, r_q, r_p)

        loss_meter.update(losses["loss"].detach().cpu().item())
        recons_meter.update(losses["Reconstruction_Loss"].detach().cpu().item())
        dist_meter.update(losses["Distance_Loss"].detach().cpu().item())

        mean_r_q        = accumulate_running_mean(mean_r_q,        r_q,        batch_idx)
        mean_r_p        = accumulate_running_mean(mean_r_p,        r_p,        batch_idx)
        mean_sampled_z_q = accumulate_running_mean(mean_sampled_z_q, sampled_z_q, batch_idx)

        per_batch_mean   = sampled_z_q.mean(0)
        distribution_vec = per_batch_mean.sum(-1)
        stats_dict["test_histogram_cache"].append(distribution_vec.detach().cpu())

        # Visualise at the very first batch of each epoch
        if batch_idx == 0:
            save_epoch_images(
                spike_input[0][0], x_recon[0][0],
                str(labels[0].item()), epoch, checkpoint_dir,
                n_timesteps=network.n_steps,
            )
            arr = mean_sampled_z_q.detach().cpu().numpy()
            fig, ax = plt.subplots()
            im = ax.imshow(arr, aspect="auto")
            fig.colorbar(im)
            ax.set_title(f"Mean sampled_z_q — Test epoch {epoch}")
            fig.savefig(
                os.path.join(hist_epoch_dir, f"epoch{epoch}_mean_sampled_z_q.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    print(
        f"Test  [{epoch}/{max_epoch}]  "
        f"Loss: {loss_meter.avg:.5f}  "
        f"Recon: {recons_meter.avg:.5f}  "
        f"MMD: {dist_meter.avg:.5f}"
    )

    stats_dict["per_epoch_test"]["loss"].append(loss_meter.avg)
    stats_dict["per_epoch_test"]["recons_loss"].append(recons_meter.avg)
    stats_dict["per_epoch_test"]["distance"].append(dist_meter.avg)
    stats_dict["per_epoch_test"]["mean_r_q"].append(float(mean_r_q.mean()))
    stats_dict["per_epoch_test"]["mean_r_p"].append(float(mean_r_p.mean()))

    return loss_meter.avg


def train_model(
    network:        VAE,
    trainloader:    torch.utils.data.DataLoader,
    testloader:     torch.utils.data.DataLoader,
    optimizer:      torch.optim.Optimizer,
    num_epochs:     int  = 15,
    checkpoint_dir: str  = "checkpoints/train",
    device:         str  = "cuda:0",
) -> dict:
    """
    Full VAE training loop.

    Runs train_one_epoch and test_one_epoch every epoch, then builds a
    histogram of the accumulated latent distribution and saves it as both
    a .npz archive and a PNG.

    Args:
        network        : VAE model.
        trainloader    : Training DataLoader.
        testloader     : Validation/test DataLoader.
        optimizer      : Optimiser.
        num_epochs     : Number of training epochs.
        checkpoint_dir : Root directory for all saved outputs.
        device         : Device string.

    Returns:
        stats_dict — complete training statistics dictionary.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    test_dir = checkpoint_dir.replace("train", "test")
    os.makedirs(test_dir, exist_ok=True)

    stats_dict = init_stats_dict()

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(
            network, trainloader, optimizer,
            epoch, num_epochs, stats_dict, checkpoint_dir, device,
        )
        test_one_epoch(
            network, testloader,
            epoch, num_epochs, stats_dict, test_dir, device,
        )

    # ── Final histogram of latent spike distribution ──────────────────────────
    hist_dir = os.path.join(checkpoint_dir, "final_hist")
    os.makedirs(hist_dir, exist_ok=True)

    cache = stats_dict["histogram_cache"]
    if cache:
        all_vals = torch.cat(cache, dim=0)
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

        bins   = 50
        counts = torch.histc(all_vals, bins=bins, min=vmin, max=vmax)
        edges  = torch.linspace(vmin, vmax, steps=bins + 1)

        np.savez(
            os.path.join(hist_dir, "final_histogram.npz"),
            counts=counts.cpu().numpy(), bin_edges=edges.cpu().numpy(),
        )

        centres = 0.5 * (edges[:-1] + edges[1:])
        widths  = edges[1:] - edges[:-1]
        fig, ax = plt.subplots()
        ax.bar(centres.numpy(), counts.numpy(), width=widths.numpy())
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_title("Final distribution of mean(z_q, batch).sum(time)")
        fig.tight_layout()
        fig.savefig(os.path.join(hist_dir, "final_histogram.png"), bbox_inches="tight")
        plt.close(fig)

        stats_dict["final_histogram"] = {"counts": counts, "bin_edges": edges}
    else:
        stats_dict["final_histogram"] = {
            "counts": torch.tensor([]), "bin_edges": torch.tensor([])
        }

    return stats_dict