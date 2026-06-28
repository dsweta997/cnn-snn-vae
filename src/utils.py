"""
Utilities: metric tracking, statistics bookkeeping, and visualisation.

AverageMeter              -- running mean of a scalar (loss / accuracy)
init_stats_dict           -- create the nested dict used by train/test loops
accumulate_running_mean   -- online mean over batches
save_epoch_images         -- save side-by-side input/reconstruction grids
save_mean_z_heatmap       -- save mean latent activity as a heatmap PNG
plot_train_test_curves    -- plot convergence curves from a stats_dict
plot_frames               -- plot individual time-bin frames of an event tensor
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec


# ── Metric tracking ───────────────────────────────────────────────────────────

class AverageMeter:
    """
    Tracks the running average of a scalar metric.

    Attributes:
        val   : Most recent observed value.
        avg   : Running average (sum / count).
        sum   : Cumulative weighted sum.
        count : Cumulative weight.

    Example:
        meter = AverageMeter()
        for loss in batch_losses:
            meter.update(loss, n=batch_size)
        print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Record a new observation (optionally weighted by n)."""
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ── Stats bookkeeping ─────────────────────────────────────────────────────────

def init_stats_dict() -> dict:
    """
    Return an empty nested statistics dictionary used by the training loop.

    Keys:
        'per_epoch'            : defaultdict of per-epoch train scalars
        'histogram_cache'      : list of per-batch latent distribution vectors
        'per_epoch_test'       : defaultdict of per-epoch test scalars
        'test_histogram_cache' : list of per-batch latent distribution vectors (test)
    """
    return {
        "per_epoch":            defaultdict(list),
        "histogram_cache":      [],
        "per_epoch_test":       defaultdict(list),
        "test_histogram_cache": [],
    }


def accumulate_running_mean(
    prev_mean:        torch.Tensor | int,
    new_batch_tensor: torch.Tensor,
    batch_idx:        int,
) -> torch.Tensor:
    """
    Incrementally update a running mean without storing all batches.

    Args:
        prev_mean        : Current running mean (or 0 before the first batch).
        new_batch_tensor : Batch tensor; mean over dim-0 is taken.
        batch_idx        : Zero-indexed batch counter.

    Returns:
        Updated running mean tensor (detached, on CPU).
    """
    new_mean = new_batch_tensor.mean(0).detach().cpu()
    if batch_idx == 0:
        return new_mean
    return (new_mean + batch_idx * prev_mean) / (batch_idx + 1)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def plot_frames(
    data_tensor: torch.Tensor | np.ndarray,
    n_cols:      int = 16,
    title:       str = "",
) -> plt.Figure:
    """
    Plot individual time-bin frames from a (H, W, T) tensor.

    Args:
        data_tensor : Shape (H, W, T).
        n_cols      : Number of columns (= time steps to display).  Default: 16.
        title       : Optional figure title.

    Returns:
        matplotlib Figure.
    """
    if isinstance(data_tensor, torch.Tensor):
        data_np = data_tensor.cpu().numpy()
    else:
        data_np = np.asarray(data_tensor)

    T = data_np.shape[2]
    n_cols = min(n_cols, T)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3, 3))
    if n_cols == 1:
        axes = [axes]

    for i in range(n_cols):
        axes[i].imshow(data_np[:, :, i], cmap="gray")
        axes[i].set_title(f"T{i + 1}", fontsize=9)
        axes[i].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def save_epoch_images(
    real_img:      torch.Tensor,
    x_recon:       torch.Tensor,
    label:         str,
    epoch:         int,
    checkpoint_dir: str,
    n_timesteps:   int = 16,
):
    """
    Save a side-by-side input / reconstruction image grid as a PNG.

    Args:
        real_img       : Shape (H, W, T) — one sample from the batch.
        x_recon        : Shape (H, W, T) — corresponding reconstruction.
        label          : Ground-truth label string (for the figure title).
        epoch          : Current epoch number (used in the filename).
        checkpoint_dir : Directory where the 'imgs/' folder is created.
        n_timesteps    : Number of time steps to display.  Default: 16.
    """
    img_dir = os.path.join(checkpoint_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"epoch_{epoch}_img.png")

    data1 = _to_numpy(real_img)
    data2 = _to_numpy(x_recon)

    T = min(n_timesteps, data1.shape[2])
    row_gap = 0.5
    fig = plt.figure(figsize=(T * 2, 2 * 2 + row_gap))
    gs  = gridspec.GridSpec(2, T, height_ratios=[1, 1], hspace=row_gap / 2)

    for i in range(T):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(data1[:, :, i], cmap="gray")
        ax.set_title(f"T{i + 1}", fontsize=7)
        ax.axis("off")

    for i in range(T):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(data2[:, :, i], cmap="gray")
        ax.axis("off")

    fig.suptitle(f"Epoch {epoch} | Label: {label} | top: input, bottom: recon", fontsize=9)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def save_mean_z_heatmap(
    mean_sampled_z_q: torch.Tensor,
    epoch:            int,
    checkpoint_dir:   str,
):
    """
    Save the mean latent activity (latent_dim × T) as a heatmap PNG.

    Args:
        mean_sampled_z_q : Shape (latent_dim, T) — running mean over the batch.
        epoch            : Current epoch (used in filename).
        checkpoint_dir   : Checkpoint root; 'hist/' sub-folder is created.
    """
    hist_dir = os.path.join(checkpoint_dir, "hist")
    os.makedirs(hist_dir, exist_ok=True)

    arr = _to_numpy(mean_sampled_z_q)
    fig, ax = plt.subplots()
    im = ax.imshow(arr, aspect="auto")
    fig.colorbar(im)
    ax.set_title(f"Mean sampled_z_q  (epoch {epoch})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Latent dim")
    out_path = os.path.join(hist_dir, f"epoch{epoch}_mean_sampled_z_q.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Convergence plots ─────────────────────────────────────────────────────────

def plot_train_test_curves(
    stats_dict: dict,
    out_dir:    str  | None = None,
    show:       bool        = True,
    prefix:     str         = "",
    eps:        float       = 1e-6,
):
    """
    Plot convergence and rate-code curves from a stats_dict.

    Produces four PNGs (when out_dir is given):
        loss_overall_log.png       -- total loss, log scale
        loss_reconstruction_log.png
        loss_mmd_log.png
        r_q_r_p_symlog.png        -- mean r_q / r_p, symlog scale

    Args:
        stats_dict : dict returned by train_model.
        out_dir    : Directory to save PNGs to (None = no save).
        show       : Call plt.show() after each figure.
        prefix     : Optional filename prefix.
        eps        : Minimum value for log-scale clamping.
    """

    def _series(block, name):
        v = block.get(name, [])
        return list(v) if v is not None else []

    def _logsafe(y):
        return [max(eps, float(v)) for v in y
                if v is not None and not (math.isnan(float(v)) or math.isinf(float(v)))]

    def _epochs(n):
        return list(range(1, n + 1))

    def _markevery(n):
        return max(1, n // 20)

    train = stats_dict.get("per_epoch",      {})
    test  = stats_dict.get("per_epoch_test", {})

    tr_loss = _series(train, "loss");         te_loss = _series(test, "loss")
    tr_rec  = _series(train, "recons_loss");  te_rec  = _series(test, "recons_loss")
    tr_mmd  = _series(train, "distance");     te_mmd  = _series(test, "distance")
    tr_rq   = _series(train, "mean_r_q");     te_rq   = _series(test, "mean_r_q")
    tr_rp   = _series(train, "mean_r_p");     te_rp   = _series(test, "mean_r_p")

    def _save_or_show(fig, fname):
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"{prefix}{fname}.png"),
                        bbox_inches="tight", dpi=160)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _plot_log(y_tr, y_te, title, ylabel, fname):
        yt = _logsafe(y_tr); yv = _logsafe(y_te)
        fig, ax = plt.subplots(figsize=(8, 5))
        if yt:
            ax.plot(_epochs(len(yt)), yt, "o-", ms=5, lw=1.8,
                    markevery=_markevery(len(yt)), label="Train")
        if yv:
            ax.plot(_epochs(len(yv)), yv, "s--", ms=5, lw=1.8,
                    markevery=_markevery(len(yv)), label="Test")
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False); fig.tight_layout()
        _save_or_show(fig, fname)

    _plot_log(tr_loss, te_loss, "Overall Loss (log)", "Loss", "loss_overall_log")
    _plot_log(tr_rec,  te_rec,  "Reconstruction Loss (log)", "Loss", "loss_reconstruction_log")
    _plot_log(tr_mmd,  te_mmd,  "MMD Loss (log)", "Loss", "loss_mmd_log")

    # r_q / r_p on symlog scale
    fig, ax = plt.subplots(figsize=(8, 5))
    for y, label, marker in [
        (tr_rq, "Train r_q", "o"), (tr_rp, "Train r_p", "^"),
        (te_rq, "Test r_q",  "s"), (te_rp, "Test r_p",  "v"),
    ]:
        if y:
            ax.plot(_epochs(len(y)), y, marker=marker, ms=5, lw=1.8,
                    markevery=_markevery(len(y)), label=label,
                    linestyle="--" if "Test" in label else "-")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean value")
    ax.set_title("r_q and r_p (Train / Test)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout()
    _save_or_show(fig, "r_q_r_p_symlog")