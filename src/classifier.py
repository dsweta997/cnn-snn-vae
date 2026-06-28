"""
Spiking classifier head that reuses a pretrained VAE encoder.

VAEClassifier
-------------
The VAE encoder, latent projection, and prior sample_layer are frozen.
Their output (latent spike tensor z_q of shape (N, latent_dim, T)) is fed
into a 3-layer feedforward spiking neural network built with snnTorch Leaky
IF neurons and a fast-sigmoid surrogate gradient.

Output is the sum of output spikes across T, used as class logits.

Training utilities
------------------
compute_metrics              -- accuracy, macro F1, precision, recall
train_one_epoch_classifier   -- one epoch of supervised training
test_classifier              -- evaluation on a dataloader
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .model import VAE
from .layers import SampledSpikeAct


class VAEClassifier(nn.Module):
    """
    Spiking classifier on top of a frozen VAE encoder.

    Architecture
    ------------
    1. Frozen VAE encoder  →  latent spikes  z_q  (N, latent_dim, T)
    2. LIF₁  (latent_dim)  [no linear; z_q is already the input current]
    3. Linear(latent_dim → 128) → LIF₂
    4. Linear(128 → num_classes) → LIF₃
    5. Sum output spikes over T  →  class logits  (N, num_classes)

    Args:
        base_model  (VAE): Trained VAE whose encoder/latent layers are reused.
        num_classes (int): Number of target classes.
        device      (str): Device string for latent sampling.
        init_beta   (float): Initial membrane decay for all Leaky neurons.
    """

    def __init__(
        self,
        base_model:  VAE,
        num_classes: int   = 10,
        device:      str   = "cuda:0",
        init_beta:   float = 0.9,
    ):
        super().__init__()

        # ── Frozen feature extractor from the pretrained VAE ─────────────────
        self.encoder            = base_model.encoder
        self.before_latent_layer = base_model.before_latent_layer
        self.sample_layer       = base_model.sample_layer
        self.device             = device
        self.n_steps            = base_model.n_steps
        self.latent_dim         = base_model.latent_dim

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.before_latent_layer.parameters():
            param.requires_grad = False
        for param in self.sample_layer.parameters():
            param.requires_grad = False

        # ── Spiking classifier layers ─────────────────────────────────────────
        spike_fn = surrogate.fast_sigmoid()

        # Layer 1: spike the raw latent input current (no linear transform)
        self.lif1 = snn.Leaky(beta=init_beta, learn_beta=True, spike_grad=spike_fn)

        # Layer 2: latent_dim → 128
        self.fc2  = nn.Linear(self.latent_dim, 128)
        self.lif2 = snn.Leaky(beta=init_beta, learn_beta=True, spike_grad=spike_fn)

        # Layer 3: 128 → num_classes
        self.fc3  = nn.Linear(128, num_classes)
        self.lif3 = snn.Leaky(beta=init_beta, learn_beta=True, spike_grad=spike_fn)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Event frame tensor, shape (N, 1, H, W, T).

        Returns:
            out         (Tensor): Summed output spikes, shape (N, num_classes).
            spk3_record (Tensor): Per-step spikes,     shape (T, N, num_classes).
        """
        # Frozen encoding: produce latent spike tensor z_q
        x = self.encoder(x)                           # (N, C, H', W', T)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N, C·H'·W', T)
        latent_x = self.before_latent_layer(x)        # (N, latent_dim, T)
        sampled_z_q, _, _ = self.gaussian_sample(latent_x, latent_x.shape[0])

        # Initialise membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_record = []

        # Unroll over time
        for t in range(self.n_steps):
            input_t = sampled_z_q[:, :, t]       # (N, latent_dim)

            spk1, mem1 = self.lif1(input_t, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_record.append(spk3)

        spk3_record = torch.stack(spk3_record, dim=2)  # (N, num_classes, T)
        out         = spk3_record.sum(dim=2)            # (N, num_classes)
        spk3_record = spk3_record.permute(2, 0, 1)     # (T, N, num_classes)
        return out, spk3_record

    def gaussian_sample(
        self,
        latent_x:   torch.Tensor | None = None,
        batch_size: int | None           = None,
        **_kwargs,
    ):
        """
        Sample z_q from posterior rates and compute prior rates r_p.
        Delegates to the same logic as VAE.gaussian_sample.
        """
        if latent_x is not None:
            eps = torch.randn(batch_size, self.latent_dim, device=self.device)
            r_p = self.sample_layer(eps)

            r_q_tiled   = latent_x.mean(-1, keepdim=True).expand_as(latent_x)
            sampled_z_q = SampledSpikeAct.apply(r_q_tiled)
            r_q         = latent_x.mean(-1)
            return sampled_z_q, r_q, r_p

        eps       = torch.randn(batch_size, self.latent_dim, device=self.device)
        r_p       = self.sample_layer(eps)
        r_p_tiled = r_p.unsqueeze(-1).expand(-1, -1, self.n_steps)
        z_p       = SampledSpikeAct.apply(r_p_tiled)
        return z_p, None, None


# ── Per-epoch training / evaluation helpers ──────────────────────────────────

def compute_metrics(output_spikes: torch.Tensor, labels: torch.Tensor):
    """
    Compute classification metrics from spike records.

    Args:
        output_spikes (Tensor): Shape (T, N, num_classes) — per-step spikes.
        labels        (Tensor): Shape (N,) ground-truth class indices.

    Returns:
        Tuple: (accuracy, macro-F1, macro-precision, macro-recall).
    """
    spike_sum = output_spikes.sum(dim=0)          # (N, num_classes)
    preds     = spike_sum.argmax(dim=-1).cpu()
    labels    = labels.cpu()
    acc       = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)
    return acc, f1, precision, recall


def train_one_epoch_classifier(
    model:      VAEClassifier,
    dataloader: torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     str,
) -> dict:
    """
    Run one supervised training epoch.

    Returns:
        dict with 'loss' and 'accuracy' averaged over all batches.
    """
    model.train()
    all_metrics = []

    for data, targets in dataloader:
        data    = data.to(device)
        targets = targets.to(device)
        spike_input = data.unsqueeze(1)   # add channel dim: (N,1,H,W,T)

        optimizer.zero_grad()
        logits, spike_record = model(spike_input)   # (N,C), (T,N,C)
        loss = criterion(spike_record, targets)
        loss.backward()
        optimizer.step()

        acc, f1, precision, recall = compute_metrics(spike_record, targets)
        all_metrics.append((loss.item(), acc, f1, precision, recall))

    means = torch.tensor(all_metrics).mean(dim=0)
    return {"loss": means[0].item(), "accuracy": means[1].item()}


def test_classifier(
    model:      VAEClassifier,
    dataloader: torch.utils.data.DataLoader,
    criterion:  nn.Module,
    device:     str,
) -> dict:
    """
    Evaluate the classifier on a dataloader.

    Returns:
        dict with 'loss' and 'accuracy' averaged over all batches.
    """
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for data, targets in dataloader:
            data    = data.to(device)
            targets = targets.to(device)
            spike_input = data.unsqueeze(1)

            logits, spike_record = model(spike_input)
            loss = criterion(spike_record, targets)

            acc, f1, precision, recall = compute_metrics(spike_record, targets)
            all_metrics.append((loss.item(), acc, f1, precision, recall))

    means = torch.tensor(all_metrics).mean(dim=0)
    return {"loss": means[0].item(), "accuracy": means[1].item()}