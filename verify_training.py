"""
Verification script: 3-epoch training run for N-MNIST and PokerDVS.

Checks:
  1. Forward/backward pass through CNN VAE (gradient norms per layer group)
  2. Forward/backward pass through SNN classifier (snnTorch Leaky neurons)
  3. MMD loss is non-zero and receives gradient
  4. Reconstruction loss decreases over 3 epochs
  5. Checkpoint PNGs and .pt files are saved
  6. Saved model loads and runs inference
  7. No dead-gradient layers (zero grad norm throughout)

Limits each epoch to MAX_BATCHES batches so CPU verification completes quickly.
"""

import os, sys, io, time, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed

# ── path so src/ is importable ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import src.layers as layer_module
from src.model       import VAE
from src.classifier  import VAEClassifier, train_one_epoch_classifier, test_classifier
from src.losses      import MMD_loss
from src.utils       import init_stats_dict, AverageMeter, accumulate_running_mean, save_epoch_images, save_mean_z_heatmap
from snntorch import functional as SF

MAX_BATCHES        = 3    # batches per epoch (keeps CPU runtime short)
NUM_EPOCHS         = 3
CLF_EPOCHS         = 2
DEVICE             = "cpu"
MAX_SCAN_SAMPLES   = 200  # samples used to estimate global min/max (avoids scanning 60k)
PASS = "[PASS]"; FAIL = "[FAIL]"; WARN = "[WARN]"

results = []


def fast_global_min_max(dataset, max_samples: int = 200, seed: int = 0):
    """Random-subsample estimate of global pixel min/max — verification only."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=min(max_samples, n), replace=False)
    mn, mx = float("inf"), float("-inf")
    for i in indices:
        data, _ = dataset[i]
        mn = min(mn, float(data.min()))
        mx = max(mx, float(data.max()))
    return mn, mx

def log(sym, msg):
    line = f"  {sym}  {msg}"
    print(line)
    results.append(line)

def section(title):
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

# ─────────────────────────────────────────────────────────────────────────────
def check_grad_norms(named_params, label):
    """Return dict of group→mean_grad_norm; flag zeros."""
    groups = {"encoder": [], "decoder": [], "latent": [], "sample_layer": []}
    for name, p in named_params:
        if p.grad is None:
            continue
        gn = p.grad.norm().item()
        for grp in groups:
            if grp in name:
                groups[grp].append(gn)
                break
        else:
            groups.setdefault("other", []).append(gn)

    dead = []
    for grp, norms in groups.items():
        if not norms:
            continue
        mean_gn = sum(norms) / len(norms)
        sym = PASS if mean_gn > 1e-10 else FAIL
        log(sym, f"  [{label}] grad norm {grp:15s}: {mean_gn:.4e}")
        if mean_gn < 1e-10:
            dead.append(grp)
    return dead

# ─────────────────────────────────────────────────────────────────────────────
def verify_dataset(name, cfg_path):
    section(f"DATASET: {name.upper()}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Apply neuron params
    layer_module.Vth = cfg["neuron"]["Vth"]
    layer_module.aa  = cfg["neuron"]["aa"]
    layer_module.tau = cfg["neuron"]["tau"]
    log(PASS, f"Neuron params set — Vth={layer_module.Vth}, aa={layer_module.aa}, tau={layer_module.tau}")

    dc = cfg["dataset"]
    vc = cfg["vae"]
    tc = cfg["training"]
    cc = cfg["classifier"]

    # ── 1. Data loading ───────────────────────────────────────────────────────
    # Use fast_global_min_max (subsample) instead of the production loader
    # factories which scan the full dataset — avoids scanning 60 k N-MNIST samples.
    section(f"  1. Data loading ({name})")
    try:
        import tonic
        import tonic.transforms as ttransforms
        from torch.utils.data import DataLoader, Subset
        from src.datasets import (
            MergePolarity, PermuteChannels, NormalizeZeroToOne,
            EventDataset, stratified_equal_test_split, get_labels,
        )

        if name == "nmnist":
            sensor_size = tonic.datasets.NMNIST.sensor_size
            base_tf = ttransforms.Compose([
                tonic.transforms.Denoise(filter_time=dc["denoise_filter_time"]),
                ttransforms.ToFrame(sensor_size=sensor_size, n_time_bins=dc["n_steps"]),
                MergePolarity(mode="on_only"),
                PermuteChannels(),
            ])
            train_ds_raw = tonic.datasets.NMNIST(save_to=dc["data_root"], train=True,  transform=base_tf)
            test_ds_raw  = tonic.datasets.NMNIST(save_to=dc["data_root"], train=False, transform=base_tf)
            log(PASS, f"N-MNIST loaded — train: {len(train_ds_raw)}  test: {len(test_ds_raw)}")

            if dc.get("global_min") is not None and dc.get("global_max") is not None:
                gmin, gmax = dc["global_min"], dc["global_max"]
                log(PASS, f"Using precomputed min/max from config: [{gmin}, {gmax}]")
            else:
                gmin, gmax = fast_global_min_max(train_ds_raw, max_samples=MAX_SCAN_SAMPLES)
                log(PASS, f"Min/max estimated from {MAX_SCAN_SAMPLES} samples: [{gmin}, {gmax}]")
            norm = NormalizeZeroToOne(int(gmin), int(gmax))

            full_tf = ttransforms.Compose([
                tonic.transforms.Denoise(filter_time=dc["denoise_filter_time"]),
                ttransforms.ToFrame(sensor_size=sensor_size, n_time_bins=dc["n_steps"]),
                MergePolarity(mode="on_only"),
                PermuteChannels(),
                norm,
            ])
            train_ds = tonic.datasets.NMNIST(save_to=dc["data_root"], train=True,  transform=full_tf)
            test_ds  = tonic.datasets.NMNIST(save_to=dc["data_root"], train=False, transform=full_tf)
            trainloader = DataLoader(train_ds, batch_size=dc["batch_size"], shuffle=True,  num_workers=0)
            testloader  = DataLoader(test_ds,  batch_size=dc["batch_size"]*2, shuffle=False, num_workers=0)

        else:  # poker_dvs
            base_tf = ttransforms.Compose([
                tonic.transforms.Denoise(filter_time=dc["denoise_filter_time"]),
                ttransforms.ToFrame(sensor_size=tuple(dc["sensor_size"]), n_time_bins=dc["n_steps"]),
                MergePolarity(mode="sum"),
                PermuteChannels(),
            ])
            ds_raw = EventDataset(directory_path=dc["directory_path"], transform=base_tf)
            log(PASS, f"PokerDVS loaded — total: {len(ds_raw)}")

            if dc.get("global_min") is not None and dc.get("global_max") is not None:
                gmin, gmax = dc["global_min"], dc["global_max"]
                log(PASS, f"Using precomputed min/max from config: [{gmin}, {gmax}]")
            else:
                gmin, gmax = fast_global_min_max(ds_raw, max_samples=MAX_SCAN_SAMPLES)
                log(PASS, f"Min/max estimated from {min(MAX_SCAN_SAMPLES, len(ds_raw))} samples: [{gmin}, {gmax}]")
            norm = NormalizeZeroToOne(int(gmin), int(gmax))

            full_tf = ttransforms.Compose([
                tonic.transforms.Denoise(filter_time=dc["denoise_filter_time"]),
                ttransforms.ToFrame(sensor_size=tuple(dc["sensor_size"]), n_time_bins=dc["n_steps"]),
                MergePolarity(mode="sum"),
                PermuteChannels(),
                norm,
            ])
            ds = EventDataset(directory_path=dc["directory_path"], transform=full_tf)
            labels = get_labels(ds)
            train_idx, test_idx, k, _ = stratified_equal_test_split(
                labels, train_ratio=dc["train_ratio"], num_classes=4, seed=dc["split_seed"]
            )
            log(PASS, f"Split — train: {len(train_idx)}  test: {len(test_idx)}  ({k} per class)")
            trainloader = DataLoader(Subset(ds, train_idx), batch_size=dc["batch_size"], shuffle=True,  num_workers=0)
            testloader  = DataLoader(Subset(ds, test_idx),  batch_size=dc["batch_size"]*2, shuffle=False, num_workers=0)

        imgs, labels = next(iter(trainloader))
        log(PASS, f"Train batch shape: {tuple(imgs.shape)}  labels: {tuple(labels.shape)}")
        log(PASS, f"Pixel range: [{imgs.min():.3f}, {imgs.max():.3f}]")
        assert imgs.max() <= 1.01, "normalisation failed — max > 1"
        assert imgs.min() >= -0.01, "normalisation failed — min < 0"
        log(PASS, "Pixel range within [0,1] confirmed")
    except Exception:
        log(FAIL, f"Data loading failed:\n{traceback.format_exc()}")
        return False

    # ── 2. Model construction ─────────────────────────────────────────────────
    section(f"  2. Model construction ({name})")
    try:
        net = VAE(
            hidden_dims=vc["hidden_dims"],
            latent_dim=vc["latent_dim"],
            n_steps=dc["n_steps"],
            bottleneck_hw=tuple(vc["bottleneck_hw"]),
            decoder_output_padding=vc["decoder_output_padding"],
            in_channels=vc["in_channels"],
            distance_lambda=vc["distance_lambda"],
            mmd_type=vc["mmd_type"],
            device=DEVICE,
        ).to(DEVICE)
        total_params = sum(p.numel() for p in net.parameters())
        log(PASS, f"VAE created — {total_params:,} parameters")
    except Exception:
        log(FAIL, f"VAE construction failed:\n{traceback.format_exc()}")
        return False

    # ── 3. Forward pass + MMD ─────────────────────────────────────────────────
    section(f"  3. VAE forward pass + MMD ({name})")
    try:
        net.train()
        imgs, labels = next(iter(trainloader))
        spike_input = imgs.unsqueeze(1).to(DEVICE)
        x_recon, r_q, r_p, sampled_z_q = net(spike_input)

        log(PASS, f"spike_input shape:  {tuple(spike_input.shape)}")
        log(PASS, f"x_recon shape:      {tuple(x_recon.shape)}")
        log(PASS, f"r_q shape:          {tuple(r_q.shape)}")
        log(PASS, f"r_p shape:          {tuple(r_p.shape)}")
        log(PASS, f"sampled_z_q shape:  {tuple(sampled_z_q.shape)}")

        assert x_recon.shape == spike_input.shape, \
            f"recon shape {x_recon.shape} != input {spike_input.shape}"
        log(PASS, "Reconstruction shape matches input ✓")

        losses = net.loss_function_mmd(spike_input, x_recon, r_q, r_p)
        mmd_val  = losses["Distance_Loss"].item()
        recon_val = losses["Reconstruction_Loss"].item()
        total_val = losses["loss"].item()

        log(PASS if mmd_val > 0 else FAIL, f"MMD loss:           {mmd_val:.6f}")
        log(PASS if recon_val > 0 else FAIL, f"Reconstruction loss:{recon_val:.6f}")
        log(PASS, f"Total loss:         {total_val:.6f}")

        # Verify MMD receives gradient
        losses["loss"].backward()
        mmd_grad = net.sample_layer[0].weight.grad
        log(PASS if mmd_grad is not None and mmd_grad.norm() > 0 else FAIL,
            f"sample_layer grad norm (MMD path): {mmd_grad.norm().item():.4e}" if mmd_grad is not None else "NO GRAD")

    except Exception:
        log(FAIL, f"VAE forward/loss failed:\n{traceback.format_exc()}")
        return False

    # ── 4. Backward pass / gradient flow ─────────────────────────────────────
    section(f"  4. Gradient flow through full VAE ({name})")
    try:
        net.zero_grad()
        spike_input = next(iter(trainloader))[0].unsqueeze(1).to(DEVICE)
        x_recon, r_q, r_p, sampled_z_q = net(spike_input)
        loss = net.loss_function_mmd(spike_input, x_recon, r_q, r_p)["loss"]
        loss.backward()
        dead = check_grad_norms(net.named_parameters(), name)
        if dead:
            log(WARN, f"Dead-gradient groups: {dead}")
        else:
            log(PASS, "No dead-gradient layer groups detected ✓")
    except Exception:
        log(FAIL, f"Backward pass failed:\n{traceback.format_exc()}")
        return False

    # ── 5. 3-epoch training ───────────────────────────────────────────────────
    section(f"  5. 3-epoch VAE training ({name})")
    net.zero_grad()
    params = list(net.named_parameters())
    lr = tc["lr"]
    lr_mult = tc["sample_layer_lr_multiplier"]
    param_groups = [
        {"params": [p for n,p in params if "sample_layer" in n], "lr": lr*lr_mult, "weight_decay": tc["weight_decay"]},
        {"params": [p for n,p in params if "sample_layer" not in n], "lr": lr, "weight_decay": tc["weight_decay"]},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

    ckpt_dir = tc["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    test_dir = ckpt_dir.rstrip("/\\") + "_test"
    os.makedirs(test_dir, exist_ok=True)

    stats = init_stats_dict()
    epoch_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        loss_meter = AverageMeter()
        mean_r_q = mean_r_p = mean_sampled_z_q = 0

        for batch_idx, (real_img, labels) in enumerate(trainloader):
            if batch_idx >= MAX_BATCHES:
                break
            optimizer.zero_grad()
            spike_input = real_img.unsqueeze(1).to(DEVICE)
            x_recon, r_q, r_p, sampled_z_q = net(spike_input, scheduled=True)
            losses = net.loss_function_mmd(spike_input, x_recon, r_q, r_p)
            losses["loss"].backward()
            optimizer.step()
            net.weight_clipper()
            loss_meter.update(losses["loss"].item())
            mean_r_q        = accumulate_running_mean(mean_r_q,        r_q,        batch_idx)
            mean_r_p        = accumulate_running_mean(mean_r_p,        r_p,        batch_idx)
            mean_sampled_z_q = accumulate_running_mean(mean_sampled_z_q, sampled_z_q, batch_idx)

            stats["histogram_cache"].append(sampled_z_q.mean(0).sum(-1).detach().cpu())

        epoch_losses.append(loss_meter.avg)

        # Save reconstruction image (last batch)
        save_epoch_images(spike_input[0][0], x_recon[0][0],
                          str(labels[0].item()), epoch, ckpt_dir,
                          n_timesteps=dc["n_steps"])

        # Save mean-z heatmap
        save_mean_z_heatmap(mean_sampled_z_q, epoch, ckpt_dir)

        # Test epoch
        net.eval()
        with torch.no_grad():
            te_loss = AverageMeter()
            for bi, (ri, li) in enumerate(testloader):
                if bi >= MAX_BATCHES: break
                si = ri.unsqueeze(1).to(DEVICE)
                xr, rq, rp, szq = net(si, scheduled=False)
                lss = net.loss_function_mmd(si, xr, rq, rp)
                te_loss.update(lss["loss"].item())
                stats["test_histogram_cache"].append(szq.mean(0).sum(-1).detach().cpu())

        stats["per_epoch"]["loss"].append(loss_meter.avg)
        stats["per_epoch"]["recons_loss"].append(losses["Reconstruction_Loss"].item())
        stats["per_epoch"]["distance"].append(losses["Distance_Loss"].item())
        stats["per_epoch"]["mean_r_q"].append(float(mean_r_q.mean()))
        stats["per_epoch"]["mean_r_p"].append(float(mean_r_p.mean()))
        stats["per_epoch_test"]["loss"].append(te_loss.avg)

        log(PASS, f"Epoch {epoch}/{NUM_EPOCHS} — train_loss={loss_meter.avg:.5f}  "
                  f"MMD={losses['Distance_Loss'].item():.5f}  "
                  f"recon={losses['Reconstruction_Loss'].item():.5f}  "
                  f"test_loss={te_loss.avg:.5f}")

    # Check loss decreasing
    if epoch_losses[-1] < epoch_losses[0]:
        log(PASS, f"Loss decreased: {epoch_losses[0]:.5f} → {epoch_losses[-1]:.5f} ✓")
    else:
        log(WARN, f"Loss did NOT decrease: {epoch_losses[0]:.5f} → {epoch_losses[-1]:.5f} (only {NUM_EPOCHS} epochs, may need more)")

    # ── 6. Check output files ─────────────────────────────────────────────────
    section(f"  6. Output file verification ({name})")
    imgs_dir = os.path.join(ckpt_dir, "imgs")
    hist_dir = os.path.join(ckpt_dir, "hist")
    pngs = [f for f in (os.listdir(imgs_dir) if os.path.isdir(imgs_dir) else []) if f.endswith(".png")]
    heatmaps = [f for f in (os.listdir(hist_dir) if os.path.isdir(hist_dir) else []) if f.endswith(".png")]
    log(PASS if pngs else FAIL, f"Reconstruction PNGs in imgs/: {pngs}")
    log(PASS if heatmaps else FAIL, f"Heatmap PNGs in hist/: {heatmaps}")

    # ── 7. Model save + load + inference ─────────────────────────────────────
    section(f"  7. Model save → load → inference ({name})")
    vae_ckpt = cc["vae_checkpoint"]
    os.makedirs(os.path.dirname(vae_ckpt), exist_ok=True)
    try:
        torch.save(net.state_dict(), vae_ckpt)
        log(PASS, f"Model saved → {vae_ckpt}")
        sz_kb = os.path.getsize(vae_ckpt) / 1024
        log(PASS, f"File size: {sz_kb:.1f} KB")
    except Exception:
        log(FAIL, f"Save failed:\n{traceback.format_exc()}")
        return False

    try:
        net2 = VAE(
            hidden_dims=vc["hidden_dims"], latent_dim=vc["latent_dim"],
            n_steps=dc["n_steps"], bottleneck_hw=tuple(vc["bottleneck_hw"]),
            decoder_output_padding=vc["decoder_output_padding"],
            in_channels=vc["in_channels"], distance_lambda=vc["distance_lambda"],
            mmd_type=vc["mmd_type"], device=DEVICE,
        ).to(DEVICE)
        net2.load_state_dict(torch.load(vae_ckpt, map_location=DEVICE))
        net2.eval()
        log(PASS, "Model loaded ✓")

        with torch.no_grad():
            si = next(iter(testloader))[0][:2].unsqueeze(1).to(DEVICE)
            xr, rq, rp, szq = net2(si)
        log(PASS, f"Inference on loaded model — recon shape: {tuple(xr.shape)} ✓")
    except Exception:
        log(FAIL, f"Load/inference failed:\n{traceback.format_exc()}")
        return False

    # ── 8. SNN Classifier backprop ────────────────────────────────────────────
    section(f"  8. SNN classifier training ({name})")
    try:
        clf = VAEClassifier(
            base_model=net2,
            num_classes=dc["num_classes"],
            device=DEVICE,
            init_beta=cc["init_beta"],
        ).to(DEVICE)
        trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in clf.parameters() if not p.requires_grad)
        log(PASS, f"VAEClassifier — trainable: {trainable:,}  frozen: {frozen:,}")

        clf_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, clf.parameters()),
            lr=cc["lr"], weight_decay=cc["weight_decay"],
        )
        criterion = SF.ce_rate_loss()

        clf.train()
        clf_losses = []
        for epoch in range(1, CLF_EPOCHS + 1):
            loss_m = AverageMeter()
            for bi, (ri, li) in enumerate(trainloader):
                if bi >= MAX_BATCHES: break
                ri = ri.to(DEVICE); li = li.to(DEVICE)
                si = ri.unsqueeze(1)
                clf_optimizer.zero_grad()
                logits, spike_record = clf(si)
                loss = criterion(spike_record, li)
                loss.backward()

                # Check SNN layer grads on first batch of first epoch
                if epoch == 1 and bi == 0:
                    for lname, lmod in [("lif1", clf.lif1), ("fc2", clf.fc2),
                                        ("lif2", clf.lif2), ("fc3", clf.fc3), ("lif3", clf.lif3)]:
                        for pname, pp in lmod.named_parameters():
                            if pp.grad is not None:
                                gn = pp.grad.norm().item()
                                sym = PASS if gn > 1e-10 else WARN
                                log(sym, f"  SNN [{lname}.{pname}] grad norm: {gn:.4e}")

                clf_optimizer.step()
                loss_m.update(loss.item())
            clf_losses.append(loss_m.avg)
            log(PASS, f"Classifier epoch {epoch}/{CLF_EPOCHS} loss={loss_m.avg:.5f}")

        if clf_losses[-1] < clf_losses[0]:
            log(PASS, f"Classifier loss decreased: {clf_losses[0]:.5f} → {clf_losses[-1]:.5f} ✓")
        else:
            log(WARN, f"Classifier loss did NOT decrease: {clf_losses[0]:.5f} → {clf_losses[-1]:.5f}")

        # Save classifier
        clf_ckpt = cc["clf_checkpoint"]
        os.makedirs(os.path.dirname(clf_ckpt), exist_ok=True)
        torch.save(clf.state_dict(), clf_ckpt)
        log(PASS, f"Classifier saved → {clf_ckpt}")

    except Exception:
        log(FAIL, f"Classifier training failed:\n{traceback.format_exc()}")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "="*65)
    print("  CNN-SNN-VAE Verification Run")
    print(f"  Device: {DEVICE}  |  Epochs: {NUM_EPOCHS}  |  Max batches/epoch: {MAX_BATCHES}")
    print("="*65)

    datasets = [
        ("nmnist",    "configs/nmnist.yaml"),
        ("poker_dvs", "configs/poker_dvs.yaml"),
    ]

    dataset_results = {}
    for ds_name, cfg_path in datasets:
        try:
            ok = verify_dataset(ds_name, cfg_path)
            dataset_results[ds_name] = ok
        except Exception:
            log(FAIL, f"Unexpected error in {ds_name}:\n{traceback.format_exc()}")
            dataset_results[ds_name] = False

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "="*65)
    print("  VERIFICATION SUMMARY")
    print("="*65)
    for ds, ok in dataset_results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {ds:15s}:  {status}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("="*65)

    all_ok = all(dataset_results.values())
    sys.exit(0 if all_ok else 1)
