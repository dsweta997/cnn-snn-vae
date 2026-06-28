"""
End-to-end training script for the N-MNIST experiment.

Stages
------
1. Load N-MNIST data via tonic (downloaded automatically on first run).
2. Train the Spiking VAE with an MMD prior-matching loss.
3. Save the trained VAE weights.
4. Freeze the VAE encoder and train a spiking classifier head.
5. Save the classifier weights and print final metrics.

Usage
-----
    python train_nmnist.py                         # uses configs/nmnist.yaml
    python train_nmnist.py --config configs/nmnist.yaml
    python train_nmnist.py --device cpu            # force CPU
"""

import argparse
import os

import torch
import torch.nn as nn
import yaml
from snntorch import functional as SF

import src.layers as layer_module
from src.classifier import VAEClassifier, test_classifier, train_one_epoch_classifier
from src.datasets import make_nmnist_loaders
from src.model import VAE
from src.train import train_model
from src.utils import plot_train_test_curves


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CNN-SNN-VAE on N-MNIST")
    p.add_argument("--config", default="configs/nmnist.yaml")
    p.add_argument("--device", default=None, help="Override device (cuda:0 / cpu)")
    p.add_argument("--vae-only",  action="store_true", help="Skip classifier training")
    p.add_argument("--clf-only",  action="store_true", help="Skip VAE training (load checkpoint)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Apply neuron hyper-parameters from config ─────────────────────────────
    layer_module.Vth = cfg["neuron"]["Vth"]
    layer_module.aa  = cfg["neuron"]["aa"]
    layer_module.tau = cfg["neuron"]["tau"]

    # ── Data ──────────────────────────────────────────────────────────────────
    dc = cfg["dataset"]
    print("Loading N-MNIST …")
    trainloader, testloader = make_nmnist_loaders(
        data_root=dc["data_root"],
        n_steps=dc["n_steps"],
        batch_size=dc["batch_size"],
        denoise_filter_time=dc["denoise_filter_time"],
        num_workers=dc["num_workers"],
    )
    print(f"  Train batches: {len(trainloader)}  Test batches: {len(testloader)}")

    # ── VAE ───────────────────────────────────────────────────────────────────
    vc = cfg["vae"]
    tc = cfg["training"]
    clf_checkpoint = cfg["classifier"]["clf_checkpoint"]
    vae_checkpoint = cfg["classifier"]["vae_checkpoint"]

    net = VAE(
        hidden_dims=vc["hidden_dims"],
        latent_dim=vc["latent_dim"],
        n_steps=dc["n_steps"],
        bottleneck_hw=tuple(vc["bottleneck_hw"]),
        decoder_output_padding=vc["decoder_output_padding"],
        in_channels=vc["in_channels"],
        distance_lambda=vc["distance_lambda"],
        mmd_type=vc["mmd_type"],
        device=device,
    ).to(device)

    if not args.clf_only:
        # ── Build optimizer: sample_layer gets a higher learning rate ─────────
        params = list(net.named_parameters())
        lr = tc["lr"]
        lr_mult = tc["sample_layer_lr_multiplier"]
        param_groups = [
            {
                "params": [p for n, p in params if "sample_layer" in n],
                "lr": lr * lr_mult,
                "weight_decay": tc["weight_decay"],
            },
            {
                "params": [p for n, p in params if "sample_layer" not in n],
                "lr": lr,
                "weight_decay": tc["weight_decay"],
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

        print(f"\nTraining VAE for {tc['num_epochs']} epochs …")
        stats = train_model(
            network=net,
            trainloader=trainloader,
            testloader=testloader,
            optimizer=optimizer,
            num_epochs=tc["num_epochs"],
            checkpoint_dir=tc["checkpoint_dir"],
            device=device,
        )

        os.makedirs(os.path.dirname(vae_checkpoint) or ".", exist_ok=True)
        torch.save(net.state_dict(), vae_checkpoint)
        print(f"VAE saved → {vae_checkpoint}")

        plot_train_test_curves(
            stats,
            out_dir=os.path.join(tc["checkpoint_dir"], "plots"),
            show=False,
        )
    else:
        print(f"Loading VAE from {vae_checkpoint} …")
        net.load_state_dict(torch.load(vae_checkpoint, map_location=device))

    # ── Classifier ────────────────────────────────────────────────────────────
    if not args.vae_only:
        cc = cfg["classifier"]
        model = VAEClassifier(
            base_model=net,
            num_classes=dc["num_classes"],
            device=device,
            init_beta=cc["init_beta"],
        ).to(device)

        clf_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cc["lr"],
            weight_decay=cc["weight_decay"],
        )
        criterion = SF.ce_rate_loss()

        print(f"\nTraining classifier for {cc['num_epochs']} epochs …")
        for epoch in range(1, cc["num_epochs"] + 1):
            tr = train_one_epoch_classifier(model, trainloader, clf_optimizer, criterion, device)
            te = test_classifier(model, testloader, criterion, device)
            print(
                f"  Epoch [{epoch}/{cc['num_epochs']}]  "
                f"Train loss: {tr['loss']:.4f}  acc: {tr['accuracy']:.4f}  |  "
                f"Test  loss: {te['loss']:.4f}  acc: {te['accuracy']:.4f}"
            )

        os.makedirs(os.path.dirname(clf_checkpoint) or ".", exist_ok=True)
        torch.save(model.state_dict(), clf_checkpoint)
        print(f"Classifier saved → {clf_checkpoint}")


if __name__ == "__main__":
    main()