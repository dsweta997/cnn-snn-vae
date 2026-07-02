# Learned Downsampling of Event Data using a Convolutional-Spiking VAE

A Masters dissertation project implementing a Variational Autoencoder (VAE) that compresses neuromorphic event-camera data using spiking neural network (SNN) layers.  The encoder and decoder use time-distributed convolutions — Conv3d with a temporal kernel of 1 — so the same spatial filter is applied independently at every time step, preserving the temporal structure of the spike representation.  The latent space is binary: posterior rates are matched to a prior distribution using Maximum Mean Discrepancy (MMD), making this an InfoVAE / Wasserstein AE variant with a spiking latent code.  A spiking classifier head is then trained on top of the frozen encoder to evaluate the quality of the learned representations.

Experiments are run on two neuromorphic datasets:

| Dataset | Classes | Resolution | Time bins | Compression ratio |
|---------|---------|-----------|-----------|------------------|
| N-MNIST | 10 (digits) | 34 × 34 | 16 | 34 × 34 / 64 ≈ 18× |
| PokerDVS | 4 (card suits) | 64 × 64 | 8 | 64 × 64 / 256 = 16× |

---

## Background

### Event cameras

Event cameras (Dynamic Vision Sensors, DVS) do not record frames at a fixed rate.  Instead, each pixel fires an asynchronous event when its log-luminance changes beyond a threshold, producing a sparse stream of `(x, y, polarity, timestamp)` tuples.  This gives extremely high temporal resolution (~1 µs) and low power consumption, but requires specialised processing.

For this work, events are accumulated into **time-binned frames**: the sensor's event stream over a recording is divided into `n_steps` equal-duration bins and the event counts are summed per pixel per bin, yielding a tensor of shape `(H, W, T)`.

### Spiking Neural Networks

Leaky Integrate-and-Fire (LIF) neurons maintain a membrane potential `u` that decays by factor `tau` each step and fires (emits a 1) when `u > Vth`.  Because the Heaviside spike function has zero gradient almost everywhere, training uses **surrogate gradients**: the backward pass substitutes a smooth box function (SpikeAct) or a shifted random threshold (SampledSpikeAct) in place of the true gradient.

### VAE with MMD prior

Standard VAEs use a KL divergence to push the posterior q(z|x) towards the prior p(z).  This project uses **Maximum Mean Discrepancy** instead (InfoVAE / WAE formulation):

```
L = MSE(x_recon, x) + λ · MMD²(r_q, r_p)
```

where `r_q` is the time-averaged posterior spike rate and `r_p = sigmoid(W·ε)`, `ε ~ N(0, I)`, is a learned prior.  MMD is estimated with a multi-kernel RBF (Gaussian) estimator.

---

## Repository structure

```
cnn-snn-vae/
│
├── src/                         # importable Python package
│   ├── __init__.py
│   ├── layers.py                # SpikeAct, LIFSpike, SampledSpikeAct, tdLinear,
│   │                            #   tdConv, tdConvTranspose, tdBatchNorm
│   ├── losses.py                # MMD_loss (linear and RBF estimators)
│   ├── model.py                 # VAE (generalised for both datasets)
│   ├── classifier.py            # VAEClassifier + per-epoch train/eval helpers
│   ├── datasets.py              # transforms, EventDataset (PokerDVS),
│   │                            #   NMNIST / PokerDVS loader factories
│   ├── train.py                 # VAE train_one_epoch / test_one_epoch / train_model
│   └── utils.py                 # AverageMeter, stats helpers, visualisation
│
├── configs/
│   ├── nmnist.yaml              # all hyperparameters for the N-MNIST experiment
│   └── poker_dvs.yaml           # all hyperparameters for the PokerDVS experiment
│
├── train_nmnist.py              # entry-point: VAE + classifier on N-MNIST
├── train_poker_dvs.py           # entry-point: VAE + classifier on PokerDVS
│
├── VAE_Clf_NMNIST_FullRun.ipynb     # original Jupyter notebook (N-MNIST)
├── VAE_Clf_PokerDVS_FullRun.ipynb   # original Jupyter notebook (PokerDVS)
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended.  The code falls back to CPU automatically if CUDA is unavailable.

---

## Datasets

### N-MNIST

Downloaded automatically by `tonic` on first run.  No manual action required.  The default cache path (`../tutorials/data`) can be changed in `configs/nmnist.yaml` under `dataset.data_root`.

### PokerDVS

Download the DVS-128 Poker card recordings and arrange them in per-class folders:

```
poker_dvs/
    heart/     *.aedat
    spade/     *.aedat
    club/      *.aedat
    diamond/   *.aedat
```

Update `dataset.directory_path` in `configs/poker_dvs.yaml` to point to this folder.

---

## Running experiments

### N-MNIST — train VAE then classifier

```bash
python train_nmnist.py
# or explicitly:
python train_nmnist.py --config configs/nmnist.yaml --device cuda:0
```

### PokerDVS

```bash
python train_poker_dvs.py --config configs/poker_dvs.yaml
```

### Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file |
| `--device DEVICE` | Override device (e.g. `cpu`, `cuda:1`) |
| `--vae-only` | Train only the VAE; skip the classifier |
| `--clf-only` | Skip VAE training; load weights from `classifier.vae_checkpoint` |

---

## Architecture details

### Time-distributed convolutions (`tdConv`, `tdConvTranspose`)

These wrap `nn.Conv3d` / `nn.ConvTranspose3d` but force the temporal kernel size to 1 and the temporal stride to 1.  Input tensors follow the `(N, C, H, W, T)` convention so that the standard PyTorch Conv3d spatial kernel acts independently on each time step.

```
Input : (N, C,  H,  W,  T)
Conv3d kernel: (k_H, k_W, 1)  — spatial only
Output: (N, C', H', W', T)    — time axis unchanged
```

### Encoder

Five `tdConv` blocks (N-MNIST) or two blocks (PokerDVS), each with a spatial stride of 2, `tdBatchNorm`, and `LIFSpike` activation.  A `tdLinear` layer then projects the flattened spatial features at each time step to `latent_dim` dimensions.

### Latent sampling

```
latent_x : (N, latent_dim, T)

r_q = mean_T(latent_x)                   shape: (N, latent_dim)
r_q_tiled = tile(r_q, T)                 shape: (N, latent_dim, T)
z_q = SampledSpikeAct(r_q_tiled)         binary: {0,1}^(N×latent_dim×T)

ε ~ N(0, I),  r_p = sigmoid(W·ε)         shape: (N, latent_dim)
```

MMD² is computed between `r_q` and `r_p` (both shape `(N, latent_dim)`).

### Decoder

A `tdLinear` expands the latent back to the bottleneck spatial size, which is then reshaped to `(N, C, H', W', T)` and upsampled through `tdConvTranspose` blocks.

### Spiking classifier (`VAEClassifier`)

The VAE encoder, latent projection, and prior `sample_layer` are frozen.  Their output `z_q` is fed through a 3-layer feedforward SNN using `snnTorch` Leaky neurons with a fast-sigmoid surrogate gradient:

```
z_q (N, latent_dim, T)
  → LIF₁          (no linear; direct current input)
  → Linear(latent_dim, 128) → LIF₂
  → Linear(128, num_classes) → LIF₃
  → sum over T → logits (N, num_classes)
```

---

## Key hyperparameters

All hyperparameters are in the YAML config files and can be changed without touching the source code.

| Parameter | N-MNIST | PokerDVS | Description |
|-----------|---------|---------|-------------|
| `n_steps` | 16 | 8 | Number of time bins T |
| `latent_dim` | 64 | 256 | Latent code dimension per time step |
| `hidden_dims` | [32,64,128,256,512] | [64,128] | Encoder channel progression |
| `Vth` | 0.2 | 0.3 | LIF spike threshold |
| `tau` | 0.35 | 0.35 | LIF membrane decay |
| `aa` | 0.5 | 0.5 | Surrogate gradient half-width |
| `distance_lambda` | 0.001 | 0.001 | MMD loss weight λ |
| `lr` (VAE) | 0.0003 | 0.005 | AdamW learning rate |
| `num_epochs` (VAE) | 15 | 150 | VAE training epochs |

---

## Outputs

After training, the following directories are created under `checkpoints/`:

```
checkpoints/<dataset>/
    train/
        imgs/               # input / reconstruction grids per epoch
        hist/               # mean latent heatmaps per epoch
        final_hist/         # final spike distribution histogram
        plots/              # loss convergence curves
    test/
        hist_test/          # test-set latent heatmaps per epoch
```

Model weights are saved to the paths specified in the config under `classifier.vae_checkpoint` and `classifier.clf_checkpoint`.

---

## Original notebooks

The two Jupyter notebooks (`VAE_Clf_NMNIST_FullRun.ipynb` and `VAE_Clf_PokerDVS_FullRun.ipynb`) contain the original, self-contained experiments including additional visualisations (spike animations, per-class reconstructions).  The modular `src/` package is a refactored extraction of the same code.

---

## Citation / acknowledgements

This project is a Masters dissertation.  The SNN layers follow the STBP (Spatio-Temporal Back Propagation) formulation:

> Wu et al., "Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks", *Frontiers in Neuroscience*, 2018.

The MMD prior is from:

> Zhao et al., "InfoVAE: Balancing Learning and Inference in Variational Autoencoders", *AAAI 2019*.
> Tolstikhin et al., "Wasserstein Auto-Encoders", *ICLR 2018*.

Event-camera data loading uses [Tonic](https://tonic.readthedocs.io/) and SNN layers use [snnTorch](https://snntorch.readthedocs.io/).