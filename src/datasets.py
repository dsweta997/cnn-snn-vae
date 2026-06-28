"""
Dataset loading and preprocessing for event-camera datasets.

Transforms
----------
MergePolarity       -- collapses ON/OFF polarity channels to a single channel
PermuteChannels     -- reorders (C,H,W) → (H,W,C) to match the (N,H,W,T)
                       convention used by the VAE (T is the last dim)
NormalizeZeroToOne  -- min-max normalisation to [0, 1]

Datasets
--------
EventDataset        -- generic Dataset for PokerDVS AEDAT files organised in
                       per-class sub-folders

Loader factories
----------------
make_nmnist_loaders        -- build train/test DataLoaders for N-MNIST
make_poker_dvs_loaders     -- build train/test DataLoaders for PokerDVS

Splitting helpers
-----------------
compute_global_min_max     -- scan a dataset for global pixel extremes
get_labels                 -- extract integer labels from any dataset
stratified_equal_test_split -- balanced train/test split with equal class sizes
class_counts_from_indices  -- diagnostic count per class
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


# ── Transforms ───────────────────────────────────────────────────────────────

class MergePolarity:
    """
    Collapse ON/OFF polarity channels into a single channel.

    For N-MNIST: keep only the ON channel (positive polarity).
    For PokerDVS: sum both channels to retain all event information.

    Args:
        mode (str): 'on_only' (default) or 'sum'.
    """

    def __init__(self, mode: str = "on_only"):
        assert mode in ("on_only", "sum"), "mode must be 'on_only' or 'sum'"
        self.mode = mode

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if self.mode == "on_only":
            return x[:, 0, :, :]        # positive polarity only  (C,H,W) → (H,W) no, stays 4D
        return x[:, 0, :, :] + x[:, 1, :, :]  # sum ON + OFF


class PermuteChannels:
    """
    Reorder tensor from (C, H, W) → (H, W, C).

    After tonic's ToFrame the shape is (C, H, W, T) which becomes (H, W, T)
    after MergePolarity drops the channel dim.  This transform produces
    the (H, W, T) layout expected by the VAE's unsqueeze(1) call, so that
    the final batch tensor is (N, 1, H, W, T).
    """

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.permute(1, 2, 0)   # (H, W, T)


class NormalizeZeroToOne:
    """
    Min-max normalisation to [0, 1].

    Args:
        min_val (float): Global minimum pixel value.
        max_val (float): Global maximum pixel value.
    """

    def __init__(self, min_val: float, max_val: float):
        self.min = float(min_val)
        self.max = float(max_val)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        denom = self.max - self.min
        if denom == 0:
            return x - self.min
        return (x - self.min) / denom


# ── PokerDVS custom Dataset ───────────────────────────────────────────────────

class EventDataset(Dataset):
    """
    PyTorch Dataset for PokerDVS AEDAT files stored in per-class folders.

    Expected directory layout:
        directory_path/
            heart/    *.aedat
            spade/    *.aedat
            club/     *.aedat
            diamond/  *.aedat

    Args:
        directory_path (str): Root directory containing class sub-folders.
        transform      (callable, optional): Transform applied to raw events.
    """

    LABEL_MAP = {"heart": 0, "spade": 1, "club": 2, "diamond": 3}

    def __init__(self, directory_path: str, transform=None):
        self.directory_path = directory_path
        self.transform      = transform
        self.files:  list[str] = []
        self.labels: list[str] = []

        for label in os.listdir(directory_path):
            label_path = os.path.join(directory_path, label)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    if file_name.endswith(".aedat"):
                        self.files.append(os.path.join(label_path, file_name))
                        self.labels.append(label)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        label     = self.labels[idx]

        events = self._read_dvs_128(file_path)
        if self.transform:
            events = self.transform(events)

        label_tensor = torch.tensor(self.label_to_index(label), dtype=torch.long)
        return events, label_tensor

    def _read_dvs_128(self, file_path: str) -> np.ndarray:
        """Parse a DVS-128 AEDAT file and return a structured event array."""
        data_version, data_start, _ = tonic.io.read_aedat_header_from_file(file_path)
        all_events = tonic.io.get_aer_events_from_file(file_path, data_version, data_start)

        addr = all_events["address"]
        t    = all_events["timeStamp"]
        x    = (addr >> 8) & 0x007F
        y    = (addr >> 1) & 0x007F
        p    = addr & 0x1

        return tonic.io.make_structured_array(x, y, t, p)

    def label_to_index(self, label: str) -> int:
        key = label.lower().strip()
        # Try plural removal ('hearts' → 'heart')
        if key not in self.LABEL_MAP and key.endswith("s"):
            key = key[:-1]
        return self.LABEL_MAP.get(key, -1)


# ── Global stat helpers ───────────────────────────────────────────────────────

def compute_global_min_max(dataset) -> tuple[float, float]:
    """Scan every sample to find the global pixel min and max for normalisation."""
    min_val, max_val = float("inf"), float("-inf")
    for i in range(len(dataset)):
        data, _ = dataset[i]
        min_val = min(min_val, float(data.min()))
        max_val = max(max_val, float(data.max()))
    return min_val, max_val


# ── Splitting helpers ─────────────────────────────────────────────────────────

def get_labels(dataset) -> list[int]:
    """
    Return integer labels for every sample in a dataset.

    Supports: `.targets`, `.labels` attributes, or fallback to __getitem__.
    Handles int, Tensor, and string labels (uses `label_to_index` for strings).
    """
    raw: list
    if hasattr(dataset, "targets"):
        raw = list(dataset.targets)
    elif hasattr(dataset, "labels"):
        raw = list(dataset.labels)
    else:
        raw = [dataset[i][1] for i in range(len(dataset))]

    out = []
    for y in raw:
        if isinstance(y, (int, np.integer)):
            out.append(int(y))
        elif isinstance(y, torch.Tensor):
            out.append(int(y.item()))
        elif isinstance(y, str):
            if not hasattr(dataset, "label_to_index"):
                raise ValueError("Dataset has string labels but no label_to_index method.")
            idx = dataset.label_to_index(y)
            if idx == -1:
                raise ValueError(f"Unknown label '{y}'.")
            out.append(int(idx))
        else:
            raise TypeError(f"Unsupported label type: {type(y)}")
    return out


def stratified_equal_test_split(
    labels:       list[int],
    train_ratio:  float = 0.9,
    num_classes:  int   = 4,
    seed:         int   = 42,
    require_at_least: int = 1,
) -> tuple[list[int], list[int], int, int]:
    """
    Build a test set with exactly the same number of samples per class.

    Returns:
        train_idx      -- sample indices for the training set
        test_idx       -- sample indices for the test set (balanced)
        k_per_class    -- number of test samples per class
        actual_test_sz -- total test size
    """
    rng = np.random.default_rng(seed)
    by_cls: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_cls[int(y)].append(i)

    classes = sorted(by_cls.keys())
    for c in classes:
        rng.shuffle(by_cls[c])

    min_count = min(len(by_cls[c]) for c in classes)
    target_test = len(labels) - int(train_ratio * len(labels))
    k = min(target_test // num_classes, min_count)
    if k < require_at_least <= min_count:
        k = require_at_least

    test_idx = sorted(idx for c in classes for idx in by_cls[c][:k])
    train_idx = sorted(set(range(len(labels))) - set(test_idx))
    return train_idx, test_idx, k, len(test_idx)


def class_counts_from_indices(labels: list[int], indices: list[int]) -> dict:
    """Count samples per class for a given index subset."""
    counts: dict[int, int] = defaultdict(int)
    for i in indices:
        counts[labels[i]] += 1
    return dict(sorted(counts.items()))


# ── Loader factories ──────────────────────────────────────────────────────────

def make_nmnist_loaders(
    data_root:   str,
    n_steps:     int   = 16,
    batch_size:  int   = 32,
    denoise_filter_time: int = 10_000,
    num_workers: int   = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build N-MNIST train and test DataLoaders.

    Applies: Denoise → ToFrame → MergePolarity (ON only) → PermuteChannels
             → NormalizeZeroToOne (fitted on the training set).

    Args:
        data_root  (str): Directory where tonic will cache the N-MNIST dataset.
        n_steps    (int): Number of time bins.  Default: 16.
        batch_size (int): Training batch size.  Default: 32.
        denoise_filter_time (int): Noise filter time constant (µs).
        num_workers (int): DataLoader workers.  Default: 0 (safe on Windows).

    Returns:
        (trainloader, testloader)
    """
    sensor_size = tonic.datasets.NMNIST.sensor_size

    base_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=denoise_filter_time),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_steps),
        MergePolarity(mode="on_only"),
        PermuteChannels(),
    ])

    train_dataset = tonic.datasets.NMNIST(
        save_to=data_root, train=True, transform=base_transform
    )
    test_dataset = tonic.datasets.NMNIST(
        save_to=data_root, train=False, transform=base_transform
    )

    # Fit normalisation on the training set only
    gmin, gmax = compute_global_min_max(train_dataset)
    norm = NormalizeZeroToOne(int(gmin), int(gmax))

    full_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=denoise_filter_time),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_steps),
        MergePolarity(mode="on_only"),
        PermuteChannels(),
        norm,
    ])

    train_dataset = tonic.datasets.NMNIST(
        save_to=data_root, train=True, transform=full_transform
    )
    test_dataset = tonic.datasets.NMNIST(
        save_to=data_root, train=False, transform=full_transform
    )

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    testloader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return trainloader, testloader


def make_poker_dvs_loaders(
    directory_path: str,
    n_steps:        int   = 8,
    batch_size:     int   = 16,
    train_ratio:    float = 0.9,
    denoise_filter_time: int = 500,
    num_workers:    int   = 0,
    seed:           int   = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build PokerDVS train and test DataLoaders with a class-balanced split.

    Applies: Denoise → ToFrame → MergePolarity (sum) → PermuteChannels
             → NormalizeZeroToOne (fitted on the full dataset).

    Args:
        directory_path (str): Root folder containing per-class AEDAT files.
        n_steps        (int): Number of time bins.  Default: 8.
        batch_size     (int): Training batch size.  Default: 16.
        train_ratio    (float): Fraction of data used for training.
        denoise_filter_time (int): Noise filter time constant (µs).
        num_workers    (int): DataLoader workers.
        seed           (int): RNG seed for the split.

    Returns:
        (trainloader, testloader)
    """
    sensor_size = (64, 64, 2)

    base_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=denoise_filter_time),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_steps),
        MergePolarity(mode="sum"),
        PermuteChannels(),
    ])

    dataset = EventDataset(directory_path=directory_path, transform=base_transform)

    # Fit normalisation on the full dataset (small enough to scan quickly)
    gmin, gmax = compute_global_min_max(dataset)
    norm = NormalizeZeroToOne(int(gmin), int(gmax))

    full_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=denoise_filter_time),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_steps),
        MergePolarity(mode="sum"),
        PermuteChannels(),
        norm,
    ])

    dataset = EventDataset(directory_path=directory_path, transform=full_transform)

    labels = get_labels(dataset)
    train_idx, test_idx, k_per_class, actual_test_size = stratified_equal_test_split(
        labels, train_ratio=train_ratio, num_classes=4, seed=seed
    )

    print(
        f"PokerDVS — Total: {len(labels)}  "
        f"Train: {len(train_idx)}  Test: {len(test_idx)}  "
        f"(test {k_per_class} per class)"
    )

    trainloader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    testloader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size * 2,
        shuffle=False, num_workers=num_workers,
    )
    return trainloader, testloader