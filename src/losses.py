"""
Loss functions used during VAE training.

MMD_loss
--------
Maximum Mean Discrepancy between the posterior rate distribution q(z|x)
and the prior p(z).  Supports a linear MMD estimator and a multi-kernel
RBF (Gaussian) estimator (MK-MMD), as used in InfoVAE / WAE.

Reference:
    Zhao et al., "InfoVAE: Balancing Learning and Inference in VAEs", AAAI 2019.
    Tolstikhin et al., "Wasserstein Auto-Encoders", ICLR 2018.
"""

import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss.

    Estimates MMD² between two distributions given finite samples.

    Two estimators are available:
        'linear' -- O(N) linear estimator via mean embedding difference.
        'rbf'    -- O(N²) multi-kernel RBF estimator with K Gaussian kernels
                    whose bandwidths are spaced geometrically by `kernel_mul`.

    Args:
        kernel_type (str): 'linear' or 'rbf'.  Default: 'rbf'.
        kernel_mul  (float): Geometric multiplier between RBF bandwidths.
        kernel_num  (int): Number of RBF kernels.
    """

    def __init__(
        self,
        kernel_type: str   = "rbf",
        kernel_mul:  float = 2.0,
        kernel_num:  int   = 5,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul  = kernel_mul
        self.kernel_num  = kernel_num

    # ── Gram matrix ──────────────────────────────────────────────────────────

    def _gaussian_kernel(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        kernel_mul:  float = 2.0,
        kernel_num:  int   = 5,
        fix_sigma:   float | None = None,
    ) -> torch.Tensor:
        """
        Build the Gram matrix K(X∪Y, X∪Y) as a sum of RBF kernels.

        Bandwidths are chosen geometrically around the median heuristic:
            σ_k = fix_sigma * kernel_mul^(k - kernel_num//2)

        Returns:
            Gram matrix of shape (2N, 2N).
        """
        n_samples = source.shape[0] + target.shape[0]
        total = torch.cat([source, target], dim=0)   # (2N, D)

        # Squared pairwise distances
        total_sq = (total ** 2).sum(dim=1, keepdim=True)   # (2N, 1)
        sq_dists = total_sq + total_sq.t() - 2.0 * total @ total.t()  # (2N, 2N)

        # Bandwidth: median heuristic when fix_sigma is None
        bandwidth = fix_sigma if fix_sigma is not None else sq_dists.mean()
        bandwidth_list = [
            bandwidth * (kernel_mul ** (i - kernel_num // 2))
            for i in range(kernel_num)
        ]

        kernel_val = sum(
            torch.exp(-sq_dists / bw) for bw in bandwidth_list
        )
        return kernel_val  # (2N, 2N)

    # ── Linear estimator ─────────────────────────────────────────────────────

    @staticmethod
    def _linear_mmd2(
        f_of_X: torch.Tensor,
        f_of_Y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear MMD²: ||E[X] - E[Y]||².

        Complexity: O(N·D).
        """
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        return delta @ delta

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MMD²(source, target).

        Args:
            source (Tensor): Samples from q, shape (N, D).
            target (Tensor): Samples from p, shape (N, D).

        Returns:
            Scalar tensor: MMD² estimate.
        """
        if self.kernel_type == "linear":
            return self._linear_mmd2(source, target)

        # RBF estimator: E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
        batch_size = source.shape[0]
        kernels = self._gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        return XX.mean() + YY.mean() - XY.mean() - YX.mean()