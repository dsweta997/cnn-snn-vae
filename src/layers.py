"""
Spiking primitives and time-distributed layer wrappers.

All layers treat the *last* tensor dimension as the time axis (T) so that a
single Conv/Linear kernel is shared across time steps — matching the
time-distributed (TD) formulation used in the dissertation.

Global neuron parameters
------------------------
Vth : float  -- Spike threshold for SpikeAct and tdBatchNorm scaling.
aa  : float  -- Half-width of the box surrogate gradient window.
tau : float  -- Membrane decay factor for LIF neurons.

These constants reproduce the exact values used in the dissertation.
Change them before importing if you want to experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Neuron hyper-parameters ───────────────────────────────────────────────────
# Defaults match the N-MNIST experiment in the dissertation.
# PokerDVS uses Vth=0.3.  Entry-point scripts override these before
# constructing the model via:  import src.layers as L; L.Vth = cfg_value
Vth: float = 0.2   # spike threshold
aa:  float = 0.5   # box surrogate half-width
tau: float = 0.35  # LIF membrane decay


# ── Surrogate-gradient spike functions ──────────────────────────────────────

class SpikeAct(torch.autograd.Function):
    """
    Heaviside spike activation with a box (straight-through) surrogate gradient.

    Forward : output = 1  if  input > Vth  else 0
    Backward: grad   = grad_out / (2*aa)   if  |input| < aa  else 0

    The box window is centered at zero (not at Vth) following the original
    STBP paper.  Vth and aa are read from module-level constants.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return torch.gt(input, Vth).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        hu = (input.abs() < aa).float() / (2.0 * aa)
        return grad_output * hu


class SampledSpikeAct(torch.autograd.Function):
    """
    Stochastic spiking activation: spike if input > U(0,1) sample.

    Replaces the fixed threshold Vth with a per-element uniform random
    threshold, making the forward pass a Bernoulli sampler with rate = input.
    The backward surrogate mirrors SpikeAct but is centred on the random
    threshold rather than zero.

    Forward : spike = 1  if  input > rand  else 0   (rand ~ U(0,1))
    Backward: grad  = grad_out / (2*aa)  if  |input - rand| < aa  else 0
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        rand = torch.rand_like(input)
        ctx.save_for_backward(input, rand)
        return torch.gt(input, rand).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, rand = ctx.saved_tensors
        hu = (input - rand).abs() < aa
        hu = hu.float() / (2.0 * aa)
        return grad_output * hu


# ── Leaky Integrate-and-Fire neuron ─────────────────────────────────────────

class LIFSpike(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron applied along the time axis.

    Processes a tensor of shape (N, C, [H, W,] T) sequentially over the last
    dimension.  Uses SpikeAct for the forward spike and surrogate backward.

    State update (soft reset):
        u[t] = tau * u[t-1] * (1 - o[t-1]) + x[t]
        o[t] = SpikeAct(u[t])

    where `tau` is the membrane decay constant (module-level default).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_steps = x.shape[-1]
        u   = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros_like(x)
        for t in range(n_steps):
            prev_out = out[..., max(t - 1, 0)]
            u, out[..., t] = self._state_update(u, prev_out, x[..., t])
        return out

    @staticmethod
    def _state_update(
        u_prev: torch.Tensor,
        o_prev: torch.Tensor,
        x_cur:  torch.Tensor,
    ):
        # tau is read from the module namespace at call time so that
        # layer_module.tau = new_value takes effect without reimporting.
        u_cur = tau * u_prev * (1.0 - o_prev) + x_cur
        o_cur = SpikeAct.apply(u_cur)
        return u_cur, o_cur


# ── Time-distributed layers ──────────────────────────────────────────────────

class tdLinear(nn.Linear):
    """
    Time-distributed fully connected layer.

    Applies the same nn.Linear(in_features, out_features) to every time step
    independently.  Input/output shape: (N, C, T).

    Optionally followed by BatchNorm and a spiking activation.

    Args:
        in_features  (int): Input feature dimension.
        out_features (int): Output feature dimension.
        bias         (bool): Learnable bias.  Default: True.
        bn           (nn.Module | None): Batch-norm module (tdBatchNorm).
        spike        (nn.Module | None): Spike activation (LIFSpike).
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:  bool = True,
        bn:    nn.Module | None = None,
        spike: nn.Module | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.bn    = bn
        self.spike = spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        x = x.transpose(1, 2)            # (N, T, C)
        y = F.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)            # (N, C', T)
        if self.bn is not None:
            # tdBatchNorm expects 5-D input; add dummy H,W dims
            y = y[:, :, None, None, :]
            y = self.bn(y)
            y = y[:, :, 0, 0, :]
        if self.spike is not None:
            y = self.spike(y)
        return y


class tdConv(nn.Conv3d):
    """
    Time-distributed convolution implemented with Conv3d.

    All convolution kernels are purely spatial: the temporal kernel size is
    forced to 1 so that the same spatial filter is applied at every time step
    independently.  Input/output shape: (N, C, H, W, T).

    Args:
        in_channels  (int): Input channels.
        out_channels (int): Output channels.
        kernel_size  (int | tuple): Spatial kernel size (1-D or 2-D).
        stride       (int | tuple): Spatial stride.  Default: 1.
        padding      (int | tuple): Spatial zero-padding.  Default: 0.
        dilation     (int | tuple): Spatial dilation.  Default: 1.
        groups       (int): Convolution groups.  Default: 1.
        bias         (bool): Learnable bias.  Default: True.
        bn           (nn.Module | None): Batch-norm applied after conv.
        spike        (nn.Module | None): Spike activation applied after bn.
        is_first_conv (bool): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int | tuple,
        stride:       int | tuple = 1,
        padding:      int | tuple = 0,
        dilation:     int | tuple = 1,
        groups:       int  = 1,
        bias:         bool = True,
        bn:           nn.Module | None = None,
        spike:        nn.Module | None = None,
        is_first_conv: bool = False,
    ):
        def _to3d(v, time_val):
            if isinstance(v, int):
                return (v, v, time_val)
            if len(v) == 1:
                return (v[0], v[0], time_val)
            return (v[0], v[1], time_val)

        kernel  = _to3d(kernel_size, 1)
        stride_ = _to3d(stride,  1)
        pad_    = _to3d(padding, 0)
        dil_    = _to3d(dilation, 1)

        super().__init__(
            in_channels, out_channels, kernel, stride_, pad_, dil_, groups, bias=bias
        )
        self.bn    = bn
        self.spike = spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv3d(x, self.weight, self.bias,
                     self.stride, self.padding, self.dilation, self.groups)
        if self.bn    is not None: x = self.bn(x)
        if self.spike is not None: x = self.spike(x)
        return x


class tdConvTranspose(nn.ConvTranspose3d):
    """
    Time-distributed transposed convolution (upsampling) with Conv3d.

    Mirrors tdConv: temporal kernel is forced to 1 and temporal stride/padding
    are fixed so the time dimension is preserved.

    Args:
        in_channels     (int): Input channels.
        out_channels    (int): Output channels.
        kernel_size     (int | tuple): Spatial kernel.
        stride          (int | tuple): Spatial stride.  Default: 1.
        padding         (int | tuple): Spatial padding.  Default: 0.
        output_padding  (int | tuple | 0): Spatial output padding for stride>1.
                         Pass a 3-tuple (H_pad, W_pad, 0) to set H/W pads
                         independently; passing 0 means no output padding.
        dilation        (int | tuple): Spatial dilation.  Default: 1.
        groups          (int): Convolution groups.  Default: 1.
        bias            (bool): Learnable bias.  Default: True.
        bn              (nn.Module | None): Batch-norm after conv.
        spike           (nn.Module | None): Spike activation after bn.
    """

    def __init__(
        self,
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int | tuple,
        stride:         int | tuple = 1,
        padding:        int | tuple = 0,
        output_padding: int | tuple = 0,
        dilation:       int | tuple = 1,
        groups:         int  = 1,
        bias:           bool = True,
        bn:             nn.Module | None = None,
        spike:          nn.Module | None = None,
    ):
        def _to3d(v, time_val):
            if isinstance(v, int):
                return (v, v, time_val)
            if len(v) == 1:
                return (v[0], v[0], time_val)
            return (v[0], v[1], time_val)

        kernel  = _to3d(kernel_size, 1)
        stride_ = _to3d(stride,  1)
        pad_    = _to3d(padding, 0)
        dil_    = _to3d(dilation, 1)

        # output_padding: 3-tuple (H,W,T) or scalar
        if isinstance(output_padding, (list, tuple)) and len(output_padding) == 3:
            op = tuple(output_padding)
        elif isinstance(output_padding, int):
            op = (output_padding, output_padding, 0)
        else:
            op = (output_padding[0], output_padding[1], 0)

        super().__init__(
            in_channels, out_channels, kernel, stride_, pad_,
            op, groups, bias=bias, dilation=dil_
        )
        self.bn    = bn
        self.spike = spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.groups, self.dilation
        )
        if self.bn    is not None: x = self.bn(x)
        if self.spike is not None: x = self.spike(x)
        return x


class tdBatchNorm(nn.BatchNorm2d):
    """
    Time-domain BatchNorm (tdBN) for 5-D inputs in spiking/temporal networks.

    Computes mean and variance over (N, T, H, W) jointly for each channel C,
    then normalises and rescales by alpha * Vth.  This formulation comes from
    'Going Deeper with Directly-Trained Larger Spiking Neural Networks'
    (Zheng et al., 2021).

    Input shape : (N, C, T, H, W)   [Conv3d convention]
    Output shape: (N, C, T, H, W)

    Args:
        num_features (int): Number of channels C.
        eps          (float): Numerical stability constant.  Default: 1e-5.
        momentum     (float): Running-stats momentum.  Default: 0.1.
        alpha        (float): Scaling coefficient in alpha * Vth.  Default: 1.0.
        affine       (bool): Learnable affine (gamma/beta).  Default: True.
    """

    def __init__(
        self,
        num_features: int,
        eps:      float = 1e-5,
        momentum: float = 0.2,   # matches original dissertation experiments
        alpha:    float = 1.0,
        affine:   bool  = True,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (N, C, *, *, *)  — any 5-D tensor; non-channel dims are
        # collapsed together, so the exact H/W/T labelling does not matter.
        N, C = input.shape[:2]

        if self.training:
            # Compute batch statistics and update running estimates
            mean = input.mean(dim=[0, 2, 3, 4], keepdim=True)
            var  = input.var( dim=[0, 2, 3, 4], keepdim=True, unbiased=False)

            n_elements = input.numel() // C
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean.view(C)
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var.view(C) * n_elements / max(n_elements - 1, 1)
                )
        else:
            # Use accumulated running statistics at evaluation time
            mean = self.running_mean.view(1, C, 1, 1, 1)
            var  = self.running_var.view( 1, C, 1, 1, 1)

        # Normalise and scale by alpha * Vth
        x_hat = (input - mean) / (var + self.eps).sqrt()
        out = self.alpha * Vth * x_hat

        if self.affine:
            gamma = self.weight.view(1, C, 1, 1, 1)
            beta  = self.bias.view(1, C, 1, 1, 1)
            out   = gamma * out + beta
        return out