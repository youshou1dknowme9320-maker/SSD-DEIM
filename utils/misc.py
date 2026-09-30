"""
Miscellaneous utilities for SSD-DEIM.

Includes learning rate scheduling, parameter statistics, model initialization,
and other helper functions used across the codebase.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import random


# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: Whether to set CuDNN to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
#  Learning Rate Schedulers
# ---------------------------------------------------------------------------

class CosineAnnealingWarmupLR(_LRScheduler):
    """Cosine annealing learning rate scheduler with linear warmup.

    Args:
        optimizer: Wrapped optimizer.
        total_iters: Total number of training iterations.
        warmup_iters: Number of warmup iterations.
        warmup_ratio: Initial LR ratio at start of warmup.
        min_lr_ratio: Minimum LR ratio at end of cosine decay.
        last_epoch: The index of last epoch / iteration.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        warmup_iters: int = 1500,
        warmup_ratio: float = 0.001,
        min_lr_ratio: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        current_iter = self.last_epoch

        if current_iter < self.warmup_iters:
            # Linear warmup
            k = current_iter / max(self.warmup_iters, 1)
            return [
                base_lr * self.warmup_ratio + base_lr * (1 - self.warmup_ratio) * k
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (current_iter - self.warmup_iters) / max(
                self.total_iters - self.warmup_iters, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            return [
                self.min_lr_ratio * base_lr
                + 0.5 * (base_lr - self.min_lr_ratio * base_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupLR(_LRScheduler):
    """Linear warmup followed by constant LR.

    Args:
        optimizer: Wrapped optimizer.
        warmup_iters: Number of warmup iterations.
        warmup_ratio: Initial LR ratio.
        last_epoch: The index of last epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int = 1000,
        warmup_ratio: float = 0.001,
        last_epoch: int = -1,
    ):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        current_iter = self.last_epoch
        if current_iter < self.warmup_iters:
            k = current_iter / max(self.warmup_iters, 1)
            return [
                base_lr * self.warmup_ratio + base_lr * (1 - self.warmup_ratio) * k
                for base_lr in self.base_lrs
            ]
        return list(self.base_lrs)


def get_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'CosineAnnealing',
    total_iters: int = 100000,
    warmup_iters: int = 1500,
    warmup_ratio: float = 0.001,
    min_lr_ratio: float = 1e-6,
) -> _LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer.
        scheduler_type: Type of scheduler.
        total_iters: Total training iterations.
        warmup_iters: Warmup iterations.
        warmup_ratio: Warmup initial LR ratio.
        min_lr_ratio: Minimum LR ratio for cosine annealing.

    Returns:
        Learning rate scheduler instance.
    """
    if scheduler_type == 'CosineAnnealing':
        return CosineAnnealingWarmupLR(
            optimizer,
            total_iters=total_iters,
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == 'LinearWarmup':
        return LinearWarmupLR(
            optimizer,
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ---------------------------------------------------------------------------
#  Model parameter statistics
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dict with total, trainable, and non-trainable parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
    }


def count_flops(model: nn.Module, input_size: tuple = (1, 3, 640, 640)) -> float:
    """Estimate FLOPs of a model (simplified estimation).

    Note: For accurate FLOP counting, use thop or fvcore.
    This provides a rough estimate based on Conv2d and Linear layers.

    Args:
        model: PyTorch model.
        input_size: Input tensor size.

    Returns:
        Estimated FLOPs in GFLOPs.
    """
    total_flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal total_flops
        if not isinstance(module, nn.Conv2d):
            return
        batch_size = input[0].size(0)
        output_h, output_w = output.size(2), output.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
            module.in_channels / module.groups
        )
        flops = kernel_ops * module.out_channels * output_h * output_w * batch_size
        total_flops += flops

    def linear_hook(module, input, output):
        nonlocal total_flops
        if not isinstance(module, nn.Linear):
            return
        batch_size = input[0].size(0)
        flops = module.in_features * module.out_features * batch_size
        total_flops += flops

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)

    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass

    for hook in hooks:
        hook.remove()

    return total_flops / 1e9  # GFLOPs


# ---------------------------------------------------------------------------
#  Model initialization
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module, init_type: str = 'kaiming') -> None:
    """Initialize weights of a module.

    Args:
        module: Module to initialize.
        init_type: Type of initialization ('kaiming', 'xavier', 'normal').
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Initialize bias of classification head with prior probability.

    Sets the bias such that the initial sigmoid probability equals prior_prob.
    b = -log((1-p)/p)

    Args:
        prior_prob: Desired initial probability.

    Returns:
        Bias value.
    """
    return -math.log((1.0 - prior_prob) / prior_prob)


# ---------------------------------------------------------------------------
#  Gradient utilities
# ---------------------------------------------------------------------------

def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Compute the gradient norm of model parameters.

    Args:
        model: PyTorch model.
        norm_type: Type of norm (2 for L2, inf for Linf).

    Returns:
        Gradient norm value.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            if norm_type == float('inf'):
                param_norm = p.grad.data.abs().max().item()
                total_norm = max(total_norm, param_norm)
            else:
                param_norm = p.grad.data.norm(norm_type).item()
                total_norm += param_norm ** norm_type
    if norm_type != float('inf'):
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


# ---------------------------------------------------------------------------
#  Activation functions
# ---------------------------------------------------------------------------

def get_activation(name: str = 'relu') -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation name ('relu', 'silu', 'gelu', 'leaky_relu', 'mish').

    Returns:
        Activation module.
    """
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'silu' or name == 'swish':
        return nn.SiLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == 'mish':
        return nn.Mish(inplace=True)
    elif name == 'none' or name is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
#  Inverse sigmoid
# ---------------------------------------------------------------------------

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid function (logit function).

    Args:
        x: Input tensor in [0, 1].
        eps: Clamping epsilon.

    Returns:
        Logit of x.
    """
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
#  Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training checkpoint.

    Args:
        path: Path to save checkpoint.
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Scheduler state.
        epoch: Current epoch.
        best_metric: Best metric value so far.
        extra: Extra data to include.
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'best_metric': best_metric,
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: str = 'cpu',
) -> dict:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        map_location: Device to map tensors to.

    Returns:
        Dict with loaded metadata (epoch, best_metric, etc.).
    """
    checkpoint = torch.load(path, map_location=map_location)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', 0.0),
    }


# ---------------------------------------------------------------------------
#  Logging / formatting
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.

    Args:
        seconds: Number of seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = '', fmt: str = ':6.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = f'{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})'
        return fmtstr
