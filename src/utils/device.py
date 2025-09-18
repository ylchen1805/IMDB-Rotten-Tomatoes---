"""Device and environment utilities."""

import random
from typing import Dict, Optional

import numpy as np
import torch


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Get the best available device.

    Args:
        device_name: Specific device name (e.g., "cuda:0", "cpu")

    Returns:
        torch.device object
    """
    if device_name:
        return torch.device(device_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    return device


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory statistics in GB
    """
    stats = {}

    if torch.cuda.is_available():
        stats["gpu_allocated"] = torch.cuda.memory_allocated() / 1e9
        stats["gpu_reserved"] = torch.cuda.memory_reserved() / 1e9
        stats["gpu_free"] = (
            torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated()
        ) / 1e9

    return stats


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU cache cleared")