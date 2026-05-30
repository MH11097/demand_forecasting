"""Reproducibility utilities."""

import random

import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    # mỗi thư viện có RNG riêng -> phải set seed cho tất cả để kết quả tái lập được 100%
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # deterministic=True chậm hơn nhưng đảm bảo cùng input → cùng output trên GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
