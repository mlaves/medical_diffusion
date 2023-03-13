"""
Utility functions.
"""

import torch


def kl_between_normal(
    mu_0: torch.Tensor,
    var_0: torch.Tensor,
    mu_1: torch.Tensor,
    var_1: torch.Tensor
) -> torch.Tensor:  # element-wise
    """
    Computes the KL divergence between two Normal distributions.
    """
    tensor = None
    for obj in (mu_0, var_0, mu_1, var_1):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None

    var_0, var_1 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (var_0, var_1)
    ]

    return 0.5 * (var_0 / var_1 + (mu_0 - mu_1).pow(2) / var_1 + var_1.log() - var_0.log() - 1.)
