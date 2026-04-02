"""Grade dropout."""

import torch

from .linear import grade_project


def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Multivector dropout, dropping out grades independently.

    Parameters
    ----------
    x : torch.Tensor
        Input data with shape (..., 16).
    p : float
        Dropout probability (assumed the same for each grade).
    training : bool
        Switches between train-time and test-time behaviour.

    Returns
    -------
    outputs : torch.Tensor
        Inputs with dropout applied, shape (..., 16).
    """

    if not training or p == 0.0:
        return x
    if p >= 1.0:
        return torch.zeros_like(x)

    grades = grade_project(x)
    keep_prob = 1.0 - p
    mask_shape = (*grades.shape[:-2], 5, 1)
    mask = (torch.rand(mask_shape, device=x.device) < keep_prob).to(dtype=x.dtype)
    grades = grades * (mask / keep_prob)
    return torch.sum(grades, dim=-2)
