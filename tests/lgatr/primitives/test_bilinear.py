"""Unit tests of bilinear primitives."""

import os

import pytest
import torch

from lgatr.primitives.bilinear import _HAS_TRITON, geometric_product
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_geometric_product,
    check_pin_equivariance,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_correctness(batch_dims):
    """Tests the geometric_product() primitive for correctness (that is, consistency with the
    clifford library).
    """
    check_consistence_with_geometric_product(geometric_product, batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_equivariance(batch_dims):
    """Tests the geometric_product() primitive for equivariance."""
    check_pin_equivariance(geometric_product, 2, batch_dims=[batch_dims] * 2, **TOLERANCES)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_TRITON,
    reason="Triton backward test requires CUDA and Triton",
)
def test_geometric_product_triton_backward_matches_dense_reference():
    old_disable = os.environ.get("LGATR_DISABLE_TRITON_GP")
    try:
        x = torch.randn(2, 1, 16, device="cuda", dtype=torch.float32, requires_grad=True)
        y = torch.randn(1, 3, 16, device="cuda", dtype=torch.float32, requires_grad=True)

        os.environ.pop("LGATR_DISABLE_TRITON_GP", None)
        out = geometric_product(x, y)
        grad_output = torch.randn_like(out)
        grad_x, grad_y = torch.autograd.grad(out, (x, y), grad_outputs=grad_output)

        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = y.detach().clone().requires_grad_(True)
        os.environ["LGATR_DISABLE_TRITON_GP"] = "1"
        out_ref = geometric_product(x_ref, y_ref)
        grad_x_ref, grad_y_ref = torch.autograd.grad(out_ref, (x_ref, y_ref), grad_outputs=grad_output)

        torch.testing.assert_close(out, out_ref, **TOLERANCES)
        torch.testing.assert_close(grad_x, grad_x_ref, **TOLERANCES)
        torch.testing.assert_close(grad_y, grad_y_ref, **TOLERANCES)
    finally:
        if old_disable is None:
            os.environ.pop("LGATR_DISABLE_TRITON_GP", None)
        else:
            os.environ["LGATR_DISABLE_TRITON_GP"] = old_disable
