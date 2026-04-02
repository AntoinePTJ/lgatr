"""Geometric product."""

import os
from functools import lru_cache
from pathlib import Path

import torch

from ..utils.einsum import cached_einsum
from .linear import DEFAULT_DEVICE, DEFAULT_DTYPE

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


@lru_cache
def _load_geometric_product_tensor(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    """Loads geometric product tensor for geometric product between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.Device or str
        Device
    dtype : torch.Dtype
        Data type

    Returns
    -------
    basis : torch.Tensor
        Geometric product tensor with shape (16, 16, 16)
    """

    # To avoid duplicate loading, base everything on float32 CPU version
    if device not in [DEFAULT_DEVICE, "cpu"] and dtype != DEFAULT_DTYPE:
        gmt = _load_geometric_product_tensor()
    else:
        filename = Path(__file__).parent.resolve() / "geometric_product.pt"
        gmt = torch.load(filename).to(DEFAULT_DTYPE).to_dense()

    return gmt.to(device=device, dtype=dtype)


@lru_cache
def _load_geometric_product_index_sign(
    device: torch.device | str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads index/sign factorization of the geometric product tensor.

    Returns
    -------
    index : torch.Tensor
        Tensor with shape (16, 16), dtype int32. ``index[i, j]`` gives the input index in y.
    sign : torch.Tensor
        Tensor with shape (16, 16), dtype float32. ``sign[i, j]`` is +/-1.
    """

    gp = _load_geometric_product_tensor(device="cpu", dtype=DEFAULT_DTYPE)
    index = torch.empty((16, 16), dtype=torch.int32)
    sign = torch.empty((16, 16), dtype=torch.float32)

    for i in range(16):
        nz = (gp[i] != 0).nonzero(as_tuple=False)
        if nz.shape[0] != 16:
            raise RuntimeError("Expected exactly 16 non-zero terms per output component.")
        nz = nz[nz[:, 0].argsort()]
        if torch.unique(nz[:, 0]).numel() != 16:
            raise RuntimeError("Unexpected geometric product tensor structure.")
        index[i] = nz[:, 1].to(torch.int32)
        sign[i] = gp[i, nz[:, 0], nz[:, 1]].to(torch.float32)

    return index.to(device=device), sign.to(device=device)


@lru_cache
def _load_geometric_product_backward_index_sign(
    device: torch.device | str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads lookup tables for geometric-product backward passes.

    Returns
    -------
    grad_x_index : torch.Tensor
        Tensor with shape (16, 16), dtype int32. ``grad_x_index[j, i]`` gives the y index used
        when accumulating ``grad_x[..., j]`` from ``grad_output[..., i]``.
    grad_x_sign : torch.Tensor
        Tensor with shape (16, 16), dtype float32. ``grad_x_sign[j, i]`` is +/-1.
    grad_y_index : torch.Tensor
        Tensor with shape (16, 16), dtype int32. ``grad_y_index[k, i]`` gives the x index used
        when accumulating ``grad_y[..., k]`` from ``grad_output[..., i]``.
    grad_y_sign : torch.Tensor
        Tensor with shape (16, 16), dtype float32. ``grad_y_sign[k, i]`` is +/-1.
    """

    forward_index, forward_sign = _load_geometric_product_index_sign(device="cpu")
    grad_x_index = forward_index.transpose(0, 1).contiguous()
    grad_x_sign = forward_sign.transpose(0, 1).contiguous()

    grad_y_index = torch.empty((16, 16), dtype=torch.int32)
    grad_y_sign = torch.empty((16, 16), dtype=torch.float32)
    for i in range(16):
        permutation = forward_index[i]
        if torch.unique(permutation).numel() != 16:
            raise RuntimeError("Expected geometric-product lookup rows to be permutations.")
        inverse = torch.empty(16, dtype=torch.int32)
        inverse_sign = torch.empty(16, dtype=torch.float32)
        for j in range(16):
            k = permutation[j].item()
            inverse[k] = j
            inverse_sign[k] = forward_sign[i, j]
        grad_y_index[:, i] = inverse
        grad_y_sign[:, i] = inverse_sign

    return (
        grad_x_index.to(device=device),
        grad_x_sign.to(device=device),
        grad_y_index.to(device=device),
        grad_y_sign.to(device=device),
    )


if _HAS_TRITON:

    @triton.jit
    def _lookup_contraction_kernel(
        lhs_ptr,
        rhs_ptr,
        index_ptr,
        sign_ptr,
        out_ptr,
        n_rows,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = row < n_rows
        comp = tl.arange(0, 16)

        lhs = tl.load(
            lhs_ptr + row[:, None] * 16 + comp[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        for i in range(16):
            rhs_index = tl.load(index_ptr + i * 16 + comp)
            rhs_perm = tl.load(
                rhs_ptr + row[:, None] * 16 + rhs_index[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            sgn = tl.load(sign_ptr + i * 16 + comp).to(tl.float32)
            acc_i = tl.sum(lhs * rhs_perm * sgn[None, :], axis=1)
            tl.store(out_ptr + row * 16 + i, acc_i, mask=valid)


def _use_triton_geometric_product(x: torch.Tensor, y: torch.Tensor) -> bool:
    if os.getenv("LGATR_DISABLE_TRITON_GP", "0") == "1":
        return False
    if not _HAS_TRITON:
        return False
    if not x.is_cuda or not y.is_cuda:
        return False
    if x.dtype != y.dtype:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return x.shape[-1] == 16 and y.shape[-1] == 16


def _lookup_contraction_triton_forward(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    index: torch.Tensor,
    sign: torch.Tensor,
) -> torch.Tensor:
    """Executes a fixed-size signed gather-and-reduce contraction on CUDA via Triton."""

    lhs_2d = lhs.reshape(-1, 16).contiguous()
    rhs_2d = rhs.reshape(-1, 16).contiguous()
    out_2d = torch.empty_like(lhs_2d)
    n_rows = lhs_2d.shape[0]
    if n_rows == 0:
        return out_2d.reshape(lhs.shape)

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _lookup_contraction_kernel[grid](
        lhs_2d,
        rhs_2d,
        index,
        sign,
        out_2d,
        n_rows,
        BLOCK_SIZE=128,
    )
    return out_2d.reshape(lhs.shape)


def _geometric_product_triton_backward(
    grad_output: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x_index, grad_x_sign, grad_y_index, grad_y_sign = _load_geometric_product_backward_index_sign(
        device=grad_output.device
    )
    grad_x = _lookup_contraction_triton_forward(grad_output, y, grad_x_index, grad_x_sign)
    grad_y = _lookup_contraction_triton_forward(grad_output, x, grad_y_index, grad_y_sign)
    return grad_x, grad_y


def _geometric_product_triton_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _lookup_contraction_triton_forward(
        x,
        y,
        *_load_geometric_product_index_sign(device=x.device),
    )


class _TritonGeometricProductFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_contig = x.contiguous()
        y_contig = y.contiguous()
        out = _geometric_product_triton_forward(x_contig, y_contig)
        ctx.save_for_backward(x_contig, y_contig)
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x, y = ctx.saved_tensors
        if _use_triton_geometric_product(x, y) and grad_output.is_cuda:
            grad_x, grad_y = _geometric_product_triton_backward(grad_output.contiguous(), x, y)
        else:
            gp = _load_geometric_product_tensor(device=grad_output.device, dtype=grad_output.dtype)
            grad_x = cached_einsum("i j k, ... i, ... k -> ... j", gp, grad_output, y)
            grad_y = cached_einsum("i j k, ... i, ... j -> ... k", gp, grad_output, x)
        return grad_x, grad_y


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the geometric product ``f(x,y) = x*y``.

    Parameters
    ----------
    x : torch.Tensor
        First input multivector with shape (..., 16).
        Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor
        Second input multivector with shape (..., 16).
        Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor
        Result with shape (..., 16).
        Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    if x.shape == y.shape:
        x_bc, y_bc = x, y
    else:
        x_bc, y_bc = torch.broadcast_tensors(x, y)

    if _use_triton_geometric_product(x_bc, y_bc):
        return _TritonGeometricProductFunction.apply(x_bc, y_bc)

    gp = _load_geometric_product_tensor(device=x_bc.device, dtype=x_bc.dtype)
    return cached_einsum("i j k, ... j, ... k -> ... i", gp, x_bc, y_bc)
