import pytest
import torch

from lgatr.nets.lgatr_slim_pseudo import (
    MLP,
    Dropout,
    GatedLinearUnit,
    LGATrSlimPseudo,
    LGATrSlimPseudoBlock,
    Linear,
    RMSNorm,
    SelfAttention,
    VectorToPseudoscalar,
    squared_norm,
)

from ...helpers.constants import BATCH_DIMS, TOLERANCES
from ...helpers.equivariance_noga import check_equivariance, check_invariance

CHANNELS = [
    (5, 1, 4, 2, 3, 2),
    (1, 4, 0, 2, 2, 3),
    (9, 3, 4, 0, 1, 1),
    (2, 7, 0, 0, 4, 2),
    (0, 1, 2, 3, 2, 1),
    (3, 0, 2, 3, 1, 4),
    (0, 0, 2, 3, 3, 2),
]


def check_parity_equivariance(function, vectors, scalars, pseudoscalars, **kwargs):
    parity = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=vectors.dtype, device=vectors.device))

    out_v, out_s, out_p = function(vectors, scalars=scalars, pseudoscalars=pseudoscalars)

    vectors_transformed = torch.einsum("ij,...j->...i", parity, vectors)
    pseudoscalars_transformed = -pseudoscalars
    out_v_of_transformed, out_s_of_transformed, out_p_of_transformed = function(
        vectors_transformed,
        scalars=scalars,
        pseudoscalars=pseudoscalars_transformed,
    )

    out_v_transformed = torch.einsum("ij,...j->...i", parity, out_v)
    out_p_transformed = -out_p

    torch.testing.assert_close(out_v_transformed, out_v_of_transformed, **kwargs)
    torch.testing.assert_close(out_s, out_s_of_transformed, **kwargs)
    torch.testing.assert_close(out_p_transformed, out_p_of_transformed, **kwargs)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_squared_norm_invariance(batch_dims):
    check_invariance(squared_norm, batch_dims=batch_dims, **TOLERANCES)


def test_VectorToPseudoscalar_flips_under_parity():
    layer = VectorToPseudoscalar(in_v_channels=4, out_p_channels=2)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0] = torch.eye(4)
        layer.weight[1] = 2.0 * torch.eye(4)

    vectors = torch.eye(4)
    pseudoscalar = layer(vectors)

    parity = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))
    parity_vectors = torch.einsum("ij,cj->ci", parity, vectors)
    parity_pseudoscalar = layer(parity_vectors)

    torch.testing.assert_close(parity_pseudoscalar, -pseudoscalar)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("dropout_prob", [0.0, 0.1, 0.5])
def test_Dropout_equivariance(batch_dims, dropout_prob):
    layer = Dropout(dropout_prob)
    layer.eval()

    # shape
    v = torch.randn(*batch_dims, 4)
    s = torch.randn(*batch_dims)
    p = torch.randn(*batch_dims)
    out_v, out_s, out_p = layer(v, scalars=s, pseudoscalars=p)
    assert out_v.shape == v.shape
    assert out_s.shape == s.shape
    assert out_p.shape == p.shape

    # equivariance
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_RMSNorm_equivariance(batch_dims):
    layer = RMSNorm()

    # shape
    v = torch.randn(*batch_dims, 4)
    s = torch.randn(*batch_dims)
    p = torch.randn(*batch_dims)
    out_v, out_s, out_p = layer(v, scalars=s, pseudoscalars=p)
    assert out_v.shape == v.shape
    assert out_s.shape == s.shape
    assert out_p.shape == p.shape

    # equivariance
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("nonlinearity", ["relu", "sigmoid", "tanh", "gelu", "silu"])
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels,in_p_channels,out_p_channels", CHANNELS)
def test_GatedLinearUnit_equivariance(
    batch_dims,
    nonlinearity,
    in_v_channels,
    out_v_channels,
    in_s_channels,
    out_s_channels,
    in_p_channels,
    out_p_channels,
):
    layer = GatedLinearUnit(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        in_p_channels=in_p_channels,
        out_p_channels=out_p_channels,
        nonlinearity=nonlinearity,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    p = torch.randn(*batch_dims, in_p_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    out_v, out_s, out_p = layer(v, s, p)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)
    assert out_p.shape == p.shape[:-1] + (out_p_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels,in_p_channels,out_p_channels", CHANNELS)
@pytest.mark.parametrize("initialization", ["default", "small"])
def test_Linear_equivariance(
    batch_dims,
    in_v_channels,
    out_v_channels,
    in_s_channels,
    out_s_channels,
    in_p_channels,
    out_p_channels,
    initialization,
):
    layer = Linear(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        in_p_channels=in_p_channels,
        out_p_channels=out_p_channels,
        initialization=initialization,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    p = torch.randn(*batch_dims, in_p_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    out_v, out_s, out_p = layer(v, s, p)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)
    assert out_p.shape == p.shape[:-1] + (out_p_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(100,)])
@pytest.mark.parametrize(
    "in_v_channels,out_v_channels,in_s_channels,out_s_channels,in_p_channels,out_p_channels",
    CHANNELS[:4],
)
def test_Linear_initialization(
    batch_dims,
    in_v_channels,
    out_v_channels,
    in_s_channels,
    out_s_channels,
    in_p_channels,
    out_p_channels,
    var_tolerance=10.0,
):
    # Test that inputs with variance 1 are roughly mapped to outputs with variance 1
    layer = Linear(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        in_p_channels=in_p_channels,
        out_p_channels=out_p_channels,
    )

    inputs_v = torch.randn(*batch_dims, in_v_channels, 4)
    inputs_s = torch.randn(*batch_dims, in_s_channels)
    inputs_p = torch.randn(*batch_dims, in_p_channels)
    outputs_v, outputs_s, outputs_p = layer(inputs_v, inputs_s, inputs_p)

    v_mean = outputs_v.cpu().detach().to(torch.float64).mean(dim=(0, 1))
    v_var = outputs_v.cpu().detach().to(torch.float64).var(dim=(0, 1))
    target_mean = torch.zeros_like(v_mean)
    target_var = torch.ones_like(v_var) / 3.0
    assert torch.all(v_mean > target_mean - 0.3)
    assert torch.all(v_mean < target_mean + 0.3)
    assert torch.all(v_var > target_var / var_tolerance)
    assert torch.all(v_var < target_var * var_tolerance)

    if out_s_channels > 0 and in_s_channels > 0:
        s_mean = outputs_s.cpu().detach().to(torch.float64).mean().item()
        s_var = outputs_s.cpu().detach().to(torch.float64).var().item()

        assert -1.0 < s_mean < 1.0
        assert 1.0 / var_tolerance < s_var < 1.0 * var_tolerance

    if out_p_channels > 0:
        assert torch.isfinite(outputs_p).all()


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels,p_channels", [(24, 14, 5)])
@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_SelfAttention_equivariance(
    batch_dims,
    v_channels,
    s_channels,
    p_channels,
    num_heads,
    attn_ratio,
):
    layer = SelfAttention(
        v_channels=v_channels,
        s_channels=s_channels,
        p_channels=p_channels,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*batch_dims, s_channels)
    p = torch.randn(*batch_dims, p_channels)
    v = torch.randn(*batch_dims, v_channels, 4)
    out_v, out_s, out_p = layer(v, s, p)
    assert out_v.shape == v.shape
    assert out_s.shape == s.shape
    assert out_p.shape == p.shape

    batch_dims = batch_dims + [v_channels]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels,p_channels", [(32, 4, 3), (16, 8, 2)])
@pytest.mark.parametrize("mlp_ratio,num_layers", [(1, 2), (2, 2), (1, 3)])
def test_MLP_equivariance(batch_dims, v_channels, s_channels, p_channels, mlp_ratio, num_layers):
    layer = MLP(
        v_channels=v_channels,
        s_channels=s_channels,
        p_channels=p_channels,
        mlp_ratio=mlp_ratio,
        num_layers=num_layers,
    )
    s = torch.randn(*batch_dims, s_channels)
    p = torch.randn(*batch_dims, p_channels)
    v = torch.randn(*batch_dims, v_channels, 4)
    batch_dims = batch_dims + [v_channels]

    # equivariance
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels,p_channels,num_heads", [(32, 4, 3, 1), (16, 8, 2, 4)])
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
def test_LGATrSlimPseudoBlock_equivariance(
    batch_dims,
    v_channels,
    s_channels,
    p_channels,
    num_heads,
    dropout_prob,
):
    layer = LGATrSlimPseudoBlock(
        v_channels=v_channels,
        s_channels=s_channels,
        p_channels=p_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
    )
    layer.eval()
    s = torch.randn(*batch_dims, s_channels)
    p = torch.randn(*batch_dims, p_channels)
    v = torch.randn(*batch_dims, v_channels, 4)
    batch_dims = batch_dims + [v_channels]

    # equivariance
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,in_p_channels,out_v_channels,out_s_channels,out_p_channels",
    [
        (4, 3, 2, 9, 2, 1),
        (2, 9, 1, 0, 3, 2),
        (3, 5, 4, 7, 0, 3),
        (8, 3, 2, 0, 0, 5),
    ],
)
@pytest.mark.parametrize(
    "hidden_v_channels,hidden_s_channels,hidden_p_channels,num_heads",
    [(32, 4, 3, 1), (16, 8, 2, 4)],
)
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_LGATrSlimPseudo_equivariance(
    batch_dims,
    in_v_channels,
    in_s_channels,
    in_p_channels,
    out_v_channels,
    out_s_channels,
    out_p_channels,
    hidden_v_channels,
    hidden_s_channels,
    hidden_p_channels,
    num_heads,
    num_blocks,
    dropout_prob,
    checkpoint_blocks,
):
    layer = LGATrSlimPseudo(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        in_p_channels=in_p_channels,
        out_p_channels=out_p_channels,
        hidden_p_channels=hidden_p_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        checkpoint_blocks=checkpoint_blocks,
    )
    layer.eval()
    s = torch.randn(*batch_dims, in_s_channels)
    p = torch.randn(*batch_dims, in_p_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    out_v, out_s, out_p = layer(v, s, p)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)
    assert out_p.shape == p.shape[:-1] + (out_p_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,in_p_channels,out_v_channels,out_s_channels,out_p_channels",
    [(4, 3, 2, 9, 2, 1)],
)
@pytest.mark.parametrize(
    "hidden_v_channels,hidden_s_channels,hidden_p_channels,num_heads,num_blocks",
    [(16, 8, 3, 4, 1)],
)
def test_LGATrSlimPseudo_equivariance_compiled(
    batch_dims,
    in_v_channels,
    in_s_channels,
    in_p_channels,
    out_v_channels,
    out_s_channels,
    out_p_channels,
    hidden_v_channels,
    hidden_s_channels,
    hidden_p_channels,
    num_heads,
    num_blocks,
    compile=True,
):
    layer = LGATrSlimPseudo(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        in_p_channels=in_p_channels,
        out_p_channels=out_p_channels,
        hidden_p_channels=hidden_p_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        compile=compile,
    )
    layer.eval()
    s = torch.randn(*batch_dims, in_s_channels)
    p = torch.randn(*batch_dims, in_p_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    out_v, out_s, out_p = layer(v, s, p)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)
    assert out_p.shape == p.shape[:-1] + (out_p_channels,)

    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)
