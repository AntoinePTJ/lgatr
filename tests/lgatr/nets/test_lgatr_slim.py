import pytest
import torch

from lgatr.layers.slim_layers import (
    SlimBlock,
    SlimDropout,
    SlimGLU,
    SlimLinear,
    SlimMLP,
    SlimRMSNorm,
    SlimSelfAttention,
    SlimVecLinear,
    SlimVectorToScalar,
)
from lgatr.nets.slim import LGATrSlim
from tests.helpers import (
    BATCH_DIMS,
    STRICT_TOLERANCES,
    TOLERANCES,
    check_equivariance,
    check_invariance,
)


# (in_v, out_v, in_s, out_s), covering the zero-channel edges on every slot.
CHANNELS = [
    (5, 1, 4, 2),
    (1, 4, 0, 2),
    (9, 3, 4, 0),
    (2, 7, 0, 0),
    (0, 1, 2, 3),
    (3, 0, 2, 3),
    (0, 0, 2, 3),
]
NONLINEARITIES = ["relu", "sigmoid", "tanh", "gelu", "silu"]

# Channels and nonlinearity cannot interact, so sweep them as a union rather than a product.
GLU_CASES = [(*channels, "sigmoid") for channels in CHANNELS]
GLU_CASES += [
    (*CHANNELS[0], nonlinearity) for nonlinearity in NONLINEARITIES if nonlinearity != "sigmoid"
]

LINEAR_CASES = [(*channels, "default") for channels in CHANNELS] + [(*CHANNELS[0], "small")]


# The vector<->scalar coupling options, as kwargs for LGATrSlim. Each entry is one ablation arm.
COUPLINGS = {
    "none": {},
    "v2s": dict(mix_v2s=True),
}


def _activate_couplings(net: torch.nn.Module) -> None:
    """Undo the zero-initialization of the coupling modules, so tests are not vacuous."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, SlimVectorToScalar):
                output = module.linear_s
            else:
                continue
            for param in output.parameters():
                param.copy_(torch.randn(param.shape, generator=generator))



@pytest.mark.parametrize("dropout_prob", [0.1, 0.5])
def test_SlimDropout_equivariance(dropout_prob: float) -> None:
    # SlimDropout preserves shapes and is SO(1, 3)-equivariant at eval time.
    layer = SlimDropout(dropout_prob)

    v = torch.randn(*BATCH_DIMS[:-1], 4, BATCH_DIMS[-1])
    s = torch.randn(*BATCH_DIMS)

    # shape and determinism in train mode (same seed -> same output; exercises the actual
    # dropout path, which is skipped in eval mode)
    layer.train()
    torch.manual_seed(0)
    train_v1, train_s1 = layer(v, scalars=s)
    torch.manual_seed(0)
    train_v2, train_s2 = layer(v, scalars=s)
    assert train_v1.shape == v.shape
    assert train_s1.shape == s.shape
    torch.testing.assert_close(train_v1, train_v2, **TOLERANCES)
    torch.testing.assert_close(train_s1, train_s2, **TOLERANCES)

    # shape at eval time
    layer.eval()
    outputs_v, outputs_s = layer(v, scalars=s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    check_equivariance(
        layer, batch_dims=BATCH_DIMS, fn_kwargs=dict(scalars=s), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("zero_channels", [None, "v", "s"])
def test_SlimRMSNorm_equivariance(zero_channels: str | None) -> None:
    # SlimRMSNorm preserves shapes and is SO(1, 3)-equivariant, including the zero-channel edge
    # cases where the affine weight is frozen (weight.numel() == 0).
    v_channels = 0 if zero_channels == "v" else BATCH_DIMS[-1]
    s_channels = 0 if zero_channels == "s" else BATCH_DIMS[-1]
    layer = SlimRMSNorm(v_channels, s_channels)
    if zero_channels == "v":
        assert not layer.weight_v.requires_grad
    if zero_channels == "s":
        assert not layer.weight_s.requires_grad

    v = torch.randn(*BATCH_DIMS[:-1], 4, v_channels)
    s = torch.randn(*BATCH_DIMS[:-1], s_channels)
    outputs_v, outputs_s = layer(v, scalars=s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS[:-1], v_channels),
        fn_kwargs=dict(scalars=s),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("in_v,out_v,in_s,out_s,nonlinearity", GLU_CASES)
def test_SlimGLU_equivariance(
    in_v: int, out_v: int, in_s: int, out_s: int, nonlinearity: str
) -> None:
    # SlimGLU produces the right output shapes and is SO(1, 3)-equivariant.
    layer = SlimGLU(
        in_v_channels=in_v,
        out_v_channels=out_v,
        in_s_channels=in_s,
        out_s_channels=out_s,
        nonlinearity=nonlinearity,
    )
    s = torch.randn(*BATCH_DIMS, in_s)
    v = torch.randn(*BATCH_DIMS, 4, in_v)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == (*BATCH_DIMS, 4, out_v)
    assert outputs_s.shape == (*BATCH_DIMS, out_s)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, in_v),
        fn_kwargs=dict(scalars=s),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("in_v,out_v,in_s,out_s,initialization", LINEAR_CASES)
def test_SlimLinear_equivariance(
    in_v: int, out_v: int, in_s: int, out_s: int, initialization: str
) -> None:
    # SlimLinear produces the right output shapes and is SO(1, 3)-equivariant.
    layer = SlimLinear(
        in_v_channels=in_v,
        out_v_channels=out_v,
        in_s_channels=in_s,
        out_s_channels=out_s,
        initialization=initialization,
    )
    s = torch.randn(*BATCH_DIMS, in_s)
    v = torch.randn(*BATCH_DIMS, 4, in_v)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == (*BATCH_DIMS, 4, out_v)
    assert outputs_s.shape == (*BATCH_DIMS, out_s)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, in_v),
        fn_kwargs=dict(scalars=s),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("in_v,out_v,in_s,out_s", CHANNELS[:4])
def test_SlimLinear_initialization(
    in_v: int, out_v: int, in_s: int, out_s: int, var_tolerance: float = 10.0
) -> None:
    # SlimLinear maps unit-variance inputs to roughly unit-variance outputs.
    layer = SlimLinear(
        in_v_channels=in_v, out_v_channels=out_v, in_s_channels=in_s, out_s_channels=out_s
    )

    inputs_v = torch.randn(100, 4, in_v)
    inputs_s = torch.randn(100, in_s)
    outputs_v, outputs_s = layer(inputs_v, inputs_s)

    v_mean = outputs_v.detach().to(torch.float64).mean(dim=(0, -1))
    v_var = outputs_v.detach().to(torch.float64).var(dim=(0, -1))
    target_mean = torch.zeros_like(v_mean)
    target_var = torch.ones_like(v_var) / 3.0
    assert torch.all(v_mean > target_mean - 0.3)
    assert torch.all(v_mean < target_mean + 0.3)
    assert torch.all(v_var > target_var / var_tolerance)
    assert torch.all(v_var < target_var * var_tolerance)

    if out_s > 0 and in_s > 0:
        s_mean = outputs_s.detach().to(torch.float64).mean().item()
        s_var = outputs_s.detach().to(torch.float64).var().item()

        assert -1.0 < s_mean < 1.0
        assert 1.0 / var_tolerance < s_var < 1.0 * var_tolerance


@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_SlimSelfAttention_equivariance(num_heads: int, attn_ratio: int) -> None:
    # SlimSelfAttention preserves shapes and is SO(1, 3)-equivariant.
    v_channels, s_channels = 24, 14
    layer = SlimSelfAttention(
        v_channels=v_channels,
        s_channels=s_channels,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*BATCH_DIMS, s_channels)
    v = torch.randn(*BATCH_DIMS, 4, v_channels)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, v_channels),
        fn_kwargs=dict(scalars=s),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("v_channels,s_channels", [(32, 4), (16, 8)])
@pytest.mark.parametrize("mlp_ratio,num_layers", [(1, 2), (2, 2), (1, 3)])
def test_SlimMLP_equivariance(
    v_channels: int, s_channels: int, mlp_ratio: int, num_layers: int
) -> None:
    # SlimMLP is SO(1, 3)-equivariant.
    layer = SlimMLP(
        v_channels=v_channels,
        s_channels=s_channels,
        mlp_ratio=mlp_ratio,
        num_layers=num_layers,
    )
    s = torch.randn(*BATCH_DIMS, s_channels)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, v_channels),
        fn_kwargs=dict(scalars=s),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("v_channels,s_channels,num_heads", [(32, 4, 1), (16, 8, 4)])
@pytest.mark.parametrize("dropout_prob", [None, 0.5])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_SlimBlock_equivariance(
    v_channels: int,
    s_channels: int,
    num_heads: int,
    dropout_prob: float | None,
    norm_elementwise_affine: bool,
) -> None:
    # SlimBlock is SO(1, 3)-equivariant at eval time.
    layer = SlimBlock(
        v_channels=v_channels,
        s_channels=s_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()
    s = torch.randn(*BATCH_DIMS, s_channels)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, v_channels),
        fn_kwargs=dict(scalars=s),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,out_v_channels,out_s_channels",
    [
        (4, 3, 9, 2),
        (2, 9, 0, 3),
        (3, 5, 7, 0),
        (8, 3, 0, 0),
    ],
)
@pytest.mark.parametrize("hidden_v_channels,hidden_s_channels,num_heads", [(32, 4, 1), (16, 8, 4)])
@pytest.mark.parametrize("num_blocks,checkpoint_blocks", [(1, False), (2, True)])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_LGATrSlim_equivariance(
    in_v_channels: int,
    in_s_channels: int,
    out_v_channels: int,
    out_s_channels: int,
    hidden_v_channels: int,
    hidden_s_channels: int,
    num_heads: int,
    num_blocks: int,
    checkpoint_blocks: bool,
    norm_elementwise_affine: bool,
) -> None:
    # LGATrSlim (full network) preserves shapes and is SO(1, 3)-equivariant at eval time. The
    # layer is in eval mode, so dropout_prob would not change anything and is fixed here.
    layer = LGATrSlim(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_prob=0.5,
        checkpoint_blocks=checkpoint_blocks,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()
    s = torch.randn(*BATCH_DIMS, in_s_channels)
    v = torch.randn(*BATCH_DIMS, in_v_channels, 4)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == (*BATCH_DIMS, out_v_channels, 4)
    assert outputs_s.shape == (*BATCH_DIMS, out_s_channels)

    check_equivariance(
        layer, batch_dims=(*BATCH_DIMS, in_v_channels), fn_kwargs=dict(scalars=s), **TOLERANCES
    )


@pytest.mark.parametrize("in_v_channels,out_v_channels", [(4, 4), (7, 3)])
def test_SlimVecLinear_equivariance(in_v_channels: int, out_v_channels: int) -> None:
    # SlimVecLinear mixes vector channels and is SO(1, 3)-equivariant.
    layer = SlimVecLinear(in_v_channels, out_v_channels)

    v = torch.randn(*BATCH_DIMS[:-1], 4, in_v_channels)
    assert layer(v).shape == (*BATCH_DIMS[:-1], 4, out_v_channels)

    check_equivariance(
        layer, batch_dims=(*BATCH_DIMS[:-1], in_v_channels), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("n_proj", [1, 4])
@pytest.mark.parametrize("v_channels,s_channels", [(4, 16), (2, 4)])
def test_SlimVectorToScalar_invariance(v_channels: int, s_channels: int, n_proj: int) -> None:
    layer = SlimVectorToScalar(v_channels, s_channels, n_proj=n_proj, zero_init=False)

    v = torch.randn(*BATCH_DIMS[:-1], 4, v_channels)
    assert layer(v).shape == (*BATCH_DIMS[:-1], s_channels)

    check_invariance(layer, batch_dims=(*BATCH_DIMS[:-1], v_channels), vector_dim=-2, **TOLERANCES)


@pytest.mark.parametrize("coupling", list(COUPLINGS))
@pytest.mark.parametrize("num_blocks", [1, 2])
def test_LGATrSlim_coupling_equivariance(coupling: str, num_blocks: int) -> None:
    in_v_channels, in_s_channels = 1, 8
    layer = LGATrSlim(
        in_v_channels=in_v_channels,
        out_v_channels=2,
        hidden_v_channels=4,
        in_s_channels=in_s_channels,
        out_s_channels=3,
        hidden_s_channels=16,
        num_blocks=num_blocks,
        num_heads=2,
        mlp_ratio=1,
        **COUPLINGS[coupling],
    )
    _activate_couplings(layer)
    layer.eval()
    s = torch.randn(*BATCH_DIMS, in_s_channels)

    check_equivariance(
        layer, batch_dims=(*BATCH_DIMS, in_v_channels), fn_kwargs=dict(scalars=s), **TOLERANCES
    )
    check_invariance(
        lambda v, scalars: layer(v, scalars)[1],
        batch_dims=(*BATCH_DIMS, in_v_channels),
        fn_kwargs=dict(scalars=s),
        **TOLERANCES,
    )


@pytest.mark.parametrize("coupling", [c for c in COUPLINGS if c != "none"])
def test_LGATrSlim_coupling_zero_init_is_a_noop(coupling: str) -> None:
    kwargs = dict(
        in_v_channels=1,
        out_v_channels=2,
        hidden_v_channels=4,
        in_s_channels=8,
        out_s_channels=3,
        hidden_s_channels=16,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=1,
    )
    baseline = LGATrSlim(**kwargs).eval()
    coupled = LGATrSlim(**kwargs, **COUPLINGS[coupling]).eval()

    # the coupling modules draw from the RNG, so seeding is not enough to align the two networks
    missing, unexpected = coupled.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(".v2s." in key for key in missing)

    v = torch.randn(*BATCH_DIMS, 1, 4)
    s = torch.randn(*BATCH_DIMS, 8)
    for expected, actual in zip(baseline(v, s), coupled(v, s), strict=True):
        torch.testing.assert_close(actual, expected, **STRICT_TOLERANCES)


@pytest.mark.parametrize("coupling", [c for c in COUPLINGS if c != "none"])
def test_LGATrSlim_coupling_flags_reach_the_model(coupling: str) -> None:
    kwargs = dict(
        in_v_channels=1,
        out_v_channels=2,
        hidden_v_channels=4,
        in_s_channels=8,
        out_s_channels=3,
        hidden_s_channels=16,
        num_blocks=1,
        num_heads=2,
        mlp_ratio=1,
    )
    baseline = LGATrSlim(**kwargs)
    coupled = LGATrSlim(**kwargs, **COUPLINGS[coupling])

    def signature(net):
        return {key: tuple(value.shape) for key, value in net.state_dict().items()}

    assert signature(coupled) != signature(baseline), f"{coupling} did not change the model"
