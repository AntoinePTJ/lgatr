"""Equivariant transformer for vector, scalar, and pseudoscalar data."""

import math

import torch
from torch import nn
from torch.nn.functional import dropout, dropout1d
from torch.utils.checkpoint import checkpoint

from ..primitives.attention import scaled_dot_product_attention
from ..utils.autocast import minimum_autocast_precision


def inner_product(x, y):
    t = x[..., 0] * y[..., 0]
    s = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return t - s


def squared_norm(x):
    return inner_product(x, x)


def get_nonlinearity(label):
    if label == "relu":
        return nn.ReLU()
    elif label == "sigmoid":
        return nn.Sigmoid()
    elif label == "tanh":
        return nn.Tanh()
    elif label == "gelu":
        return nn.GELU()
    elif label == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported nonlinearity type: {label}")


class VectorToPseudoscalar(nn.Module):
    """Maps vectors to pseudoscalars through a learned oriented 4-volume.

    The module first projects the input channels to four learned Lorentz vectors for each output
    pseudoscalar channel, then computes the determinant of the resulting 4x4 matrix. This produces
    a parity-odd scalar quantity.
    """

    def __init__(self, in_v_channels: int, out_p_channels: int):
        super().__init__()
        self._in_v_channels = in_v_channels
        self._out_p_channels = out_p_channels
        self.weight = nn.Parameter(torch.empty(out_p_channels, 4, in_v_channels))
        self.reset_parameters()

    def reset_parameters(self, factor: float = 1.0):
        fan_in = max(self._in_v_channels, 1)
        bound = factor / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, a=-bound, b=bound)

    def forward(self, vectors):
        """Compute pseudoscalars from vector inputs.

        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., out_p_channels) representing pseudoscalar features.
        """
        projected = torch.einsum("...cM,pac->...paM", vectors, self.weight)
        return torch.linalg.det(projected)


class Dropout(nn.Module):
    """Dropout module for scalar, pseudoscalar, and vector features.

    For vector features, the same dropout mask is applied to all four components of each vector.
    """

    def __init__(self, dropout_prob: float):
        super().__init__()
        self._dropout_prob = dropout_prob

    def forward(self, vectors, scalars, pseudoscalars):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor
            A tensor of shape (..., p_channels) representing pseudoscalar features.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the dropped out vectors, scalars, and pseudoscalars.
        """
        # have to reshape vectors because dropout1d constrains input shape
        v = vectors.reshape(-1, 4)
        out_v = dropout1d(v, p=self._dropout_prob, training=self.training)
        out_v = out_v.reshape(vectors.shape)

        out_s = dropout(scalars, p=self._dropout_prob, training=self.training)
        out_p = dropout(pseudoscalars, p=self._dropout_prob, training=self.training)
        return out_v, out_s, out_p


class RMSNorm(nn.Module):
    """Normalize jointly over vector, scalar, and pseudoscalar features.

    For vectors, we use the absolute value of the squared norm because otherwise negative norms are possible.
    """

    def __init__(self, epsilon: float = 0.01):
        super().__init__()
        self.epsilon = epsilon

    @minimum_autocast_precision(torch.float32)
    def forward(self, vectors, scalars, pseudoscalars):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor
            A tensor of shape (..., p_channels) representing pseudoscalar features.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the normalized vectors, scalars, and
            pseudoscalars.
        """
        v_squared_norm = squared_norm(vectors).abs()
        s_squared_norm = scalars.square()
        p_squared_norm = pseudoscalars.square()
        sum_squared_norms = (
            v_squared_norm.sum(dim=-1) + s_squared_norm.sum(dim=-1) + p_squared_norm.sum(dim=-1)
        )
        mean_squared_norms = sum_squared_norms / (
            vectors.shape[-2] + scalars.shape[-1] + pseudoscalars.shape[-1]
        )
        norm = torch.rsqrt(mean_squared_norms + self.epsilon).unsqueeze(-1)

        vectors_out = vectors * norm.unsqueeze(-1)
        scalars_out = scalars * norm
        pseudoscalars_out = pseudoscalars * norm
        return vectors_out, scalars_out, pseudoscalars_out


class Linear(nn.Module):
    """Linear operations for vector, scalar, and pseudoscalar features.

    Supports optional mixing between vector and scalar features to improve expressivity.
    Pseudoscalar outputs receive both a linear pseudoscalar contribution and a parity-odd
    contribution generated from the vector inputs through ``VectorToPseudoscalar``.
    """

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        in_p_channels: int,
        out_p_channels: int,
        bias: bool = True,
        initialization: str = "default",
    ):
        """
        Parameters
        ----------
        in_v_channels : int
            Number of input vector channels.
        out_v_channels : int
            Number of output vector channels.
        in_s_channels : int
            Number of input scalar channels.
        out_s_channels : int
            Number of output scalar channels.
        in_p_channels : int
            Number of input pseudoscalar channels.
        out_p_channels : int
            Number of output pseudoscalar channels.
        bias : bool, optional
            Whether to include a bias term in the scalar linear layer, by default True.
        initialization : str, optional
            Initialization method for weights, by default "default".
            The alternative "small" initializes weights to smaller values,
            which might improve stability in attention projections.
        """
        super().__init__()
        self._in_v_channels = in_v_channels
        self._out_v_channels = out_v_channels
        self._in_s_channels = in_s_channels
        self._out_s_channels = out_s_channels
        self._bias = bias
        self._in_p_channels = in_p_channels
        self._out_p_channels = out_p_channels

        self.weight_v = nn.Parameter(
            torch.empty(
                (
                    out_v_channels,
                    in_v_channels,
                )
            )
        )
        self.linear_s = nn.Linear(in_s_channels, out_s_channels, bias=bias)
        self.linear_p = nn.Linear(in_p_channels, out_p_channels, bias=False)
        self.vector_to_p = VectorToPseudoscalar(in_v_channels, out_p_channels)

        self.reset_parameters(initialization)

    def forward(self, vectors, scalars, pseudoscalars):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor
            A tensor of shape (..., p_channels) representing pseudoscalar features.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the transformed vectors, scalars, and pseudoscalars.
        """
        vectors_out = self.weight_v @ vectors
        scalars_out = self.linear_s(scalars)
        pseudoscalars_out = self.linear_p(pseudoscalars) + self.vector_to_p(vectors)
        return vectors_out, scalars_out, pseudoscalars_out

    def reset_parameters(self, initialization, additional_factor=1.0):
        if initialization == "default":
            v_factor = additional_factor
            s_factor = additional_factor
            p_factor = additional_factor
        elif initialization == "small":
            v_factor = 0.1 * additional_factor
            s_factor = 0.1 * additional_factor
            p_factor = 0.1 * additional_factor
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        fan_in = max(self._in_v_channels, 1)
        bound = v_factor / math.sqrt(fan_in)
        nn.init.uniform_(self.weight_v, a=-bound, b=bound)

        fan_in = max(self._in_s_channels, 1)
        bound = s_factor / math.sqrt(fan_in)
        nn.init.uniform_(self.linear_s.weight, a=-bound, b=bound)

        fan_in = max(self._in_p_channels, 1)
        bound = p_factor / math.sqrt(fan_in)
        nn.init.uniform_(self.linear_p.weight, a=-bound, b=bound)
        self.vector_to_p.reset_parameters(p_factor)


class GatedLinearUnit(nn.Module):
    """Gated linear unit (GLU) for vector, scalar, and pseudoscalar features.

    Scalar and pseudoscalar gates are computed from scalar features,
    while vector gates are computed from inner products of vector features.
    """

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        in_p_channels: int,
        out_p_channels: int,
        nonlinearity: str = "gelu",
    ):
        super().__init__()
        self.out_s_channels = out_s_channels
        self.out_p_channels = out_p_channels
        self.linear = Linear(
            in_v_channels=in_v_channels,
            out_v_channels=3 * out_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=2 * out_s_channels + out_p_channels,
            in_p_channels=in_p_channels,
            out_p_channels=out_p_channels,
        )
        self.nonlinearity = get_nonlinearity(nonlinearity)

    def forward(self, vectors, scalars, pseudoscalars):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor
            A tensor of shape (..., p_channels) representing pseudoscalar features.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the transformed vectors, scalars, and
            pseudoscalars.
        """
        v_full, s_full, p_pre = self.linear(vectors, scalars, pseudoscalars)
        v_pre, v_gates_1, v_gates_2 = v_full.chunk(3, dim=-2)
        s_pre = s_full[..., : self.out_s_channels]
        s_gates = s_full[..., self.out_s_channels : 2 * self.out_s_channels]
        p_gates = s_full[..., 2 * self.out_s_channels :]

        v_gates = inner_product(v_gates_1, v_gates_2).unsqueeze(-1)
        vectors_out = self.nonlinearity(v_gates) * v_pre
        scalars_out = self.nonlinearity(s_gates) * s_pre
        pseudoscalars_out = self.nonlinearity(p_gates) * p_pre
        return vectors_out, scalars_out, pseudoscalars_out


class SelfAttention(nn.Module):
    """Self-attention module for Lorentz vectors, scalars, and pseudoscalars."""

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        num_heads: int,
        attn_ratio: int = 1,
        dropout_prob: float | None = None,
    ):
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * s_channels // num_heads, 4)
        self.hidden_p_channels = max(attn_ratio * p_channels // num_heads, 1)
        self.num_heads = num_heads

        metric = torch.tensor([1.0, -1.0, -1.0, -1.0])
        self.register_buffer("metric", metric)

        self.linear_in = Linear(
            in_v_channels=v_channels,
            out_v_channels=3 * self.hidden_v_channels * self.num_heads,
            in_s_channels=s_channels,
            out_s_channels=3 * self.hidden_s_channels * self.num_heads,
            in_p_channels=p_channels,
            out_p_channels=3 * self.hidden_p_channels * self.num_heads,
            initialization="small",
        )
        self.linear_out = Linear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=s_channels,
            in_p_channels=self.hidden_p_channels * self.num_heads,
            out_p_channels=p_channels,
            initialization="small",
        )
        self.norm = RMSNorm()
        if dropout_prob is not None:
            self.dropout = Dropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_reshape(self, qkv_v, qkv_s, qkv_p):
        qkv_v = (
            qkv_v.unflatten(-2, (3, self.hidden_v_channels, self.num_heads))
            .movedim(-4, 0)
            .movedim(-2, -4)
        )
        qkv_s = (
            qkv_s.unflatten(-1, (3, self.hidden_s_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )
        qkv_p = (
            qkv_p.unflatten(-1, (3, self.hidden_p_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )

        # normalize for stability (important)
        qkv_v, qkv_s, qkv_p = self.norm(qkv_v, qkv_s, qkv_p)

        q_v, k_v, v_v = qkv_v.unbind(0)
        q_s, k_s, v_s = qkv_s.unbind(0)
        q_p, k_p, v_p = qkv_p.unbind(0)

        q_v_mod = q_v * self.metric
        q = torch.cat([q_v_mod.flatten(start_dim=-2), q_s, q_p], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s, k_p], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s, v_p], dim=-1)
        return q, k, v

    def _post_reshape(self, out):
        v_end = self.hidden_v_channels * 4
        s_end = v_end + self.hidden_s_channels
        h_v = out[..., :v_end].reshape(*out.shape[:-1], self.hidden_v_channels, 4)
        h_s = out[..., v_end:s_end]
        h_p = out[..., s_end:]

        h_v = h_v.movedim(-3, -4).flatten(-3, -2)
        h_s = h_s.movedim(-2, -3).flatten(-2, -1)
        h_p = h_p.movedim(-2, -3).flatten(-2, -1)
        return h_v, h_s, h_p

    def forward(self, vectors, scalars, pseudoscalars, **attn_kwargs):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor
            A tensor of shape (..., p_channels) representing pseudoscalar features.
        **attn_kwargs : dict
            Additional keyword arguments for the attention function.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the transformed vectors, scalars, and
            pseudoscalars.
        """
        qkv_v, qkv_s, qkv_p = self.linear_in(vectors, scalars, pseudoscalars)

        q, k, v = self._pre_reshape(qkv_v, qkv_s, qkv_p)
        out = scaled_dot_product_attention(q, k, v, **attn_kwargs)
        h_v, h_s, h_p = self._post_reshape(out)

        out_v, out_s, out_p = self.linear_out(h_v, h_s, h_p)

        if self.dropout is not None:
            out_v, out_s, out_p = self.dropout(out_v, out_s, out_p)
        return out_v, out_s, out_p


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) for vector, scalar, and pseudoscalar features."""

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        num_layers: int = 2,
        dropout_prob: float | None = None,
    ):
        super().__init__()
        assert num_layers >= 2
        layers = []

        v_channels_list = [v_channels] + [mlp_ratio * v_channels] * (num_layers - 1) + [v_channels]
        s_channels_list = [s_channels] + [mlp_ratio * s_channels] * (num_layers - 1) + [s_channels]
        p_channels_list = [p_channels] * (num_layers + 1)

        for i in range(num_layers - 1):
            layers.append(
                GatedLinearUnit(
                    in_v_channels=v_channels_list[i],
                    out_v_channels=v_channels_list[i + 1],
                    in_s_channels=s_channels_list[i],
                    out_s_channels=s_channels_list[i + 1],
                    in_p_channels=p_channels_list[i],
                    out_p_channels=p_channels_list[i + 1],
                    nonlinearity=nonlinearity,
                )
            )
            if dropout_prob is not None:
                layers.append(Dropout(dropout_prob))
        layers.append(
            Linear(
                in_v_channels=v_channels_list[-2],
                out_v_channels=v_channels_list[-1],
                in_s_channels=s_channels_list[-2],
                out_s_channels=s_channels_list[-1],
                in_p_channels=p_channels_list[-2],
                out_p_channels=p_channels_list[-1],
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, vectors, scalars, pseudoscalars):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor
            A tensor of shape (..., p_channels) representing pseudoscalar features.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the normalized vectors, scalars, and
            pseudoscalars.
        """
        v, s, p = vectors, scalars, pseudoscalars

        for layer in self.layers:
            v, s, p = layer(v, scalars=s, pseudoscalars=p)

        return v, s, p


class LGATrSlimPseudoBlock(nn.Module):
    """A single block of the pseudoscalar-extended L-GATr-slim,
    consisting of self-attention and MLP layers, pre-norm and residual connections."""

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
    ):
        super().__init__()

        self.norm = RMSNorm()

        self.attention = SelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            p_channels=p_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
        )

        self.mlp = MLP(
            v_channels=v_channels,
            s_channels=s_channels,
            p_channels=p_channels,
            nonlinearity=nonlinearity,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

    def forward(self, vectors, scalars, pseudoscalars, **attn_kwargs):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor
            A tensor of shape (..., p_channels) representing pseudoscalar features.
        **attn_kwargs : dict
            Additional keyword arguments for the attention function.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the normalized vectors, scalars, and
            pseudoscalars.
        """
        h_v, h_s, h_p = self.norm(vectors, scalars, pseudoscalars)

        h_v, h_s, h_p = self.attention(
            h_v,
            h_s,
            h_p,
            **attn_kwargs,
        )

        outputs_v = vectors + h_v
        outputs_s = scalars + h_s
        outputs_p = pseudoscalars + h_p

        h_v, h_s, h_p = self.norm(outputs_v, outputs_s, outputs_p)

        h_v, h_s, h_p = self.mlp(h_v, h_s, h_p)

        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s
        outputs_p = outputs_p + h_p

        return outputs_v, outputs_s, outputs_p


class LGATrSlimPseudo(nn.Module):
    """L-GATr-slim network with an additional pseudoscalar stream."""

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        hidden_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        hidden_s_channels: int,
        in_p_channels: int,
        out_p_channels: int,
        hidden_p_channels: int,
        num_blocks: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        checkpoint_blocks: bool = False,
        compile: bool = False,
    ):
        """
        Parameters
        ----------
        in_v_channels : int
            Number of input vector channels.
        out_v_channels : int
            Number of output vector channels.
        hidden_v_channels : int
            Number of hidden vector channels.
        in_s_channels : int
            Number of input scalar channels.
        out_s_channels : int
            Number of output scalar channels.
        hidden_s_channels : int
            Number of hidden scalar channels.
        in_p_channels : int
            Number of input pseudoscalar channels.
        out_p_channels : int
            Number of output pseudoscalar channels.
        hidden_p_channels : int
            Number of hidden pseudoscalar channels.
        num_blocks : int
            Number of Lorentz Transformer blocks.
        num_heads : int
            Number of attention heads.
        nonlinearity : str, optional
            Nonlinearity type for MLP layers, by default "gelu".
        mlp_ratio : int, optional
            Expansion ratio for MLP hidden layers, by default 2.
        attn_ratio : int, optional
            Expansion ratio for attention hidden layers, by default 1.
        num_layers_mlp : int, optional
            Number of layers in MLP, by default 2.
        dropout_prob : float | None, optional
            Dropout probability, by default None.
        checkpoint_blocks : bool, optional
            Whether to use gradient checkpointing for blocks, by default False.
        compile : bool, optional
            Whether to compile the model with torch.compile, by default False.
        """
        super().__init__()

        self.linear_in = Linear(
            in_v_channels=in_v_channels,
            in_s_channels=in_s_channels,
            in_p_channels=in_p_channels,
            out_v_channels=hidden_v_channels,
            out_s_channels=hidden_s_channels,
            out_p_channels=hidden_p_channels,
        )

        self.blocks = nn.ModuleList(
            [
                LGATrSlimPseudoBlock(
                    v_channels=hidden_v_channels,
                    s_channels=hidden_s_channels,
                    p_channels=hidden_p_channels,
                    num_heads=num_heads,
                    nonlinearity=nonlinearity,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    num_layers_mlp=num_layers_mlp,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )

        self.linear_out = Linear(
            in_v_channels=hidden_v_channels,
            in_s_channels=hidden_s_channels,
            in_p_channels=hidden_p_channels,
            out_v_channels=out_v_channels,
            out_s_channels=out_s_channels,
            out_p_channels=out_p_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks

        self.compile = compile
        if compile:
            # ugly hack to make torch.compile convenient for users
            # the clean solution is model = torch.compile(model, **kwargs) outside of the constructor
            # note that we need fullgraph=False because of the torch.compiler.disable for attention
            self.__class__ = torch.compile(self.__class__, dynamic=True, mode="default")

    def forward(self, vectors, scalars, pseudoscalars = None, **attn_kwargs):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        pseudoscalars : torch.Tensor, optional
            A tensor of shape (..., p_channels) representing pseudoscalar features.
        **attn_kwargs : dict
            Additional keyword arguments for the attention function.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the normalized vectors, scalars and pseudoscalars.
        """

        if pseudoscalars is None and self.linear_in._in_p_channels == 0:
            pseudoscalars = torch.empty(scalars.shape[:-1] + (0,), device=scalars.device, dtype=scalars.dtype)
        else:
            assert pseudoscalars is not None, "Pseudoscalar input cannot be None if the model expects pseudoscalar channels."

        h_v, h_s, h_p = self.linear_in(vectors, scalars, pseudoscalars)

        for block in self.blocks:
            if self._checkpoint_blocks:
                h_v, h_s, h_p = checkpoint(
                    block,
                    h_v,
                    h_s,
                    h_p,
                    use_reentrant=False,
                    **attn_kwargs,
                )
            else:
                h_v, h_s, h_p = block(h_v, h_s, h_p, **attn_kwargs)

        outputs_v, outputs_s, outputs_p = self.linear_out(h_v, h_s, h_p)
        return outputs_v, outputs_s, outputs_p
