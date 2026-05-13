import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .laplacian_fast_attn import FastCudaLaplacianLinearAttention
from .rope import apply_2d_rope


class PyramidPatchEmbedding(nn.Module):
    def __init__(
            self, in_channels: int, embed_dim: int, kernel_size: int, stride: int, padding: int
            ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class SpatialMultiHeadAttention(nn.Module):
    def __init__(
            self, dim: int, num_heads: int, use_rope: bool = False, rope_base: float = 10000.0,
            dropout: float = 0.0
            ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        if num_tokens != height * width:
            raise ValueError(f"Expected {height * width} tokens, got {num_tokens}")

        qkv = self.qkv(x).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            q = apply_2d_rope(q, height, width, self.rope_base)
            k = apply_2d_rope(k, height, width, self.rope_base)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PyramidBlock(nn.Module):
    def __init__(
            self, dim: int, num_heads: int, mlp_ratio: float, attn_type: str,
            pool_ratio: int, lambda_scale: float, ns_iters: int, use_rope: bool,
            rope_base: float, laplacian_backend: str = "cuda",
            ):
        super().__init__()
        self.attn_type = attn_type
        self.norm1 = nn.LayerNorm(dim)
        if attn_type == "vanilla":
            self.attn = SpatialMultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                use_rope=use_rope,
                rope_base=rope_base,
            )
        elif attn_type == "laplacian":
            if laplacian_backend != "cuda":
                raise ValueError("Laplacian PVT attention supports only laplacian_backend='cuda'")

            self.attn = FastCudaLaplacianLinearAttention(
                dim=dim,
                num_heads=num_heads,
                lambda_scale=lambda_scale,
                pool_ratio=pool_ratio,
                ns_iters=ns_iters,
                use_rope=use_rope,
                rope_base=rope_base,
            )
        else:
            raise ValueError("attn_type must be 'vanilla' or 'laplacian'")

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), height, width)
        x = x + self.mlp(self.norm2(x))
        return x


class PyramidStage(nn.Module):
    def __init__(
            self, in_channels: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float,
            patch_size: int, stride: int, padding: int, attn_type: str, pool_ratio: int,
            lambda_scale: float, ns_iters: int, use_rope: bool, rope_base: float,
            laplacian_backend: str = "cuda",
            ):
        super().__init__()
        self.patch_embed = PyramidPatchEmbedding(in_channels, embed_dim, patch_size, stride, padding)
        self.blocks = nn.ModuleList(
            [
                PyramidBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_type=attn_type,
                    pool_ratio=pool_ratio,
                    lambda_scale=lambda_scale,
                    ns_iters=ns_iters,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    laplacian_backend=laplacian_backend,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x, height, width = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, height, width)
        x = self.norm(x)
        return x, height, width


class PyramidVisionBackbone(nn.Module):
    """
    PVT-style hierarchical backbone with optional RoPE in each stage.
    Returns a pooled feature vector from the final stage.
    """

    def __init__(
            self, img_size: int = 224, in_channels: int = 3,
            embed_dims: tuple[int, ...] = (64, 128, 320, 512),
            depths: tuple[int, ...] = (2, 2, 2, 2),
            num_heads: tuple[int, ...] = (1, 2, 5, 8),
            mlp_ratios: tuple[float, ...] = (8.0, 8.0, 4.0, 4.0),
            patch_sizes: tuple[int, ...] = (7, 3, 3, 3),
            strides: tuple[int, ...] = (4, 2, 2, 2),
            paddings: tuple[int, ...] = (3, 1, 1, 1),
            pool_ratios: tuple[int, ...] = (8, 4, 2, 1),
            attn_type: str = "laplacian", lambda_scale: float = 4.0, ns_iters: int = 5,
            use_rope: bool = True, rope_base: float = 10000.0,
            laplacian_backend: str = "cuda",
            ):
        super().__init__()
        stage_lengths = {
            len(embed_dims), len(depths), len(num_heads), len(mlp_ratios),
            len(patch_sizes), len(strides), len(paddings), len(pool_ratios)
        }
        if len(stage_lengths) != 1:
            raise ValueError("All stage-wise PVT config lists must have the same length")

        self.img_size = img_size
        self.out_dim = embed_dims[-1]
        self.stages = nn.ModuleList()

        current_channels = in_channels
        for embed_dim, depth, heads, mlp_ratio, patch_size, stride, padding, pool_ratio in zip(
                embed_dims, depths, num_heads, mlp_ratios, patch_sizes, strides, paddings, pool_ratios
                ):
            stage = PyramidStage(
                in_channels=current_channels,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                stride=stride,
                padding=padding,
                attn_type=attn_type,
                pool_ratio=pool_ratio,
                lambda_scale=lambda_scale,
                ns_iters=ns_iters,
                use_rope=use_rope,
                rope_base=rope_base,
                laplacian_backend=laplacian_backend,
            )
            self.stages.append(stage)
            current_channels = embed_dim

    def forward_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage_index, stage in enumerate(self.stages):
            x, height, width = stage(x)
            feature_map = x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], height, width)
            features.append(feature_map)
            if stage_index < len(self.stages) - 1:
                x = feature_map

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_feature_maps(x)
        return features[-1].flatten(2).mean(dim=-1)
