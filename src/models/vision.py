import torch
import torch.nn as nn
from .vanilla_attn import MultiHeadAttention
from .laplacian_attn import LaplacianLinearAttention
from .laplacian_fast_attn import FastCudaLaplacianLinearAttention


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    def __init__(
            self, dim, num_heads, attn_type="vanilla", grid_size=14,
            attn_kwargs=None, laplacian_backend="torch", laplacian_fallback_to_torch=True
            ):
        super().__init__()
        self.attn_type = attn_type
        self.grid_size = grid_size
        attn_kwargs = attn_kwargs or {}

        self.norm1 = nn.LayerNorm(dim)
        if attn_type == "vanilla":
            self.attn = MultiHeadAttention(d_model=dim, num_heads=num_heads)
        elif attn_type == "laplacian":
            if laplacian_backend == "torch":
                attn_cls = LaplacianLinearAttention
                backend_kwargs = {}
            elif laplacian_backend == "cuda":
                attn_cls = FastCudaLaplacianLinearAttention
                backend_kwargs = {"fallback_to_torch": laplacian_fallback_to_torch}
            else:
                raise ValueError("laplacian_backend must be 'torch' or 'cuda'")

            self.attn = attn_cls(
                dim=dim,
                num_heads=num_heads,
                lambda_scale=attn_kwargs.get("lambda_scale", 4.0),
                pool_ratio=attn_kwargs.get("pool_ratio", 2),
                ns_iters=attn_kwargs.get("ns_iters", 5),
                **backend_kwargs,
            )
            self.cls_norm = nn.LayerNorm(dim)
            # Keep a lightweight CLS-to-token path so the classification token is updated
            # even though Laplacian attention operates over the spatial grid only.
            self.cls_attn = MultiHeadAttention(d_model=dim, num_heads=num_heads)
        else:
            raise ValueError("attn_type must be 'vanilla' or 'laplacian'")

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        if self.attn_type == "vanilla":
            norm_x = self.norm1(x)
            x = x + self.attn(norm_x, norm_x, norm_x)
        else:
            cls_token, spatial_x = x[:, :1], x[:, 1:]

            norm_spatial = self.norm1(spatial_x)
            spatial_x = spatial_x + self.attn(norm_spatial, self.grid_size, self.grid_size)

            cls_context = torch.cat((cls_token, spatial_x), dim=1)
            norm_cls_context = self.cls_norm(cls_context)
            cls_token = cls_token + self.cls_attn(
                norm_cls_context[:, :1],
                norm_cls_context,
                norm_cls_context,
            )

            x = torch.cat((cls_token, spatial_x), dim=1)

        x = x + self.mlp(self.norm2(x))
        return x


class VisionBackbone(nn.Module):
    """
    Acts solely as a feature extractor.
    Returns the [CLS] token representation for classification tasks.
    """
    def __init__(
            self, img_size=224, patch_size=16,
            dim=384, depth=6, num_heads=6, attn_type="vanilla",
            lambda_scale=4.0, pool_ratio=2, ns_iters=5,
            laplacian_backend="torch", laplacian_fallback_to_torch=True
            ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, dim))

        attn_kwargs = {
            "lambda_scale": lambda_scale,
            "pool_ratio": pool_ratio,
            "ns_iters": ns_iters,
        }

        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim, num_heads, attn_type, self.patch_embed.grid_size, attn_kwargs,
                    laplacian_backend=laplacian_backend,
                    laplacian_fallback_to_torch=laplacian_fallback_to_torch,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # return the CLS token
        return x[:, 0]
