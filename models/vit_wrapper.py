import torch
import torch.nn as nn
from .vanilla_attn import MultiHeadAttention
from .laplacian_attn import LaplacianLinearAttention


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Flatten spatial dimensions and transpose: (B, C, H, W) -> (B, N, C)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, attn_type="vanilla", grid_size=14):
        super().__init__()
        self.attn_type = attn_type
        self.grid_size = grid_size

        self.norm1 = nn.LayerNorm(dim)
        if attn_type == "vanilla":
            self.attn = MultiHeadAttention(d_model=dim, num_heads=num_heads)
        elif attn_type == "laplacian":
            self.attn = LaplacianLinearAttention(dim=dim, num_heads=num_heads)
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
            # Vanilla expects (Q, K, V)
            norm_x = self.norm1(x)
            x = x + self.attn(norm_x, norm_x, norm_x)
        else:
            # Laplacian expects (x, H_sp, W_sp)
            x = x + self.attn(self.norm1(x), self.grid_size, self.grid_size)

        x = x + self.mlp(self.norm2(x))
        return x


class CompareViT(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, num_classes=1000,
            dim=384, depth=6, num_heads=6, attn_type="vanilla"
            ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, dim))

        self.blocks = nn.ModuleList(
            [
                ViTBlock(dim, num_heads, attn_type, self.patch_embed.grid_size)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Note: We slice off the cls_token for the spatial Laplacian attention
        # and add it back, or just process the tokens. For simplicity in this blueprint,
        # we process everything but pass grid_size to Laplacian.
        for block in self.blocks:
            if block.attn_type == "laplacian":
                # Laplacian expects strict spatial grid, exclude cls_token temporarily
                cls_token, spatial_x = x[:, :1], x[:, 1:]
                spatial_x = block(spatial_x)
                x = torch.cat((cls_token, spatial_x), dim=1)
            else:
                x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])