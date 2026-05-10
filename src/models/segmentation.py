import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pvt import PyramidVisionBackbone


class PyramidSegmentationModel(nn.Module):
    """
    Lightweight semantic segmentation model built on the existing PVT backbone.
    Multi-scale PVT feature maps are projected to a shared decoder dimension,
    fused at the highest backbone resolution, and upsampled to the input size.
    """

    def __init__(
            self,
            num_classes: int,
            img_size: int = 224,
            in_channels: int = 3,
            embed_dims: tuple[int, ...] = (64, 128, 320, 512),
            depths: tuple[int, ...] = (2, 2, 2, 2),
            num_heads: tuple[int, ...] = (1, 2, 5, 8),
            mlp_ratios: tuple[float, ...] = (8.0, 8.0, 4.0, 4.0),
            patch_sizes: tuple[int, ...] = (7, 3, 3, 3),
            strides: tuple[int, ...] = (4, 2, 2, 2),
            paddings: tuple[int, ...] = (3, 1, 1, 1),
            pool_ratios: tuple[int, ...] = (8, 4, 2, 1),
            attn_type: str = "laplacian",
            lambda_scale: float = 4.0,
            ns_iters: int = 5,
            use_rope: bool = True,
            rope_base: float = 10000.0,
            laplacian_backend: str = "cuda",
            decoder_dim: int = 128,
            dropout: float = 0.1,
            ):
        super().__init__()
        self.backbone = PyramidVisionBackbone(
            img_size=img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            patch_sizes=patch_sizes,
            strides=strides,
            paddings=paddings,
            pool_ratios=pool_ratios,
            attn_type=attn_type,
            lambda_scale=lambda_scale,
            ns_iters=ns_iters,
            use_rope=use_rope,
            rope_base=rope_base,
            laplacian_backend=laplacian_backend,
        )
        self.projections = nn.ModuleList(
            [nn.Conv2d(dim, decoder_dim, kernel_size=1) for dim in embed_dims]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(embed_dims), decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone.forward_feature_maps(x)
        target_size = features[0].shape[-2:]

        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)

        logits = self.classifier(self.fuse(torch.cat(projected, dim=1)))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
