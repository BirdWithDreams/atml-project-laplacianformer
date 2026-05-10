import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.models.pvt import PyramidVisionBackbone


def _make_group_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    if num_channels < 1 or num_groups < 1:
        raise ValueError("num_channels and num_groups must be positive for GroupNorm")
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class SegmentationHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            hidden_channels: int = 64,
            norm_groups: int = 32,
            num_layers: int = 2,
            dropout: float = 0.0,
            ):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(max(int(num_layers), 1)):
            layers.extend(
                [
                    nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    _make_group_norm(hidden_channels, norm_groups),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            current_channels = hidden_channels
        self.refine = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        features = F.interpolate(
            features,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.classifier(self.refine(features))


class PyramidSegmentationModel(nn.Module):
    """
    Lightweight semantic segmentation model built on the existing PVT backbone.
    Multi-scale PVT feature maps are projected, fused with GroupNorm, upsampled
    as features, and refined by a learned segmentation head at full resolution.
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
            decoder_norm_groups: int = 32,
            segmentation_head_dim: int = 64,
            segmentation_head_layers: int = 2,
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
            nn.Conv2d(
                decoder_dim * len(embed_dims),
                decoder_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _make_group_norm(decoder_dim, decoder_norm_groups),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(decoder_dim, decoder_norm_groups),
            nn.GELU(),
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_dim,
            num_classes=num_classes,
            hidden_channels=segmentation_head_dim,
            norm_groups=decoder_norm_groups,
            num_layers=segmentation_head_layers,
            dropout=dropout * 0.5,
        )

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

        fused = self.fuse(torch.cat(projected, dim=1))
        return self.segmentation_head(fused, input_size)


class TorchvisionSegmentationModel(nn.Module):
    """Thin wrapper around torchvision segmentation models for pipeline sanity checks."""

    def __init__(
            self,
            num_classes: int,
            architecture: str = "deeplabv3_resnet50",
            weights: str | None = "DEFAULT",
            weights_backbone: str | None = "DEFAULT",
            aux_loss: bool | None = None,
            ):
        super().__init__()
        self.architecture = architecture
        self.model = self._build_model(
            architecture=architecture,
            num_classes=num_classes,
            weights=weights,
            weights_backbone=weights_backbone,
            aux_loss=aux_loss,
        )
        self.backbone = self.model.backbone
        decoder_modules = [self.model.classifier]
        aux_classifier = getattr(self.model, "aux_classifier", None)
        if aux_classifier is not None:
            decoder_modules.append(aux_classifier)
        self.segmentation_head = nn.ModuleList(decoder_modules)
        self.projections = nn.ModuleList()
        self.fuse = nn.Identity()

    @staticmethod
    def _resolve_weights(weights_enum, value: str | None):
        if value is None or str(value).lower() in {"none", "null", "false"}:
            return None
        if str(value).upper() == "DEFAULT":
            return weights_enum.DEFAULT
        return weights_enum[value]

    def _build_model(
            self,
            architecture: str,
            num_classes: int,
            weights: str | None,
            weights_backbone: str | None,
            aux_loss: bool | None,
            ) -> nn.Module:
        if architecture == "deeplabv3_resnet50":
            from torchvision.models import ResNet50_Weights
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            resolved_weights = self._resolve_weights(DeepLabV3_ResNet50_Weights, weights)
            resolved_backbone_weights = None
            if resolved_weights is None:
                resolved_backbone_weights = self._resolve_weights(ResNet50_Weights, weights_backbone)
            return torchvision.models.segmentation.deeplabv3_resnet50(
                weights=resolved_weights,
                weights_backbone=resolved_backbone_weights,
                num_classes=num_classes,
                aux_loss=aux_loss,
            )

        if architecture == "fcn_resnet50":
            from torchvision.models import ResNet50_Weights
            from torchvision.models.segmentation import FCN_ResNet50_Weights

            resolved_weights = self._resolve_weights(FCN_ResNet50_Weights, weights)
            resolved_backbone_weights = None
            if resolved_weights is None:
                resolved_backbone_weights = self._resolve_weights(ResNet50_Weights, weights_backbone)
            return torchvision.models.segmentation.fcn_resnet50(
                weights=resolved_weights,
                weights_backbone=resolved_backbone_weights,
                num_classes=num_classes,
                aux_loss=aux_loss,
            )

        raise ValueError(f"Unknown torchvision segmentation architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]
