import lightning as L
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from src.models.vision import VisionBackbone, PatchEmbedding, ViTBlock


class DetectionBackbone(nn.Module):
    """
    ViT backbone that returns a 2D feature map for detection.
    Uses either vanilla or laplacian attention.
    """
    def __init__(self, img_size=224, patch_size=16, dim=384, depth=6, num_heads=6, attn_type="vanilla"):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, attn_type, self.patch_embed.grid_size)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim
        self.grid_size = self.patch_embed.grid_size

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        for block in self.blocks:
            if block.attn_type == "laplacian":
                x = block(x)
            else:
                x = block(x)
        x = self.norm(x)
        # reshape to 2D feature map: (B, C, H, W)
        H = W = self.grid_size
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return {"0": x}


class ObjectDetectionTask(L.LightningModule):
    def __init__(
            self,
            num_classes: int = 91,
            lr: float = 1e-4,
            weight_decay: float = 0.0005,
            optimizer: str = "AdamW",
            model_cfg: dict = None
            ):
        super().__init__()
        self.save_hyperparameters()

        if model_cfg is None:
            model_cfg = {"attn_type": "vanilla", "dim": 384, "depth": 6, "num_heads": 6}

        backbone = DetectionBackbone(
            img_size=224,
            patch_size=16,
            dim=model_cfg.get("dim", 384),
            depth=model_cfg.get("depth", 6),
            num_heads=model_cfg.get("num_heads", 6),
            attn_type=model_cfg.get("attn_type", "vanilla")
        )

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torch.ops.torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        ) if False else None

        from torchvision.ops import MultiScaleRoIAlign
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=224,
            max_size=224
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        targets = list(targets)
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        preds = self.model(images)
        total_iou = 0.0
        count = 0
        for pred, target in zip(preds, targets):
            if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
                iou = box_iou(pred["boxes"], target["boxes"])
                total_iou += iou.max(dim=1).values.mean().item()
                count += 1
        if count > 0:
            self.log("val/mean_iou", total_iou / count, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        preds = self.model(images)
        total_iou = 0.0
        count = 0
        for pred, target in zip(preds, targets):
            if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
                iou = box_iou(pred["boxes"], target["boxes"])
                total_iou += iou.max(dim=1).values.mean().item()
                count += 1
        if count > 0:
            self.log("test/mean_iou", total_iou / count)

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }