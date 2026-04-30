import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Metric

from src.models.segmentation import PyramidSegmentationModel


class SegmentationMetrics(Metric):
    def __init__(self, num_classes: int, ignore_index: int = 255, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state(
            "intersections",
            default=torch.zeros(num_classes, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "unions",
            default=torch.zeros(num_classes, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        valid = target != self.ignore_index
        preds = preds[valid]
        target = target[valid]
        if target.numel() == 0:
            return

        self.correct += (preds == target).sum().float()
        self.total += target.numel()

        for class_index in range(self.num_classes):
            pred_mask = preds == class_index
            target_mask = target == class_index
            self.intersections[class_index] += (pred_mask & target_mask).sum().float()
            self.unions[class_index] += (pred_mask | target_mask).sum().float()

    def compute(self) -> dict[str, torch.Tensor]:
        valid_classes = self.unions > 0
        if valid_classes.any():
            mean_iou = (self.intersections[valid_classes] / self.unions[valid_classes]).mean()
        else:
            mean_iou = torch.tensor(0.0, device=self.intersections.device)

        pixel_acc = self.correct / torch.clamp(self.total, min=1.0)
        return {"mIoU": mean_iou, "pixel_acc": pixel_acc}


class SemanticSegmentationTask(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            lr: float = 3e-4,
            weight_decay: float = 0.05,
            optimizer: str = "AdamW",
            model_cfg: dict = None,
            ignore_index: int = 255,
            ):
        super().__init__()
        self.save_hyperparameters()

        if model_cfg is None:
            model_cfg = {"attn_type": "laplacian", "backbone_type": "pvt"}

        backbone_type = model_cfg.get("backbone_type", "pvt")
        if backbone_type != "pvt":
            raise ValueError("SemanticSegmentationTask currently supports backbone_type='pvt'")

        self.model = PyramidSegmentationModel(
            num_classes=num_classes,
            img_size=model_cfg.get("img_size", 224),
            embed_dims=tuple(model_cfg.get("embed_dims", [64, 128, 320, 512])),
            depths=tuple(model_cfg.get("depths", [2, 2, 2, 2])),
            num_heads=tuple(model_cfg.get("num_heads", [1, 2, 5, 8])),
            mlp_ratios=tuple(model_cfg.get("mlp_ratios", [8.0, 8.0, 4.0, 4.0])),
            patch_sizes=tuple(model_cfg.get("patch_sizes", [7, 3, 3, 3])),
            strides=tuple(model_cfg.get("strides", [4, 2, 2, 2])),
            paddings=tuple(model_cfg.get("paddings", [3, 1, 1, 1])),
            pool_ratios=tuple(model_cfg.get("pool_ratios", [8, 4, 2, 1])),
            attn_type=model_cfg.get("attn_type", "laplacian"),
            lambda_scale=model_cfg.get("lambda_scale", 4.0),
            ns_iters=model_cfg.get("ns_iters", 5),
            use_rope=model_cfg.get("use_rope", True),
            rope_base=model_cfg.get("rope_base", 10000.0),
            laplacian_backend=model_cfg.get("laplacian_backend", "torch"),
            laplacian_fallback_to_torch=model_cfg.get("laplacian_fallback_to_torch", True),
            decoder_dim=model_cfg.get("decoder_dim", 128),
            dropout=model_cfg.get("dropout", 0.1),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
        self.test_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            valid = masks != self.hparams.ignore_index
            pixel_acc = (preds[valid] == masks[valid]).float().mean() if valid.any() else torch.tensor(0.0)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/pixel_acc", pixel_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.val_metrics(preds, masks)
        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log("val/mIoU", metrics["mIoU"], prog_bar=True)
        self.log("val/pixel_acc", metrics["pixel_acc"])
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.test_metrics(preds, masks)
        self.log("test/loss", loss)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log("test/mIoU", metrics["mIoU"])
        self.log("test/pixel_acc", metrics["pixel_acc"])
        self.test_metrics.reset()

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
