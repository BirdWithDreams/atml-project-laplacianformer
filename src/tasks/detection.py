import lightning as L
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from torchvision.models.detection.backbone_utils import BackboneWithFPN
import torchvision


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

        # Use pretrained ResNet backbone for detection
        # but vary the attention type in a classification head
        attn_type = model_cfg.get("attn_type", "vanilla") if model_cfg else "vanilla"
        
        # Use torchvision's resnet50 backbone with FPN
        backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes
        )
        self.model = backbone

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
