import math

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from src.models.pvt import PyramidVisionBackbone
from src.models.vision import VisionBackbone

class CVClassificationTask(L.LightningModule):
    def __init__(
            self,
            num_classes: int = 100,
            lr: float = 3e-4,
            weight_decay: float = 0.05,
            optimizer: str = "AdamW",
            scheduler: str = "warmup_cosine",
            warmup_epochs: int = 0,
            model_cfg: dict = None
            ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Backbone via model_cfg (which comes from Hydra)
        if model_cfg is None:
            model_cfg = {"attn_type": "vanilla", "dim": 384, "depth": 6, "num_heads": 6}

        backbone_type = model_cfg.get("backbone_type", "vit")
        if backbone_type == "vit":
            self.backbone = VisionBackbone(
                img_size=model_cfg.get("img_size", 224),
                patch_size=model_cfg.get("patch_size", 16),
                dim=model_cfg.get("dim", 384),
                depth=model_cfg.get("depth", 6),
                num_heads=model_cfg.get("num_heads", 6),
                attn_type=model_cfg.get("attn_type", "vanilla"),
                lambda_scale=model_cfg.get("lambda_scale", 4.0),
                pool_ratio=model_cfg.get("pool_ratio", 2),
                ns_iters=model_cfg.get("ns_iters", 5),
            )
            dim = model_cfg.get("dim", 384)
        elif backbone_type == "pvt":
            self.backbone = PyramidVisionBackbone(
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
            )
            dim = self.backbone.out_dim
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        self.head = nn.Linear(dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_rec = MulticlassRecall(num_classes=num_classes, average="macro")

    def _validate_targets(self, targets: torch.Tensor):
        if targets.numel() == 0:
            return

        min_target = int(targets.detach().min().item())
        max_target = int(targets.detach().max().item())
        num_classes = int(self.hparams.num_classes)
        if min_target < 0 or max_target >= num_classes:
            raise ValueError(
                "CV classification target labels are outside the configured class range: "
                f"min={min_target}, max={max_target}, num_classes={num_classes}. "
                "Check datamodule.num_classes against the dataset class folders."
            )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self._validate_targets(y)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self._validate_targets(y)
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.val_acc(logits, y)
        self.val_prec(logits, y)
        self.val_rec(logits, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/precision", self.val_prec)
        self.log("val/recall", self.val_rec)

    def test_step(self, batch, batch_idx):
        x, y = batch
        self._validate_targets(y)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler_name = str(self.hparams.scheduler).lower()
        if scheduler_name in {"none", "null", "false"}:
            return optimizer

        max_epochs = int(self.trainer.max_epochs)
        if max_epochs <= 0:
            raise ValueError(
                "CV classification schedulers require trainer.max_epochs > 0, "
                f"got {max_epochs}."
            )

        warmup_epochs = int(self.hparams.warmup_epochs)
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}.")
        if warmup_epochs >= max_epochs:
            raise ValueError(
                "warmup_epochs must be smaller than trainer.max_epochs; "
                f"got warmup_epochs={warmup_epochs}, max_epochs={max_epochs}."
            )

        if scheduler_name in {"cosine", "warmup_cosine"}:
            def lr_lambda(current_epoch: int) -> float:
                current_epoch = max(int(current_epoch), 0)
                if warmup_epochs > 0 and current_epoch < warmup_epochs:
                    return float(current_epoch + 1) / float(warmup_epochs)

                decay_epochs = max(max_epochs - warmup_epochs, 1)
                decay_epoch = min(max(current_epoch - warmup_epochs, 0), decay_epochs)
                decay_progress = float(decay_epoch) / float(decay_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * decay_progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(
                "CVClassificationTask supports scheduler='warmup_cosine', "
                f"scheduler='cosine', or scheduler='none'; got {self.hparams.scheduler!r}."
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
