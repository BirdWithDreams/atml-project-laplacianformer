import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from src.models.text import TextBackbone

class NLPClassificationTask(L.LightningModule):
    def __init__(
            self,
            num_classes: int = 2,
            lr: float = 2e-5,
            weight_decay: float = 0.01,
            optimizer: str = "AdamW",
            model_cfg: dict = None,
            vocab_size: int = 30522,
            max_seq_len: int = 128
            ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Backbone via model_cfg
        if model_cfg is None:
            model_cfg = {"attn_type": "vanilla", "dim": 384, "depth": 6, "num_heads": 6}

        self.backbone = TextBackbone(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dim=model_cfg.get("dim", 384),
            depth=model_cfg.get("depth", 6),
            num_heads=model_cfg.get("num_heads", 6),
            attn_type=model_cfg.get("attn_type", "vanilla"),
            lambda_scale=model_cfg.get("lambda_scale", 4.0),
            pool_ratio=model_cfg.get("pool_ratio", 2),
            ns_iters=model_cfg.get("ns_iters", 5),
        )

        dim = model_cfg.get("dim", 384)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_rec = MulticlassRecall(num_classes=num_classes, average="macro")

    def forward(self, input_ids, attention_mask=None):
        features = self.backbone(input_ids, attention_mask)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.train_acc(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        self.val_acc(logits, labels)
        self.val_prec(logits, labels)
        self.val_rec(logits, labels)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/precision", self.val_prec)
        self.log("val/recall", self.val_rec)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("label")
        if labels is None or torch.all(labels < 0):
            return

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.log("test/loss", loss)

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
