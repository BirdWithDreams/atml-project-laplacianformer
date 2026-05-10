import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Metric

from src.models.segmentation import PyramidSegmentationModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
            log_segmentation_images: bool = True,
            num_log_segmentation_images: int = 4,
            log_segmentation_images_every_n_epochs: int = 1,
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
            laplacian_backend=model_cfg.get("laplacian_backend", "cuda"),
            decoder_dim=model_cfg.get("decoder_dim", 128),
            dropout=model_cfg.get("dropout", 0.1),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
        self.test_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
        self._segmentation_log_counts = {"val": 0, "test": 0}
        self._segmentation_class_labels = self._build_class_labels(num_classes, ignore_index)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        batch_size = images.shape[0]

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            valid = masks != self.hparams.ignore_index
            pixel_acc = (preds[valid] == masks[valid]).float().mean() if valid.any() else torch.tensor(0.0)

        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/pixel_acc_step", pixel_acc, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log("train/pixel_acc_epoch", pixel_acc, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        batch_size = images.shape[0]

        self.val_metrics(preds, masks)
        self._maybe_log_segmentation_images("val", images, masks, preds, batch_idx)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_start(self):
        self._segmentation_log_counts["val"] = 0

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
        batch_size = images.shape[0]

        self.test_metrics(preds, masks)
        self._maybe_log_segmentation_images("test", images, masks, preds, batch_idx)
        self.log("test/loss", loss, batch_size=batch_size)

    def on_test_epoch_start(self):
        self._segmentation_log_counts["test"] = 0

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log("test/mIoU", metrics["mIoU"])
        self.log("test/pixel_acc", metrics["pixel_acc"])
        self.test_metrics.reset()

    @staticmethod
    def _build_class_labels(num_classes: int, ignore_index: int) -> dict[int, str]:
        class_labels = {
            class_index: "background" if class_index == 0 else f"class_{class_index}"
            for class_index in range(num_classes)
        }
        class_labels[ignore_index] = "ignore"
        return class_labels

    def _should_log_segmentation_images(self, split: str) -> bool:
        if not self.hparams.log_segmentation_images:
            return False
        if int(self.hparams.num_log_segmentation_images) <= 0:
            return False
        if getattr(self.trainer, "sanity_checking", False):
            return False
        if not getattr(self.trainer, "is_global_zero", True):
            return False

        if split == "val":
            every_n_epochs = max(int(self.hparams.log_segmentation_images_every_n_epochs), 1)
            return self.current_epoch % every_n_epochs == 0
        return split == "test"

    def _maybe_log_segmentation_images(
            self,
            split: str,
            images: torch.Tensor,
            masks: torch.Tensor,
            preds: torch.Tensor,
            batch_idx: int,
            ):
        if not self._should_log_segmentation_images(split):
            return

        log_limit = int(self.hparams.num_log_segmentation_images)
        already_logged = self._segmentation_log_counts.get(split, 0)
        remaining = log_limit - already_logged
        if remaining <= 0:
            return

        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger is not None else None
        if experiment is None or not hasattr(experiment, "log"):
            return

        try:
            import wandb
        except ImportError:
            return

        num_images = min(remaining, images.shape[0])
        wandb_images = []
        for sample_index in range(num_images):
            log_index = already_logged + sample_index
            wandb_images.append(
                wandb.Image(
                    self._to_display_image(images[sample_index]),
                    masks={
                        "prediction": {
                            "mask_data": self._to_display_mask(preds[sample_index]),
                            "class_labels": self._segmentation_class_labels,
                        },
                        "ground_truth": {
                            "mask_data": self._to_display_mask(masks[sample_index]),
                            "class_labels": self._segmentation_class_labels,
                        },
                    },
                    caption=f"{split} epoch={self.current_epoch} batch={batch_idx} sample={log_index}",
                )
            )

        if not wandb_images:
            return

        experiment.log(
            {f"{split}/segmentation_masks": wandb_images},
            commit=False,
        )
        self._segmentation_log_counts[split] = already_logged + num_images

    @staticmethod
    def _to_display_image(image: torch.Tensor):
        mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
        image = (image.detach() * std + mean).clamp(0.0, 1.0)
        image = image.permute(1, 2, 0).cpu().numpy()
        return (image * 255).astype("uint8")

    @staticmethod
    def _to_display_mask(mask: torch.Tensor):
        return mask.detach().to(torch.int32).cpu().numpy()

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
