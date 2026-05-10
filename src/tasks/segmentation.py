import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

from src.models.segmentation import PyramidSegmentationModel, TorchvisionSegmentationModel


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
        self.add_state(
            "pred_pixels",
            default=torch.zeros(num_classes, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "target_pixels",
            default=torch.zeros(num_classes, dtype=torch.float),
            dist_reduce_fx="sum",
        )

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
            self.pred_pixels[class_index] += pred_mask.sum().float()
            self.target_pixels[class_index] += target_mask.sum().float()

    def compute(self) -> dict[str, torch.Tensor]:
        valid_classes = self.unions > 0
        iou_per_class = self.intersections / torch.clamp(self.unions, min=1.0)
        if valid_classes.any():
            mean_iou = iou_per_class[valid_classes].mean()
        else:
            mean_iou = torch.tensor(0.0, device=self.intersections.device)

        class_indices = torch.arange(self.num_classes, device=self.unions.device)
        foreground_classes = valid_classes & (class_indices != 0)
        if foreground_classes.any():
            foreground_miou = iou_per_class[foreground_classes].mean()
        else:
            foreground_miou = torch.tensor(0.0, device=self.intersections.device)

        pixel_acc = self.correct / torch.clamp(self.total, min=1.0)
        pred_background_fraction = self.pred_pixels[0] / torch.clamp(self.pred_pixels.sum(), min=1.0)
        target_background_fraction = self.target_pixels[0] / torch.clamp(self.target_pixels.sum(), min=1.0)
        return {
            "mIoU": mean_iou,
            "foreground_mIoU": foreground_miou,
            "pixel_acc": pixel_acc,
            "pred_background_fraction": pred_background_fraction,
            "target_background_fraction": target_background_fraction,
        }


class FocalDiceLoss(nn.Module):
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = 255,
            focal_gamma: float = 2.0,
            focal_alpha: float | list[float] | tuple[float, ...] | None = None,
            class_weights: list[float] | tuple[float, ...] | None = None,
            focal_weight: float = 1.0,
            dice_weight: float = 1.0,
            dice_smooth: float = 1e-5,
            dice_present_classes_only: bool = True,
            ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha_scalar = (
            float(focal_alpha) if isinstance(focal_alpha, (int, float)) else None
        )
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.dice_smooth = float(dice_smooth)
        self.dice_present_classes_only = bool(dice_present_classes_only)

        class_alpha = None
        if focal_alpha is not None and self.focal_alpha_scalar is None:
            class_alpha = torch.as_tensor(list(focal_alpha), dtype=torch.float)
            if class_alpha.numel() != self.num_classes:
                raise ValueError(
                    "focal_alpha must be a scalar or provide one weight per class; "
                    f"got {class_alpha.numel()} weights for {self.num_classes} classes."
                )
        ce_weights = None
        if class_weights is not None:
            ce_weights = torch.as_tensor(list(class_weights), dtype=torch.float)
            if ce_weights.numel() != self.num_classes:
                raise ValueError(
                    "class_weights must provide one weight per class; "
                    f"got {ce_weights.numel()} weights for {self.num_classes} classes."
                )
        self.register_buffer("class_alpha", class_alpha, persistent=False)
        self.register_buffer("ce_weights", ce_weights, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        focal_loss = self._focal_loss(logits, target, valid)
        dice_loss = self._dice_loss(logits, target, valid)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

    def _focal_loss(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
            valid: torch.Tensor,
            ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            weight=self.ce_weights.to(logits.device) if self.ce_weights is not None else None,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt).pow(self.focal_gamma) * ce_loss

        if self.class_alpha is not None:
            safe_target = target.clamp(min=0, max=self.num_classes - 1)
            focal_loss = focal_loss * self.class_alpha.to(logits.device)[safe_target]
        elif self.focal_alpha_scalar is not None:
            background_alpha = 1.0 - self.focal_alpha_scalar
            alpha_t = torch.where(
                target == 0,
                target.new_full(target.shape, background_alpha, dtype=logits.dtype),
                target.new_full(target.shape, self.focal_alpha_scalar, dtype=logits.dtype),
            )
            focal_loss = focal_loss * alpha_t

        return focal_loss[valid].mean()

    def _dice_loss(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
            valid: torch.Tensor,
            ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid_mask = valid.unsqueeze(1).to(dtype=probs.dtype)
        safe_target = target.masked_fill(~valid, 0)
        target_one_hot = F.one_hot(safe_target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)

        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        reduce_dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.dice_smooth) / (denominator + self.dice_smooth)
        if self.dice_present_classes_only:
            present_classes = target_one_hot.sum(dim=reduce_dims) > 0
            if present_classes.any():
                return 1.0 - dice[present_classes].mean()
        return 1.0 - dice.mean()


class WarmupPolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iters: int,
            warmup_iters: int = 1500,
            power: float = 0.9,
            ):
        max_iters = int(max_iters)
        warmup_iters = int(warmup_iters)
        power = float(power)
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive for WarmupPolyLR, got {max_iters}.")
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}.")
        if warmup_iters >= max_iters:
            raise ValueError(
                "warmup_iters must be smaller than max_iters for WarmupPolyLR; "
                f"got warmup_iters={warmup_iters}, max_iters={max_iters}."
            )

        def lr_lambda(current_step: int) -> float:
            current_step = min(max(int(current_step), 0), max_iters)
            if warmup_iters > 0 and current_step < warmup_iters:
                return float(current_step + 1) / float(warmup_iters)

            decay_iters = max(max_iters - warmup_iters, 1)
            decay_step = min(max(current_step - warmup_iters, 0), decay_iters)
            decay_progress = float(decay_step) / float(decay_iters)
            return max(1.0 - decay_progress, 0.0) ** power

        super().__init__(optimizer, lr_lambda=lr_lambda)


class FocalDiceLoss(nn.Module):
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = 255,
            focal_gamma: float = 2.0,
            focal_alpha: float | list[float] | tuple[float, ...] | None = None,
            focal_weight: float = 1.0,
            dice_weight: float = 1.0,
            dice_smooth: float = 1e-5,
            ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha_scalar = (
            float(focal_alpha) if isinstance(focal_alpha, (int, float)) else None
        )
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.dice_smooth = float(dice_smooth)

        class_alpha = None
        if focal_alpha is not None and self.focal_alpha_scalar is None:
            class_alpha = torch.as_tensor(list(focal_alpha), dtype=torch.float)
            if class_alpha.numel() != self.num_classes:
                raise ValueError(
                    "focal_alpha must be a scalar or provide one weight per class; "
                    f"got {class_alpha.numel()} weights for {self.num_classes} classes."
                )
        self.register_buffer("class_alpha", class_alpha, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        focal_loss = self._focal_loss(logits, target, valid)
        dice_loss = self._dice_loss(logits, target, valid)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

    def _focal_loss(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
            valid: torch.Tensor,
            ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt).pow(self.focal_gamma) * ce_loss

        if self.class_alpha is not None:
            safe_target = target.clamp(min=0, max=self.num_classes - 1)
            focal_loss = focal_loss * self.class_alpha.to(logits.device)[safe_target]
        elif self.focal_alpha_scalar is not None:
            background_alpha = 1.0 - self.focal_alpha_scalar
            alpha_t = torch.where(
                target == 0,
                target.new_full(target.shape, background_alpha, dtype=logits.dtype),
                target.new_full(target.shape, self.focal_alpha_scalar, dtype=logits.dtype),
            )
            focal_loss = focal_loss * alpha_t

        return focal_loss[valid].mean()

    def _dice_loss(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
            valid: torch.Tensor,
            ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid_mask = valid.unsqueeze(1).to(dtype=probs.dtype)
        safe_target = target.masked_fill(~valid, 0)
        target_one_hot = F.one_hot(safe_target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)

        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        reduce_dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.dice_smooth) / (denominator + self.dice_smooth)
        return 1.0 - dice.mean()


class WarmupPolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iters: int,
            warmup_iters: int = 1500,
            power: float = 0.9,
            ):
        max_iters = int(max_iters)
        warmup_iters = int(warmup_iters)
        power = float(power)
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive for WarmupPolyLR, got {max_iters}.")
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}.")
        if warmup_iters >= max_iters:
            raise ValueError(
                "warmup_iters must be smaller than max_iters for WarmupPolyLR; "
                f"got warmup_iters={warmup_iters}, max_iters={max_iters}."
            )

        def lr_lambda(current_step: int) -> float:
            current_step = min(max(int(current_step), 0), max_iters)
            if warmup_iters > 0 and current_step < warmup_iters:
                return float(current_step + 1) / float(warmup_iters)

            decay_iters = max(max_iters - warmup_iters, 1)
            decay_step = min(max(current_step - warmup_iters, 0), decay_iters)
            decay_progress = float(decay_step) / float(decay_iters)
            return max(1.0 - decay_progress, 0.0) ** power

        super().__init__(optimizer, lr_lambda=lr_lambda)


class SemanticSegmentationTask(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            lr: float = 3e-4,
            weight_decay: float = 0.05,
            optimizer: str = "AdamW",
            optimizer_betas: tuple[float, float] | list[float] = (0.9, 0.999),
            decoder_lr_multiplier: float = 10.0,
            scheduler: str = "warmup_poly",
            max_iters: int = 80000,
            warmup_iters: int = 1500,
            poly_power: float = 0.9,
            model_cfg: dict = None,
            ignore_index: int = 255,
            log_segmentation_images: bool = True,
            num_log_segmentation_images: int = 4,
            log_segmentation_images_every_n_epochs: int = 1,
            focal_gamma: float = 2.0,
            focal_alpha: float | list[float] | tuple[float, ...] | None = None,
            focal_loss_weight: float = 1.0,
            dice_loss_weight: float = 1.0,
            dice_smooth: float = 1e-5,
            ):
        super().__init__()
        self.save_hyperparameters()

        if model_cfg is None:
            model_cfg = {"attn_type": "laplacian", "backbone_type": "pvt"}

        backbone_type = model_cfg.get("backbone_type", "pvt")
        if backbone_type == "pvt":
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
                decoder_norm_groups=model_cfg.get("decoder_norm_groups", 32),
                segmentation_head_dim=model_cfg.get("segmentation_head_dim", 64),
                segmentation_head_layers=model_cfg.get("segmentation_head_layers", 2),
                dropout=model_cfg.get("dropout", 0.1),
            )
        elif backbone_type == "torchvision":
            self.model = TorchvisionSegmentationModel(
                num_classes=num_classes,
                architecture=model_cfg.get("architecture", "deeplabv3_resnet50"),
                weights=model_cfg.get("weights", "DEFAULT"),
                weights_backbone=model_cfg.get("weights_backbone", "DEFAULT"),
                aux_loss=model_cfg.get("aux_loss", None),
            )
        else:
            raise ValueError(
                "SemanticSegmentationTask supports backbone_type='pvt' or 'torchvision', "
                f"got {backbone_type!r}."
            )

        if backbone_type == "pvt" and model_cfg.get("backbone_checkpoint_path", None):
            self._load_backbone_checkpoint(model_cfg["backbone_checkpoint_path"])

        self.criterion = FocalDiceLoss(
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
            decoder_norm_groups=model_cfg.get("decoder_norm_groups", 32),
            segmentation_head_dim=model_cfg.get("segmentation_head_dim", 64),
            dropout=model_cfg.get("dropout", 0.1),
        )
        self.criterion = FocalDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            focal_weight=focal_loss_weight,
            dice_weight=dice_loss_weight,
            dice_smooth=dice_smooth,
        )
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
        self.test_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
        self._segmentation_log_counts = {"val": 0, "test": 0}
        self._segmentation_class_labels = self._build_class_labels(num_classes, ignore_index)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _load_backbone_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_state = self.model.backbone.state_dict()
        candidate_state = {}

        for key, value in state_dict.items():
            stripped_key = key.removeprefix("module.")
            matched_prefixed_key = False
            for prefix in ("model.backbone.", "backbone."):
                if stripped_key.startswith(prefix):
                    candidate_state[stripped_key.removeprefix(prefix)] = value
                    matched_prefixed_key = True
                    break
            if not matched_prefixed_key and stripped_key in backbone_state:
                candidate_state[stripped_key] = value

        compatible_state = {
            key: value
            for key, value in candidate_state.items()
            if (
                key in backbone_state
                and hasattr(value, "shape")
                and tuple(value.shape) == tuple(backbone_state[key].shape)
            )
        }
        if not compatible_state:
            raise ValueError(f"No compatible backbone tensors found in checkpoint: {checkpoint_path}")

        self.model.backbone.load_state_dict(compatible_state, strict=False)

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
        self.log("val/foreground_mIoU", metrics["foreground_mIoU"])
        self.log("val/pixel_acc", metrics["pixel_acc"])
        self.log("val/pred_background_fraction", metrics["pred_background_fraction"])
        self.log("val/target_background_fraction", metrics["target_background_fraction"])
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
        self.log("test/foreground_mIoU", metrics["foreground_mIoU"])
        self.log("test/pixel_acc", metrics["pixel_acc"])
        self.log("test/pred_background_fraction", metrics["pred_background_fraction"])
        self.log("test/target_background_fraction", metrics["target_background_fraction"])
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
        if self.hparams.optimizer != "AdamW":
            raise ValueError(
                "SemanticSegmentationTask supports only AdamW because it uses "
                "backbone/decoder parameter groups."
            )

        betas = tuple(float(beta) for beta in self.hparams.optimizer_betas)
        if len(betas) != 2:
            raise ValueError(f"optimizer_betas must contain two values, got {betas}.")

        base_lr = float(self.hparams.lr)
        decoder_lr = base_lr * float(self.hparams.decoder_lr_multiplier)
        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "backbone",
                    "params": self.model.backbone.parameters(),
                    "lr": base_lr,
                },
                {
                    "name": "decoder",
                    "params": [
                        *self.model.projections.parameters(),
                        *self.model.fuse.parameters(),
                        *self.model.segmentation_head.parameters(),
                    ],
                    "lr": decoder_lr,
                },
            ],
            betas=betas,
            weight_decay=float(self.hparams.weight_decay),
        )

        if self.hparams.scheduler != "warmup_poly":
            raise ValueError(
                "SemanticSegmentationTask supports only scheduler='warmup_poly', "
                f"got {self.hparams.scheduler!r}."
            )

        scheduler = WarmupPolyLR(
            optimizer,
            max_iters=int(self.hparams.max_iters),
            warmup_iters=int(self.hparams.warmup_iters),
            power=float(self.hparams.poly_power),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
