import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
import sys
import logging
from datetime import datetime
from pathlib import Path
import re

import torch
torch.set_float32_matmul_precision('high')


# Loguru interception for standard logging
class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_loguru():
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])


def build_wandb_tags(cfg: DictConfig) -> list[str]:
    automatic_tags = []

    if HydraConfig.initialized():
        choices = HydraConfig.get().runtime.choices
        for group_name in ("task", "model", "datamodule", "optimizer", "trainer"):
            choice = choices.get(group_name)
            if choice:
                automatic_tags.append(f"{group_name}:{choice}")

    # Fall back to explicit config names for contexts where Hydra runtime
    # choices are unavailable, such as direct unit tests.
    for group_name in ("task", "model", "datamodule", "optimizer"):
        group_cfg = cfg.get(group_name)
        group_value = group_cfg.get("name") if group_cfg is not None else None
        if group_value:
            automatic_tags.append(f"{group_name}:{group_value}")

    extra_tags = cfg.logger.get("extra_tags", [])
    tags = []
    seen = set()
    for tag in [*automatic_tags, *extra_tags]:
        tag = str(tag).strip()
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)

    return tags


def get_accumulate_grad_batches(cfg: DictConfig) -> int:
    accumulate_grad_batches = int(cfg.trainer.get("accumulate_grad_batches", 1))
    if accumulate_grad_batches < 1:
        raise ValueError(
            "trainer.accumulate_grad_batches must be a positive integer, "
            f"got {accumulate_grad_batches}."
        )
    return accumulate_grad_batches


def sanitize_path_component(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = value.strip("._")
    return value or "unnamed"


def build_checkpoint_callback(cfg: DictConfig) -> ModelCheckpoint:
    checkpoint_dir = cfg.logger.get("checkpoint_dir", None)
    if checkpoint_dir is None:
        run_name = sanitize_path_component(str(cfg.logger.name))
        project_name = sanitize_path_component(str(cfg.logger.project))
        checkpoint_root = Path(str(cfg.logger.save_dir)) / project_name / run_name

        if cfg.logger.get("checkpoint_add_timestamp", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_root = checkpoint_root / timestamp

        checkpoint_dir = checkpoint_root / "checkpoints"

    checkpoint_callback_kwargs = {
        "dirpath": str(checkpoint_dir),
        "filename": cfg.logger.get("checkpoint_filename", "epoch={epoch}-step={step}"),
        "save_top_k": int(cfg.logger.get("checkpoint_save_top_k", 1)),
        "save_last": bool(cfg.logger.get("checkpoint_save_last", True)),
        "auto_insert_metric_name": bool(
            cfg.logger.get("checkpoint_auto_insert_metric_name", False)
        ),
    }
    monitor = cfg.logger.get("checkpoint_monitor", None)
    if monitor is not None:
        checkpoint_callback_kwargs["monitor"] = monitor
        checkpoint_callback_kwargs["mode"] = cfg.logger.get("checkpoint_mode", "min")

    return ModelCheckpoint(**checkpoint_callback_kwargs)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    setup_loguru()
    logger.info(f"Starting training with config:\n{OmegaConf.to_yaml(cfg)}")

    L.seed_everything(cfg.seed)

    # 1. Setup DataModule
    if cfg.task.name == "cv_classification":
        from src.datamodules.cv_datamodule import CVDataModule
        datamodule = CVDataModule(
            data_dir=cfg.datamodule.get("data_dir", "./data"),
            dataset_name=cfg.datamodule.dataset_name,
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
            image_size=cfg.model.get("img_size", cfg.datamodule.get("image_size", 224)),
        )
        num_classes = cfg.datamodule.num_classes
    elif cfg.task.name == "nlp_classification":
        from src.datamodules.nlp_datamodule import NLPDataModule
        datamodule = NLPDataModule(
            model_name=cfg.datamodule.get("model_name", "bert-base-uncased"),
            dataset_name=cfg.datamodule.dataset_name,
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
            max_length=cfg.datamodule.max_length,
            dataset_path=cfg.datamodule.get("dataset_path", None),
            dataset_config_name=cfg.datamodule.get("dataset_config_name", None),
            text_column=cfg.datamodule.get("text_column", None),
            text_pair_column=cfg.datamodule.get("text_pair_column", None),
            label_column=cfg.datamodule.get("label_column", "label"),
            train_split=cfg.datamodule.get("train_split", "train"),
            validation_split=cfg.datamodule.get("validation_split", None),
            test_split=cfg.datamodule.get("test_split", "test"),
            validation_size=cfg.datamodule.get("validation_size", 0.1),
            subset_seed=cfg.datamodule.get("subset_seed", cfg.seed),
        )
        num_classes = cfg.datamodule.num_classes
    elif cfg.task.name == "ner_task":
        from src.datamodules.ner_datamodule import NERDataModule
        datamodule = NERDataModule(
            dataset_name=cfg.datamodule.dataset_name,
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
            max_length=cfg.datamodule.max_length,
            label_column=cfg.datamodule.get("label_column", None),
            min_freq=cfg.datamodule.get("min_freq", 1),
            unk_replace_prob=cfg.datamodule.get("unk_replace_prob", 0.01),
        )
        label_names = datamodule.get_label_names()
        num_classes = len(label_names)
    elif cfg.task.name == "semantic_segmentation":
        from src.datamodules.segmentation_datamodule import SegmentationDataModule
        datamodule = SegmentationDataModule(
            data_dir=cfg.datamodule.get("data_dir", "./data"),
            dataset_name=cfg.datamodule.dataset_name,
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
            image_size=cfg.model.get("img_size", cfg.datamodule.get("image_size", 224)),
            train_min_scale=cfg.datamodule.get("train_min_scale", 0.5),
            train_max_scale=cfg.datamodule.get("train_max_scale", 2.0),
            num_classes=cfg.datamodule.num_classes,
            ignore_index=cfg.datamodule.get("ignore_index", 255),
            download=cfg.datamodule.get("download", True),
            cityscapes_mode=cfg.datamodule.get("cityscapes_mode", "fine"),
            val_fraction=cfg.datamodule.get("val_fraction", 0.1),
            test_fraction=cfg.datamodule.get("test_fraction", 0.1),
            max_train_samples=cfg.datamodule.get("max_train_samples", None),
            max_val_samples=cfg.datamodule.get("max_val_samples", None),
            max_test_samples=cfg.datamodule.get("max_test_samples", None),
            subset_seed=cfg.datamodule.get("subset_seed", cfg.seed),
        )
        num_classes = cfg.datamodule.num_classes
    elif cfg.task.name == "detection":
        from src.datamodules.detection_datamodule import VOCDataModule
        datamodule = VOCDataModule(
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers
        )
        num_classes = cfg.datamodule.num_classes
    else:
        raise ValueError(f"Unknown task: {cfg.task.name}")

    # 2. Setup Task & Model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    optimizer_name = cfg.optimizer.optimizer
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay
    
    if cfg.task.name == "cv_classification":
        from src.tasks.classification_cv import CVClassificationTask
        task = CVClassificationTask(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer_name,
            scheduler=cfg.optimizer.get("scheduler", "warmup_cosine"),
            warmup_epochs=cfg.optimizer.get("warmup_epochs", 0),
            model_cfg=model_cfg
        )
    elif cfg.task.name == "nlp_classification":
        from src.tasks.classification_nlp import NLPClassificationTask
        task = NLPClassificationTask(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer_name,
            model_cfg=model_cfg,
            vocab_size=len(datamodule.tokenizer),
            max_seq_len=cfg.datamodule.max_length,
            label_smoothing=cfg.task.get("label_smoothing", 0.0),
        )
    elif cfg.task.name == "ner_task":
        from src.tasks.ner_task import NERTask
        id2label = {i: label for i, label in enumerate(label_names)}

        task = NERTask(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer_name,
            model_cfg=model_cfg,
            vocab_size=datamodule.vocab_size,
            max_seq_len=cfg.datamodule.max_length,
            id2label=id2label
        )
    elif cfg.task.name == "semantic_segmentation":
        from src.tasks.segmentation import SemanticSegmentationTask
        task = SemanticSegmentationTask(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer_name,
            optimizer_betas=cfg.optimizer.get("betas", [0.9, 0.999]),
            decoder_lr_multiplier=cfg.optimizer.get("decoder_lr_multiplier", 10.0),
            scheduler=cfg.optimizer.get("scheduler", "warmup_poly"),
            max_iters=cfg.optimizer.get("max_iters", cfg.trainer.get("max_steps", 80000)),
            warmup_iters=cfg.optimizer.get("warmup_iters", 1500),
            poly_power=cfg.optimizer.get("poly_power", 0.9),
            model_cfg=model_cfg,
            ignore_index=cfg.datamodule.get("ignore_index", 255),
            log_segmentation_images=cfg.logger.get("log_segmentation_images", True),
            num_log_segmentation_images=cfg.logger.get("num_log_segmentation_images", 4),
            log_segmentation_images_every_n_epochs=cfg.logger.get(
                "log_segmentation_images_every_n_epochs",
                1,
            ),
            focal_gamma=cfg.task.get("focal_gamma", 2.0),
            focal_alpha=cfg.task.get("focal_alpha", None),
            class_weights=cfg.task.get("class_weights", None),
            focal_loss_weight=cfg.task.get("focal_loss_weight", 1.0),
            dice_loss_weight=cfg.task.get("dice_loss_weight", 1.0),
            dice_smooth=cfg.task.get("dice_smooth", 1e-5),
            dice_present_classes_only=cfg.task.get("dice_present_classes_only", True),
        )
    elif cfg.task.name == "generation_nlp":
        from src.tasks.generation_nlp import NLPGenerationTask
        task = NLPGenerationTask(
            vocab_size=len(datamodule.tokenizer),
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer_name,
            model_cfg=model_cfg,
            pad_token_id=datamodule.tokenizer.pad_token_id,
            bos_token_id=datamodule.tokenizer.bos_token_id or datamodule.tokenizer.pad_token_id,
            eos_token_id=datamodule.tokenizer.eos_token_id
        )

    elif cfg.task.name == "detection":
        from src.tasks.detection import ObjectDetectionTask
        task = ObjectDetectionTask(
            num_classes=num_classes,
            lr=cfg.task.lr,
            weight_decay=cfg.task.weight_decay,
            optimizer=cfg.task.optimizer,
            model_cfg=model_cfg
        )
    task = torch.compile(task)

    # 3. Setup Logger
    wandb_tags = build_wandb_tags(cfg)
    logger.info(f"Using W&B tags: {wandb_tags}")
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.get("entity", None),
        name=cfg.logger.name,
        tags=wandb_tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=cfg.logger.save_dir,
    )

    # 4. Setup Trainer
    accumulate_grad_batches = get_accumulate_grad_batches(cfg)
    checkpoint_callback = build_checkpoint_callback(cfg)
    logger.info(f"Saving checkpoints to: {checkpoint_callback.dirpath}")
    logger.info(
        "Using gradient accumulation: "
        f"{accumulate_grad_batches} step(s) with datamodule.batch_size={cfg.datamodule.batch_size}"
    )
    trainer_kwargs = {
        "max_epochs": cfg.trainer.max_epochs,
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "precision": cfg.trainer.precision,
        "accumulate_grad_batches": accumulate_grad_batches,
        "gradient_clip_val": cfg.trainer.get("gradient_clip_val", None),
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "logger": wandb_logger,
        "callbacks": [checkpoint_callback],
    }
    if cfg.trainer.get("max_steps", None) is not None:
        trainer_kwargs["max_steps"] = int(cfg.trainer.max_steps)

    trainer = L.Trainer(
        **trainer_kwargs,
    )

    # 5. Train
    logger.info("Starting Trainer.fit()...")
    trainer.fit(model=task, datamodule=datamodule)

    datamodule.setup("test")
    if getattr(datamodule, "has_test_labels", True) and getattr(datamodule, "test_dataset", object()) is not None:
        logger.info("Starting Trainer.test()...")
        trainer.test(model=task, datamodule=datamodule)
    else:
        logger.warning("Skipping Trainer.test() because the configured test split has no public labels.")


if __name__ == "__main__":
    main()
