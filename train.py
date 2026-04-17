import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
import sys
import logging

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
            max_length=cfg.datamodule.max_length
        )
        num_classes = cfg.datamodule.num_classes
    elif cfg.task.name == "ner_task":
        from src.datamodules.ner_datamodule import NERDataModule
        datamodule = NERDataModule(
            model_name=cfg.datamodule.get("model_name", "bert-base-cased"),
            dataset_name=cfg.datamodule.dataset_name,
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
            max_length=cfg.datamodule.max_length
        )
        num_classes = cfg.datamodule.num_classes
    elif cfg.task.name == "detection":
        from src.datamodules.detection_datamodule import VOCDataModule
        datamodule = VOCDataModule(
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers
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
    
    if cfg.task.name == "cv_classification":
        from src.tasks.classification_cv import CVClassificationTask
        task = CVClassificationTask(
            num_classes=num_classes,
            lr=cfg.task.lr,
            weight_decay=cfg.task.weight_decay,
            optimizer=cfg.task.optimizer,
            model_cfg=model_cfg
        )
    elif cfg.task.name == "nlp_classification":
        from src.tasks.classification_nlp import NLPClassificationTask
        task = NLPClassificationTask(
            num_classes=num_classes,
            lr=cfg.task.lr,
            weight_decay=cfg.task.weight_decay,
            optimizer=cfg.task.optimizer,
            model_cfg=model_cfg,
            vocab_size=len(datamodule.tokenizer),
            max_seq_len=cfg.datamodule.max_length,
        )
    elif cfg.task.name == "ner_task":
        from src.tasks.ner_task import NERTask
        from datasets import load_dataset
        
        # Grab class names to properly inform seqeval about BIO tags
        ds = load_dataset(cfg.datamodule.dataset_name, split="train")
        # Most datasets use either 'ner_tags' or 'tags'
        tag_col = "ner_tags" if "ner_tags" in ds.features else "tags" if "tags" in ds.features else None
        
        id2label = None
        if tag_col and hasattr(ds.features[tag_col], "feature"):
            class_names = ds.features[tag_col].feature.names
            id2label = {i: v for i, v in enumerate(class_names)}

        task = NERTask(
            num_classes=num_classes,
            lr=cfg.task.lr,
            weight_decay=cfg.task.weight_decay,
            optimizer=cfg.task.optimizer,
            model_cfg=model_cfg,
            vocab_size=len(datamodule.tokenizer),
            max_seq_len=cfg.datamodule.max_length,
            id2label=id2label
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
    if cfg.task.name != "detection":
        task = torch.compile(task)
        
    # 3. Setup Logger
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 4. Setup Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=wandb_logger,
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
