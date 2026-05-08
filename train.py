import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
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


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    setup_loguru()
    logger.info(f"Starting training with config:\n{OmegaConf.to_yaml(cfg)}")

    L.seed_everything(cfg.seed)

    # 1. Setup DataModule
    if cfg.task.name == "cv_classification":
        from src.datamodules.cv_datamodule import CVDataModule
        datamodule = CVDataModule(
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
            num_classes=cfg.datamodule.num_classes,
            ignore_index=cfg.datamodule.get("ignore_index", 255),
            download=cfg.datamodule.get("download", True),
            coco_year=cfg.datamodule.get("coco_year", "2017"),
            max_train_samples=cfg.datamodule.get("max_train_samples", None),
            max_val_samples=cfg.datamodule.get("max_val_samples", None),
            max_test_samples=cfg.datamodule.get("max_test_samples", None),
            subset_seed=cfg.datamodule.get("subset_seed", cfg.seed),
        )
        num_classes = cfg.datamodule.num_classes
    elif cfg.task.name == "generation_nlp":
        from src.tasks.generation_nlp import NLPGenerationTask
        from src.datamodules.seq2seq_datamodule import Seq2SeqDataModule
        from src.models.text_seq2seq import TextSeq2SeqBackbone        

        # 1. INITIALIZE DATAMODULE FIRST
        # Convert the Hydra config to a standard dictionary and remove 'name'
        dm_kwargs = dict(cfg.datamodule)
        dm_kwargs.pop("name", None) 
        
        # Unpack the cleaned dictionary
        datamodule = Seq2SeqDataModule(**dm_kwargs)

        # 2. INITIALIZE BACKBONE SECOND
        # Now we can safely call len(datamodule.tokenizer)
        backbone = TextSeq2SeqBackbone(
            src_vocab_size=len(datamodule.tokenizer),
            tgt_vocab_size=len(datamodule.tokenizer),
            dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            depth=cfg.model.depth,
            attn_type=cfg.model.attn_type,
            lambda_scale=cfg.model.get("lambda_scale", 4.0),
            ns_iters=cfg.model.get("ns_iters", 5),
            dropout=cfg.model.get("dropout", 0.1)
        )

        # 3. INITIALIZE LIGHTNING TASK THIRD
        # Wrap the backbone in your LightningModule (adjust 'GenerationNLPTask' if you named it differently)
        task = NLPGenerationTask(
            model=backbone,
            cfg=cfg,
            tokenizer=datamodule.tokenizer # Passing this is helpful for calculating ROUGE scores later
        )
        
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
            model_cfg=model_cfg,
            ignore_index=cfg.datamodule.get("ignore_index", 255),
            log_segmentation_images=cfg.logger.get("log_segmentation_images", True),
            num_log_segmentation_images=cfg.logger.get("num_log_segmentation_images", 4),
            log_segmentation_images_every_n_epochs=cfg.logger.get(
                "log_segmentation_images_every_n_epochs",
                1,
            ),
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

    if cfg.trainer.get("compile", True):
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
    logger.info(
        "Using gradient accumulation: "
        f"{accumulate_grad_batches} step(s) with datamodule.batch_size={cfg.datamodule.batch_size}"
    )
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=accumulate_grad_batches,
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
