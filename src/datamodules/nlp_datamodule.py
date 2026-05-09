import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import ClassLabel, DatasetDict, load_dataset
from transformers import AutoTokenizer

class NLPDataModule(L.LightningDataModule):
    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            dataset_name: str = "sst2",
            batch_size: int = 32,
            num_workers: int = 4,
            max_length: int = 128,
            dataset_path: str | None = None,
            dataset_config_name: str | None = None,
            text_column: str | None = None,
            text_pair_column: str | None = None,
            label_column: str = "label",
            train_split: str = "train",
            validation_split: str | None = None,
            test_split: str | None = "test",
            validation_size: float = 0.1,
            subset_seed: int = 42,
            ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.dataset_path = dataset_path
        self.dataset_config_name = dataset_config_name
        self.text_column = text_column
        self.text_pair_column = text_pair_column
        self.label_column = label_column
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split
        self.validation_size = validation_size
        self.subset_seed = subset_seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.has_test_labels = True

        if self.dataset_path is None:
            if self.dataset_name == "sst2":
                self.dataset_path = "glue"
                self.dataset_config_name = self.dataset_config_name or self.dataset_name
            else:
                self.dataset_path = self.dataset_name

        if self.validation_split is None and self.dataset_path == "glue":
            self.validation_split = "validation"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def _collate_batch(batch):
        output = {
            "input_ids": torch.tensor([example["input_ids"] for example in batch], dtype=torch.long),
            "attention_mask": torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long),
        }
        if "label" in batch[0]:
            output["label"] = torch.tensor([example["label"] for example in batch], dtype=torch.long)
        return output

    def prepare_data(self):
        if self.dataset_name == "ag_news":
            load_dataset("ag_news")
        else:
            load_dataset("glue", self.dataset_name)

    def setup(self, stage=None):
        if self.dataset_name == "ag_news":
            dataset = load_dataset("ag_news")
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            if stage == "fit" or stage is None:
                self.train_dataset = tokenized_datasets["train"]
                self.val_dataset = tokenized_datasets["test"]
            if stage == "test":
                self.test_dataset = tokenized_datasets["test"]
        else:
            dataset = load_dataset("glue", self.dataset_name)
            def tokenize_function(examples):
                return self.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=self.max_length)
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            if stage == "fit" or stage is None:
                self.train_dataset = tokenized_datasets["train"]
                self.val_dataset = tokenized_datasets["validation"]
            if stage == "test":
                self.test_dataset = tokenized_datasets["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_batch,
        )
