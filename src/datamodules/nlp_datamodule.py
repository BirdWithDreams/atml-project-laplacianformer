import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class NLPDataModule(L.LightningDataModule):
    def __init__(
            self, model_name: str = "bert-base-uncased", dataset_name: str = "sst2",
            batch_size: int = 32, num_workers: int = 4, max_length: int = 128
            ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.has_test_labels = True

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
<<<<<<< HEAD
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
=======
        # Using GLUE benchmark's SST-2
        load_dataset("glue", self.dataset_name)

    @staticmethod
    def _split_has_labels(split) -> bool:
        if "label" not in split.column_names or len(split) == 0:
            return False

        sample_size = min(128, len(split))
        labels = split[:sample_size]["label"]
        return any(int(label) >= 0 for label in labels)

    def setup(self, stage=None):
        dataset = load_dataset("glue", self.dataset_name)

        def tokenize_function(examples):
            return self.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=self.max_length)

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset["train"].column_names if col != "label"],
        )

        if stage == "fit" or stage is None:
            self.train_dataset = tokenized_datasets["train"]
            self.val_dataset = tokenized_datasets["validation"]
        
        if stage == "test":
            if "test" in tokenized_datasets:
                self.has_test_labels = self._split_has_labels(tokenized_datasets["test"])
                self.test_dataset = tokenized_datasets["test"] if self.has_test_labels else None
            else:
                self.has_test_labels = False
                self.test_dataset = None
>>>>>>> 731ec0b (Add hierarchical Pyramid Vision Transformer (PVT) backbone with Laplacian attention, refine RoPE utilities, update classification tasks to support configurable backbones, and improve datamodule flexibility with dynamic dataset parameters.)

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
