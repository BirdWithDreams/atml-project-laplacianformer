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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
