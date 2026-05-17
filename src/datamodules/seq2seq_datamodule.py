import lightning as L
import torch
from torch.utils.data import DataLoader
import pyarrow as pa

# Compatibility shim for datasets versions expecting PyExtensionType.
if not hasattr(pa, "PyExtensionType"):
    pa.PyExtensionType = pa.ExtensionType

from datasets import load_dataset
from transformers import AutoTokenizer

class Seq2SeqDataModule(L.LightningDataModule):
    def __init__(
            self, 
            model_name: str = "t5-small", 
            dataset_name: str = "cnn_dailymail",
            dataset_config: str = "3.0.0",
            source_column: str = "article",
            target_column: str = "highlights",
            batch_size: int = 16, 
            num_workers: int = 4, 
            max_source_length: int = 512,
            max_target_length: int = 128
        ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.source_column = source_column
        self.target_column = target_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load tokenizer. We use a Seq2Seq tokenizer like T5 or BART as a base.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _collate_batch(batch):
        output = {
            "input_ids": torch.tensor([example["input_ids"] for example in batch], dtype=torch.long),
            "attention_mask": torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long),
            "labels": torch.tensor([example["labels"] for example in batch], dtype=torch.long),
        }
        return output

    def prepare_data(self):
        # Download the dataset to disk
        if self.dataset_config:
            load_dataset(self.dataset_name, self.dataset_config)
        else:
            load_dataset(self.dataset_name)

    def setup(self, stage=None):
        if self.dataset_config:
            dataset = load_dataset(self.dataset_name, self.dataset_config)
        else:
            dataset = load_dataset(self.dataset_name)

        def tokenize_function(examples):
            # Handle standard flat columns vs nested 'translation' dicts (like WMT)
            if "translation" in examples:
                sources = [ex[self.source_column] for ex in examples["translation"]]
                targets = [ex[self.target_column] for ex in examples["translation"]]
            else:
                sources = examples[self.source_column]
                targets = examples[self.target_column]

            # Tokenize the source texts (Encoder Inputs)
            model_inputs = self.tokenizer(
                sources, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_source_length
            )

            # Tokenize the target texts (Decoder Inputs / Labels)
            # Using text_target ensures the tokenizer uses the target language vocabulary if applicable
            labels = self.tokenizer(
                text_target=targets, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_target_length
            )

            # Add the tokenized targets to the model inputs under the key "labels"
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Apply tokenization across the dataset
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = tokenized_datasets["train"]
            # Some datasets use "validation", others use "val"
            val_split = "validation" if "validation" in tokenized_datasets else "val"
            self.val_dataset = tokenized_datasets[val_split]
            
        if stage == "test":
            self.test_dataset = tokenized_datasets.get("test", None)

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
        if self.test_dataset is None:
            raise ValueError("No 'test' split found in this dataset.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_batch,
        )