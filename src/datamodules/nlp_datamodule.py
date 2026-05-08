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

        # Using GLUE benchmark's SST-2
        load_dataset("glue", self.dataset_name)

        self._load_raw_dataset()

    def _load_raw_dataset(self):
        if self.dataset_config_name is not None:
            return load_dataset(self.dataset_path, self.dataset_config_name)
        return load_dataset(self.dataset_path)

    @staticmethod
    def _infer_text_column(column_names: list[str]) -> str:
        for candidate in ("text", "sentence", "content", "review"):
            if candidate in column_names:
                return candidate
        raise ValueError(
            "Could not infer text column. Set datamodule.text_column explicitly. "
            f"Available columns: {column_names}"
        )

    def _normalize_label_column(self, dataset: DatasetDict) -> DatasetDict:
        if self.label_column == "label":
            return dataset

        renamed_splits = {}
        for split_name, split_dataset in dataset.items():
            if self.label_column in split_dataset.column_names:
                renamed_splits[split_name] = split_dataset.rename_column(self.label_column, "label")
            else:
                renamed_splits[split_name] = split_dataset
        return DatasetDict(renamed_splits)

    def _make_validation_split(self, train_dataset):
        split_kwargs = {
            "test_size": self.validation_size,
            "seed": self.subset_seed,
        }
        if isinstance(train_dataset.features.get("label"), ClassLabel):
            split_kwargs["stratify_by_column"] = "label"

        try:
            return train_dataset.train_test_split(**split_kwargs)
        except (ImportError, RuntimeError, TypeError, ValueError):
            split_kwargs.pop("stratify_by_column", None)
            return train_dataset.train_test_split(**split_kwargs)

    def _select_raw_splits(self, dataset: DatasetDict, stage: str | None) -> DatasetDict:
        selected = {}

        if stage in ("fit", None):
            if self.train_split not in dataset:
                raise ValueError(f"Missing train split '{self.train_split}' in {list(dataset.keys())}")

            train_dataset = dataset[self.train_split]
            if self.validation_split is not None and self.validation_split in dataset:
                selected["train"] = train_dataset
                selected["validation"] = dataset[self.validation_split]
            else:
                if not 0 < self.validation_size < 1:
                    raise ValueError(
                        "A dataset without a validation split requires "
                        "0 < datamodule.validation_size < 1."
                    )
                split = self._make_validation_split(train_dataset)
                selected["train"] = split["train"]
                selected["validation"] = split["test"]

        if stage in ("test", None) and self.test_split is not None and self.test_split in dataset:
            selected["test"] = dataset[self.test_split]

        return DatasetDict(selected)


    @staticmethod
    def _split_has_labels(split) -> bool:
        if "label" not in split.column_names or len(split) == 0:
            return False

        sample_size = min(128, len(split))
        labels = split[:sample_size]["label"]
        return any(int(label) >= 0 for label in labels)

    def setup(self, stage=None):
        dataset = self._normalize_label_column(self._load_raw_dataset())
        selected_splits = self._select_raw_splits(dataset, stage)

        if len(selected_splits) == 0:
            return

        reference_split = next(iter(selected_splits.values()))
        text_column = self.text_column or self._infer_text_column(reference_split.column_names)
        if text_column not in reference_split.column_names:
            raise ValueError(
                f"Configured text column '{text_column}' is missing from "
                f"{reference_split.column_names}"
            )
        if self.text_pair_column is not None and self.text_pair_column not in reference_split.column_names:
            raise ValueError(
                f"Configured text pair column '{self.text_pair_column}' is missing from "
                f"{reference_split.column_names}"
            )

        def tokenize_function(examples):
            text = examples[text_column]
            text_pair = examples[self.text_pair_column] if self.text_pair_column is not None else None
            return self.tokenizer(
                text,
                text_pair,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        tokenized_splits = {}
        for split_name, split_dataset in selected_splits.items():
            remove_columns = [col for col in split_dataset.column_names if col != "label"]
            tokenized_splits[split_name] = split_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=remove_columns,
            )
        tokenized_datasets = DatasetDict(tokenized_splits)

        if stage == "fit" or stage is None:
            self.train_dataset = tokenized_datasets["train"]
            self.val_dataset = tokenized_datasets["validation"]
        
        if stage == "test" or stage is None:
            if "test" in tokenized_datasets:
                self.has_test_labels = self._split_has_labels(tokenized_datasets["test"])
                self.test_dataset = tokenized_datasets["test"] if self.has_test_labels else None
            else:
                self.has_test_labels = False
                self.test_dataset = None

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
