import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import ClassLabel, Sequence
from datasets import load_dataset
from transformers import AutoTokenizer


TNER_ONTONOTES5_LABEL2ID = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12,
    "B-PERCENT": 13,
    "I-PERCENT": 14,
    "B-ORDINAL": 15,
    "B-MONEY": 16,
    "I-MONEY": 17,
    "B-WORK_OF_ART": 18,
    "I-WORK_OF_ART": 19,
    "B-FAC": 20,
    "B-TIME": 21,
    "I-CARDINAL": 22,
    "B-LOC": 23,
    "B-QUANTITY": 24,
    "I-QUANTITY": 25,
    "I-NORP": 26,
    "I-LOC": 27,
    "B-PRODUCT": 28,
    "I-TIME": 29,
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36,
}
TNER_ONTONOTES5_LABELS = [
    label for label, _ in sorted(TNER_ONTONOTES5_LABEL2ID.items(), key=lambda item: item[1])
]

KNOWN_LABEL_NAMES = {
    "tner/ontonotes5": TNER_ONTONOTES5_LABELS,
}


class NERDataModule(L.LightningDataModule):
    def __init__(
            self, dataset_name: str = "eriktks/conll2003", model_name: str = "bert-base-cased",
            batch_size: int = 32, num_workers: int = 4, max_length: int = 128,
            label_column: str | None = None
            ):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.label_column = label_column
        self.tokens_key: str | None = None
        self.labels_key: str | None = None
        self.label_names: list[str] | None = None
        self.id2label: dict[int, str] | None = None
        self.label2id: dict[str, int] | None = None
        # add_prefix_space is required for some models for token classification with fast tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    def _load_dataset(self):
        return load_dataset(self.dataset_name)

    def _resolve_schema(self, dataset):
        train_columns = set(dataset["train"].column_names)
        if "tokens" in train_columns:
            tokens_key = "tokens"
        elif "words" in train_columns:
            tokens_key = "words"
        else:
            raise ValueError(
                f"Could not find a token column in {self.dataset_name}. "
                f"Expected one of ['tokens', 'words'], got {sorted(train_columns)}."
            )

        if self.label_column is not None:
            if self.label_column not in train_columns:
                raise ValueError(
                    f"Configured label_column={self.label_column!r} is not present "
                    f"in {self.dataset_name}. Available columns: {sorted(train_columns)}."
                )
            labels_key = self.label_column
        elif "ner_tags" in train_columns:
            labels_key = "ner_tags"
        elif "tags" in train_columns:
            labels_key = "tags"
        else:
            raise ValueError(
                f"Could not find an NER label column in {self.dataset_name}. "
                f"Expected one of ['ner_tags', 'tags'], got {sorted(train_columns)}."
            )

        label_names = self._extract_label_names(dataset, labels_key)
        self.tokens_key = tokens_key
        self.labels_key = labels_key
        self.label_names = label_names
        self.id2label = {i: label for i, label in enumerate(label_names)}
        self.label2id = {label: i for i, label in self.id2label.items()}

    def _extract_label_names(self, dataset, labels_key: str) -> list[str]:
        feature = dataset["train"].features[labels_key]
        if isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
            return list(feature.feature.names)
        if isinstance(feature, ClassLabel):
            return list(feature.names)

        dataset_key = self.dataset_name.lower()
        if dataset_key in KNOWN_LABEL_NAMES:
            return KNOWN_LABEL_NAMES[dataset_key]

        raise ValueError(
            f"Dataset {self.dataset_name} uses integer labels in column {labels_key!r} "
            "without ClassLabel metadata. Add its label names to KNOWN_LABEL_NAMES."
        )

    def get_label_names(self) -> list[str]:
        if self.label_names is None:
            self._resolve_schema(self._load_dataset())
        return list(self.label_names)

    def _label_to_id(self, label_value) -> int:
        if isinstance(label_value, str):
            try:
                return self.label2id[label_value]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown NER label {label_value!r} for dataset {self.dataset_name}."
                ) from exc

        label_id = int(label_value)
        if label_id < 0 or label_id >= len(self.label_names):
            raise ValueError(
                f"NER label id {label_id} is outside the configured label map "
                f"for {self.dataset_name} with {len(self.label_names)} labels."
            )
        return label_id

    @staticmethod
    def _collate_batch(batch):
        return {
            "input_ids": torch.tensor([example["input_ids"] for example in batch], dtype=torch.long),
            "attention_mask": torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long),
            "labels": torch.tensor([example["labels"] for example in batch], dtype=torch.long),
        }

    def prepare_data(self):
        self._load_dataset()

    def setup(self, stage=None):
        dataset = self._load_dataset()
        self._resolve_schema(dataset)
        
        # Tokenize and align labels
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples[self.tokens_key],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(examples[self.labels_key]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(self._label_to_id(label[word_idx]))
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        
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
