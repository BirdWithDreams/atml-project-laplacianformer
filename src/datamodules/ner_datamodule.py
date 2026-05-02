import lightning as L
import torch
from collections import Counter
from torch.utils.data import DataLoader
from datasets import ClassLabel, Sequence
from datasets import load_dataset


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
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(
            self, dataset_name: str = "eriktks/conll2003", batch_size: int = 32,
            num_workers: int = 4, max_length: int = 128, label_column: str | None = None,
            min_freq: int = 1, unk_replace_prob: float = 0.01
            ):
        super().__init__()
        if not 0.0 <= unk_replace_prob <= 1.0:
            raise ValueError("unk_replace_prob must be in [0.0, 1.0]")
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.label_column = label_column
        self.min_freq = min_freq
        self.unk_replace_prob = unk_replace_prob
        self.tokens_key: str | None = None
        self.labels_key: str | None = None
        self.label_names: list[str] | None = None
        self.id2label: dict[int, str] | None = None
        self.label2id: dict[str, int] | None = None
        self.itos: list[str] | None = None
        self.stoi: dict[str, int] | None = None
        self.pad_id = 0
        self.unk_id = 1

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

    @property
    def vocab_size(self) -> int:
        if self.itos is None:
            dataset = self._load_dataset()
            self._resolve_schema(dataset)
            self._build_vocab(dataset["train"])
        return len(self.itos)

    def _build_vocab(self, train_dataset):
        counter = Counter()
        for example in train_dataset:
            counter.update(str(token) for token in example[self.tokens_key])

        tokens = [
            token
            for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if count >= self.min_freq
        ]
        self.itos = [self.PAD_TOKEN, self.UNK_TOKEN, *tokens]
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi[self.PAD_TOKEN]
        self.unk_id = self.stoi[self.UNK_TOKEN]

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

    def _token_to_id(self, token: str) -> int:
        return self.stoi.get(str(token), self.unk_id)

    @staticmethod
    def _collate_batch(batch):
        return {
            "input_ids": torch.tensor([example["input_ids"] for example in batch], dtype=torch.long),
            "attention_mask": torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long),
            "labels": torch.tensor([example["labels"] for example in batch], dtype=torch.long),
        }

    def _collate_train_batch(self, batch):
        batch = self._collate_batch(batch)
        if self.unk_replace_prob <= 0.0:
            return batch

        input_ids = batch["input_ids"].clone()
        ordinary_tokens = (batch["attention_mask"] == 1) & (input_ids != self.unk_id)
        replace_mask = (
            torch.rand(input_ids.shape, device=input_ids.device) < self.unk_replace_prob
        ) & ordinary_tokens
        input_ids[replace_mask] = self.unk_id
        batch["input_ids"] = input_ids
        return batch

    def prepare_data(self):
        self._load_dataset()

    def setup(self, stage=None):
        dataset = self._load_dataset()
        self._resolve_schema(dataset)
        self._build_vocab(dataset["train"])

        def encode_examples(examples):
            input_ids = []
            attention_masks = []
            labels = []

            for tokens, token_labels in zip(examples[self.tokens_key], examples[self.labels_key]):
                tokens = list(tokens)[:self.max_length]
                token_labels = list(token_labels)[:self.max_length]
                sequence_length = len(tokens)
                padding_length = self.max_length - sequence_length

                input_ids.append(
                    [self._token_to_id(token) for token in tokens]
                    + [self.pad_id] * padding_length
                )
                attention_masks.append([1] * sequence_length + [0] * padding_length)
                labels.append(
                    [self._label_to_id(label) for label in token_labels]
                    + [-100] * padding_length
                )

            return {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
            }

        encoded_datasets = dataset.map(
            encode_examples,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        
        if stage == "fit" or stage is None:
            self.train_dataset = encoded_datasets["train"]
            self.val_dataset = encoded_datasets["validation"]
        
        if stage == "test":
            self.test_dataset = encoded_datasets["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_train_batch,
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
