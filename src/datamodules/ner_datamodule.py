import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class NERDataModule(L.LightningDataModule):
    def __init__(
            self, dataset_name: str = "eriktks/conll2003", model_name: str = "bert-base-cased",
            batch_size: int = 32, num_workers: int = 4, max_length: int = 128
            ):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        # add_prefix_space is required for some models for token classification with fast tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    def _load_dataset(self):
        return load_dataset(self.dataset_name)

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
        
        # Tokenize and align labels
        def tokenize_and_align_labels(examples):
            # Most NER datasets use 'tokens' or 'words' and 'ner_tags'
            tokens_key = "tokens" if "tokens" in examples else "words"
            labels_key = "ner_tags" if "ner_tags" in examples else "tags"
            
            tokenized_inputs = self.tokenizer(
                examples[tokens_key],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(examples[labels_key]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
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
