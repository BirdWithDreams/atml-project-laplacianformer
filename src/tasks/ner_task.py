import lightning as L
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import evaluate
from torchmetrics import Metric
from src.models.text_ner import TextBackboneNER

class SeqevalMetric(Metric):
    # To use seqeval inside torchmetrics natively without issues.
    # It requires storing lists of predictions.
    def __init__(self, id2label, **kwargs):
        super().__init__(**kwargs)
        self.id2label = id2label
        self.metric = evaluate.load("seqeval")
        
        # State lists
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("references", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for k in range(preds.shape[0]):
            pred = preds[k].cpu().tolist()
            ref = target[k].cpu().tolist()

            true_predictions = [
                self.id2label[p] for (p, l) in zip(pred, ref) if l != -100
            ]
            true_labels = [
                self.id2label[l] for (p, l) in zip(pred, ref) if l != -100
            ]
            
            # Avoid empty sequence issues.
            if len(true_predictions) == 0:
                continue

            self.predictions.append(true_predictions)
            self.references.append(true_labels)

    def compute(self):
        if len(self.predictions) == 0:
            return {}

        results = self.metric.compute(predictions=self.predictions, references=self.references)
        
        flat_predictions = [p for preds in self.predictions for p in preds]
        flat_references = [r for refs in self.references for r in refs]
        
        # Token-level metrics (Secondary metric)
        token_f1 = f1_score(flat_references, flat_predictions, average="macro", zero_division=0)

        final_metrics = {
            "Entity_Prec": torch.tensor(results["overall_precision"]),
            "Entity_Rec": torch.tensor(results["overall_recall"]),
            "Entity_F1": torch.tensor(results["overall_f1"]),
            "Token_F1": torch.tensor(token_f1)
        }
        
        # Per-entity type secondary metrics
        for key, value in results.items():
            if isinstance(value, dict) and 'f1' in value:
                final_metrics[f"{key}_F1"] = torch.tensor(value['f1'])
                
        return final_metrics

class NERTask(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            lr: float = 2e-5,
            weight_decay: float = 0.01,
            optimizer: str = "AdamW",
            model_cfg: dict = None,
            vocab_size: int = 30522,
            max_seq_len: int = 128,
            id2label: dict = None
            ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Backbone via model_cfg
        if model_cfg is None:
            model_cfg = {"attn_type": "vanilla", "dim": 384, "depth": 6, "num_heads": 6}

        self.backbone = TextBackboneNER(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dim=model_cfg.get("dim", 384),
            depth=model_cfg.get("depth", 6),
            num_heads=model_cfg.get("num_heads", 6),
            attn_type=model_cfg.get("attn_type", "vanilla")
        )

        dim = model_cfg.get("dim", 384)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        if id2label is None:
            # Fallback if no labels dict provided (e.g. Conll2003)
            # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
            self.id2label = {
                0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 
                5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
            }
        else:
            self.id2label = id2label
            
        # Metrics
        self.val_metric = SeqevalMetric(id2label=self.id2label)
        self.test_metric = SeqevalMetric(id2label=self.id2label)

    def forward(self, input_ids, attention_mask=None):
        features = self.backbone(input_ids, attention_mask)
        return self.head(features) # (B, seq_len, num_classes)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        # Flatten for CrossEntropyLoss
        loss = self.criterion(logits.view(-1, self.hparams.num_classes), labels.view(-1))

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits.view(-1, self.hparams.num_classes), labels.view(-1))
        
        preds = torch.argmax(logits, dim=-1)
        self.val_metric(preds, labels)

        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.val_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k == "Entity_F1"))
        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits.view(-1, self.hparams.num_classes), labels.view(-1))
        
        preds = torch.argmax(logits, dim=-1)
        self.test_metric(preds, labels)
        
        self.log("test/loss", loss)

    def on_test_epoch_end(self):
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
        self.test_metric.reset()

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
