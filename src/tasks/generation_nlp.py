import lightning as L
import torch
import torch.nn as nn
import evaluate
from src.models.text_seq2seq import TextSeq2SeqBackbone

class NLPGenerationTask(L.LightningModule):
    def __init__(
            self,
            vocab_size: int = 32128, # Default for T5
            lr: float = 5e-5,
            weight_decay: float = 0.01,
            optimizer: str = "AdamW",
            model_cfg: dict = None,
            pad_token_id: int = 0,
            bos_token_id: int = 1, # Beginning of Sequence
            eos_token_id: int = 2, # End of Sequence
            tokenizer = None
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer
        
        # 1. Initialize Backbone
        if model_cfg is None:
            model_cfg = {"attn_type": "vanilla", "dim": 512, "depth": 6, "num_heads": 8}
            
        self.backbone = TextSeq2SeqBackbone(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            max_seq_len=model_cfg.get("max_seq_len", 512),
            dim=model_cfg.get("dim", 512),
            depth=model_cfg.get("depth", 6),
            num_heads=model_cfg.get("num_heads", 8),
            attn_type=model_cfg.get("attn_type", "vanilla")
        )
        
        # 2. Loss Function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_token_id)
        
        # 3. Metrics
        # Using ROUGE for summarization. Use "sacrebleu" if doing translation.
        self.rouge_metric = evaluate.load("rouge")
        self.val_preds = []
        self.val_targets = []

    def forward(self, input_ids, decoder_input_ids, src_mask=None):
        return self.backbone(input_ids, decoder_input_ids, src_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        src_mask = batch["attention_mask"]
        labels = batch["labels"] # Target sequence
        
        # Teacher Forcing: Shift labels right for decoder input
        # e.g., labels = [SOS, token1, token2, EOS]
        # decoder_input = [SOS, token1, token2]
        # target = [token1, token2, EOS]
        decoder_input_ids = labels[:, :-1]
        target_labels = labels[:, 1:]

        logits = self(input_ids, decoder_input_ids, src_mask)
        
        # Flatten for CrossEntropyLoss: (B * seq_len, vocab_size) vs (B * seq_len)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        src_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # 1. Calculate Standard Loss (Teacher Forcing)
        decoder_input_ids = labels[:, :-1]
        target_labels = labels[:, 1:]
        logits = self(input_ids, decoder_input_ids, src_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
        self.log("val/loss", loss, prog_bar=True)

        # 2. Generate text for metrics (limit to first few batches to save time)
        if batch_idx < 2 and self.tokenizer is not None:
            # Generate a short sequence to track quality
            generated_ids = self.generate(input_ids, src_mask, max_new_tokens=30)
            
            # Decode predictions and labels
            decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Replace -100 (if used by HF datasets) with pad token before decoding
            safe_labels = torch.where(labels != -100, labels, self.hparams.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(safe_labels, skip_special_tokens=True)
            
            self.val_preds.extend(decoded_preds)
            self.val_targets.extend(decoded_labels)
            
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            results = self.rouge_metric.compute(predictions=self.val_preds, references=self.val_targets)
            self.log("val/rougeL", results["rougeL"], prog_bar=True)
            
            # Print a sample to the console to monitor learning
            print(f"\n--- Epoch {self.current_epoch} Generation Sample ---")
            print(f"Target: {self.val_targets[0]}")
            print(f"Pred:   {self.val_preds[0]}")
            print("---------------------------------------\n")
            
            self.val_preds.clear()
            self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        src_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        decoder_input_ids = labels[:, :-1]
        target_labels = labels[:, 1:]

        logits = self(input_ids, decoder_input_ids, src_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
        
        self.log("test/loss", loss)

    @torch.no_grad()
    def generate(self, input_ids, src_mask=None, max_new_tokens=50):
        """
        Autoregressive greedy decoding for inference.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Initialize decoder sequence with the BOS token
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.hparams.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Track which sequences in the batch have finished (hit EOS)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self(input_ids, decoder_input_ids, src_mask)
            
            # Get the logits for the last time step
            next_token_logits = logits[:, -1, :]
            
            # Greedy search: take the argmax
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # If a sequence is already finished, force the next token to be a pad token
            next_tokens = next_tokens * unfinished_sequences + self.hparams.pad_token_id * (~unfinished_sequences)
            
            # Append the new token to the sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update unfinished sequences (if they just hit EOS, mark them as finished)
            unfinished_sequences = unfinished_sequences & (next_tokens != self.hparams.eos_token_id)
            
            # Stop early if all sequences in the batch have output an EOS token
            if not unfinished_sequences.any():
                break
                
        return decoder_input_ids

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