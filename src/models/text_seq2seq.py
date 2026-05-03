import torch
import torch.nn as nn
from src.models.vanilla_attn import Transformer, create_target_mask

class TextSeq2SeqBackbone(nn.Module):
    """
    Encoder-Decoder backbone for generative NLP tasks (Translation, Summarization).
    Wraps the existing Transformer architecture.
    """
    def __init__(
            self, 
            src_vocab_size: int = 32128, # Default for T5
            tgt_vocab_size: int = 32128,
            max_seq_len: int = 512,
            dim: int = 384, 
            depth: int = 6, 
            num_heads: int = 6, 
            attn_type: str = "vanilla", 
            # Note: attn_type handling would require modifying vanilla_attn.py 
            # to accept Laplacian blocks in the Encoder/Decoder layers.
        ):
        super().__init__()
        
        # We map depth to num_layers, and calculate a standard d_ff (usually 4x dim)
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=dim,
            num_heads=num_heads,
            num_layers=depth,
            d_ff=dim * 4, 
            max_len=max_seq_len,
            dropout=0.1
        )

    def forward(self, input_ids, decoder_input_ids, src_mask=None):
        """
        Args:
            input_ids: (Batch, Src_Seq_Len)
            decoder_input_ids: (Batch, Tgt_Seq_Len)
            src_mask: (Batch, Src_Seq_Len)
        """
        device = decoder_input_ids.device
        tgt_seq_len = decoder_input_ids.size(1)
        
        # 1. Format the Source Mask
        # The existing Transformer expects src_mask of shape (B, 1, 1, Src_Seq_Len)
        if src_mask is not None:
            src_mask = src_mask[:, None, None, :]

        # 2. Create the Causal Target Mask
        # Prevents the decoder from looking at future tokens
        tgt_mask = create_target_mask(tgt_seq_len, device)

        # 3. Forward pass
        logits = self.transformer(
            src=input_ids,
            tgt=decoder_input_ids,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        return logits