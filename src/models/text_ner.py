import torch
import torch.nn as nn
from .text import TextTransformerBlock

class TextBackboneNER(nn.Module):
    """
    Text Transformer backbone that returns token sequences for Token Classification (NER).
    """
    def __init__(
            self, vocab_size=30522, max_seq_len=128,
            dim=384, depth=6, num_heads=6, attn_type="vanilla"
            ):
        super().__init__()
        # Standard embedding
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        
        self.blocks = nn.ModuleList(
            [
                TextTransformerBlock(dim, num_heads, attn_type)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids, attention_mask=None):
        B, seq_len = input_ids.shape
        
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(B, seq_len)
        
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
