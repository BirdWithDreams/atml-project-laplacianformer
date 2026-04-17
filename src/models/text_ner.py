import torch
import torch.nn as nn
from .text import TextTransformerBlock

class TextBackboneNER(nn.Module):
    """
    Text Transformer backbone that returns token sequences for Token Classification (NER).
    """
    def __init__(
            self, vocab_size=30522, max_seq_len=128,
            dim=384, depth=6, num_heads=6, attn_type="vanilla",
            lambda_scale=4.0, pool_ratio=2, ns_iters=5
            ):
        super().__init__()
        # Standard embedding
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        attn_kwargs = {
            "lambda_scale": lambda_scale,
            "pool_ratio": pool_ratio,
            "ns_iters": ns_iters,
        }
        
        self.blocks = nn.ModuleList(
            [
                TextTransformerBlock(dim, num_heads, attn_type, attn_kwargs)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids, attention_mask=None):
        B, seq_len = input_ids.shape
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.pos_embed.num_embeddings}"
            )
        
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(B, seq_len)
        
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x
