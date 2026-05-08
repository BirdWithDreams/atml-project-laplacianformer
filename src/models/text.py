import torch
import torch.nn as nn
from .vanilla_attn import MultiHeadAttention
from .laplacian_1d_attn import CudaLaplacian1DLinearAttention

class TextTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, attn_type="vanilla", attn_kwargs=None, dropout=0.0):
        super().__init__()
        self.attn_type = attn_type
        attn_kwargs = attn_kwargs or {}
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        if attn_type == "vanilla":
            self.attn = MultiHeadAttention(d_model=dim, num_heads=num_heads)
        elif attn_type == "laplacian":
            laplacian_backend = attn_kwargs.get("laplacian_backend", "cuda_1d")
            if laplacian_backend != "cuda_1d":
                raise ValueError("Text Laplacian attention supports only laplacian_backend='cuda_1d'")

            self.attn = CudaLaplacian1DLinearAttention(
                dim=dim,
                num_heads=num_heads,
                lambda_scale=attn_kwargs.get("lambda_scale", 4.0),
                pool_ratio=attn_kwargs.get("pool_ratio", 2),
                ns_iters=attn_kwargs.get("ns_iters", 5),
            )
        else:
            raise ValueError("attn_type must be 'vanilla' or 'laplacian'")

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, attention_mask=None):
        if self.attn_type == "vanilla":
            norm_x = self.norm1(x)
            x = x + self.dropout(self.attn(norm_x, norm_x, norm_x, attention_mask))
        else:
            x = x + self.dropout(self.attn(self.norm1(x), attention_mask=attention_mask))

        x = x + self.dropout(self.mlp(self.norm2(x)))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class TextBackbone(nn.Module):
    """
    Text Transformer backbone that returns pooled token representation.
    """
    def __init__(
            self, vocab_size=30522, max_seq_len=128,
            dim=384, depth=6, num_heads=6, attn_type="vanilla",
            lambda_scale=4.0, pool_ratio=2, ns_iters=5,
            laplacian_backend="cuda_1d",
            dropout=0.0,
            ):
        super().__init__()
        # Standard embedding
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.embed_dropout = nn.Dropout(dropout)

        attn_kwargs = {
            "lambda_scale": lambda_scale,
            "pool_ratio": pool_ratio,
            "ns_iters": ns_iters,
            "laplacian_backend": laplacian_backend,
        }
        
        # [CLS] token equivalent is usually first token or pooled. In BERT, it's the first token.
        self.blocks = nn.ModuleList(
            [
                TextTransformerBlock(dim, num_heads, attn_type, attn_kwargs, dropout)
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
        
        x = self.embed_dropout(self.token_embed(input_ids) + self.pos_embed(positions))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        # Use simple average pooling over the sequence for regression/classification, 
        # or we could return x[:, 0] if a CLS token was explicitly prepended. 
        # For simplicity, we use mean pooling masked by attention_mask.
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(x).float()
            sum_embeddings = torch.sum(x * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = x.mean(dim=1)
            
        return pooled_output
