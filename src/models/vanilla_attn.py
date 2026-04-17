import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. Embeddings & Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the
    tokens in the sequence. Uses sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the positional encodings once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) to allow broadcasting across batch size
        pe = pe.unsqueeze(0)

        # Register as a buffer so it is saved in the state_dict but not updated by gradients
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Standard embedding layer multiplied by sqrt(d_model) as specified in the paper."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Computes attention weights and applied them to values."""
        # q, k, v shape: (batch_size, num_heads, seq_len, d_k)

        # Matrix multiply Q and K^T, then scale
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Apply mask (e.g., to prevent looking ahead in the decoder)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multiply weights by V
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear projections and split into heads: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            if mask.dim() == 2:
                # Key padding mask: (B, N) -> (B, 1, 1, N)
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                # Explicit attention mask: (B, N, N) -> (B, 1, N, N)
                mask = mask[:, None, :, :]

        # 2. Apply attention
        x, attn = self.scaled_dot_product_attention(q, k, v, mask)

        # 3. Concatenate heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final linear projection
        return self.w_o(x)


# ---------------------------------------------------------------------------
# 3. Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    """Two linear transformations with a ReLU activation in between."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original paper used ReLU
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 4. Encoder and Decoder Layers
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Consists of Multi-Head Self-Attention and a Position-wise Feed-Forward Network.
    Uses "Post-Layer Normalization" (Sublayer -> Add & Norm) as in the original paper.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Self-Attention + Add & Norm
        attn_out = self.self_attn(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Feed-Forward + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class DecoderLayer(nn.Module):
    """
    Consists of Masked Self-Attention, Cross-Attention (to Encoder output), and FFN.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, x: torch.Tensor, enc_output: torch.Tensor,
            src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None
            ) -> torch.Tensor:
        # 1. Masked Self-Attention (prevents attending to future tokens)
        attn_out = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-Attention (Queries from Decoder, Keys/Values from Encoder)
        cross_out = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


# ---------------------------------------------------------------------------
# 5. The Complete Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    The complete model consisting of an Encoder, a Decoder, and a final linear layer.
    """

    def __init__(
            self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
            num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048,
            max_len: int = 5000, dropout: float = 0.1
            ):
        super().__init__()

        # Source and Target Embeddings + Positional Encodings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stacked Encoder Layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Stacked Decoder Layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final linear projection to target vocabulary size
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.src_embedding(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
            self, tgt: torch.Tensor, enc_output: torch.Tensor,
            src_mask: torch.Tensor, tgt_mask: torch.Tensor
            ) -> torch.Tensor:
        x = self.pos_encoding(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(
            self, src: torch.Tensor, tgt: torch.Tensor,
            src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None
            ) -> torch.Tensor:

        # Generate the encoded representation of the source
        enc_output = self.encode(src, src_mask)

        # Generate the decoded representation
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)

        # Map back to vocabulary dimensions
        return self.generator(dec_output)


# ---------------------------------------------------------------------------
# 6. Utility: Mask Generation
# ---------------------------------------------------------------------------

def create_target_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Creates a lower-triangular mask for the target sequence to prevent
    the model from looking ahead at future tokens.
    """
    # Shape: (1, size, size)
    mask = torch.tril(torch.ones((size, size), device=device)).unsqueeze(0)
    return mask  # 1s mean "allow to attend", 0s mean "mask out"
