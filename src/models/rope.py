import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _build_1d_rope_cache(
        positions: torch.Tensor, dim: int, base: float, dtype: torch.dtype
        ) -> tuple[torch.Tensor, torch.Tensor]:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim)
    )
    freqs = torch.einsum("n,d->nd", positions.to(torch.float32), inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype=dtype)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype=dtype)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def apply_2d_rope(
        x: torch.Tensor, height: int, width: int, base: float = 10000.0
        ) -> torch.Tensor:
    """
    Applies axial 2D rotary embeddings to spatial tokens.

    Args:
        x: Tensor of shape (B, heads, N, head_dim) with N = height * width.
    """
    if x.shape[-2] != height * width:
        raise ValueError(f"Expected {height * width} tokens, got {x.shape[-2]}")

    head_dim = x.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError(f"2D RoPE requires head_dim divisible by 4, got {head_dim}")

    half_dim = head_dim // 2
    y_positions = torch.arange(height, device=x.device).repeat_interleave(width)
    x_positions = torch.arange(width, device=x.device).repeat(height)

    cos_y, sin_y = _build_1d_rope_cache(y_positions, half_dim, base, x.dtype)
    cos_x, sin_x = _build_1d_rope_cache(x_positions, half_dim, base, x.dtype)

    y_part = x[..., :half_dim]
    x_part = x[..., half_dim:]

    y_rot = y_part * cos_y + _rotate_half(y_part) * sin_y
    x_rot = x_part * cos_x + _rotate_half(x_part) * sin_x
    return torch.cat((y_rot, x_rot), dim=-1)
