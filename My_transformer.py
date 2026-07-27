# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Self-Attention Head
# ----------------------------
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # FIX: scale by sqrt(head_size), not sqrt(embedding_dim)
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out

# ----------------------------
# Multi-Head Attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            [SelfAttentionHead(embedding_dim, block_size, head_size, dropout)
             for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

# ----------------------------
# Feed Forward Network
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# My Transformer
# ----------------------------
class Trans(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads, dropout)
        self.ffwd = FeedForward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
