"""
model.py — Your own GPT-style AI model, built from scratch!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """This is the 'brain' that lets the AI focus on important words."""
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_size // heads

        self.keys    = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values  = nn.Linear(embed_size, embed_size)
        self.out     = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = self.queries(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.values(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        # Mask future tokens (so the AI can't "cheat" by looking ahead)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    """One 'layer' of thinking for the AI."""
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.attn  = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff    = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class MyGPT(nn.Module):
    """
    Your own GPT-style AI!
    
    Config:
        vocab_size  — how many unique characters/words it knows
        embed_size  — how 'smart' each token is (bigger = smarter but slower)
        num_heads   — parallel attention streams
        num_layers  — depth of thinking (more = smarter but slower)
        max_len     — max context length (how much it can 'remember')
    """
    def __init__(self, vocab_size, embed_size=256, num_heads=8, num_layers=6, max_len=256, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed   = nn.Embedding(max_len, embed_size)
        self.blocks      = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.max_len = max_len

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        """Generate text from a starting prompt."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling (keeps output creative but coherent)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
