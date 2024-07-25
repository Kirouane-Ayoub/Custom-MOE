# Causal scaled dot product self-Attention Head
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings
from dataset import vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(settings.n_embd, head_size, bias=False)
        self.query = nn.Linear(settings.n_embd, head_size, bias=False)
        self.value = nn.Linear(settings.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(settings.block_size, settings.block_size))
        )

        self.dropout = nn.Dropout(settings.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out


# Multi-Headed Self Attention
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(settings.n_embd, settings.n_embd)
        self.dropout = nn.Dropout(settings.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Expert module
class Expert(nn.Module):
    """An MLP representing each expert"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(settings.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Router module
class Router(nn.Module):
    def __init__(self, n_embed: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output: torch.Tensor):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


# Sparse Mixture of Experts (SMoE)
class MoE(nn.Module):
    def __init__(self, n_embed: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = Router(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


# Transformer Block with Multi-Head Attention and SMoE
class Block(nn.Module):
    """Mixture of Experts Transformer block"""

    def __init__(self, n_embed: int, n_head: int, num_experts: int, top_k: int):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = MoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x


# Sparse MoE Language Model
class SMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, settings.n_embd)
        self.position_embedding_table = nn.Embedding(
            settings.block_size, settings.n_embd
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    settings.n_embd,
                    settings.n_head,
                    settings.num_experts,
                    settings.top_k,
                )
                for _ in range(settings.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(settings.n_embd)
        self.lm_head = nn.Linear(settings.n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -settings.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
