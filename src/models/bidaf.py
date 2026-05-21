# BiDAF (Seo et al., 2017) reimplementation, adapted for the three tiers.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BiDAFConfig:
    vocab_size: int
    emb_dim: int = 300
    hidden: int = 100
    dropout: float = 0.2
    task: str = "extractive" 
    pad_id: int = 0


class Highway(nn.Module):
    def __init__(self, dim: int, n_layers: int = 2):
        super().__init__()
        self.gates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.transforms = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for g, t in zip(self.gates, self.transforms):
            gate = torch.sigmoid(g(x))
            trans = F.relu(t(x))
            x = gate * trans + (1 - gate) * x
        return x


class BiDAFAttention(nn.Module):

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.w_h = nn.Linear(dim, 1, bias=False) 
        self.w_u = nn.Linear(dim, 1, bias=False) 
        self.w_hu = nn.Parameter(torch.empty(dim))
        nn.init.xavier_uniform_(self.w_hu.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H: torch.Tensor, 
        U: torch.Tensor, 
        ctx_mask: torch.Tensor, 
        q_mask: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = H.shape
        J = U.shape[1]
        s1 = self.w_h(H) 
        s2 = self.w_u(U).transpose(1, 2)          
        s3 = torch.einsum("btd,bjd,d->btj", H, U, self.w_hu) 
        S = s1 + s2 + s3 

        # Context2Query
        q_mask_b = q_mask.unsqueeze(1).to(S.dtype)
        S_q = S.masked_fill(q_mask_b == 0, -1e9)
        a = F.softmax(S_q, dim=-1)  
        U_tilde = torch.bmm(a, U) 

        # Query2Context
        S_c_mask = (q_mask_b == 0).to(S.dtype) * -1e9
        b_logits = (S + S_c_mask).max(dim=-1).values
        ctx_mask_b = ctx_mask.to(S.dtype)
        b = F.softmax(b_logits.masked_fill(ctx_mask_b == 0, -1e9), dim=-1)
        H_tilde = torch.einsum("bt,btd->bd", b, H).unsqueeze(1).expand(-1, T, -1)

        G = torch.cat([H, U_tilde, H * U_tilde, H * H_tilde], dim=-1)
        return self.dropout(G), S


class BiDAF(nn.Module):
    def __init__(self, cfg: BiDAFConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden
        self.embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.proj = nn.Linear(cfg.emb_dim, 2 * D)
        self.highway = Highway(2 * D, n_layers=2)
        self.ctx_lstm = nn.LSTM(2 * D, D, batch_first=True, bidirectional=True)
        self.q_lstm = nn.LSTM(2 * D, D, batch_first=True, bidirectional=True)
        self.attn = BiDAFAttention(2 * D, dropout=cfg.dropout)
        self.model_lstm1 = nn.LSTM(8 * D, D, batch_first=True, bidirectional=True)
        self.model_lstm2 = nn.LSTM(2 * D, D, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.task == "extractive":
            self.start_lin = nn.Linear(10 * D, 1)
            self.end_lin = nn.Linear(10 * D, 1)
        elif cfg.task == "boolq":
            self.cls = nn.Linear(8 * D, 2)
        else:
            raise ValueError(cfg.task)

    @staticmethod
    def _pack_lstm(lstm: nn.LSTM, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return out

    def init_embeddings(self, vectors: torch.Tensor, freeze: bool = True) -> None:
        assert vectors.shape == self.embed.weight.shape, \
            f"emb shape mismatch: vectors {vectors.shape} vs embed {self.embed.weight.shape}"
        with torch.no_grad():
            self.embed.weight.copy_(vectors)
        if freeze:
            self.embed.weight.requires_grad_(False)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
        x = self.embed(ids)
        x = self.proj(x)
        x = self.highway(x)
        x = self.dropout(x)
        return self._pack_lstm(lstm, x, mask)

    def forward(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
    ):
        H = self.encode(ctx_ids, ctx_mask, self.ctx_lstm)
        U = self.encode(q_ids, q_mask, self.q_lstm)
        G, S = self.attn(H, U, ctx_mask, q_mask)
        M1 = self._pack_lstm(self.model_lstm1, G, ctx_mask)
        M2 = self._pack_lstm(self.model_lstm2, M1, ctx_mask)

        if self.cfg.task == "extractive":
            start = self.start_lin(torch.cat([G, M1], dim=-1)).squeeze(-1)
            end = self.end_lin(torch.cat([G, M2], dim=-1)).squeeze(-1)
            start = start.masked_fill(ctx_mask == 0, -1e9)
            end = end.masked_fill(ctx_mask == 0, -1e9)
            return start, end, S

        mask = ctx_mask.to(G.dtype).unsqueeze(-1)
        G_pool = (G * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.cls(G_pool)
        return logits, S
