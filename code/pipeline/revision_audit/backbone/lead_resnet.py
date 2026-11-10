"""Revision backbone: lead-aware 1D residual encoder + masked attention-pooling receiver.

Transmitter side (per lead l, run independently on each available lead):
    x_l [1, T] --stem conv--> h + e_lead[l] (learned lead-ID bias, broadcast over time)
    --> residual stages --> global average pool --> Linear --> z_l in R^32
  The lead ID enters only at the transmitter, as an additive learned embedding
  after the stem, so the transmitted 32-dim z_l (16 complex symbols) already
  carries the lead identity in its content; nothing about the lead index is
  sent separately.

Receiver side (sees only the received z_l of the transmitted leads + a mask):
    masked attention pooling: a_l = w^T tanh(W z_l); softmax over the present
    leads only (absent leads get -inf); fused = sum_l a_l z_l
    --> MLP head 32 -> 64 -> 5 logits.

Missing/untransmitted leads are never zero-filled into the pool: the mask
removes them from the softmax. Training samples a random lead subset per
record (lead dropout) so one model supports every subset size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        s = x if self.short is None else self.short(x)
        return self.act(self.b2(self.c2(self.act(self.b1(self.c1(x))))) + s)


class LeadAwareEncoder(nn.Module):
    def __init__(self, widths=(32, 64, 96, 128), blocks_per_stage=2, latent_dim=32, num_leads=12):
        super().__init__()
        c0 = widths[0]
        self.stem = nn.Sequential(nn.Conv1d(1, c0, 15, 2, 7, bias=False), nn.BatchNorm1d(c0), nn.ReLU(inplace=True))
        self.lead_emb = nn.Embedding(num_leads, c0)
        nn.init.normal_(self.lead_emb.weight, std=0.1)
        layers, cin = [], c0
        for i, w in enumerate(widths):
            for j in range(blocks_per_stage):
                layers.append(Block(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
        self.body = nn.Sequential(*layers)
        self.proj = nn.Linear(cin, latent_dim)
        self.latent_dim = latent_dim

    def encode_pairs(self, x_pairs, lead_ids):
        """x_pairs: [N, T] single-lead signals, lead_ids: [N] -> [N, d]. Each lead is encoded on its own."""
        h = self.stem(x_pairs.unsqueeze(1))
        h = self.body(h + self.lead_emb(lead_ids).unsqueeze(-1))
        return self.proj(h.mean(-1))

    def forward(self, x, lead_idx=None):
        """x: [B, L, T] -> z [B, L, d]. lead_idx: [L] lead ids of the L columns (default 0..L-1)."""
        b, l, t = x.shape
        if lead_idx is None:
            lead_idx = torch.arange(l, device=x.device)
        h = self.stem(x.reshape(b * l, 1, t))
        e = self.lead_emb(lead_idx).repeat(b, 1).unsqueeze(-1)  # [B*L, C, 1]
        h = self.body(h + e)
        return self.proj(h.mean(-1)).reshape(b, l, self.latent_dim)


class AttnPoolReceiver(nn.Module):
    def __init__(self, latent_dim=32, hidden=64, num_classes=5, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(hidden, num_classes))

    def pool(self, z, mask):
        a = self.score(z).squeeze(-1).masked_fill(mask <= 0, float("-inf"))
        w = torch.softmax(a, dim=1) * (mask > 0)  # exact zeros for absent leads
        return (w.unsqueeze(-1) * z).sum(1)

    def forward(self, z, mask):
        return self.head(self.pool(z, mask))


class LeadAwareModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.encoder = LeadAwareEncoder(**kw)
        self.receiver = AttnPoolReceiver(latent_dim=self.encoder.latent_dim)

    def forward(self, x, mask):
        """Encodes only the leads with mask==1 (the transmitted ones); absent leads never enter the encoder."""
        b, l, t = x.shape
        sel = mask > 0
        lead_ids = torch.arange(l, device=x.device).expand(b, l)[sel]
        z = x.new_zeros(b, l, self.encoder.latent_dim)
        z[sel] = self.encoder.encode_pairs(x[sel], lead_ids)
        return self.receiver(z, mask)
