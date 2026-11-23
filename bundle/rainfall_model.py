"""Operational (v3) gated multimodal fusion network. Shared by the Colab notebook
(written there as rainfall_model.py) and the portal, so saved weights load unchanged.

Modalities: 'nwp' (vector, MLP encoder), 'imd' and 'sat' (5-day sequences, GRU encoders),
'img' (MODIS 32x32 RGB, CNN encoder). Context: learned region embedding + day-of-year sin/cos."""
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, n_in, emb=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, emb), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class SeqEncoder(nn.Module):
    def __init__(self, n_in, hidden=32, emb=32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, emb), nn.ReLU())

    def forward(self, x):
        _, h = self.gru(x)
        return self.out(h[-1])


class ImageEncoder(nn.Module):
    def __init__(self, emb=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.out = nn.Sequential(nn.Linear(32, emb), nn.ReLU())

    def forward(self, img):
        return self.out(self.net(img).flatten(1))


class FusionNet(nn.Module):
    """modalities: ordered list from {'nwp','imd','sat','img'}; dims: n features per tabular modality."""

    def __init__(self, modalities, dims, n_regions=4, ctx_dim=2, fusion="gated", use_region=True,
                 emb=32, region_dim=8):
        super().__init__()
        self.modalities, self.fusion, self.use_region = list(modalities), fusion, use_region
        enc = {}
        for m in self.modalities:
            if m == "img":
                enc[m] = ImageEncoder(emb)
            elif m == "nwp":
                enc[m] = MLPEncoder(dims[m], emb)
            else:
                enc[m] = SeqEncoder(dims[m], emb=emb)
        self.encoders = nn.ModuleDict(enc)
        n_mod = len(self.modalities)
        self.region_emb = nn.Embedding(n_regions, region_dim) if use_region else None
        ctx_in = ctx_dim + (region_dim if use_region else 0)
        if fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(n_mod * emb + ctx_in, 64), nn.ReLU(), nn.Linear(64, n_mod))
            trunk_in = emb + ctx_in
        else:
            trunk_in = n_mod * emb + ctx_in
        self.trunk = nn.Sequential(nn.Linear(trunk_in, 64), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(64, 32), nn.ReLU())
        self.reg_head, self.cls_head = nn.Linear(32, 1), nn.Linear(32, 1)

    def forward(self, b):
        E = torch.stack([self.encoders[m](b[m]) for m in self.modalities], 1)          # (B, M, emb)
        c = torch.cat([b["ctx"], self.region_emb(b["rid"])], 1) if self.use_region else b["ctx"]
        gates = None
        if self.fusion == "gated":
            gates = torch.softmax(self.gate(torch.cat([E.flatten(1), c], 1)), dim=1)
            fused = (gates.unsqueeze(-1) * E).sum(1)
        else:
            fused = E.flatten(1)
        h = self.trunk(torch.cat([fused, c], 1))
        return self.reg_head(h).squeeze(-1), self.cls_head(h).squeeze(-1), gates
