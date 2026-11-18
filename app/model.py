from pathlib import Path

import torch
from torch import nn
from extractors import build_extractor


class RegAgeHead2(nn.Module):
    def __init__(self, in_size=512):
        super().__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm(in_size), nn.Linear(in_size, 1024), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(1024, 256), nn.SiLU(), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.linear(x).clamp(0, 100)


class AgeRegressor(nn.Module):
    def __init__(self, extractor, weights, train_mean):
        super().__init__()
        self.encoder = build_extractor(extractor, Path(weights))
        self.tdnn = nn.Sequential(nn.Conv1d(self.encoder.output_dim, 1024, 1),
                                  nn.ReLU(), nn.Conv1d(1024, 1024, 1))
        self.maxout_a = nn.Linear(2048, 512)
        self.maxout_b = nn.Linear(2048, 512)
        self.norm = nn.BatchNorm1d(512, affine=False)
        self.head = RegAgeHead2()
        nn.init.constant_(self.head.linear[-1].bias, train_mean)

    def forward(self, wave):
        x = self.tdnn(self.encoder(wave))
        pooled = torch.cat((x.mean(-1), x.std(-1, unbiased=False)), dim=1)
        segment = torch.maximum(self.maxout_a(pooled), self.maxout_b(pooled))
        return self.head(self.norm(segment)).squeeze(-1)
