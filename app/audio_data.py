import math
import random

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


def read_audio(path):
    waveform, rate = torchaudio.load(str(path), backend="ffmpeg")
    if rate <= 0 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    return waveform, rate


class AgeDataset(Dataset):
    def __init__(self, path, training=False, seconds=4):
        self.samples = int(seconds * 16000)
        self.training = training
        self.rows = []
        with path.open() as f:
            for line in f:
                corpus, speaker, rest = line.strip().split(maxsplit=2)
                path, age = rest.rsplit(maxsplit=1)
                self.rows.append((corpus, speaker, path, float(age)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        _, _, path, age = self.rows[index]
        waveform, rate = read_audio(path)
        x = waveform.mean(dim=0)
        if rate != 16000:
            x = torchaudio.functional.resample(x, rate, 16000)
        extra = len(x) - self.samples
        if extra > 0:
            start = random.randint(0, extra) if self.training else extra // 2
            x = x[start:start + self.samples]
        elif extra < 0:
            x = torch.nn.functional.pad(x, (0, -extra))
        peak = x.abs().max()
        if peak > 1e-6:
            x = x / peak
        return x, torch.tensor(age, dtype=torch.float32)


class FullEpochBatches(Sampler):
    """Merge a final singleton into the preceding batch (BatchNorm and CCC)."""
    def __init__(self, size, batch):
        if min(size, batch) < 2:
            raise ValueError("Training requires at least two examples per batch")
        self.size = size
        self.batch = batch

    def __iter__(self):
        order = torch.randperm(self.size).tolist()
        start = 0
        while start < self.size:
            end = start + self.batch
            if end == self.size - 1:
                end += 1
            yield order[start:end]
            start = end

    def __len__(self):
        return math.ceil(self.size / self.batch) - int(self.size % self.batch == 1)
