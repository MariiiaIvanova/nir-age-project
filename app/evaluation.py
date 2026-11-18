"""Validation metrics calculated per corpus; equal corpus weight for selection."""
import csv
import math
from collections import defaultdict
from benchmark import TESTS
from runtime import write_json
import torch
from torch.utils.data import DataLoader
from audio_data import AgeDataset

CORPORA = ('timit', 'nnces', 'agevoxceleb')


@torch.inference_mode()
def validate(model, splits, args):
    model.eval()
    metrics = {}
    for corpus in CORPORA:
        ds = AgeDataset(splits / f'{corpus}_val.txt', seconds=args.seconds)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed))
        predictions, targets = [], []
        for wave, age in loader:
            predictions.extend(model(wave.to(args.device)).cpu().tolist())
            targets.extend(age.tolist())
        p, t = torch.tensor(predictions, dtype=torch.float64), torch.tensor(targets, dtype=torch.float64)
        if not len(t) or not torch.isfinite(p).all():
            raise ValueError(f'Invalid validation predictions: {corpus}')
        delta = p - t
        denominator = p.var(unbiased=False) + t.var(unbiased=False) + (p.mean()-t.mean()).square()
        ccc = 2*((p-p.mean())*(t-t.mean())).mean()/denominator.clamp_min(1e-8)
        metrics[corpus] = dict(n=len(t), mae=delta.abs().mean().item(),
                              rmse=math.sqrt(delta.square().mean().item()),
                              bias=delta.mean().item(), ccc=ccc.item())
    return dict(macro_mae=sum(m['mae'] for m in metrics.values())/len(metrics), corpora=metrics)


@torch.inference_mode()
def evaluate(model, splits, run, args):
    model.eval()
    metrics = {}
    for corpus in TESTS:
        print(f"Testing {corpus}...", flush=True)
        dataset = AgeDataset(splits / f"{corpus}_test.txt", seconds=args.seconds)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        total = squared = signed = 0.
        speakers = defaultdict(list)
        offset = 0
        tmp = run / f"{corpus}.csv.tmp"
        with tmp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["speaker", "path", "age", "prediction"])
            for wave, _ in loader:
                for pred in model(wave.to(args.device)).cpu().tolist():
                    if not math.isfinite(pred):
                        raise ValueError(f"Nonfinite prediction on {corpus}")
                    _, speaker, path, age = dataset.rows[offset]
                    delta = pred - age
                    total += abs(delta)
                    squared += delta * delta
                    signed += delta
                    speakers[speaker].append(abs(delta))
                    writer.writerow([speaker, path, age, pred])
                    offset += 1
        tmp.replace(run / f"{corpus}.csv")
        metrics[corpus] = dict(n=offset, mae=total / offset, rmse=math.sqrt(squared / offset),
                               bias=signed / offset,
                               speaker_mae=sum(sum(v)/len(v) for v in speakers.values())/len(speakers))
        print(f"Test {corpus}: {metrics[corpus]}", flush=True)
        write_json(run / "metrics.json", metrics)
