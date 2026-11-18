import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio_data import AgeDataset, FullEpochBatches
from benchmark import LISTS
from model import AgeRegressor
from evaluation import validate, evaluate
from runtime import RunTimer, write_json


def loss_value(pred, target, kind):
    if kind == "mse":
        return (pred - target).square().mean()
    p = pred - pred.mean()
    t = target - target.mean()
    denom = p.square().mean() + t.square().mean() + (pred.mean() - target.mean()).square()
    return 1 - 2 * (p * t).mean() / denom.clamp_min(1e-8)


def update(model, optimizer, batch, device, loss_name):
    wave, target = (v.to(device) for v in batch)
    optimizer.zero_grad(set_to_none=True)
    pred = model(wave)
    loss = loss_value(pred, target, loss_name)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), norm.item()



def main(args, timer):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Check NVIDIA Container Toolkit and GPU_ID.")
    if not args.weights.is_file():
        raise SystemExit(f"Missing {args.weights}. Provide the local checkpoint for --extractor; download it with tools/download_weights.py.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    splits = args.splits.resolve(strict=True)
    audit = json.loads((splits / "audit.json").read_text())
    hashes = {n: hashlib.sha256((splits / f"{n}.txt").read_bytes()).hexdigest() for n in LISTS}
    if hashes != audit["prepared_sha256"]:
        raise ValueError("Prepared lists differ from the audit")
    partition = json.loads((splits / "partition.json").read_text())
    if partition["source_sha256"] != hashes:
        raise ValueError("Partition does not match prepared benchmark")
    partition_hashes = {n: hashlib.sha256((splits / f"{n}.txt").read_bytes()).hexdigest()
                        for n in ("train", "timit_val", "nnces_val", "agevoxceleb_val")}
    if partition_hashes != partition["files_sha256"]:
        raise ValueError("Train/validation files changed")
    dataset = AgeDataset(splits / "train.txt", training=True, seconds=args.seconds)
    model = AgeRegressor(args.extractor, args.weights, sum(r[-1] for r in dataset.rows)/len(dataset)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.smoke:
        i = next(i for i,r in enumerate(dataset.rows) if r[-1] != dataset.rows[0][-1])
        batch = torch.utils.data.default_collate([dataset[0], dataset[i]])
        loss, norm = update(model, optimizer, batch, args.device, args.loss)
        if norm == 0:
            raise ValueError("Zero gradient")
        model.eval()
        with torch.no_grad():
            print("loss", loss, "gradient_norm", norm, "predictions", model(batch[0].to(args.device)).tolist())
        print("Smoke optimizer: AdamW; partition hashes verified", flush=True)
        return

    run = args.run
    ckpt = run / "last.pt"
    signature = {k: v for k, v in vars(args).items()
                 if k not in ("weights", "splits", "run", "resume", "smoke")}
    signature.update(protocol="speaker_validation_v1", selection="validation_macro_mae", partition_sha256=partition_hashes, optimizer="adamw", splits_sha256=hashes, weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_score = float("inf")
    best_epoch = None
    if args.resume:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state["config"] != signature:
            raise ValueError("Resume configuration or input files changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["random"])
        torch.set_rng_state(state["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        history = state["history"]
        best_score = state["best_score"]
        best_epoch = state["best_epoch"]
        best_weights = state["best_model"]
        del state
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", signature)
    timer.active = True
    timer.save()
    print(f"Training: {len(dataset)} records; AdamW; {args.epochs} epochs; validation each epoch", flush=True)
    loader = DataLoader(dataset, batch_sampler=FullEpochBatches(len(dataset), args.batch_size),
                        num_workers=args.workers)
    for epoch in range(len(history) + 1, args.epochs + 1):
        model.train()
        total = count = 0
        with timer.stage("train", epoch=epoch):
            for batch in loader:
                value, _ = update(model, optimizer, batch, args.device, args.loss)
                n = len(batch[1])
                total += value * n
                count += n
        print(f"Epoch {epoch}: train loss={total/count:.6f}; validating...", flush=True)
        with timer.stage("validation", epoch=epoch):
            val = validate(model, splits, args)
        history.append(dict(epoch=epoch, loss=total/count, lr=optimizer.param_groups[0]["lr"], n=count,
                            validation=val))
        if val["macro_mae"] < best_score:
            best_score, best_epoch = val["macro_mae"], epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_tmp = run / "best.tmp"
            torch.save(dict(config=signature, model=best_weights, epoch=best_epoch,
                            validation_macro_mae=best_score), best_tmp)
            best_tmp.replace(run / "best.pt")
        scheduler.step()
        tmp = ckpt.with_suffix(".tmp")
        torch.save(dict(config=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), history=history, best_score=best_score,
                        best_epoch=best_epoch, best_model=best_weights, random=random.getstate(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if args.device == "cuda" else None), tmp)
        tmp.replace(ckpt)
        write_json(run / "history.json", history)
        print(history[-1], flush=True)
    write_json(run / "history.json", history)
    # last.pt also contains best weights, making resume robust to interrupted writes.
    model.load_state_dict(best_weights)
    torch.save(dict(config=signature, model=best_weights, epoch=best_epoch,
                    validation_macro_mae=best_score), run / "best.tmp")
    (run / "best.tmp").replace(run / "best.pt")
    write_json(run / "selection.json", dict(epoch=best_epoch, validation_macro_mae=best_score))
    print(f"Testing best epoch {best_epoch} (validation macro MAE={best_score:.4f})", flush=True)
    with timer.stage("test"):
        evaluate(model, splits, run, args)
    print("Training and all five tests completed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extractor", default="whisper_tiny")
    p.add_argument("--splits", type=Path, default=Path("/output/validation_seed42"))
    p.add_argument("--weights", type=Path, default=Path("/weights/tiny.pt"))
    p.add_argument("--run", type=Path, default=Path("/output/runs/whisper_tiny_ccc"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--loss", choices=["ccc", "mse"], default="ccc")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seconds", type=float, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not (0.1 <= args.seconds <= 30 and args.epochs > 0 and args.batch_size >= 2 and args.workers >= 0):
        p.error("Invalid duration, epoch count, batch size or worker count")
    with RunTimer(args.run, args.device, enabled=not args.smoke) as timer:
        main(args, timer)
